#!/usr/bin/env python3
"""
Stage 2: Extract purified attention + compute hallucination detection scores + AUROC.

Loads generated captions, runs a single forward pass per image with DetectionAdapter
injected, collects original + purified attention and sink stats, computes per-object
detection scores, and evaluates AUROC using CHAIR labels.
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add detection src to path
DETECTION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = DETECTION_ROOT.parent
sys.path.insert(0, str(DETECTION_ROOT / "src"))

# Also add PAS src for CHAIR import (PAS sits as a sibling project)
PAS_SRC = PROJECT_ROOT.parent / "pas" / "src"
if str(PAS_SRC) not in sys.path:
    sys.path.insert(0, str(PAS_SRC))

from sinkdetect.adapter import DetectionAdapter, PER_HEAD_LAYERS, get_llm_layers, inject_detection_adapter
from sinkdetect.chair import (
    add_token_labels,
    evaluate_chair,
    find_first_mentions,
    load_chair_evaluator,
)
from sinkdetect.grounding import _safe_normalize, _sinks_within_visual, _strip_sinks
from sinkdetect.scoring import compute_all_scores, compute_auroc
from sinkdetect.sink_utils import find_vis_bounds
from sinkdetect.utils import (
    build_caption_prompt,
    load_generation_json,
    load_model_and_processor,
    partition_tokens,
)


def _set_adapter_bounds(layers, vis_start: int, vis_end: int) -> None:
    """Push dynamic visual-token bounds into every DetectionAdapter layer."""
    for layer in layers:
        adapter = layer.self_attn
        if isinstance(adapter, DetectionAdapter):
            adapter.vis_start = vis_start
            adapter.vis_end = vis_end
            adapter.text_start = vis_end

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Purified attention hallucination detection")
    p.add_argument("--model_path", type=str,
                   default="/data/common_model/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/")
    p.add_argument("--coco_path", type=str, default="/data/common_dataset/coco-2014-dataset/")
    p.add_argument("--generation_json", type=str, required=True,
                   help="Path to Stage 1 output (generation.json)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--chair_pkl", type=str,
                   default=str(PROJECT_ROOT.parent / "pas" / "data" / "chair_coco.pkl"))
    p.add_argument("--ratio", type=float, default=0.5,
                   help="Top-mass visual masking ratio for purification")
    p.add_argument("--start_layer", type=int, default=0)
    p.add_argument("--end_layer", type=int, default=32,
                   help="Number of layers in the model (32 for LLaVA-1.5-7B)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of images to process (0 = all)")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="This shard's index (0..num_shards-1). Slices generation.json.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of parallel shards. When >1, output files "
                        "get a _shard{i} suffix so the merge step can pick them up.")
    p.add_argument("--save_shape_cache", action="store_true",
                   help="Save per-mention visual object/null rows for fast "
                        "metric recomputation without another model forward.")
    p.add_argument("--compute_no_rope_attention", action="store_true",
                   help="Also recompute attention from pre-RoPE Q/K and cache/score "
                        "no_rope_* shape branches. Slower and more memory intensive.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load generation data ──
    logger.info("Loading generation data from %s", args.generation_json)
    gen_data = load_generation_json(args.generation_json)
    if args.limit > 0:
        gen_data = gen_data[:args.limit]
    if args.num_shards > 1:
        gen_data = gen_data[args.shard_idx :: args.num_shards]
        logger.info("  shard %d/%d: %d entries", args.shard_idx, args.num_shards, len(gen_data))
    else:
        logger.info("  %d entries loaded", len(gen_data))

    shard_suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""

    # ── CHAIR evaluation ──
    logger.info("Loading CHAIR evaluator from %s", args.chair_pkl)
    evaluator = load_chair_evaluator(args.chair_pkl)

    # Prepare CHAIR input
    chair_data = [
        {"image_id": entry["image_id"], "caption": entry["caption"]}
        for entry in gen_data
    ]
    logger.info("Running CHAIR evaluation on %d captions ...", len(chair_data))
    chair_input_path = output_dir / f"chair_input{shard_suffix}.json"
    eval_dicts, chair_metrics = evaluate_chair(
        evaluator, data=chair_data, json_path=str(chair_input_path)
    )
    logger.info("  CHAIRi=%.4f, CHAIRs=%.4f", chair_metrics["CHAIRi"], chair_metrics["CHAIRs"])

    # Token-level labeling
    # IMPORTANT: pass only the generated portion (NOT the full output_ids).
    # The positions returned by add_token_labels are relative to the start of
    # this sequence; we add prompt_end_idx later to get absolute positions.
    logger.info("Adding token-level hallucination labels ...")
    from transformers import LlavaProcessor
    processor = LlavaProcessor.from_pretrained(args.model_path)

    sequences = [
        processor.tokenizer.encode(entry["caption"], add_special_tokens=False)
        for entry in gen_data
    ]
    eval_dicts = add_token_labels(eval_dicts, sequences, processor, evaluator)
    logger.info("  Token labeling complete")

    # ── Load model with DetectionAdapter ──
    logger.info("Loading model from %s ...", args.model_path)
    model, _ = load_model_and_processor(args.model_path, device)

    n_layers = args.end_layer - args.start_layer
    logger.info("Injecting DetectionAdapter (layers %d-%d, ratio=%.2f, no_rope=%s) ...",
                args.start_layer, args.end_layer, args.ratio, args.compute_no_rope_attention)
    inject_detection_adapter(
        model,
        args.start_layer,
        args.end_layer,
        device,
        ratio=args.ratio,
        compute_no_rope_attention=args.compute_no_rope_attention,
    )

    # ── Forward pass loop ──
    all_scores = {}
    all_labels = []
    shape_cache = _init_shape_cache() if args.save_shape_cache else None
    prompt_text = build_caption_prompt()

    # Cross-image null buffer: stores per-layer instruction null distributions
    # from recently processed images for cross-image CVG computation.
    # Each entry is a list of (n_v,) tensors, one per layer.
    _CROSS_NULL_BUFFER_SIZE = 20
    cross_image_null_buffer: list[list[torch.Tensor]] = []

    layers = get_llm_layers(model)

    for idx, (entry, eval_dict) in enumerate(tqdm(
        zip(gen_data, eval_dicts),
        total=len(gen_data),
        desc="Extracting purified attention",
    )):
        image_id = entry["image_id"]
        caption = entry["caption"]
        prompt_end_idx = entry["prompt_end_idx"]

        # Find first mentions for this image
        first_mentions = find_first_mentions(eval_dict)
        if not first_mentions:
            continue

        # Reconstruct inputs (image + prompt + caption = teacher forcing)
        img_path = Path(args.coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        image = Image.open(img_path).convert("RGB")
        full_text = prompt_text + " " + caption

        inputs = processor(images=image, text=full_text, return_tensors="pt").to(
            device, dtype=torch.float16
        )

        # Compute dynamic visual-token bounds from the actual input_ids and push
        # them into every DetectionAdapter layer before the forward pass.
        img_token_id = getattr(processor, "image_token_id", None) or getattr(
            processor, "image_token_index", 32000
        )
        try:
            vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], img_token_id)
        except ValueError:
            logger.warning("No image tokens in input_ids for image %d", image_id)
            continue
        _set_adapter_bounds(layers, vis_start, vis_end)

        with torch.inference_mode():
            outputs = model.forward(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
            )

        # Collect attention from adapter instances
        orig_attn_layers = []
        sink_only_attn_layers = []
        topmass_only_attn_layers = []
        purified_attn_layers = []
        no_rope_attn_layers = []
        no_rope_sink_only_attn_layers = []
        no_rope_topmass_only_attn_layers = []
        no_rope_purified_attn_layers = []
        sink_stats_layers = []
        per_head_attn: dict[int, torch.Tensor] = {}

        for i, layer in enumerate(layers):
            adapter = layer.self_attn
            if isinstance(adapter, DetectionAdapter):
                if adapter.last_original_attn is not None:
                    orig_attn_layers.append(adapter.last_original_attn)
                    sink_only_attn_layers.append(adapter.last_sink_only_attn)
                    topmass_only_attn_layers.append(adapter.last_topmass_only_attn)
                    purified_attn_layers.append(adapter.last_purified_attn)
                    if args.compute_no_rope_attention:
                        no_rope_attn_layers.append(adapter.last_no_rope_attn)
                        no_rope_sink_only_attn_layers.append(adapter.last_no_rope_sink_only_attn)
                        no_rope_topmass_only_attn_layers.append(adapter.last_no_rope_topmass_only_attn)
                        no_rope_purified_attn_layers.append(adapter.last_no_rope_purified_attn)
                    sink_stats_layers.append(adapter.last_sink_stats)
                    # Collect per-head attention for key layers
                    if adapter.last_per_head_attn is not None:
                        per_head_attn[adapter.layer_idx] = adapter.last_per_head_attn

        if not orig_attn_layers:
            logger.warning("No attention collected for image %d", image_id)
            del inputs, outputs
            continue

        # Verify prompt_end_idx matches
        output_ids = inputs["input_ids"][0]
        try:
            computed_pei = _find_prompt_end_in_inputs(output_ids.tolist())
            if computed_pei != prompt_end_idx:
                logger.debug("prompt_end_idx adjusted: %d -> %d for image %d",
                             prompt_end_idx, computed_pei, image_id)
                prompt_end_idx = computed_pei
        except ValueError:
            pass  # use the original prompt_end_idx

        # Token partitioning
        token_masks = partition_tokens(output_ids, processor, prompt_end_idx)

        # Extract current image's instruction null distributions for cross-image buffer
        instruction_mask = token_masks["instruction_mask"]
        current_null_layers = []
        n_v = vis_end - vis_start
        for l, A in enumerate(orig_attn_layers):
            sink_rel = _sinks_within_visual(
                sink_stats_layers[l].get("sink_positions", []),
                vis_start, vis_end,
            )
            instr_rows = A[instruction_mask, vis_start:vis_end].float()
            if instr_rows.shape[0] > 0:
                a_null = instr_rows.mean(dim=0)
            else:
                a_null = torch.full((n_v,), 1.0 / n_v, device=A.device)
            a_null_clean = _safe_normalize(_strip_sinks(a_null, sink_rel))
            current_null_layers.append(a_null_clean)

        # Add to buffer (maintain rolling window)
        cross_image_null_buffer.append(current_null_layers)
        if len(cross_image_null_buffer) > _CROSS_NULL_BUFFER_SIZE:
            cross_image_null_buffer.pop(0)

        # Prepare cross-image nulls (exclude current image = last entry)
        cross_nulls_for_scoring = cross_image_null_buffer[:-1] if len(cross_image_null_buffer) > 1 else None

        if shape_cache is not None:
            cache_variants = {
                "orig": orig_attn_layers,
                "sink_only": sink_only_attn_layers,
                "topmass_only": topmass_only_attn_layers,
                "purified": purified_attn_layers,
            }
            if args.compute_no_rope_attention:
                cache_variants.update({
                    "no_rope": no_rope_attn_layers,
                    "no_rope_sink_only": no_rope_sink_only_attn_layers,
                    "no_rope_topmass_only": no_rope_topmass_only_attn_layers,
                    "no_rope_purified": no_rope_purified_attn_layers,
                })
            _append_shape_cache(
                cache=shape_cache,
                mentions=first_mentions,
                image_id=image_id,
                prompt_end_idx=prompt_end_idx,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
                sink_stats_layers=sink_stats_layers,
                variants=cache_variants,
            )

        # Compute shape scores only: CVG / concentration / CLC, plus optional
        # per-head and cross-image null variants. PAS-style attention-mass
        # baselines are intentionally not emitted in current runs.
        scores, labels = compute_all_scores(
            first_mentions,
            orig_attn_layers,
            sink_only_attn_layers,
            topmass_only_attn_layers,
            purified_attn_layers,
            sink_stats_layers,
            token_masks,
            prompt_end_idx,
            vis_start=vis_start,
            vis_end=vis_end,
            no_rope_attn_layers=no_rope_attn_layers if args.compute_no_rope_attention else None,
            no_rope_sink_only_attn_layers=(
                no_rope_sink_only_attn_layers if args.compute_no_rope_attention else None
            ),
            no_rope_topmass_only_attn_layers=(
                no_rope_topmass_only_attn_layers if args.compute_no_rope_attention else None
            ),
            no_rope_purified_attn_layers=(
                no_rope_purified_attn_layers if args.compute_no_rope_attention else None
            ),
            per_head_attn=per_head_attn if per_head_attn else None,
            cross_image_nulls=cross_nulls_for_scoring,
        )

        # Merge into all_scores. Some optional score families (e.g.
        # cross-image nulls) may be unavailable for early samples, so keep every
        # score vector aligned with all_labels by padding missing values as NaN.
        _merge_scores_aligned(all_scores, all_labels, scores, labels)

        del inputs, outputs
        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Compute AUROC ──
    if not all_labels:
        logger.error("No labels collected. Cannot compute AUROC.")
        return

    roc_auc = compute_auroc(all_scores, all_labels)

    # ── Report results ──
    n_total = len(all_labels)
    n_hallu = sum(all_labels)
    n_non_hallu = n_total - n_hallu

    logger.info("=" * 60)
    logger.info("SinkDetect Results")
    logger.info("=" * 60)
    logger.info("Total object mentions: %d (hallucinated: %d, grounded: %d)",
                n_total, n_hallu, n_non_hallu)
    logger.info("CHAIRi=%.4f, CHAIRs=%.4f",
                chair_metrics["CHAIRi"], chair_metrics["CHAIRs"])
    logger.info("")

    # Top global scores
    logger.info("Top AUROC scores (global):")
    for key, val in roc_auc.items():
        if key.startswith("global_"):
            logger.info("  %s: %.4f", key, val)

    # Best score
    if roc_auc:
        best_key = next(iter(roc_auc))
        best_val = roc_auc[best_key]
        logger.info("")
        logger.info("Best AUROC: %.4f (%s)", best_val, best_key)

    # ── Save results ──
    metrics = {
        "objects": {
            "total": n_total,
            "hallucinated": n_hallu,
            "non_hallucinated": n_non_hallu,
        },
        "chair_metrics": chair_metrics,
        "config": {
            "ratio": args.ratio,
            "start_layer": args.start_layer,
            "end_layer": args.end_layer,
            "model_path": args.model_path,
            "generation_json": args.generation_json,
            "compute_no_rope_attention": args.compute_no_rope_attention,
        },
        "roc_auc": roc_auc,
    }

    metrics_path = output_dir / f"metrics{shard_suffix}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save raw scores
    raw_path = output_dir / f"raw_scores{shard_suffix}.npz"
    save_dict = {k: np.array(v, dtype=np.float32) for k, v in all_scores.items()}
    save_dict["labels"] = np.array(all_labels, dtype=np.int32)
    np.savez(raw_path, **save_dict)
    logger.info("Raw scores saved to %s", raw_path)

    if shape_cache is not None and shape_cache["labels"]:
        cache_path = output_dir / f"shape_cache{shard_suffix}.npz"
        _save_shape_cache(shape_cache, cache_path)
        logger.info("Shape cache saved to %s", cache_path)


def _find_prompt_end_in_inputs(output_ids_list):
    """Find prompt end index from the full input sequence."""
    pattern = (319, 1799, 9047, 13566, 29901)
    n = len(pattern)
    for j in range(len(output_ids_list) - n + 1):
        if tuple(output_ids_list[j : j + n]) == pattern:
            return j + n
    raise ValueError("Cannot find prompt end idx")


def _merge_scores_aligned(
    all_scores: dict[str, list],
    all_labels: list[int],
    scores: dict[str, list],
    labels: list[int],
) -> None:
    """Append one sample's scores while preserving score/label alignment."""
    n_prev = len(all_labels)
    n_new = len(labels)

    for key in list(all_scores.keys()):
        vals = scores.get(key)
        if vals is None:
            all_scores[key].extend([np.nan] * n_new)
            continue
        if len(vals) != n_new:
            logger.warning(
                "Score %s has %d values for %d labels; padding/truncating.",
                key, len(vals), n_new,
            )
        padded = list(vals[:n_new]) + [np.nan] * max(0, n_new - len(vals))
        all_scores[key].extend(padded)

    for key, vals in scores.items():
        if key in all_scores:
            continue
        if len(vals) != n_new:
            logger.warning(
                "New score %s has %d values for %d labels; padding/truncating.",
                key, len(vals), n_new,
            )
        padded = list(vals[:n_new]) + [np.nan] * max(0, n_new - len(vals))
        all_scores[key] = [np.nan] * n_prev + padded

    all_labels.extend(labels)


def _init_shape_cache() -> dict:
    cache = {
        "labels": [],
        "image_ids": [],
        "token_pos": [],
        "words": [],
        "sink_mask": [],
    }
    for name in (
        "orig",
        "sink_only",
        "topmass_only",
        "purified",
        "no_rope",
        "no_rope_sink_only",
        "no_rope_topmass_only",
        "no_rope_purified",
    ):
        cache[f"{name}_obj"] = []
        cache[f"{name}_null"] = []
    return cache


def _append_shape_cache(
    cache: dict,
    mentions: list[dict],
    image_id: int,
    prompt_end_idx: int,
    instruction_mask: torch.Tensor,
    vis_start: int,
    vis_end: int,
    sink_stats_layers: list[dict],
    variants: dict[str, list[torch.Tensor]],
) -> None:
    n_layers = len(sink_stats_layers)
    n_v = vis_end - vis_start

    sink_mask_layers = []
    for layer_stats in sink_stats_layers:
        mask = np.zeros(n_v, dtype=np.bool_)
        for pos in layer_stats.get("sink_positions", []):
            if vis_start <= pos < vis_end:
                mask[pos - vis_start] = True
        sink_mask_layers.append(mask)
    sink_mask = np.stack(sink_mask_layers, axis=0)

    for mention in mentions:
        token_pos = mention["pos"] + prompt_end_idx
        cache["labels"].append(1 if mention["hallucinated"] else 0)
        cache["image_ids"].append(int(image_id))
        cache["token_pos"].append(int(token_pos))
        cache["words"].append(str(mention["word"]))
        cache["sink_mask"].append(sink_mask)

        for name, attn_layers in variants.items():
            obj_rows = []
            null_rows = []
            for A in attn_layers:
                obj_rows.append(A[token_pos, vis_start:vis_end].float().detach().cpu().numpy())
                instr_rows = A[instruction_mask, vis_start:vis_end].float()
                if instr_rows.shape[0] == 0:
                    null_rows.append(np.full((n_v,), 1.0 / n_v, dtype=np.float32))
                else:
                    null_rows.append(instr_rows.mean(dim=0).detach().cpu().numpy())
            cache[f"{name}_obj"].append(np.stack(obj_rows, axis=0).astype(np.float16))
            cache[f"{name}_null"].append(np.stack(null_rows, axis=0).astype(np.float16))


def _save_shape_cache(cache: dict, path: Path) -> None:
    arrays = {
        "labels": np.asarray(cache["labels"], dtype=np.int32),
        "image_ids": np.asarray(cache["image_ids"], dtype=np.int64),
        "token_pos": np.asarray(cache["token_pos"], dtype=np.int32),
        "words": np.asarray(cache["words"], dtype="U64"),
        "sink_mask": np.stack(cache["sink_mask"], axis=0).astype(np.bool_),
    }
    for name in (
        "orig",
        "sink_only",
        "topmass_only",
        "purified",
        "no_rope",
        "no_rope_sink_only",
        "no_rope_topmass_only",
        "no_rope_purified",
    ):
        if cache[f"{name}_obj"]:
            arrays[f"{name}_obj"] = np.stack(cache[f"{name}_obj"], axis=0).astype(np.float16)
            arrays[f"{name}_null"] = np.stack(cache[f"{name}_null"], axis=0).astype(np.float16)
    np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    main()
