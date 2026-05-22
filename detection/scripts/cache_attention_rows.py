#!/usr/bin/env python3
"""Stage 2a: cache only visual attention rows needed by shape metrics.

This stage runs the teacher-forced VLM forward once, but saves compact rows
instead of full attention maps. The cache is sufficient for recomputing CVG,
concentration, CLC, sink removal, top-mass ablations, and no-RoPE ablations.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

DETECTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DETECTION_ROOT.parent
sys.path.insert(0, str(DETECTION_ROOT / "src"))
PAS_SRC = PROJECT_ROOT.parent / "pas" / "src"
if str(PAS_SRC) not in sys.path:
    sys.path.insert(0, str(PAS_SRC))

from sinkdetect.adapter import DetectionAdapter, get_llm_layers, inject_detection_adapter
from sinkdetect.chair import add_token_labels, evaluate_chair, find_first_mentions, load_chair_evaluator
from sinkdetect.sink_utils import find_vis_bounds
from sinkdetect.utils import build_caption_prompt, load_generation_json, load_model_and_processor, partition_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache compact attention rows for SinkDetect metrics")
    p.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--coco_path", type=str, default="/data/common_dataset/coco-2014-dataset/")
    p.add_argument("--generation_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--chair_pkl", type=str, default=str(PROJECT_ROOT.parent / "pas" / "data" / "chair_coco.pkl"))
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--cache_layers", type=str, default="0,1,2,3,4",
                   help="Comma-separated layer indices to cache. Use first 5 by default.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--local_window", type=int, default=16)
    p.add_argument("--local_max_tokens", type=int, default=8)
    return p.parse_args()


def _parse_layers(spec: str) -> list[int]:
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not layers:
        raise ValueError("--cache_layers must contain at least one layer")
    return layers


def _set_adapter_bounds(layers, vis_start: int, vis_end: int) -> None:
    for layer in layers:
        adapter = layer.self_attn
        if isinstance(adapter, DetectionAdapter):
            adapter.vis_start = vis_start
            adapter.vis_end = vis_end
            adapter.text_start = vis_end


def _norm_token(token: str) -> str:
    token = token.replace("▁", "").replace("Ġ", "").replace("Ċ", "")
    token = token.lstrip("Ġ▁").lower()
    return re.sub(r"[^a-z0-9]+", "", token)


def _object_token_ids(processor, evaluator) -> set[int]:
    words = set()
    inv = getattr(evaluator, "inverse_synonym_dict", {})
    words.update(str(k) for k in inv.keys())
    words.update(str(v) for v in inv.values())
    words.update({"image", "picture", "photo", "photograph", "scene"})

    ids: set[int] = set()
    for word in words:
        for tok_id in processor.tokenizer.encode(word, add_special_tokens=False):
            ids.add(int(tok_id))
    return ids


def _clean_instruction_mask(
    output_ids: torch.Tensor,
    instruction_mask: torch.Tensor,
    object_token_ids: set[int],
) -> tuple[torch.Tensor, bool]:
    clean = instruction_mask.clone()
    if object_token_ids:
        bad = torch.tensor(sorted(object_token_ids), device=output_ids.device, dtype=output_ids.dtype)
        clean &= ~torch.isin(output_ids, bad)
    valid = bool(clean.sum().item() >= 3)
    return clean if valid else instruction_mask, valid


def _nearest_nonobject_positions(
    token_pos: int,
    output_mask: torch.Tensor,
    object_positions: set[int],
    max_tokens: int,
    window: int,
) -> list[int]:
    candidates = torch.nonzero(output_mask, as_tuple=False).flatten().tolist()
    candidates = [
        int(pos)
        for pos in candidates
        if pos != token_pos
        and pos not in object_positions
        and abs(int(pos) - token_pos) <= window
    ]
    candidates.sort(key=lambda pos: (abs(pos - token_pos), pos))
    return candidates[:max_tokens]


def _mean_row_or_uniform(A: torch.Tensor, rows: torch.Tensor | list[int], vis_start: int, vis_end: int) -> tuple[np.ndarray, bool]:
    n_v = vis_end - vis_start
    if isinstance(rows, list):
        if not rows:
            return np.full((n_v,), 1.0 / n_v, dtype=np.float32), False
        idx = torch.tensor(rows, dtype=torch.long, device=A.device)
    else:
        idx = torch.nonzero(rows, as_tuple=False).flatten()
        if idx.numel() == 0:
            return np.full((n_v,), 1.0 / n_v, dtype=np.float32), False
    row = A[idx, vis_start:vis_end].float().mean(dim=0)
    return row.detach().cpu().numpy().astype(np.float16), True


def _init_cache(layer_indices: list[int]) -> dict:
    cache = {
        "labels": [],
        "image_ids": [],
        "token_pos": [],
        "words": [],
        "valid_instr_null": [],
        "valid_local_null": [],
        "layer_indices": np.asarray(layer_indices, dtype=np.int32),
        "sink_mask": [],
    }
    for prefix in ("orig", "no_rope"):
        cache[f"{prefix}_obj"] = []
        cache[f"{prefix}_instr_null"] = []
        cache[f"{prefix}_local_null"] = []
    return cache


def _append_cache_rows(
    cache: dict,
    mentions: list[dict],
    image_id: int,
    prompt_end_idx: int,
    vis_start: int,
    vis_end: int,
    output_mask: torch.Tensor,
    instr_mask: torch.Tensor,
    valid_instr: bool,
    orig_layers: list[torch.Tensor],
    no_rope_layers: list[torch.Tensor],
    sink_stats_layers: list[dict],
    local_window: int,
    local_max_tokens: int,
) -> None:
    n_v = vis_end - vis_start
    object_positions = set()
    for m in mentions:
        q = m["pos"] + prompt_end_idx
        object_positions.update({q, q + 1, q + 2})

    sink_mask_layers = []
    for stats in sink_stats_layers:
        mask = np.zeros(n_v, dtype=np.bool_)
        for pos in stats.get("sink_positions", []):
            if vis_start <= pos < vis_end:
                mask[pos - vis_start] = True
        sink_mask_layers.append(mask)
    sink_mask = np.stack(sink_mask_layers, axis=0)

    for mention in mentions:
        token_pos = mention["pos"] + prompt_end_idx
        local_pos = _nearest_nonobject_positions(
            token_pos,
            output_mask,
            object_positions,
            max_tokens=local_max_tokens,
            window=local_window,
        )
        cache["labels"].append(1 if mention["hallucinated"] else 0)
        cache["image_ids"].append(int(image_id))
        cache["token_pos"].append(int(token_pos))
        cache["words"].append(str(mention["word"]))
        cache["valid_instr_null"].append(bool(valid_instr))
        cache["valid_local_null"].append(bool(local_pos))
        cache["sink_mask"].append(sink_mask)

        for name, layers in (("orig", orig_layers), ("no_rope", no_rope_layers)):
            obj_rows, instr_rows, local_rows = [], [], []
            for A in layers:
                obj_rows.append(A[token_pos, vis_start:vis_end].float().detach().cpu().numpy().astype(np.float16))
                instr_row, _ = _mean_row_or_uniform(A, instr_mask, vis_start, vis_end)
                local_row, _ = _mean_row_or_uniform(A, local_pos, vis_start, vis_end)
                instr_rows.append(instr_row)
                local_rows.append(local_row)
            cache[f"{name}_obj"].append(np.stack(obj_rows, axis=0))
            cache[f"{name}_instr_null"].append(np.stack(instr_rows, axis=0))
            cache[f"{name}_local_null"].append(np.stack(local_rows, axis=0))


def _save_cache(cache: dict, path: Path) -> None:
    arrays = {
        "labels": np.asarray(cache["labels"], dtype=np.int32),
        "image_ids": np.asarray(cache["image_ids"], dtype=np.int64),
        "token_pos": np.asarray(cache["token_pos"], dtype=np.int32),
        "words": np.asarray(cache["words"], dtype="U64"),
        "valid_instr_null": np.asarray(cache["valid_instr_null"], dtype=np.bool_),
        "valid_local_null": np.asarray(cache["valid_local_null"], dtype=np.bool_),
        "layer_indices": cache["layer_indices"],
        "sink_mask": np.stack(cache["sink_mask"], axis=0).astype(np.bool_),
    }
    for prefix in ("orig", "no_rope"):
        for row_name in ("obj", "instr_null", "local_null"):
            arrays[f"{prefix}_{row_name}"] = np.stack(cache[f"{prefix}_{row_name}"], axis=0).astype(np.float16)
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = _parse_layers(args.cache_layers)
    if layer_indices != list(range(min(layer_indices), max(layer_indices) + 1)):
        raise ValueError("cache_attention_rows currently expects contiguous cache_layers")

    gen_data = load_generation_json(args.generation_json)
    if args.limit > 0:
        gen_data = gen_data[:args.limit]
    if args.num_shards > 1:
        gen_data = gen_data[args.shard_idx :: args.num_shards]
        logger.info("shard %d/%d: %d entries", args.shard_idx, args.num_shards, len(gen_data))
    suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""

    evaluator = load_chair_evaluator(args.chair_pkl)
    chair_input_path = output_dir / f"chair_input{suffix}.json"
    chair_data = [{"image_id": e["image_id"], "caption": e["caption"]} for e in gen_data]
    eval_dicts, chair_metrics = evaluate_chair(evaluator, data=chair_data, json_path=str(chair_input_path))

    from transformers import LlavaProcessor
    processor = LlavaProcessor.from_pretrained(args.model_path)
    sequences = [processor.tokenizer.encode(e["caption"], add_special_tokens=False) for e in gen_data]
    eval_dicts = add_token_labels(eval_dicts, sequences, processor, evaluator)
    object_token_ids = _object_token_ids(processor, evaluator)

    model, _ = load_model_and_processor(args.model_path, device)
    inject_detection_adapter(
        model,
        min(layer_indices),
        max(layer_indices) + 1,
        device,
        ratio=args.ratio,
        compute_no_rope_attention=True,
    )
    layers = get_llm_layers(model)
    cache = _init_cache(layer_indices)
    prompt_text = build_caption_prompt()

    for idx, (entry, eval_dict) in enumerate(tqdm(
        zip(gen_data, eval_dicts),
        total=len(gen_data),
        desc="Caching attention rows",
    )):
        mentions = find_first_mentions(eval_dict)
        if not mentions:
            continue
        image_id = entry["image_id"]
        img_path = Path(args.coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt_text + " " + entry["caption"], return_tensors="pt").to(
            device, dtype=torch.float16
        )
        img_token_id = getattr(processor, "image_token_id", None) or getattr(processor, "image_token_index", 32000)
        vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], img_token_id)
        _set_adapter_bounds(layers, vis_start, vis_end)

        with torch.inference_mode():
            outputs = model.forward(**inputs, output_attentions=True, output_hidden_states=False)

        output_ids = inputs["input_ids"][0]
        prompt_end_idx = entry["prompt_end_idx"]
        token_masks = partition_tokens(output_ids, processor, prompt_end_idx)
        instr_mask, valid_instr = _clean_instruction_mask(
            output_ids,
            token_masks["instruction_mask"],
            object_token_ids,
        )

        orig_layers, no_rope_layers, sink_stats = [], [], []
        for layer_idx in layer_indices:
            adapter = layers[layer_idx].self_attn
            if not isinstance(adapter, DetectionAdapter) or adapter.last_original_attn is None:
                raise RuntimeError(f"Missing adapter attention for layer {layer_idx}")
            orig_layers.append(adapter.last_original_attn)
            no_rope_layers.append(adapter.last_no_rope_attn)
            sink_stats.append(adapter.last_sink_stats)

        _append_cache_rows(
            cache=cache,
            mentions=mentions,
            image_id=image_id,
            prompt_end_idx=prompt_end_idx,
            vis_start=vis_start,
            vis_end=vis_end,
            output_mask=token_masks["output_mask"],
            instr_mask=instr_mask,
            valid_instr=valid_instr,
            orig_layers=orig_layers,
            no_rope_layers=no_rope_layers,
            sink_stats_layers=sink_stats,
            local_window=args.local_window,
            local_max_tokens=args.local_max_tokens,
        )

        del inputs, outputs
        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    out = output_dir / f"attention_row_cache{suffix}.npz"
    _save_cache(cache, out)
    metrics = {
        "objects": {
            "total": int(len(cache["labels"])),
            "hallucinated": int(np.sum(cache["labels"])),
            "non_hallucinated": int(len(cache["labels"]) - np.sum(cache["labels"])),
        },
        "chair_metrics": chair_metrics,
        "cache_layers": layer_indices,
        "row_cache": str(out),
    }
    (output_dir / f"attention_row_cache_metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
