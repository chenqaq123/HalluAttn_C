#!/usr/bin/env python3
"""Cache raw, pre-RoPE, and sink-clean per-head grounding features.

The LVLM is never modified. Read-only forward pre-hooks capture Q/K before
RoPE, while ordinary ``output_attentions`` supplies the raw control branch.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm

DETECTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DETECTION_ROOT.parent
sys.path.insert(0, str(DETECTION_ROOT / "src"))

from sinkdetect.sink_utils import auto_detect_sinks, find_vis_bounds  # noqa: E402
from sinkdetect.utils import (  # noqa: E402
    build_caption_prompt,
    load_model_and_processor,
    partition_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "entropy_obj",
    "top1_obj",
    "top5_obj",
    "max_over_mean_obj",
    "jsd_obj_local",
    "delta_top1",
    "delta_top5",
    "delta_entropy",
]
_EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--object_cache",
        default=str(
            PROJECT_ROOT
            / "detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl"
        ),
    )
    p.add_argument("--coco_path", default="/srv/common_dataset/coco-2014-dataset")
    p.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--layers", default="22,31")
    p.add_argument("--local_window", type=int, default=16)
    p.add_argument("--local_max_tokens", type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit_images", type=int, default=0)
    p.add_argument("--check_forward_identity", action="store_true")
    p.add_argument("--reference_per_head_glob", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _parse_layers(spec: str) -> list[int]:
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not layers:
        raise ValueError("--layers must contain at least one layer")
    return layers


def _resolve_llm_layers(model) -> torch.nn.ModuleList:
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
    ]
    for get_layers in candidates:
        try:
            layers = get_layers(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList):
            return layers
    raise RuntimeError(f"Cannot locate decoder layers in {type(model).__name__}")


class PreRoPECapture:
    """Read-only pre-hook that captures per-head Q/K before RoPE."""

    def __init__(self, attention_module):
        self.attention = attention_module
        self.query: torch.Tensor | None = None
        self.key: torch.Tensor | None = None
        self.key_norms: torch.Tensor | None = None

    def __call__(self, module, args, kwargs) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None:
            raise RuntimeError("Attention pre-hook did not receive hidden_states")
        config = module.config
        heads = int(config.num_attention_heads)
        kv_heads = int(getattr(config, "num_key_value_heads", heads))
        head_dim = int(getattr(module, "head_dim", config.hidden_size // heads))
        batch, sequence, _ = hidden.shape
        query = module.q_proj(hidden).view(batch, sequence, heads, head_dim).transpose(1, 2)
        key = module.k_proj(hidden).view(batch, sequence, kv_heads, head_dim).transpose(1, 2)
        self.query = query[0].float().detach()
        self.key = key[0].float().detach()
        self.key_norms = key.float().norm(p=2, dim=-1).mean(dim=(0, 1)).detach()

    def no_rope_visual_rows(
        self,
        query_positions: torch.Tensor,
        vis_start: int,
        vis_end: int,
    ) -> torch.Tensor:
        if self.query is None or self.key is None:
            raise RuntimeError("Pre-RoPE hook did not capture Q/K")
        query = self.query[:, query_positions, :]
        key = self.key[:, vis_start:vis_end, :]
        repeats = query.shape[0] // key.shape[0]
        if repeats > 1:
            key = key.repeat_interleave(repeats, dim=0)
        scaling = float(
            getattr(self.attention, "scaling", query.shape[-1] ** -0.5)
        )
        logits = torch.einsum("hrd,hvd->hrv", query, key) * scaling
        # All requested queries follow the visual span. Restricting the softmax
        # to visual keys is exactly equivalent after visual L1 normalization.
        return torch.softmax(logits, dim=-1)


def _load_records(path: str) -> dict[int, list[dict]]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    by_image: dict[int, list[dict]] = {}
    for record in records:
        by_image.setdefault(int(record["image_id"]), []).append(record)
    for mentions in by_image.values():
        mentions.sort(key=lambda row: int(row["token_pos"]))
    return by_image


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


def _normalize_visual(rows: torch.Tensor, keep_mask: torch.Tensor | None) -> torch.Tensor:
    rows = rows.float()
    if keep_mask is not None:
        rows = rows * keep_mask.to(rows.dtype)
    sums = rows.sum(dim=-1, keepdim=True)
    if keep_mask is None:
        fallback = torch.full_like(rows, 1.0 / rows.shape[-1])
    else:
        denominator = keep_mask.sum().clamp_min(1).to(rows.dtype)
        fallback = (keep_mask.to(rows.dtype) / denominator).expand_as(rows)
    return torch.where(sums > _EPS, rows / sums.clamp_min(_EPS), fallback)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(_EPS).log()).sum(dim=-1)


def _summarize_rows(
    visual_rows: torch.Tensor,
    object_index: int,
    local_indices: list[int],
    keep_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return (heads, features) from a shared set of visual query rows."""
    obj = _normalize_visual(visual_rows[:, object_index], keep_mask)
    local = _normalize_visual(visual_rows[:, local_indices], keep_mask).mean(dim=1)
    local = _normalize_visual(local, keep_mask)

    entropy_obj = _entropy(obj)
    entropy_local = _entropy(local)
    object_sorted = obj.sort(dim=-1, descending=True).values
    local_sorted = local.sort(dim=-1, descending=True).values
    top1_obj = object_sorted[:, 0]
    top5_obj = object_sorted[:, :5].sum(dim=-1)
    top1_local = local_sorted[:, 0]
    top5_local = local_sorted[:, :5].sum(dim=-1)
    active = obj.shape[-1] if keep_mask is None else int(keep_mask.sum().item())
    midpoint = 0.5 * (obj + local)
    jsd = 0.5 * (
        (obj * (obj.clamp_min(_EPS).log() - midpoint.clamp_min(_EPS).log())).sum(dim=-1)
        + (local * (local.clamp_min(_EPS).log() - midpoint.clamp_min(_EPS).log())).sum(dim=-1)
    )
    return torch.stack(
        [
            entropy_obj,
            top1_obj,
            top5_obj,
            top1_obj * max(active, 1),
            jsd,
            top1_obj - top1_local,
            top5_obj - top5_local,
            entropy_local - entropy_obj,
        ],
        dim=-1,
    )


def _load_reference(glob_pattern: str | None) -> dict[tuple[int, int], dict[int, np.ndarray]]:
    if not glob_pattern:
        return {}
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(glob_pattern)
    reference: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=True) as cache:
            layers = [int(x) for x in cache["layer_indices"]]
            image_ids = np.asarray(cache["image_ids"])
            token_positions = np.asarray(cache["token_pos"])
            features = np.asarray(cache["feats"], dtype=np.float32)
        for index, (image_id, token_pos) in enumerate(zip(image_ids, token_positions)):
            reference[(int(image_id), int(token_pos))] = {
                layer: features[index, layer_index]
                for layer_index, layer in enumerate(layers)
            }
    return reference


def _prepare_inputs(processor, image_path: Path, caption: str, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    try:
        return processor(
            images=image,
            text=build_caption_prompt() + " " + caption,
            return_tensors="pt",
        ).to(device, dtype=torch.float16)
    finally:
        image.close()


def main() -> None:
    args = parse_args()
    selected_layers = _parse_layers(args.layers)
    by_image = _load_records(args.object_cache)
    image_ids = sorted(by_image)[args.shard_idx :: args.num_shards]
    if args.limit_images > 0:
        image_ids = image_ids[: args.limit_images]

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(
        args.model_path, device, attn_implementation="eager"
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = _resolve_llm_layers(model)
    if selected_layers[-1] >= len(layers):
        raise ValueError(f"Requested layer {selected_layers[-1]} from {len(layers)} layers")

    first_inputs = None
    baseline_logits = None
    baseline_attentions: dict[int, torch.Tensor] = {}
    if args.check_forward_identity and image_ids:
        first_image = image_ids[0]
        image_path = (
            Path(args.coco_path) / "val2014" / f"COCO_val2014_{first_image:012d}.jpg"
        )
        first_inputs = _prepare_inputs(
            processor, image_path, str(by_image[first_image][0]["caption"]), device
        )
        with torch.inference_mode():
            baseline = model(
                **first_inputs, output_attentions=True, output_hidden_states=False, use_cache=False
            )
        baseline_logits = baseline.logits.detach().float().cpu()
        baseline_attentions = {
            layer: baseline.attentions[layer][0].detach().float().cpu()
            for layer in selected_layers
        }
        del baseline

    hooks = {layer: PreRoPECapture(layers[layer].self_attn) for layer in selected_layers}
    handles = [
        layers[layer].self_attn.register_forward_pre_hook(hooks[layer], with_kwargs=True)
        for layer in selected_layers
    ]
    reference = _load_reference(args.reference_per_head_glob)
    checked_reference_rows = 0
    max_reference_error = 0.0
    max_reference_error_by_feature = np.zeros(4, dtype=np.float64)
    max_forward_logit_error = None
    max_forward_attention_error = None

    output: dict[str, list] = {
        key: []
        for key in (
            "object_id",
            "image_id",
            "token_pos",
            "gen_pos",
            "label",
            "word",
            "local_count",
            "sink_count",
            "raw",
            "no_rope",
            "clean",
        )
    }
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    try:
        for image_index, image_id in enumerate(
            tqdm(image_ids, desc=f"clean-head-{args.shard_idx}")
        ):
            records = by_image[image_id]
            image_path = (
                Path(args.coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
            )
            if not image_path.exists():
                logger.warning("Image not found: %s", image_path)
                continue
            inputs = first_inputs if image_index == 0 and first_inputs is not None else _prepare_inputs(
                processor, image_path, str(records[0]["caption"]), device
            )
            vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], int(image_token_id))
            prompt_end = int(records[0]["prompt_end_idx"])
            token_masks = partition_tokens(inputs["input_ids"][0], processor, prompt_end)
            object_positions: set[int] = set()
            for record in records:
                position = int(record["token_pos"])
                object_positions.update({position, position + 1, position + 2})
            valid_mentions: list[tuple[dict, list[int]]] = []
            for record in records:
                token_pos = int(record["token_pos"])
                if token_pos >= inputs["input_ids"].shape[1]:
                    raise ValueError(f"Object token {token_pos} is outside image {image_id} input")
                local_positions = _nearest_nonobject_positions(
                    token_pos,
                    token_masks["output_mask"],
                    object_positions,
                    args.local_max_tokens,
                    args.local_window,
                )
                if local_positions:
                    valid_mentions.append((record, local_positions))

            with torch.inference_mode():
                result = model(
                    **inputs, output_attentions=True, output_hidden_states=False, use_cache=False
                )
            if image_index == 0 and baseline_logits is not None:
                max_forward_logit_error = float(
                    (baseline_logits - result.logits.detach().float().cpu()).abs().max().item()
                )
                max_forward_attention_error = max(
                    float(
                        (
                            baseline_attentions[layer]
                            - result.attentions[layer][0].detach().float().cpu()
                        )
                        .abs()
                        .max()
                        .item()
                    )
                    for layer in selected_layers
                )
                logger.info(
                    "Read-only hook identity: logits %.3g, raw attention %.3g",
                    max_forward_logit_error,
                    max_forward_attention_error,
                )

            query_positions = sorted(
                {
                    position
                    for record, local in valid_mentions
                    for position in [int(record["token_pos"]), *local]
                }
            )
            query_lookup = {position: index for index, position in enumerate(query_positions)}
            query_tensor = torch.tensor(query_positions, dtype=torch.long, device=device)
            per_mention = [
                {"raw": [], "no_rope": [], "clean": [], "sink_count": []}
                for _ in valid_mentions
            ]
            for layer in selected_layers:
                raw_visual = result.attentions[layer][0][
                    :, query_tensor, vis_start:vis_end
                ]
                no_rope_visual = hooks[layer].no_rope_visual_rows(
                    query_tensor, vis_start, vis_end
                )
                key_norms = hooks[layer].key_norms
                if key_norms is None:
                    raise RuntimeError(f"Layer {layer} did not capture key norms")
                visual_positions = torch.arange(vis_start, vis_end, device=device)
                sinks = auto_detect_sinks(key_norms, visual_positions)
                keep_mask = torch.ones(vis_end - vis_start, dtype=torch.bool, device=device)
                keep_mask[sinks - vis_start] = False

                for mention_index, (record, local_positions) in enumerate(valid_mentions):
                    object_index = query_lookup[int(record["token_pos"])]
                    local_indices = [query_lookup[position] for position in local_positions]
                    raw_features = _summarize_rows(
                        raw_visual, object_index, local_indices, None
                    )
                    no_rope_features = _summarize_rows(
                        no_rope_visual, object_index, local_indices, None
                    )
                    clean_features = _summarize_rows(
                        no_rope_visual, object_index, local_indices, keep_mask
                    )
                    per_mention[mention_index]["raw"].append(raw_features.cpu().numpy())
                    per_mention[mention_index]["no_rope"].append(
                        no_rope_features.cpu().numpy()
                    )
                    per_mention[mention_index]["clean"].append(
                        clean_features.cpu().numpy()
                    )
                    per_mention[mention_index]["sink_count"].append(int(sinks.numel()))

                    reference_features = reference.get(
                        (image_id, int(record["token_pos"])), {}
                    ).get(layer)
                    if reference_features is not None and checked_reference_rows < 100:
                        current = raw_features[:, :4].cpu().numpy().copy()
                        current[:, 1:] *= -1.0
                        error = np.abs(current - reference_features)
                        max_reference_error = max(max_reference_error, float(error.max()))
                        max_reference_error_by_feature = np.maximum(
                            max_reference_error_by_feature, error.max(axis=0)
                        )
                        checked_reference_rows += 1

            for (record, local_positions), mention_features in zip(
                valid_mentions, per_mention
            ):
                for key in ("object_id", "image_id", "token_pos", "gen_pos", "label", "word"):
                    output[key].append(record[key])
                output["local_count"].append(len(local_positions))
                output["sink_count"].append(mention_features["sink_count"])
                for branch in ("raw", "no_rope", "clean"):
                    output[branch].append(np.stack(mention_features[branch], axis=0))

            del inputs, result
            if image_index % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    required_reference_rows = min(100, len(output["label"]) * len(selected_layers))
    if reference and checked_reference_rows < required_reference_rows:
        raise AssertionError(
            f"Only checked {checked_reference_rows}/{required_reference_rows} "
            "available raw-reference layer rows"
        )
    if checked_reference_rows and max_reference_error > 1e-4:
        logger.warning(
            "Legacy raw-cache drift (diagnostic only): max=%.6g, by_feature=%s",
            max_reference_error,
            max_reference_error_by_feature.tolist(),
        )
    if max_forward_logit_error is not None and max_forward_logit_error > 1e-6:
        raise AssertionError(f"Read-only hooks changed logits: {max_forward_logit_error}")
    if max_forward_attention_error is not None and max_forward_attention_error > 1e-6:
        raise AssertionError(
            f"Read-only hooks changed raw attention: {max_forward_attention_error}"
        )

    arrays = {
        "object_id": np.asarray(output["object_id"], dtype=np.int64),
        "image_id": np.asarray(output["image_id"], dtype=np.int64),
        "token_pos": np.asarray(output["token_pos"], dtype=np.int32),
        "gen_pos": np.asarray(output["gen_pos"], dtype=np.int32),
        "label": np.asarray(output["label"], dtype=np.int32),
        "word": np.asarray(output["word"], dtype="U64"),
        "local_count": np.asarray(output["local_count"], dtype=np.int16),
        "sink_count": np.asarray(output["sink_count"], dtype=np.int16),
        "layer_indices": np.asarray(selected_layers, dtype=np.int32),
        "feature_names": np.asarray(FEATURE_NAMES, dtype="U32"),
        "raw": np.asarray(output["raw"], dtype=np.float32),
        "no_rope": np.asarray(output["no_rope"], dtype=np.float32),
        "clean": np.asarray(output["clean"], dtype=np.float32),
        "transformers_version": np.asarray(transformers.__version__),
        "max_forward_logit_error": np.asarray(
            np.nan if max_forward_logit_error is None else max_forward_logit_error
        ),
        "max_forward_attention_error": np.asarray(
            np.nan if max_forward_attention_error is None else max_forward_attention_error
        ),
        "reference_rows_checked": np.asarray(checked_reference_rows, dtype=np.int32),
        "max_reference_error": np.asarray(max_reference_error, dtype=np.float64),
        "max_reference_error_by_feature": max_reference_error_by_feature,
    }
    for branch in ("raw", "no_rope", "clean"):
        if not np.isfinite(arrays[branch]).all():
            raise AssertionError(f"Non-finite values in {branch} features")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    selected_mentions = sum(len(by_image[image_id]) for image_id in image_ids)
    logger.info(
        "Wrote %s: N=%d, layers=%s, valid-local=%.4f, max-ref-error=%.3g",
        out_path,
        len(arrays["label"]),
        selected_layers,
        len(arrays["label"]) / max(selected_mentions, 1),
        max_reference_error,
    )


if __name__ == "__main__":
    main()
