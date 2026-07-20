#!/usr/bin/env python3
"""Cache all-layer no-RoPE spatial allocation features for object mentions."""

from __future__ import annotations

import argparse
import gc
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
from sinkdetect.spatial_allocation import (  # noqa: E402
    FEATURE_NAMES,
    remove_attention_keys,
    spatial_allocation_features,
)
from sinkdetect.utils import build_caption_prompt, load_model_and_processor  # noqa: E402
from sinkdetect.value_evidence import causal_attention_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object_cache",
        default=str(
            PROJECT_ROOT
            / "detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl"
        ),
    )
    parser.add_argument("--coco_path", default="/srv/common_dataset/coco-2014-dataset")
    parser.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit_images", type=int, default=0)
    parser.add_argument("--check_forward_identity", action="store_true")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_records(path: str) -> dict[int, list[dict]]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    by_image: dict[int, list[dict]] = {}
    for record in records:
        by_image.setdefault(int(record["image_id"]), []).append(record)
    for mentions in by_image.values():
        mentions.sort(key=lambda row: int(row["token_pos"]))
    return by_image


def _resolve_llm_layers(model) -> torch.nn.ModuleList:
    candidates = [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.model.language_model.model.layers,
    ]
    for resolve in candidates:
        try:
            layers = resolve(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList):
            return layers
    raise RuntimeError(f"Cannot locate decoder layers in {type(model).__name__}")


def _parse_layers(spec: str, layer_count: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(layer_count))
    selected: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    layers = sorted(selected)
    if not layers or layers[0] < 0 or layers[-1] >= layer_count:
        raise ValueError(f"Invalid --layers={spec!r} for {layer_count} layers")
    return layers


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


def _qk_states(
    attention, normalized_hidden: torch.Tensor, query_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    config = attention.config
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(attention, "head_dim", config.hidden_size // heads))
    sequence = normalized_hidden.shape[0]
    query = attention.q_proj(normalized_hidden[query_positions])
    query = query.view(query_positions.numel(), heads, head_dim).transpose(0, 1)
    key = attention.k_proj(normalized_hidden)
    key = key.view(sequence, kv_heads, head_dim).transpose(0, 1)
    return query, key


def _empty_arrays(layer_count: int, heads: int) -> dict[str, np.ndarray]:
    feature_shape = (0, layer_count, 2, heads, len(FEATURE_NAMES))
    return {
        "no_rope": np.empty(feature_shape, dtype=np.float32),
        "sink_clean": np.empty(feature_shape, dtype=np.float32),
        "sink_mass": np.empty((0, layer_count, 2, heads), dtype=np.float32),
        "sink_count": np.empty((0, layer_count), dtype=np.int16),
    }


def main() -> None:
    args = parse_args()
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
    selected_layers = _parse_layers(args.layers, len(layers))
    heads = int(layers[selected_layers[0]].self_attn.config.num_attention_heads)

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    keys = (
        "object_id",
        "image_id",
        "token_pos",
        "gen_pos",
        "label",
        "word",
        "no_rope",
        "sink_clean",
        "sink_mass",
        "sink_count",
    )
    output: dict[str, list] = {key: [] for key in keys}
    max_forward_logit_error = 0.0
    max_clean_row_sum_error = 0.0
    max_branch_difference = 0.0
    identity_checked = False

    for image_index, image_id in enumerate(
        tqdm(image_ids, desc=f"spatial-allocation-{args.shard_idx}")
    ):
        records = by_image[image_id]
        image_path = (
            Path(args.coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        )
        if not image_path.exists():
            logger.warning("Image not found: %s", image_path)
            continue
        inputs = _prepare_inputs(processor, image_path, str(records[0]["caption"]), device)
        input_ids = inputs["input_ids"][0]
        valid_records = [
            record for record in records if int(record["token_pos"]) + 2 < input_ids.numel()
        ]
        if not valid_records:
            del inputs
            continue

        pre_positions = torch.tensor(
            [int(record["token_pos"]) for record in valid_records],
            dtype=torch.long,
            device=device,
        )
        query_positions = torch.stack((pre_positions, pre_positions + 1), dim=1).reshape(-1)
        vis_start, vis_end = find_vis_bounds(input_ids, int(image_token_id))
        if vis_end - vis_start != 576:
            raise ValueError(
                f"Expected 576 visual patches, got {vis_end - vis_start} for image {image_id}"
            )

        baseline_logits = None
        if args.check_forward_identity and not identity_checked:
            with torch.inference_mode():
                baseline = model(
                    **inputs,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                )
            baseline_logits = baseline.logits.detach().float()
            del baseline

        with torch.inference_mode():
            result = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                use_cache=False,
            )
            if baseline_logits is not None:
                max_forward_logit_error = float(
                    (baseline_logits - result.logits.float()).abs().max().item()
                )
                identity_checked = True

            image_no_rope = []
            image_sink_clean = []
            image_sink_mass = []
            image_sink_count = []
            visual_positions = torch.arange(vis_start, vis_end, device=device)
            for layer_index in selected_layers:
                layer = layers[layer_index]
                attention = layer.self_attn
                residual = result.hidden_states[layer_index][0]
                normalized_hidden = layer.input_layernorm(residual)
                query, key = _qk_states(attention, normalized_hidden, query_positions)
                no_rope_attention = causal_attention_rows(
                    query,
                    key,
                    query_positions,
                    scaling=float(
                        getattr(attention, "scaling", query.shape[-1] ** -0.5)
                    ),
                )
                key_norms = key.float().norm(p=2, dim=-1).mean(dim=0)
                sinks = auto_detect_sinks(key_norms, visual_positions)
                sink_clean_attention = remove_attention_keys(no_rope_attention, sinks)
                max_clean_row_sum_error = max(
                    max_clean_row_sum_error,
                    float((sink_clean_attention.sum(dim=-1) - 1.0).abs().max().item()),
                )

                base_features = spatial_allocation_features(
                    no_rope_attention, vis_start, vis_end
                )
                clean_features = spatial_allocation_features(
                    sink_clean_attention, vis_start, vis_end
                )
                max_branch_difference = max(
                    max_branch_difference,
                    float((base_features - clean_features).abs().max().item()),
                )
                sink_mass = no_rope_attention[:, :, sinks].sum(dim=-1)

                mention_count = len(valid_records)
                image_no_rope.append(
                    base_features.permute(1, 0, 2)
                    .reshape(mention_count, 2, heads, -1)
                    .cpu()
                    .numpy()
                )
                image_sink_clean.append(
                    clean_features.permute(1, 0, 2)
                    .reshape(mention_count, 2, heads, -1)
                    .cpu()
                    .numpy()
                )
                image_sink_mass.append(
                    sink_mass.transpose(0, 1)
                    .reshape(mention_count, 2, heads)
                    .cpu()
                    .numpy()
                )
                image_sink_count.append(int(sinks.numel()))

        no_rope_array = np.stack(image_no_rope, axis=1)
        sink_clean_array = np.stack(image_sink_clean, axis=1)
        sink_mass_array = np.stack(image_sink_mass, axis=1)
        sink_count_array = np.broadcast_to(
            np.asarray(image_sink_count, dtype=np.int16),
            (len(valid_records), len(selected_layers)),
        ).copy()
        for mention_index, record in enumerate(valid_records):
            for key in ("object_id", "image_id", "token_pos", "gen_pos", "label", "word"):
                output[key].append(record[key])
            output["no_rope"].append(no_rope_array[mention_index])
            output["sink_clean"].append(sink_clean_array[mention_index])
            output["sink_mass"].append(sink_mass_array[mention_index])
            output["sink_count"].append(sink_count_array[mention_index])

        del inputs, result, baseline_logits
        if image_index % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    numeric = _empty_arrays(len(selected_layers), heads)
    if output["label"]:
        for key in numeric:
            numeric[key] = np.asarray(output[key], dtype=numeric[key].dtype)
    arrays = {
        "object_id": np.asarray(output["object_id"], dtype=np.int64),
        "image_id": np.asarray(output["image_id"], dtype=np.int64),
        "token_pos": np.asarray(output["token_pos"], dtype=np.int32),
        "gen_pos": np.asarray(output["gen_pos"], dtype=np.int32),
        "label": np.asarray(output["label"], dtype=np.int32),
        "word": np.asarray(output["word"], dtype="U64"),
        "layer_indices": np.asarray(selected_layers, dtype=np.int32),
        "step_names": np.asarray(["pre", "post"], dtype="U8"),
        "feature_names": np.asarray(FEATURE_NAMES, dtype="U32"),
        "top_fraction": np.asarray(0.10, dtype=np.float32),
        "connectivity": np.asarray("8-neighbor", dtype="U16"),
        "transformers_version": np.asarray(transformers.__version__),
        "identity_checked": np.asarray(identity_checked),
        "max_forward_logit_error": np.asarray(max_forward_logit_error),
        "max_clean_row_sum_error": np.asarray(max_clean_row_sum_error),
        "max_branch_difference": np.asarray(max_branch_difference),
        **numeric,
    }
    for key in ("no_rope", "sink_clean", "sink_mass"):
        if not np.isfinite(arrays[key]).all():
            raise AssertionError(f"Non-finite values in {key}")
    if identity_checked and max_forward_logit_error > 1e-6:
        raise AssertionError(f"Output flags changed logits: {max_forward_logit_error}")
    if max_clean_row_sum_error > 1e-6:
        raise AssertionError(f"Sink-clean rows do not sum to one: {max_clean_row_sum_error}")
    if output["label"] and max_branch_difference == 0.0:
        raise AssertionError("Sink deletion did not change any cached feature")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    logger.info(
        "Wrote %s: N=%d layers=%d row-sum-error=%.3g branch-diff=%.3g",
        out_path,
        len(arrays["label"]),
        len(selected_layers),
        max_clean_row_sum_error,
        max_branch_difference,
    )


if __name__ == "__main__":
    main()
