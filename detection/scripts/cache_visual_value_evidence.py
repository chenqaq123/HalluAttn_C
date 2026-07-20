#!/usr/bin/env python3
"""Cache target-aligned visual value evidence from a frozen LLaVA forward."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from tqdm import tqdm

DETECTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DETECTION_ROOT.parent
sys.path.insert(0, str(DETECTION_ROOT / "src"))

from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import build_caption_prompt, load_model_and_processor  # noqa: E402
from sinkdetect.value_evidence import (  # noqa: E402
    CONFIDENCE_FEATURE_NAMES,
    JOINT_FEATURE_NAMES,
    PER_HEAD_FEATURE_NAMES,
    causal_attention_rows,
    confidence_features,
    evidence_features,
    project_source_contributions,
    reconstruct_attention_output,
    repeat_kv,
)

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


def _resolve_language_components(model):
    candidates = [
        lambda m: (m.language_model.model.layers, m.language_model.model.norm, m.language_model.lm_head),
        lambda m: (m.model.language_model.layers, m.model.language_model.norm, m.lm_head),
        lambda m: (m.model.language_model.model.layers, m.model.language_model.model.norm, m.model.language_model.lm_head),
    ]
    for resolve in candidates:
        try:
            layers, final_norm, lm_head = resolve(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList) and hasattr(lm_head, "weight"):
            return layers, final_norm, lm_head
    raise RuntimeError(f"Cannot locate language-model components in {type(model).__name__}")


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


def _qkv_states(attention, normalized_hidden: torch.Tensor, query_positions: torch.Tensor):
    config = attention.config
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(attention, "head_dim", config.hidden_size // heads))
    sequence = normalized_hidden.shape[0]
    query = attention.q_proj(normalized_hidden[query_positions])
    query = query.view(query_positions.numel(), heads, head_dim).transpose(0, 1)
    key = attention.k_proj(normalized_hidden)
    key = key.view(sequence, kv_heads, head_dim).transpose(0, 1)
    value = attention.v_proj(normalized_hidden)
    value = value.view(sequence, kv_heads, head_dim).transpose(0, 1)
    return query, key, value


def _direct_attention_output(
    attention_rows: torch.Tensor,
    value_states: torch.Tensor,
    attention_module,
    source_start: int = 0,
    source_end: int | None = None,
) -> torch.Tensor:
    if source_end is None:
        source_end = attention_rows.shape[-1]
    values = repeat_kv(
        value_states, attention_rows.shape[0] // value_states.shape[0]
    ).float()
    head_output = torch.einsum(
        "hrv,hvd->hrd",
        attention_rows[:, :, source_start:source_end].float(),
        values[:, source_start:source_end],
    )
    flattened = head_output.transpose(0, 1).reshape(head_output.shape[1], -1)
    projected = F.linear(
        flattened.float(), attention_module.o_proj.weight.float(), bias=None
    )
    return projected.float()


def _branch_features(
    attention_rows: torch.Tensor,
    value_states: torch.Tensor,
    attention_module,
    residual_rows: torch.Tensor,
    target_ids: torch.Tensor,
    final_norm,
    unembedding_weight: torch.Tensor,
    vis_start: int,
    vis_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contributions, mass = project_source_contributions(
        attention_rows,
        value_states,
        attention_module.o_proj.weight,
        vis_start,
        vis_end,
    )
    per_head, joint = evidence_features(
        residual_rows,
        contributions,
        mass,
        target_ids,
        final_norm,
        unembedding_weight,
    )
    return per_head, joint, contributions


def _empty_arrays(layer_count: int, heads: int) -> dict[str, np.ndarray]:
    return {
        "raw": np.empty((0, layer_count, 2, heads, len(PER_HEAD_FEATURE_NAMES)), np.float32),
        "no_rope": np.empty((0, layer_count, 2, heads, len(PER_HEAD_FEATURE_NAMES)), np.float32),
        "joint_raw": np.empty((0, layer_count, 2, len(JOINT_FEATURE_NAMES)), np.float32),
        "joint_no_rope": np.empty((0, layer_count, 2, len(JOINT_FEATURE_NAMES)), np.float32),
        "confidence": np.empty((0, 2, len(CONFIDENCE_FEATURE_NAMES)), np.float32),
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
    layers, final_norm, lm_head = _resolve_language_components(model)
    selected_layers = _parse_layers(args.layers, len(layers))
    heads = int(layers[selected_layers[0]].self_attn.config.num_attention_heads)

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    output: dict[str, list] = {
        key: []
        for key in (
            "object_id",
            "image_id",
            "token_pos",
            "gen_pos",
            "label",
            "word",
            "target_token_id",
            "target_token",
            "raw",
            "no_rope",
            "joint_raw",
            "joint_no_rope",
            "confidence",
        )
    }
    max_forward_logit_error = 0.0
    max_reconstruction_error = 0.0
    max_head_sum_error = 0.0
    max_raw_mass_error = 0.0
    max_branch_difference = 0.0
    identity_checked = False

    for image_index, image_id in enumerate(
        tqdm(image_ids, desc=f"visual-value-{args.shard_idx}")
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
        mention_positions = torch.stack([pre_positions, pre_positions + 1], dim=1)
        query_positions = mention_positions.reshape(-1)
        object_target_ids = input_ids[pre_positions + 1]
        evidence_target_ids = object_target_ids[:, None].expand(-1, 2).reshape(-1)
        vis_start, vis_end = find_vis_bounds(input_ids, int(image_token_id))

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
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )
            if baseline_logits is not None:
                error = float((baseline_logits - result.logits.float()).abs().max().item())
                max_forward_logit_error = max(max_forward_logit_error, error)
                identity_checked = True

            confidence, _ = confidence_features(
                result.logits[0], input_ids, query_positions
            )
            confidence = confidence.reshape(len(valid_records), 2, -1)
            image_raw = []
            image_no_rope = []
            image_joint_raw = []
            image_joint_no_rope = []

            for layer_index in selected_layers:
                layer = layers[layer_index]
                attention = layer.self_attn
                residual = result.hidden_states[layer_index][0]
                normalized_hidden = layer.input_layernorm(residual)
                query, key, value = _qkv_states(
                    attention, normalized_hidden, query_positions
                )
                raw_attention = result.attentions[layer_index][0][
                    :, query_positions, :
                ].float()
                no_rope_attention = causal_attention_rows(
                    query,
                    key,
                    query_positions,
                    scaling=float(getattr(attention, "scaling", query.shape[-1] ** -0.5)),
                )
                residual_rows = residual[query_positions].float()

                raw, joint_raw, raw_contributions = _branch_features(
                    raw_attention,
                    value,
                    attention,
                    residual_rows,
                    evidence_target_ids,
                    final_norm,
                    lm_head.weight,
                    vis_start,
                    vis_end,
                )
                no_rope, joint_no_rope, no_rope_contributions = _branch_features(
                    no_rope_attention,
                    value,
                    attention,
                    residual_rows,
                    evidence_target_ids,
                    final_norm,
                    lm_head.weight,
                    vis_start,
                    vis_end,
                )

                reconstructed = reconstruct_attention_output(
                    raw_attention, value, attention.o_proj.weight
                )
                direct = _direct_attention_output(raw_attention, value, attention)
                max_reconstruction_error = max(
                    max_reconstruction_error,
                    float((reconstructed - direct).abs().max().item()),
                )
                raw_joint_direct = _direct_attention_output(
                    raw_attention, value, attention, vis_start, vis_end
                )
                no_rope_joint_direct = _direct_attention_output(
                    no_rope_attention, value, attention, vis_start, vis_end
                )
                max_head_sum_error = max(
                    max_head_sum_error,
                    float(
                        torch.maximum(
                            (raw_joint_direct - raw_contributions.sum(dim=1)).abs().max(),
                            (no_rope_joint_direct - no_rope_contributions.sum(dim=1)).abs().max(),
                        ).item()
                    ),
                )
                raw_mass = raw_attention[:, :, vis_start:vis_end].sum(dim=-1).transpose(0, 1)
                max_raw_mass_error = max(
                    max_raw_mass_error,
                    float((raw[..., 0] - raw_mass).abs().max().item()),
                )
                max_branch_difference = max(
                    max_branch_difference,
                    float((raw_attention - no_rope_attention).abs().max().item()),
                )

                shape = (len(valid_records), 2, heads, len(PER_HEAD_FEATURE_NAMES))
                joint_shape = (len(valid_records), 2, len(JOINT_FEATURE_NAMES))
                image_raw.append(raw.reshape(shape).cpu().numpy())
                image_no_rope.append(no_rope.reshape(shape).cpu().numpy())
                image_joint_raw.append(joint_raw.reshape(joint_shape).cpu().numpy())
                image_joint_no_rope.append(
                    joint_no_rope.reshape(joint_shape).cpu().numpy()
                )

        per_head_raw = np.stack(image_raw, axis=1)
        per_head_no_rope = np.stack(image_no_rope, axis=1)
        joint_raw_array = np.stack(image_joint_raw, axis=1)
        joint_no_rope_array = np.stack(image_joint_no_rope, axis=1)
        for mention_index, record in enumerate(valid_records):
            for key in ("object_id", "image_id", "token_pos", "gen_pos", "label", "word"):
                output[key].append(record[key])
            target_id = int(object_target_ids[mention_index].item())
            output["target_token_id"].append(target_id)
            output["target_token"].append(
                processor.tokenizer.decode([target_id], skip_special_tokens=False).strip()
            )
            output["raw"].append(per_head_raw[mention_index])
            output["no_rope"].append(per_head_no_rope[mention_index])
            output["joint_raw"].append(joint_raw_array[mention_index])
            output["joint_no_rope"].append(joint_no_rope_array[mention_index])
            output["confidence"].append(confidence[mention_index].cpu().numpy())

        del inputs, result, baseline_logits
        if image_index % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    numeric = _empty_arrays(len(selected_layers), heads)
    if output["label"]:
        for key in numeric:
            numeric[key] = np.asarray(output[key], dtype=np.float32)
    arrays = {
        "object_id": np.asarray(output["object_id"], dtype=np.int64),
        "image_id": np.asarray(output["image_id"], dtype=np.int64),
        "token_pos": np.asarray(output["token_pos"], dtype=np.int32),
        "gen_pos": np.asarray(output["gen_pos"], dtype=np.int32),
        "label": np.asarray(output["label"], dtype=np.int32),
        "word": np.asarray(output["word"], dtype="U64"),
        "target_token_id": np.asarray(output["target_token_id"], dtype=np.int32),
        "target_token": np.asarray(output["target_token"], dtype="U64"),
        "layer_indices": np.asarray(selected_layers, dtype=np.int32),
        "step_names": np.asarray(["pre", "post"], dtype="U8"),
        "per_head_feature_names": np.asarray(PER_HEAD_FEATURE_NAMES, dtype="U32"),
        "joint_feature_names": np.asarray(JOINT_FEATURE_NAMES, dtype="U32"),
        "confidence_feature_names": np.asarray(CONFIDENCE_FEATURE_NAMES, dtype="U32"),
        "transformers_version": np.asarray(transformers.__version__),
        "max_forward_logit_error": np.asarray(max_forward_logit_error),
        "max_reconstruction_error": np.asarray(max_reconstruction_error),
        "max_head_sum_error": np.asarray(max_head_sum_error),
        "max_raw_mass_error": np.asarray(max_raw_mass_error),
        "max_branch_difference": np.asarray(max_branch_difference),
        "identity_checked": np.asarray(identity_checked),
        **numeric,
    }
    for key in numeric:
        if not np.isfinite(arrays[key]).all():
            raise AssertionError(f"Non-finite values in {key}")
    if identity_checked and max_forward_logit_error > 1e-6:
        raise AssertionError(f"Output flags changed logits: {max_forward_logit_error}")
    if max_reconstruction_error > 1e-4:
        raise AssertionError(
            f"Per-head output reconstruction error is {max_reconstruction_error}"
        )
    if max_head_sum_error > 1e-4:
        raise AssertionError(f"Per-head source sum error is {max_head_sum_error}")
    if max_raw_mass_error > 1e-6:
        raise AssertionError(f"Cached raw visual mass error is {max_raw_mass_error}")
    if output["label"] and max_branch_difference == 0.0:
        raise AssertionError("Raw and no-RoPE branches are identical")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    logger.info(
        "Wrote %s: N=%d layers=%d reconstruction=%.3g branch-diff=%.3g",
        out_path,
        len(arrays["label"]),
        len(selected_layers),
        max_reconstruction_error,
        max_branch_difference,
    )


if __name__ == "__main__":
    main()
