#!/usr/bin/env python3
"""Cache POPE target-token per-head attention-shape rows.

This is the POPE-side counterpart of detection/scripts/cache_per_head_rows.py.
It does not generate answers. For each POPE question, it runs one forward pass on
LLaVA's question prompt and extracts per-head visual-attention shape features at
question target-object tokens. The rows align with semantic-neighbor audit CSVs,
so they can later test whether calibrated LH-Shape transfers to POPE gating and
related-present false-positive controls.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import load_model_and_processor  # noqa: E402
from src.data import iter_pope_records, load_pope_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = ["entropy", "neg_top1_mass", "neg_top5_mass", "neg_max_over_mean"]
_EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache POPE per-head target-token attention-shape features")
    p.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--audit_csv", default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--per_head_layers", default="0,1,10,22,31")
    p.add_argument("--target_token_policy", choices=["first", "last", "mean"], default="mean")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_in_8bit", action="store_true", help="Load LLaVA with bitsandbytes 8-bit weights for low-memory cache runs")
    p.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA with bitsandbytes 4-bit weights for low-memory cache runs")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def parse_layers(spec: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    if not layers:
        raise ValueError("--per_head_layers must contain at least one layer")
    return layers


def load_cache_model_and_processor(args: argparse.Namespace, device: torch.device):
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Choose at most one of --load_in_8bit and --load_in_4bit")
    if not (args.load_in_8bit or args.load_in_4bit):
        return load_model_and_processor(
            args.model_path,
            device,
            cache_dir=args.cache_dir or None,
            attn_implementation="eager",
        )

    from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaProcessor

    kwargs = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    processor = LlavaProcessor.from_pretrained(args.model_path, **kwargs)
    quant_config = BitsAndBytesConfig(
        load_in_8bit=bool(args.load_in_8bit),
        load_in_4bit=bool(args.load_in_4bit),
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map={"": int(args.device)},
        attn_implementation="eager",
        **kwargs,
    )
    vision_config = getattr(model.config, "vision_config", None)
    if getattr(processor, "patch_size", None) is None and vision_config is not None:
        processor.patch_size = getattr(vision_config, "patch_size", None)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    additional_image_tokens = getattr(processor, "num_additional_image_tokens", None)
    if additional_image_tokens is None or int(additional_image_tokens) == 0:
        processor.num_additional_image_tokens = 1
    model.eval()
    return model, processor


def read_audit_rows(path: Path, splits: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    split_set = set(splits)
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] in split_set:
                rows[(row["split"], str(row["question_id"]))] = row
    if not rows:
        raise ValueError(f"No audit rows found in {path} for splits={splits}")
    return rows


def pope_prompt(question: str) -> str:
    return f"<image>\nUSER: {question} Please just answer yes or no.\nASSISTANT:"


def head_shape_features(rows: torch.Tensor) -> torch.Tensor:
    """rows: (H, n_v) raw attention over visual tokens. Returns (H, F)."""
    rows = rows.float()
    rows = rows / rows.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    ent = -(rows * rows.clamp_min(_EPS).log()).sum(dim=-1)
    sorted_rows, _ = rows.sort(dim=-1, descending=True)
    top1 = sorted_rows[:, 0]
    top5 = sorted_rows[:, :5].sum(dim=-1)
    mean = rows.mean(dim=-1)
    max_over_mean = top1 / mean.clamp_min(_EPS)
    return torch.stack([ent, -top1, -top5, -max_over_mean], dim=-1)


def candidate_tokenizations(tokenizer, target: str) -> list[list[int]]:
    candidates = []
    for text in (target, " " + target, target.replace(" ", "_"), " " + target.replace(" ", "_")):
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [int(x) for x in ids]
        if ids and ids not in candidates:
            candidates.append(ids)
    return candidates


def find_subsequence(sequence: list[int], pattern: list[int], end_before: int | None = None) -> tuple[int, int] | None:
    if end_before is not None:
        sequence = sequence[:end_before]
    n = len(pattern)
    if n == 0 or len(sequence) < n:
        return None
    matches = []
    for start in range(0, len(sequence) - n + 1):
        if sequence[start:start + n] == pattern:
            matches.append((start, start + n))
    return matches[-1] if matches else None


def assistant_start(input_ids: list[int], tokenizer) -> int:
    patterns = []
    for text in ("ASSISTANT:", " ASSISTANT:", "\nASSISTANT:"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            patterns.append([int(x) for x in ids])
    best = None
    for pattern in patterns:
        match = find_subsequence(input_ids, pattern)
        if match is not None and (best is None or match[0] > best[0]):
            best = match
    return best[0] if best else len(input_ids)


def find_target_span(input_ids: list[int], tokenizer, target: str) -> tuple[int, int, list[int]]:
    end_before = assistant_start(input_ids, tokenizer)
    for pattern in candidate_tokenizations(tokenizer, target):
        match = find_subsequence(input_ids, pattern, end_before=end_before)
        if match is not None:
            return match[0], match[1], pattern
    decoded = tokenizer.decode(input_ids[:end_before], skip_special_tokens=False)
    raise ValueError(f"Could not locate target {target!r} in prompt tokens. Prompt prefix: {decoded!r}")


def target_positions(start: int, end: int, policy: str) -> list[int]:
    if policy == "first":
        return [start]
    if policy == "last":
        return [end - 1]
    return list(range(start, end))


def iter_joined_records(args: argparse.Namespace, splits: list[str], audit_rows: dict[tuple[str, str], dict[str, str]]) -> list[dict]:
    joined = []
    for split in splits:
        for record in iter_pope_records(args.pope_dir, args.coco_path, split, limit=args.limit):
            key = (split, str(record["question_id"]))
            audit = audit_rows.get(key)
            if audit is None:
                continue
            joined.append({**record, **audit, "split": split, "question_id": str(record["question_id"])})
    if args.num_shards > 1:
        joined = joined[args.shard_idx::args.num_shards]
    return joined


def main() -> None:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    layers = parse_layers(args.per_head_layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""

    audit_rows = read_audit_rows(Path(args.audit_csv), splits)
    records = iter_joined_records(args, splits, audit_rows)
    logger.info("Caching %d POPE rows for splits=%s shard=%d/%d", len(records), splits, args.shard_idx, args.num_shards)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_cache_model_and_processor(args, device)
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    feats_all: list[np.ndarray] = []
    labels_absent: list[int] = []
    image_ids: list[int] = []
    question_ids: list[str] = []
    split_values: list[str] = []
    targets: list[str] = []
    negative_types: list[str] = []
    token_start_values: list[int] = []
    token_end_values: list[int] = []
    token_pos_values: list[int] = []
    missing_targets = []

    for step, record in enumerate(tqdm(records, desc="Caching POPE per-head features"), start=1):
        target = str(record["target"])
        prompt = pope_prompt(str(record["question"]))
        image = load_pope_image(record)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
        image.close()
        input_ids = [int(x) for x in inputs["input_ids"][0].detach().cpu().tolist()]
        try:
            target_start, target_end, _pattern = find_target_span(input_ids, tokenizer, target)
        except ValueError as exc:
            missing_targets.append({"split": record["split"], "question_id": record["question_id"], "target": target, "error": str(exc)})
            del inputs
            continue
        positions = target_positions(target_start, target_end, args.target_token_policy)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        representative_pos = positions[-1]
        vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], image_token_id)

        with torch.inference_mode():
            outputs = model.forward(**inputs, output_attentions=True, output_hidden_states=False)
        per_layer = []
        for layer_idx in layers:
            attn = outputs.attentions[layer_idx][0]
            rows = attn[:, position_tensor, vis_start:vis_end].float().mean(dim=1)
            per_layer.append(head_shape_features(rows).detach().cpu().numpy())
        feats_all.append(np.stack(per_layer, axis=0).astype(np.float32))
        labels_absent.append(1 if str(record["label"]).lower() == "no" else 0)
        image_ids.append(int(record["image_id"]))
        question_ids.append(str(record["question_id"]))
        split_values.append(str(record["split"]))
        targets.append(target)
        negative_types.append(str(record.get("negative_type", "")))
        token_start_values.append(int(target_start))
        token_end_values.append(int(target_end))
        token_pos_values.append(int(representative_pos))

        del inputs, outputs
        if step % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    out = output_dir / f"pope_per_head_row_cache{suffix}.npz"
    if feats_all:
        feats = np.stack(feats_all, axis=0).astype(np.float32)
    else:
        feats = np.zeros((0, len(layers), 0, len(FEATURE_NAMES)), dtype=np.float32)
    np.savez_compressed(
        out,
        feats=feats,
        labels=np.asarray(labels_absent, dtype=np.int32),
        label_name=np.asarray(["target_absent"], dtype="U32"),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        question_ids=np.asarray(question_ids, dtype="U32"),
        splits=np.asarray(split_values, dtype="U16"),
        targets=np.asarray(targets, dtype="U64"),
        negative_types=np.asarray(negative_types, dtype="U64"),
        token_pos=np.asarray(token_pos_values, dtype=np.int32),
        target_token_start=np.asarray(token_start_values, dtype=np.int32),
        target_token_end=np.asarray(token_end_values, dtype=np.int32),
        layer_indices=np.asarray(layers, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U32"),
        target_token_policy=np.asarray([args.target_token_policy], dtype="U16"),
    )
    metrics = {
        "rows_requested": len(records),
        "rows_cached": len(labels_absent),
        "missing_targets": len(missing_targets),
        "missing_target_examples": missing_targets[:10],
        "splits": splits,
        "layers": layers,
        "target_token_policy": args.target_token_policy,
        "label_name": "target_absent",
        "absent_rows": int(np.sum(labels_absent)),
        "present_rows": int(len(labels_absent) - np.sum(labels_absent)),
        "cache": str(out),
    }
    metrics_path = output_dir / f"pope_per_head_row_cache_metrics{suffix}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)
    logger.info("Wrote %s", metrics_path)


if __name__ == "__main__":
    main()
