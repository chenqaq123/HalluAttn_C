#!/usr/bin/env python3
"""Diagnostic Stage: cache PER-HEAD attention shape features.

The main row cache stores mean-over-heads visual attention only. To test
whether a position-independent grounding signal survives in attention but is
washed out by averaging over heads/layers, we need per-head rows. This script
runs the teacher-forced VLM forward once (no DetectionAdapter needed; it reads
``outputs.attentions`` directly) and, for each object mention, computes a small
set of *shape* features of every head's normalized visual-attention
distribution at the chosen layers.

Output (per shard) ``per_head_row_cache[_shard{i}].npz``:
    feats        (N, L, H, F) float32  -- shape features per (layer, head)
    labels       (N,) int32
    token_pos    (N,) int32
    image_ids    (N,) int64
    words        (N,) U64
    layer_indices(L,) int32
    feature_names(F,) U32            -- names of the F shape features

Features F per head (all on the L1-normalized visual row, sinks NOT stripped --
the diagnostic question is about per-head granularity, not sinks):
    0 entropy        H(r)              (high = diffuse)
    1 neg_top1_mass  -max(r)
    2 neg_top5_mass  -sum top-5(r)
    3 neg_max_over_mean  -(max/mean)
Sign convention matches scoring: higher feature value => more "hallucination-like".
"""

from __future__ import annotations

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

DETECTION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DETECTION_ROOT.parent
sys.path.insert(0, str(DETECTION_ROOT / "src"))
PAS_SRC = PROJECT_ROOT.parent / "pas" / "src"
if str(PAS_SRC) not in sys.path:
    sys.path.insert(0, str(PAS_SRC))

from sinkdetect.chair import (
    add_token_labels,
    evaluate_chair,
    find_first_mentions,
    load_chair_evaluator,
)
from sinkdetect.sink_utils import find_vis_bounds
from sinkdetect.utils import (
    build_caption_prompt,
    load_generation_json,
    load_model_and_processor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = ["entropy", "neg_top1_mass", "neg_top5_mass", "neg_max_over_mean"]
_EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache per-head attention shape features")
    p.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--coco_path", type=str, default="/data/common_dataset/coco-2014-dataset/")
    p.add_argument("--generation_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--chair_pkl", type=str, default=str(PROJECT_ROOT.parent / "pas" / "data" / "chair_coco.pkl"))
    p.add_argument("--per_head_layers", type=str, default="0,1,10,22,31",
                   help="Comma-separated decoder layer indices to extract per-head features from.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def _parse_layers(spec: str) -> list[int]:
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not layers:
        raise ValueError("--per_head_layers must contain at least one layer")
    return layers


def _head_shape_features(rows: torch.Tensor) -> torch.Tensor:
    """rows: (H, n_v) raw attention over visual tokens. Returns (H, F) features."""
    rows = rows.float()
    rows = rows / rows.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    ent = -(rows * rows.clamp_min(_EPS).log()).sum(dim=-1)
    sorted_, _ = rows.sort(dim=-1, descending=True)
    top1 = sorted_[:, 0]
    top5 = sorted_[:, :5].sum(dim=-1)
    mean = rows.mean(dim=-1)
    max_over_mean = top1 / mean.clamp_min(_EPS)
    feats = torch.stack([ent, -top1, -top5, -max_over_mean], dim=-1)  # (H, F)
    return feats


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_head_layers = _parse_layers(args.per_head_layers)

    gen_data = load_generation_json(args.generation_json)
    if args.limit > 0:
        gen_data = gen_data[: args.limit]
    if args.num_shards > 1:
        gen_data = gen_data[args.shard_idx :: args.num_shards]
        logger.info("shard %d/%d: %d entries", args.shard_idx, args.num_shards, len(gen_data))
    suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""

    evaluator = load_chair_evaluator(args.chair_pkl)
    chair_input_path = output_dir / f"per_head_chair_input{suffix}.json"
    chair_data = [{"image_id": e["image_id"], "caption": e["caption"]} for e in gen_data]
    eval_dicts, _ = evaluate_chair(evaluator, data=chair_data, json_path=str(chair_input_path))

    from transformers import LlavaProcessor

    processor = LlavaProcessor.from_pretrained(args.model_path)
    sequences = [processor.tokenizer.encode(e["caption"], add_special_tokens=False) for e in gen_data]
    eval_dicts = add_token_labels(eval_dicts, sequences, processor, evaluator)

    # eager attn implementation is required so outputs.attentions is populated.
    model, _ = load_model_and_processor(args.model_path, device, attn_implementation="eager")
    prompt_text = build_caption_prompt()

    feats_all: list[np.ndarray] = []
    labels: list[int] = []
    image_ids: list[int] = []
    token_positions: list[int] = []
    words: list[str] = []

    for idx, (entry, eval_dict) in enumerate(tqdm(
        zip(gen_data, eval_dicts), total=len(gen_data), desc="Caching per-head features",
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
        inputs = processor(
            images=image, text=prompt_text + " " + entry["caption"], return_tensors="pt"
        ).to(device, dtype=torch.float16)
        img_token_id = getattr(processor, "image_token_id", None) or getattr(
            processor, "image_token_index", 32000
        )
        vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], img_token_id)

        with torch.inference_mode():
            outputs = model.forward(
                **inputs, output_attentions=True, output_hidden_states=False
            )
        # outputs.attentions: tuple(num_layers) of (batch, n_heads, q, k)
        attn_layers = [outputs.attentions[l][0] for l in per_head_layers]  # each (H, q, k)
        prompt_end_idx = entry["prompt_end_idx"]

        for mention in mentions:
            token_pos = mention["pos"] + prompt_end_idx
            per_layer = []
            for A in attn_layers:
                rows = A[:, token_pos, vis_start:vis_end]  # (H, n_v)
                per_layer.append(_head_shape_features(rows).cpu().numpy())  # (H, F)
            feats_all.append(np.stack(per_layer, axis=0).astype(np.float32))  # (L, H, F)
            labels.append(1 if mention["hallucinated"] else 0)
            image_ids.append(int(image_id))
            token_positions.append(int(token_pos))
            words.append(str(mention["word"]))

        del inputs, outputs, attn_layers
        if idx % 4 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    out = output_dir / f"per_head_row_cache{suffix}.npz"
    np.savez_compressed(
        out,
        feats=np.stack(feats_all, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        token_pos=np.asarray(token_positions, dtype=np.int32),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        words=np.asarray(words, dtype="U64"),
        layer_indices=np.asarray(per_head_layers, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U32"),
    )
    logger.info("Wrote %s  (N=%d, L=%d, F=%d)", out, len(labels), len(per_head_layers), len(FEATURE_NAMES))


if __name__ == "__main__":
    main()
