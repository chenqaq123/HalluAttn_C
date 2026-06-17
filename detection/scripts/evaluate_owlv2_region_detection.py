#!/usr/bin/env python3
"""Evaluate OWLv2 region evidence on CHAIR object-mention detection.

The POPE experiments showed that OWLv2 target scores are strong but vulnerable
to related-object false positives. This script checks whether the same region
evidence is useful for the CHAIR object-mention hallucination detection task
under the standard position-controlled metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.data.build_object_cache import read_jsonl
from detection.baselines.metrics import compute_metric_bundle

OWL_CACHE_UTILS = PROJECT_ROOT / "mitigation" / "scripts"
if str(OWL_CACHE_UTILS) not in sys.path:
    sys.path.insert(0, str(OWL_CACHE_UTILS))
from owlv2_cache_utils import encode_image_object_scores, load_score_cache, save_score_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OWLv2 region scores on CHAIR object mentions")
    p.add_argument("--object_cache", default="detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--coco_path", default="/home/chenguanxu/common_dataset/coco-2014-dataset")
    p.add_argument("--output_dir", default="detection/baselines/results/owlv2_region_detection")
    p.add_argument("--owlv2_model_path", default="/home/chenguanxu/common_model/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_neighbors", type=int, default=10)
    p.add_argument("--two_stage_low", type=float, default=0.10)
    p.add_argument("--two_stage_high", type=float, default=0.16)
    p.add_argument("--two_stage_margin", type=float, default=-0.15)
    p.add_argument("--image_score_cache", default="", help="Optional NPZ cache from OWLv2 image-object scoring")
    p.add_argument("--save_image_score_cache", default="", help="Optional NPZ path to save image-object scores")
    return p.parse_args()


def canonical_image_key(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"


def coco_image_path(coco_path: Path, image_id: int) -> Path:
    return coco_path / "val2014" / canonical_image_key(image_id)


def read_neighbors(path: Path, top_k: int) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        obj: [item["object"] for item in items[:top_k]]
        for obj, items in raw.items()
    }


def two_stage_yes(target_score: float, margin: float, low: float, high: float, margin_threshold: float) -> bool:
    return target_score > high or (target_score > low and margin > margin_threshold)


def two_stage_continuous_score(target_score: float, margin: float, low: float, high: float, margin_threshold: float) -> float:
    high_branch = target_score - high
    medium_branch = min(target_score - low, margin - margin_threshold)
    return max(high_branch, medium_branch)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    records = read_jsonl(args.object_cache)
    if args.limit > 0:
        records = records[: args.limit]
    neighbors = read_neighbors(Path(args.neighbors_json), top_k=args.top_neighbors)

    object_names = {str(record["word"]) for record in records}
    for word in list(object_names):
        object_names.update(neighbors.get(word, []))
    object_list = sorted(object_names)
    prompts = [f"a photo of a {obj}" for obj in object_list]

    unique_images = sorted({int(record["image_id"]) for record in records})
    image_keys = [canonical_image_key(image_id) for image_id in unique_images]
    if args.image_score_cache:
        image_scores_by_key = load_score_cache(args.image_score_cache, image_keys, object_list)
    else:
        processor = Owlv2Processor.from_pretrained(args.owlv2_model_path, local_files_only=True)
        model = Owlv2ForObjectDetection.from_pretrained(args.owlv2_model_path, local_files_only=True).to(device)
        model.eval()

        image_scores_by_key: dict[str, dict[str, float]] = {}
        coco_path = Path(args.coco_path)
        for start in range(0, len(unique_images), args.batch_size):
            batch_ids = unique_images[start:start + args.batch_size]
            batch_keys = [canonical_image_key(image_id) for image_id in batch_ids]
            images = [Image.open(coco_image_path(coco_path, image_id)).convert("RGB") for image_id in batch_ids]
            score_batch = encode_image_object_scores(model, processor, images, prompts, object_list, device)
            for image in images:
                image.close()
            for image_key, scores in zip(batch_keys, score_batch):
                image_scores_by_key[image_key] = scores
        if args.save_image_score_cache:
            save_score_cache(args.save_image_score_cache, image_scores_by_key, object_list)

    scored_rows: list[dict] = []
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int32)
    gen_pos = np.asarray([int(record["gen_pos"]) for record in records], dtype=np.int32)
    scores = {
        "owlv2_target_absence": np.full(len(records), np.nan, dtype=np.float32),
        "owlv2_margin_absence": np.full(len(records), np.nan, dtype=np.float32),
        "owlv2_neighbor_presence": np.full(len(records), np.nan, dtype=np.float32),
        "owlv2_two_stage_absence": np.full(len(records), np.nan, dtype=np.float32),
        "owlv2_two_stage_binary": np.full(len(records), np.nan, dtype=np.float32),
    }

    for idx, record in enumerate(records):
        target = str(record["word"])
        per_object = image_scores_by_key[canonical_image_key(int(record["image_id"]))]
        target_score = per_object[target]
        neighbor_scores = [(name, per_object[name]) for name in neighbors.get(target, []) if name in per_object]
        if neighbor_scores:
            best_neighbor, best_neighbor_score = max(neighbor_scores, key=lambda item: item[1])
            margin = target_score - best_neighbor_score
        else:
            best_neighbor, best_neighbor_score = "", 0.0
            margin = target_score
        two_stage_score = two_stage_continuous_score(
            target_score,
            margin,
            args.two_stage_low,
            args.two_stage_high,
            args.two_stage_margin,
        )
        two_stage_present = two_stage_yes(
            target_score,
            margin,
            args.two_stage_low,
            args.two_stage_high,
            args.two_stage_margin,
        )
        scores["owlv2_target_absence"][idx] = -target_score
        scores["owlv2_margin_absence"][idx] = -margin
        scores["owlv2_neighbor_presence"][idx] = best_neighbor_score
        scores["owlv2_two_stage_absence"][idx] = -two_stage_score
        scores["owlv2_two_stage_binary"][idx] = 0.0 if two_stage_present else 1.0
        scored_rows.append({
            **record,
            "target_score": target_score,
            "best_neighbor": best_neighbor,
            "best_neighbor_score": best_neighbor_score,
            "tdev_margin": margin,
            "two_stage_score": two_stage_score,
            "two_stage_present": int(two_stage_present),
            "neighbors": "|".join(neighbors.get(target, [])),
        })

    metrics = {
        name: compute_metric_bundle(labels, values, gen_pos)
        for name, values in scores.items()
    }
    payload = {
        "object_cache": args.object_cache,
        "neighbors_json": args.neighbors_json,
        "owlv2_model_path": args.owlv2_model_path,
        "samples": len(records),
        "hallucinated": int(labels.sum()),
        "object_count": len(object_list),
        "top_neighbors": args.top_neighbors,
        "two_stage_low": args.two_stage_low,
        "two_stage_high": args.two_stage_high,
        "two_stage_margin": args.two_stage_margin,
        "image_score_cache": args.image_score_cache,
        "save_image_score_cache": args.save_image_score_cache,
        "score_direction": "larger score means more likely hallucinated",
        "metrics": metrics,
    }
    write_csv(output_dir / "owlv2_region_detection_scores.csv", scored_rows)
    np.savez(output_dir / "owlv2_region_detection_scores.npz", labels=labels, gen_pos=gen_pos, **scores)
    with (output_dir / "owlv2_region_detection_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'owlv2_region_detection_metrics.json'}")


if __name__ == "__main__":
    main()
