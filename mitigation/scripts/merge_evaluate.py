#!/usr/bin/env python3
"""Merge mitigation shards and evaluate one method on POPE or CHAIR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from src.evaluation import basic_caption_stats, pope_metrics, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["pope", "chair"], required=True)
    p.add_argument("--method_dir", required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--chair_pkl", default="")
    p.add_argument("--invalid_policy", choices=["error", "as_wrong", "drop"], default="error")
    return p.parse_args()


def merge_shards(method_dir: Path, num_shards: int) -> list[dict]:
    rows = []
    for shard_idx in range(num_shards):
        path = method_dir / f"shard{shard_idx}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing shard result: {path}")
        rows.extend(read_jsonl(path))
    key = "question_id" if rows and "question_id" in rows[0] else "image_id"
    rows.sort(key=lambda row: row[key])
    output = method_dir / "predictions.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def evaluate_chair(rows: list[dict], chair_pkl: str, method_dir: Path) -> dict:
    if not chair_pkl:
        raise ValueError("--chair_pkl is required for CHAIR evaluation")
    from sinkdetect.chair import evaluate_chair, load_chair_evaluator

    evaluator = load_chair_evaluator(chair_pkl)
    data = [{"image_id": row["image_id"], "caption": row["caption"]} for row in rows]
    per_sample, overall = evaluate_chair(
        evaluator,
        data=data,
        json_path=str(method_dir / "chair_input.json"),
    )
    mention_counts = []
    hallucinated_counts = []
    with (method_dir / "chair_per_sample.jsonl").open("w", encoding="utf-8") as f:
        for sample in per_sample:
            generated = list(sample.get("mscoco_generated_words", []))
            grounded = set(sample.get("mscoco_gt_words", []))
            hallucinated = sum(word not in grounded for word in generated)
            mention_counts.append(len(generated))
            hallucinated_counts.append(hallucinated)
            f.write(json.dumps({
                "image_id": sample.get("image_id"),
                "caption": sample.get("caption", ""),
                "object_mentions": len(generated),
                "hallucinated_mentions": hallucinated,
            }) + "\n")
    stats = basic_caption_stats(rows)
    stats.update({
        "mean_object_mentions": sum(mention_counts) / len(mention_counts) if mention_counts else 0.0,
        "mean_hallucinated_mentions": sum(hallucinated_counts) / len(hallucinated_counts) if hallucinated_counts else 0.0,
        "total_object_mentions": sum(mention_counts),
        "total_hallucinated_mentions": sum(hallucinated_counts),
    })
    return {"chair": overall, "caption_stats": stats}


def main() -> None:
    args = parse_args()
    method_dir = Path(args.method_dir)
    rows = merge_shards(method_dir, args.num_shards)
    metrics = pope_metrics(rows, invalid_policy=args.invalid_policy) if args.task == "pope" else evaluate_chair(rows, args.chair_pkl, method_dir)
    write_json(method_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
