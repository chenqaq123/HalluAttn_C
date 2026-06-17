#!/usr/bin/env python3
"""Compare attention interventions against vanilla on POPE or CHAIR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))

from src.evaluation import intervention_delta, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task_dir", required=True)
    p.add_argument("--methods", default="vanilla,pai,clearsight,visattnsink")
    return p.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_ids(path: Path) -> set:
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.add(row.get("question_id", row.get("image_id")))
    return ids


def main() -> None:
    args = parse_args()
    task_dir = Path(args.task_dir)
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    vanilla = _load(task_dir / "vanilla" / "metrics.json")
    vanilla_ids = _prediction_ids(task_dir / "vanilla" / "predictions.jsonl")
    comparison = {"vanilla": vanilla, "methods": {}}
    is_pope = "yes_rate" in vanilla
    for method in methods:
        if method == "vanilla":
            continue
        method_ids = _prediction_ids(task_dir / method / "predictions.jsonl")
        if method_ids != vanilla_ids:
            missing = len(vanilla_ids - method_ids)
            extra = len(method_ids - vanilla_ids)
            raise ValueError(f"{method} sample IDs differ from vanilla: missing={missing}, extra={extra}")
        metrics = _load(task_dir / method / "metrics.json")
        if is_pope:
            delta = intervention_delta(metrics, vanilla)
        else:
            base_chair, this_chair = vanilla["chair"], metrics["chair"]
            base_stats, this_stats = vanilla["caption_stats"], metrics["caption_stats"]
            delta = {
                "delta_CHAIRi": this_chair["CHAIRi"] - base_chair["CHAIRi"],
                "delta_CHAIRs": this_chair["CHAIRs"] - base_chair["CHAIRs"],
                "delta_mean_words": this_stats["mean_words"] - base_stats["mean_words"],
                "delta_mean_object_mentions": (
                    this_stats["mean_object_mentions"] - base_stats["mean_object_mentions"]
                ),
                "delta_mean_hallucinated_mentions": (
                    this_stats["mean_hallucinated_mentions"] - base_stats["mean_hallucinated_mentions"]
                ),
            }
        comparison["methods"][method] = {"metrics": metrics, "delta_vs_vanilla": delta}
    write_json(task_dir / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
