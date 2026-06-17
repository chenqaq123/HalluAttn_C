#!/usr/bin/env python3
"""Post-hoc OWLv2 region-score variants for CHAIR object-mention detection.

This script reuses the per-mention OWLv2 scores emitted by
`evaluate_owlv2_region_detection.py` and evaluates cheap score variants without
rerunning OWLv2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import compute_metric_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate post-hoc OWLv2 region-score variants")
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/owlv2_region_posthoc_scores",
    )
    p.add_argument("--two_stage_low", type=float, default=0.10)
    p.add_argument("--two_stage_high", type=float, default=0.16)
    p.add_argument("--two_stage_margin", type=float, default=-0.15)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--dominance_alphas", default="0.25,0.50,1.00,2.00")
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_values(spec: str) -> list[float]:
    values = [float(item) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Value list cannot be empty")
    return values


def two_stage_score(target: np.ndarray, margin: np.ndarray, low: float, high: float, margin_threshold: float) -> np.ndarray:
    high_branch = target - high
    medium_branch = np.minimum(target - low, margin - margin_threshold)
    return np.maximum(high_branch, medium_branch)


def metric_row(name: str, values: np.ndarray, labels: np.ndarray, positions: np.ndarray) -> dict:
    metrics = compute_metric_bundle(labels, values, positions)
    return {
        "score": name,
        "overall_auroc": metrics["overall_auroc"],
        "within_bin_auroc": metrics["within_bin_auroc"],
        "matched_pair_auroc": metrics["matched_pair_auroc"],
        "matched_pair_count": metrics["matched_pair_count"],
        "residual_auroc": metrics["residual_auroc"],
        "finite_count": metrics["finite_count"],
    }


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
    rows = read_rows(Path(args.scores_csv))

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    positions = np.asarray([int(row["gen_pos"]) for row in rows], dtype=np.int32)
    target = np.asarray([float(row["target_score"]) for row in rows], dtype=np.float32)
    neighbor = np.asarray([float(row["best_neighbor_score"]) for row in rows], dtype=np.float32)
    margin = np.asarray([float(row["tdev_margin"]) for row in rows], dtype=np.float32)

    score_arrays: dict[str, np.ndarray] = {
        "owlv2_target_absence": -target,
        "owlv2_margin_absence": -margin,
        "owlv2_neighbor_presence": neighbor,
        "owlv2_two_stage_absence": -two_stage_score(
            target,
            margin,
            args.two_stage_low,
            args.two_stage_high,
            args.two_stage_margin,
        ),
        "owlv2_hybrid_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_margin,
        ),
        "owlv2_hybrid_mcc_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_mcc_margin,
        ),
    }
    dominance = np.maximum(neighbor - target, 0.0)
    for alpha in parse_values(args.dominance_alphas):
        key = f"owlv2_target_absence_plus_neighbor_dominance_{alpha:g}"
        score_arrays[key] = -target + alpha * dominance

    metric_rows = [
        metric_row(name, values, labels, positions)
        for name, values in score_arrays.items()
    ]
    metric_rows.sort(key=lambda row: row["within_bin_auroc"] or 0.0, reverse=True)
    write_csv(output_dir / "owlv2_region_posthoc_metrics.csv", metric_rows)
    np.savez(output_dir / "owlv2_region_posthoc_scores.npz", labels=labels, gen_pos=positions, **score_arrays)
    payload = {
        "scores_csv": args.scores_csv,
        "samples": len(rows),
        "hallucinated": int(labels.sum()),
        "score_direction": "larger score means more likely hallucinated",
        "two_stage": {
            "low": args.two_stage_low,
            "high": args.two_stage_high,
            "margin": args.two_stage_margin,
        },
        "hybrid_positive_branch": {
            "low": args.hybrid_low,
            "high": args.hybrid_high,
            "margin": args.hybrid_margin,
            "mcc_margin": args.hybrid_mcc_margin,
        },
        "dominance_alphas": parse_values(args.dominance_alphas),
        "metrics": {row["score"]: row for row in metric_rows},
    }
    with (output_dir / "owlv2_region_posthoc_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'owlv2_region_posthoc_metrics.csv'}")


if __name__ == "__main__":
    main()
