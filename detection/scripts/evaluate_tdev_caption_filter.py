#!/usr/bin/env python3
"""Simulate caption-side object mention filtering with TDEV scores.

This is a post-hoc mitigation proxy, not a caption re-generation experiment.
Given per-object CHAIR labels and region evidence scores, it asks how many
hallucinated object mentions would be removed by abstaining on high-risk
mentions, and how many grounded mentions would be lost at the same time.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TDEV caption object-mention filtering")
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_caption_filter",
    )
    p.add_argument("--remove_fracs", default="0.02,0.05,0.10,0.20")
    p.add_argument("--grounded_loss_caps", default="0.01,0.02,0.05,0.10")
    p.add_argument("--two_stage_low", type=float, default=0.10)
    p.add_argument("--two_stage_high", type=float, default=0.16)
    p.add_argument("--two_stage_margin", type=float, default=-0.15)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
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


def build_scores(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    target = np.asarray([float(row["target_score"]) for row in rows], dtype=np.float64)
    neighbor = np.asarray([float(row["best_neighbor_score"]) for row in rows], dtype=np.float64)
    margin = np.asarray([float(row["tdev_margin"]) for row in rows], dtype=np.float64)
    dominance = np.maximum(neighbor - target, 0.0)
    return {
        "target_absence": -target,
        "margin_absence": -margin,
        "two_stage_absence": -two_stage_score(
            target,
            margin,
            args.two_stage_low,
            args.two_stage_high,
            args.two_stage_margin,
        ),
        "hybrid_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_margin,
        ),
        "hybrid_mcc_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_mcc_margin,
        ),
        f"target_absence_plus_neighbor_dominance_{args.neighbor_dominance_alpha:g}": (
            -target + args.neighbor_dominance_alpha * dominance
        ),
    }


def baseline_summary(labels: np.ndarray, image_ids: np.ndarray) -> dict[str, float | int]:
    total = int(labels.size)
    hallucinated = int(labels.sum())
    grounded = total - hallucinated
    image_hallu = 0
    for image_id in np.unique(image_ids):
        image_hallu += int(labels[image_ids == image_id].sum() > 0)
    return {
        "images": int(np.unique(image_ids).size),
        "mentions": total,
        "hallucinated_mentions": hallucinated,
        "grounded_mentions": grounded,
        "mention_hallucination_rate": hallucinated / total if total else 0.0,
        "image_hallucination_rate": image_hallu / np.unique(image_ids).size if image_ids.size else 0.0,
    }


def evaluate_mask(labels: np.ndarray, image_ids: np.ndarray, remove_mask: np.ndarray, score_name: str, policy: str) -> dict:
    total = int(labels.size)
    hallucinated = int(labels.sum())
    grounded = total - hallucinated
    removed = int(remove_mask.sum())
    removed_hallu = int(labels[remove_mask].sum())
    removed_grounded = removed - removed_hallu
    remaining_mask = ~remove_mask
    remaining_total = int(remaining_mask.sum())
    remaining_hallu = int(labels[remaining_mask].sum())
    remaining_grounded = remaining_total - remaining_hallu
    image_hallu = 0
    for image_id in np.unique(image_ids):
        mask = (image_ids == image_id) & remaining_mask
        image_hallu += int(labels[mask].sum() > 0)
    return {
        "score": score_name,
        "policy": policy,
        "removed_mentions": removed,
        "removed_fraction": removed / total if total else 0.0,
        "removed_hallucinated": removed_hallu,
        "removed_grounded": removed_grounded,
        "removal_precision": removed_hallu / removed if removed else 0.0,
        "hallucinated_reduction": removed_hallu / hallucinated if hallucinated else 0.0,
        "grounded_loss": removed_grounded / grounded if grounded else 0.0,
        "remaining_mentions": remaining_total,
        "remaining_hallucinated": remaining_hallu,
        "remaining_grounded": remaining_grounded,
        "remaining_mention_hallucination_rate": remaining_hallu / remaining_total if remaining_total else 0.0,
        "remaining_image_hallucination_rate": image_hallu / np.unique(image_ids).size if image_ids.size else 0.0,
    }


def top_fraction_mask(values: np.ndarray, frac: float) -> np.ndarray:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"Removal fraction must be in (0,1): {frac}")
    n_remove = max(1, int(round(values.size * frac)))
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:n_remove]] = True
    return mask


def best_under_grounded_cap(values: np.ndarray, labels: np.ndarray, cap: float) -> np.ndarray:
    if cap < 0.0:
        raise ValueError(f"Grounded loss cap must be non-negative: {cap}")
    grounded = int((labels == 0).sum())
    max_grounded_removed = int(np.floor(cap * grounded))
    order = np.argsort(-values, kind="mergesort")
    best_k = 0
    grounded_removed = 0
    for idx, row_idx in enumerate(order, start=1):
        grounded_removed += int(labels[row_idx] == 0)
        if grounded_removed > max_grounded_removed:
            break
        best_k = idx
    mask = np.zeros(values.size, dtype=bool)
    if best_k:
        mask[order[:best_k]] = True
    return mask


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.scores_csv))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    scores = build_scores(rows, args)
    remove_fracs = parse_values(args.remove_fracs)
    grounded_loss_caps = parse_values(args.grounded_loss_caps)

    summary_rows = []
    for score_name, values in scores.items():
        for frac in remove_fracs:
            mask = top_fraction_mask(values, frac)
            summary_rows.append(evaluate_mask(labels, image_ids, mask, score_name, f"top_{frac:g}"))
        for cap in grounded_loss_caps:
            mask = best_under_grounded_cap(values, labels, cap)
            summary_rows.append(evaluate_mask(labels, image_ids, mask, score_name, f"grounded_loss_cap_{cap:g}"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tdev_caption_filter_metrics.csv", summary_rows)
    payload = {
        "scores_csv": args.scores_csv,
        "baseline": baseline_summary(labels, image_ids),
        "score_direction": "larger score means higher risk and is removed first",
        "remove_fracs": remove_fracs,
        "grounded_loss_caps": grounded_loss_caps,
        "metrics": summary_rows,
    }
    with (output_dir / "tdev_caption_filter_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'tdev_caption_filter_metrics.csv'}")


if __name__ == "__main__":
    main()
