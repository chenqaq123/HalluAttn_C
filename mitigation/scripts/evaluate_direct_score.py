#!/usr/bin/env python3
"""Evaluate a calibrated direct yes/no score on POPE-style prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate and evaluate a direct POPE score")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--score_field", default="tdev_margin")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--threshold_mode", choices=["zero", "calibrate_mcc"], default="calibrate_mcc")
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = row["prediction"]
        gold = row["label"]
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall_tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def apply_threshold(rows: list[dict], score_field: str, threshold: float) -> list[dict]:
    return [
        {**row, "prediction": "yes" if float(row[score_field]) > threshold else "no"}
        for row in rows
    ]


def choose_threshold(rows: list[dict], score_field: str) -> tuple[float, dict]:
    scores = sorted({float(row[score_field]) for row in rows})
    candidates = [scores[0] - 1e-6, scores[-1] + 1e-6]
    candidates.extend((a + b) / 2 for a, b in zip(scores, scores[1:]))
    best_threshold = 0.0
    best_metrics: dict | None = None
    for threshold in candidates:
        vals = metrics(apply_threshold(rows, score_field, threshold))
        if best_metrics is None or vals["mcc"] > best_metrics["mcc"]:
            best_threshold = threshold
            best_metrics = vals
    assert best_metrics is not None
    return best_threshold, best_metrics


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "positive":
        return [row for row in rows if row["label"] == "yes"]
    if subset == "negative":
        return [row for row in rows if row["label"] == "no"]
    return [row for row in rows if row.get("negative_type") == subset]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    rows = read_rows(Path(args.predictions_csv))
    if args.score_field not in rows[0]:
        raise KeyError(f"{args.score_field!r} is not a column in {args.predictions_csv}")

    calibration_rows = [row for row in rows if row["split"] == args.calibration_split]
    if args.threshold_mode == "zero":
        threshold = 0.0
        calibration_metrics = metrics(apply_threshold(calibration_rows, args.score_field, threshold))
    else:
        threshold, calibration_metrics = choose_threshold(calibration_rows, args.score_field)

    predicted = apply_threshold(rows, args.score_field, threshold)
    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    summary_rows = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    with (output_dir / "direct_score_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predicted[0].keys()))
        writer.writeheader()
        writer.writerows(predicted)
    with (output_dir / "direct_score_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "predictions_csv": args.predictions_csv,
        "score_field": args.score_field,
        "threshold": threshold,
        "threshold_mode": args.threshold_mode,
        "calibration_split": args.calibration_split,
        "calibration_metrics": calibration_metrics,
    }
    with (output_dir / "direct_score_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
