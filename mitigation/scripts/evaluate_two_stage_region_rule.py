#!/usr/bin/env python3
"""Evaluate a two-stage region verifier on POPE predictions.

Rule:

    yes if target_score > high_threshold
       or target_score > low_threshold and tdev_margin > margin_threshold

The rule preserves high-confidence target detections while requiring a
target-vs-neighbor check for medium-confidence detections. It can be calibrated
for ordinary MCC or for a semantic-aware objective that penalizes false
positives on related-present negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import product
from pathlib import Path

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a two-stage OWLv2 region verifier")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["direct", "gate"], default="direct")
    p.add_argument("--result_root", default="", help="Required for --mode gate")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--objective", choices=["mcc", "semantic_penalty"], default="semantic_penalty")
    p.add_argument("--related_penalty", type=float, default=2.0)
    p.add_argument("--tpr_floor", type=float, default=0.85)
    p.add_argument("--low_values", default="0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.25,0.30")
    p.add_argument("--high_values", default="0.08,0.10,0.12,0.16,0.20,0.25,0.30,0.40,0.50,0.70")
    p.add_argument("--margin_values", default="-0.50,-0.40,-0.30,-0.20,-0.15,-0.10,-0.05,0.00,0.02,0.05,0.10")
    return p.parse_args()


def parse_values(spec: str) -> list[float]:
    values = [float(item) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Grid value list cannot be empty")
    return values


def normalize_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def attach_base_predictions(rows: list[dict], result_root: Path, splits: list[str], method: str) -> list[dict]:
    base_by_key = {}
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for row in read_jsonl(pred_path):
            base_by_key[(split, str(row["question_id"]))] = normalize_yes_no(row.get("text", row.get("prediction", "")))
    return [{**row, "base_prediction": base_by_key[(row["split"], str(row["question_id"]))]} for row in rows]


def predict_yes(row: dict, low: float, high: float, margin: float, mode: str) -> bool:
    if mode == "gate" and row["base_prediction"] != "yes":
        return False
    target_score = float(row["target_score"])
    tdev_margin = float(row["tdev_margin"])
    return target_score > high or (target_score > low and tdev_margin > margin)


def apply_rule(rows: list[dict], low: float, high: float, margin: float, mode: str) -> list[dict]:
    predicted = []
    for row in rows:
        pred = "yes" if predict_yes(row, low, high, margin, mode) else "no"
        predicted.append({
            **row,
            "prediction": pred,
            "low_threshold": low,
            "high_threshold": high,
            "margin_threshold": margin,
        })
    return predicted


def metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = invalid = 0
    for row in rows:
        pred = row["prediction"]
        gold = row["label"]
        if pred == "invalid":
            invalid += 1
            continue
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
        "invalid": invalid,
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


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "positive":
        return [row for row in rows if row["label"] == "yes"]
    if subset == "negative":
        return [row for row in rows if row["label"] == "no"]
    return [row for row in rows if row.get("negative_type") == subset]


def objective_value(rows: list[dict], objective: str, related_penalty: float, tpr_floor: float) -> tuple[float, dict, dict]:
    vals = metrics(rows)
    related_vals = metrics(subset_rows(rows, "negative_related_present"))
    value = vals["mcc"]
    if objective == "semantic_penalty":
        value -= related_penalty * related_vals["fpr"]
        if vals["recall_tpr"] < tpr_floor:
            value -= 10.0
    return value, vals, related_vals


def calibrate(
    rows: list[dict],
    lows: list[float],
    highs: list[float],
    margins: list[float],
    mode: str,
    objective: str,
    related_penalty: float,
    tpr_floor: float,
) -> tuple[float, float, float, float, dict, dict]:
    best = None
    for low, high, margin in product(lows, highs, margins):
        if high < low:
            continue
        predicted = apply_rule(rows, low, high, margin, mode)
        score, vals, related_vals = objective_value(predicted, objective, related_penalty, tpr_floor)
        if best is None or score > best[0]:
            best = (score, low, high, margin, vals, related_vals)
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    rows = read_csv(Path(args.predictions_csv))
    required = {"target_score", "tdev_margin", "split", "question_id", "label", "negative_type"}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Missing columns in {args.predictions_csv}: {sorted(missing)}")

    if args.mode == "gate":
        if not args.result_root:
            raise ValueError("--result_root is required for --mode gate")
        rows = attach_base_predictions(rows, Path(args.result_root), splits, args.base_method)
    else:
        rows = [{**row, "base_prediction": ""} for row in rows]

    calibration_rows = [row for row in rows if row["split"] == args.calibration_split]
    score, low, high, margin, calibration_metrics, calibration_related_metrics = calibrate(
        calibration_rows,
        parse_values(args.low_values),
        parse_values(args.high_values),
        parse_values(args.margin_values),
        args.mode,
        args.objective,
        args.related_penalty,
        args.tpr_floor,
    )
    predicted = apply_rule(rows, low, high, margin, args.mode)

    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    summary_rows = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    with (output_dir / "two_stage_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predicted[0].keys()))
        writer.writeheader()
        writer.writerows(predicted)
    with (output_dir / "two_stage_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "predictions_csv": args.predictions_csv,
        "mode": args.mode,
        "base_method": args.base_method if args.mode == "gate" else "",
        "calibration_split": args.calibration_split,
        "objective": args.objective,
        "related_penalty": args.related_penalty,
        "tpr_floor": args.tpr_floor,
        "objective_value": score,
        "low_threshold": low,
        "high_threshold": high,
        "margin_threshold": margin,
        "calibration_metrics": calibration_metrics,
        "calibration_related_metrics": calibration_related_metrics,
    }
    with (output_dir / "two_stage_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
