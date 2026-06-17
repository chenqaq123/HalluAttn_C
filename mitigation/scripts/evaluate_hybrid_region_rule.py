#!/usr/bin/env python3
"""Evaluate a hybrid OWLv2 TDEV rule on POPE predictions.

The two-stage gate is precise but recall-limited because it can only suppress
vanilla "yes" answers. This hybrid keeps that precision filter and adds a
conservative rescue path for vanilla "no" answers when target region evidence is
very strong:

    if vanilla says yes:
        yes if target_score > high or target_score > low and margin > margin
    else:
        yes if target_score > rescue_high and margin > rescue_margin
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_two_stage_region_rule import (  # noqa: E402
    attach_base_predictions,
    metrics,
    parse_values,
    read_csv,
    subset_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a hybrid OWLv2 TDEV region rule")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--result_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--objective", choices=["mcc", "semantic_penalty"], default="semantic_penalty")
    p.add_argument("--related_penalty", type=float, default=2.0)
    p.add_argument("--tpr_floor", type=float, default=0.80)
    p.add_argument("--low_values", default="0.02,0.04,0.08,0.12")
    p.add_argument("--high_values", default="0.10,0.12,0.16,0.20,0.30")
    p.add_argument("--margin_values", default="-0.30,-0.20,-0.10,0.00")
    p.add_argument("--rescue_high_values", default="0.30,0.40,0.50,0.60,0.70")
    p.add_argument("--rescue_margin_values", default="-0.10,0.00,0.05,0.10")
    return p.parse_args()


def predict_yes(row: dict, low: float, high: float, margin: float, rescue_high: float, rescue_margin: float) -> bool:
    target_score = float(row["target_score"])
    tdev_margin = float(row["tdev_margin"])
    gate_yes = target_score > high or (target_score > low and tdev_margin > margin)
    rescue_yes = target_score > rescue_high and tdev_margin > rescue_margin
    if row["base_prediction"] == "yes":
        return gate_yes
    return rescue_yes


def apply_rule(
    rows: list[dict],
    low: float,
    high: float,
    margin: float,
    rescue_high: float,
    rescue_margin: float,
) -> list[dict]:
    predicted = []
    for row in rows:
        pred = "yes" if predict_yes(row, low, high, margin, rescue_high, rescue_margin) else "no"
        predicted.append({
            **row,
            "prediction": pred,
            "low_threshold": low,
            "high_threshold": high,
            "margin_threshold": margin,
            "rescue_high_threshold": rescue_high,
            "rescue_margin_threshold": rescue_margin,
        })
    return predicted


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
    rescue_highs: list[float],
    rescue_margins: list[float],
    objective: str,
    related_penalty: float,
    tpr_floor: float,
) -> tuple[float, float, float, float, float, float, dict, dict]:
    best = None
    for low, high, margin, rescue_high, rescue_margin in product(lows, highs, margins, rescue_highs, rescue_margins):
        if high < low:
            continue
        predicted = apply_rule(rows, low, high, margin, rescue_high, rescue_margin)
        score, vals, related_vals = objective_value(predicted, objective, related_penalty, tpr_floor)
        if best is None or score > best[0]:
            best = (score, low, high, margin, rescue_high, rescue_margin, vals, related_vals)
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

    rows = attach_base_predictions(rows, Path(args.result_root), splits, args.base_method)
    calibration_rows = [row for row in rows if row["split"] == args.calibration_split]
    (
        score,
        low,
        high,
        margin,
        rescue_high,
        rescue_margin,
        calibration_metrics,
        calibration_related_metrics,
    ) = calibrate(
        calibration_rows,
        parse_values(args.low_values),
        parse_values(args.high_values),
        parse_values(args.margin_values),
        parse_values(args.rescue_high_values),
        parse_values(args.rescue_margin_values),
        args.objective,
        args.related_penalty,
        args.tpr_floor,
    )
    predicted = apply_rule(rows, low, high, margin, rescue_high, rescue_margin)

    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    summary_rows = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    with (output_dir / "hybrid_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predicted[0].keys()))
        writer.writeheader()
        writer.writerows(predicted)
    with (output_dir / "hybrid_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "predictions_csv": args.predictions_csv,
        "result_root": args.result_root,
        "base_method": args.base_method,
        "calibration_split": args.calibration_split,
        "objective": args.objective,
        "related_penalty": args.related_penalty,
        "tpr_floor": args.tpr_floor,
        "objective_value": score,
        "low_threshold": low,
        "high_threshold": high,
        "margin_threshold": margin,
        "rescue_high_threshold": rescue_high,
        "rescue_margin_threshold": rescue_margin,
        "calibration_metrics": calibration_metrics,
        "calibration_related_metrics": calibration_related_metrics,
    }
    with (output_dir / "hybrid_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'hybrid_metrics.csv'}")


if __name__ == "__main__":
    main()
