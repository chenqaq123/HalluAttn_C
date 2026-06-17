#!/usr/bin/env python3
"""Audit calibration sensitivity for the hybrid OWLv2 TDEV rule.

The hybrid gate-plus-rescue result is currently the strongest lightweight TDEV
variant on POPE. This script checks whether that gain is stable across common
calibration objectives and TPR floors without rerunning OWLv2.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hybrid_region_rule import apply_rule, calibrate  # noqa: E402
from evaluate_two_stage_region_rule import (  # noqa: E402
    attach_base_predictions,
    metrics,
    parse_values,
    read_csv,
    subset_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep hybrid TDEV calibration settings")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--result_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--objectives", default="mcc,semantic_penalty")
    p.add_argument("--related_penalties", default="1.0,2.0")
    p.add_argument("--tpr_floors", default="0.75,0.80,0.85,0.90")
    p.add_argument("--low_values", default="0.02,0.04")
    p.add_argument("--high_values", default="0.10,0.12,0.16")
    p.add_argument("--margin_values", default="-0.30,-0.20,-0.10")
    p.add_argument("--rescue_high_values", default="0.40,0.50,0.60")
    p.add_argument("--rescue_margin_values", default="-0.10,0.00")
    return p.parse_args()


def _split_items(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def _prefixed(prefix: str, vals: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in vals.items()}


def _summarize(predicted: list[dict]) -> dict:
    macro = metrics(predicted)
    adversarial = [row for row in predicted if row["split"] == "adversarial"]
    related = metrics(subset_rows(adversarial, "negative_related_present"))
    plain = metrics(subset_rows(adversarial, "negative_absent_plain"))
    random_rows = [row for row in predicted if row["split"] == "random"]
    popular_rows = [row for row in predicted if row["split"] == "popular"]
    adversarial_all = metrics(adversarial)
    out = {}
    out.update(_prefixed("macro", macro))
    out.update(_prefixed("random", metrics(random_rows)))
    out.update(_prefixed("popular", metrics(popular_rows)))
    out.update(_prefixed("adversarial", adversarial_all))
    out.update(_prefixed("adversarial_related", related))
    out.update(_prefixed("adversarial_plain", plain))
    out["adversarial_related_minus_plain_fpr"] = related["fpr"] - plain["fpr"]
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _split_items(args.splits)
    rows = read_csv(Path(args.predictions_csv))
    rows = attach_base_predictions(rows, Path(args.result_root), splits, args.base_method)
    calibration_rows = [row for row in rows if row["split"] == args.calibration_split]

    lows = parse_values(args.low_values)
    highs = parse_values(args.high_values)
    margins = parse_values(args.margin_values)
    rescue_highs = parse_values(args.rescue_high_values)
    rescue_margins = parse_values(args.rescue_margin_values)
    objectives = _split_items(args.objectives)
    penalties = [float(value) for value in _split_items(args.related_penalties)]
    floors = [float(value) for value in _split_items(args.tpr_floors)]

    summary_rows = []
    for objective in objectives:
        active_penalties = penalties if objective == "semantic_penalty" else [0.0]
        for related_penalty in active_penalties:
            for tpr_floor in floors:
                (
                    score,
                    low,
                    high,
                    margin,
                    rescue_high,
                    rescue_margin,
                    cal_metrics,
                    cal_related,
                ) = calibrate(
                    calibration_rows,
                    lows,
                    highs,
                    margins,
                    rescue_highs,
                    rescue_margins,
                    objective,
                    related_penalty,
                    tpr_floor,
                )
                predicted = apply_rule(rows, low, high, margin, rescue_high, rescue_margin)
                summary = {
                    "objective": objective,
                    "related_penalty": related_penalty,
                    "tpr_floor": tpr_floor,
                    "objective_value": score,
                    "low_threshold": low,
                    "high_threshold": high,
                    "margin_threshold": margin,
                    "rescue_high_threshold": rescue_high,
                    "rescue_margin_threshold": rescue_margin,
                }
                summary.update(_prefixed("calibration", cal_metrics))
                summary.update(_prefixed("calibration_related", cal_related))
                summary.update(_summarize(predicted))
                summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            row["objective"],
            row["related_penalty"],
            row["tpr_floor"],
        )
    )
    csv_path = output_dir / "hybrid_calibration_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "predictions_csv": args.predictions_csv,
        "result_root": args.result_root,
        "base_method": args.base_method,
        "splits": splits,
        "calibration_split": args.calibration_split,
        "objectives": objectives,
        "related_penalties": penalties,
        "tpr_floors": floors,
        "rows": len(summary_rows),
        "csv": str(csv_path),
    }
    with (output_dir / "hybrid_calibration_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
