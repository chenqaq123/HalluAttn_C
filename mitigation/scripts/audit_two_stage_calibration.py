#!/usr/bin/env python3
"""Audit calibration sensitivity for the two-stage OWLv2 region verifier.

The main two-stage evaluator selects one threshold triple for one objective.
This script sweeps multiple semantic penalties and TPR floors without rerunning
OWLv2, so paper claims can report whether the TDEV trade-off is stable rather
than cherry-picked.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_two_stage_region_rule import (  # noqa: E402
    apply_rule,
    attach_base_predictions,
    calibrate,
    metrics,
    parse_values,
    read_csv,
    subset_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep two-stage TDEV calibration settings")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--result_root", default="", help="Required when gate mode is included")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--modes", default="direct,gate")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--objectives", default="mcc,semantic_penalty")
    p.add_argument("--related_penalties", default="1.0,2.0")
    p.add_argument("--tpr_floors", default="0.75,0.80,0.85,0.90")
    p.add_argument("--low_values", default="0.04,0.08,0.12")
    p.add_argument("--high_values", default="0.12,0.20,0.30")
    p.add_argument("--margin_values", default="-0.20,-0.10,0.00")
    return p.parse_args()


def _split_items(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def _prefixed(prefix: str, vals: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in vals.items()}


def _summarize(predicted: list[dict], splits: list[str]) -> dict:
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

    rows = read_csv(Path(args.predictions_csv))
    splits = _split_items(args.splits)
    lows = parse_values(args.low_values)
    highs = parse_values(args.high_values)
    margins = parse_values(args.margin_values)
    modes = _split_items(args.modes)
    objectives = _split_items(args.objectives)
    penalties = [float(value) for value in _split_items(args.related_penalties)]
    floors = [float(value) for value in _split_items(args.tpr_floors)]

    rows_by_mode: dict[str, list[dict]] = {}
    for mode in modes:
        if mode == "gate":
            if not args.result_root:
                raise ValueError("--result_root is required when gate mode is included")
            rows_by_mode[mode] = attach_base_predictions(rows, Path(args.result_root), splits, args.base_method)
        elif mode == "direct":
            rows_by_mode[mode] = [{**row, "base_prediction": ""} for row in rows]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    summary_rows = []
    for mode in modes:
        mode_rows = rows_by_mode[mode]
        calibration_rows = [row for row in mode_rows if row["split"] == args.calibration_split]
        for objective in objectives:
            active_penalties = penalties if objective == "semantic_penalty" else [0.0]
            for related_penalty in active_penalties:
                for tpr_floor in floors:
                    score, low, high, margin, cal_metrics, cal_related = calibrate(
                        calibration_rows,
                        lows,
                        highs,
                        margins,
                        mode,
                        objective,
                        related_penalty,
                        tpr_floor,
                    )
                    predicted = apply_rule(mode_rows, low, high, margin, mode)
                    summary = {
                        "mode": mode,
                        "objective": objective,
                        "related_penalty": related_penalty,
                        "tpr_floor": tpr_floor,
                        "objective_value": score,
                        "low_threshold": low,
                        "high_threshold": high,
                        "margin_threshold": margin,
                    }
                    summary.update(_prefixed("calibration", cal_metrics))
                    summary.update(_prefixed("calibration_related", cal_related))
                    summary.update(_summarize(predicted, splits))
                    summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            row["mode"],
            row["objective"],
            row["related_penalty"],
            row["tpr_floor"],
        )
    )
    csv_path = output_dir / "two_stage_calibration_sweep.csv"
    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    payload = {
        "predictions_csv": args.predictions_csv,
        "result_root": args.result_root,
        "base_method": args.base_method,
        "splits": splits,
        "calibration_split": args.calibration_split,
        "modes": modes,
        "objectives": objectives,
        "related_penalties": penalties,
        "tpr_floors": floors,
        "rows": len(summary_rows),
        "csv": str(csv_path),
    }
    with (output_dir / "two_stage_calibration_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
