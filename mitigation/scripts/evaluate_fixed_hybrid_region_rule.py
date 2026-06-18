#!/usr/bin/env python3
"""Apply fixed hybrid OWLv2 TDEV thresholds to POPE predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_hybrid_region_rule import apply_rule  # noqa: E402
from evaluate_two_stage_region_rule import metrics, normalize_yes_no, read_csv, read_jsonl, subset_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fixed hybrid TDEV thresholds")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--result_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--splits", default="adversarial")
    p.add_argument("--low_threshold", type=float, default=0.04)
    p.add_argument("--high_threshold", type=float, default=0.12)
    p.add_argument("--margin_threshold", type=float, default=-0.20)
    p.add_argument("--rescue_high_threshold", type=float, default=0.50)
    p.add_argument("--rescue_margin_threshold", type=float, default=-0.10)
    return p.parse_args()


def attach_available_base_predictions(rows: list[dict], result_root: Path, splits: list[str], method: str) -> tuple[list[dict], int]:
    base_by_key = {}
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for row in read_jsonl(pred_path):
            base_by_key[(split, str(row["question_id"]))] = normalize_yes_no(row.get("text", row.get("prediction", "")))
    attached = []
    missing = 0
    for row in rows:
        key = (row["split"], str(row["question_id"]))
        if key not in base_by_key:
            missing += 1
            continue
        attached.append({**row, "base_prediction": base_by_key[key]})
    return attached, missing


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    rows = [row for row in read_csv(Path(args.predictions_csv)) if row["split"] in splits]
    rows, missing_base_rows = attach_available_base_predictions(rows, Path(args.result_root), splits, args.base_method)
    if not rows:
        raise ValueError("No TDEV rows matched available base predictions")
    predicted = apply_rule(
        rows,
        args.low_threshold,
        args.high_threshold,
        args.margin_threshold,
        args.rescue_high_threshold,
        args.rescue_margin_threshold,
    )

    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    summary_rows = []
    for split in splits + (["macro"] if len(splits) > 1 else []):
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    with (output_dir / "fixed_hybrid_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predicted[0].keys()))
        writer.writeheader()
        writer.writerows(predicted)
    with (output_dir / "fixed_hybrid_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "predictions_csv": args.predictions_csv,
        "result_root": args.result_root,
        "base_method": args.base_method,
        "splits": splits,
        "matched_rows": len(rows),
        "missing_base_rows": missing_base_rows,
        "low_threshold": args.low_threshold,
        "high_threshold": args.high_threshold,
        "margin_threshold": args.margin_threshold,
        "rescue_high_threshold": args.rescue_high_threshold,
        "rescue_margin_threshold": args.rescue_margin_threshold,
        "subsets": {f"{row['split']}:{row['subset']}": row for row in summary_rows},
    }
    with (output_dir / "fixed_hybrid_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'fixed_hybrid_metrics.csv'}")


if __name__ == "__main__":
    main()
