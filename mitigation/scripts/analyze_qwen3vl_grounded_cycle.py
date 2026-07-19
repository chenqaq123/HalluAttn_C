#!/usr/bin/env python3
"""Analyze the pre-registered grounded re-identification cycle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Grounded-cycle JSONL files")
    parser.add_argument("--source_inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path_text in paths:
        with Path(path_text).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    by_id = {str(row["question_id"]): row for row in rows}
    return sorted(by_id.values(), key=lambda row: int(row["question_id"]))


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": len(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tpr + 1 - fpr),
        "tpr": tpr,
        "fpr": fpr,
        "mcc": ((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "yes_rate": float(predictions.mean()),
    }


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs)
    source_rows = read_rows(args.source_inputs)
    if not rows:
        raise ValueError("No grounded-cycle rows found")

    supported = [row for row in rows if row["cycle_support_found"]]
    true_rows = [row for row in rows if row["label"] == "yes"]
    false_rows = [row for row in rows if row["label"] == "no"]
    supported_true = [row for row in supported if row["label"] == "yes"]
    supported_false = [row for row in supported if row["label"] == "no"]
    actions = [row for row in rows if row["action"]]
    stage1_gate = (
        len(true_rows) == 10
        and len(false_rows) == 4
        and len(supported_true) >= 8
        and len(supported_false) <= 1
        and bool(supported)
        and len(supported_true) / len(supported) >= 0.8
    )

    conflict_ids = {
        str(row["question_id"])
        for row in source_rows
        if row.get("global_support_verification", {}).get(
            "local_identity_proposals", []
        )
    }
    cycle_by_id = {str(row["question_id"]): row for row in rows}
    full_coverage = conflict_ids.issubset(cycle_by_id)
    full_behavior = None
    stage2_gate = None
    if full_coverage:
        labels = np.asarray(
            [row["label"] == "yes" for row in source_rows], dtype=np.int64
        )
        baseline = np.asarray(
            [row["baseline_margin"] >= 0 for row in source_rows], dtype=bool
        )
        predictions = baseline.copy()
        for index, row in enumerate(source_rows):
            cycle = cycle_by_id.get(str(row["question_id"]))
            if cycle is not None and cycle["action"]:
                predictions[index] = False
        baseline_behavior = binary_metrics(labels, baseline)
        behavior = binary_metrics(labels, predictions)
        action_count = int(np.sum(baseline & ~predictions))
        corrections = int(np.sum((labels == 0) & baseline & ~predictions))
        positive_vetoes = int(np.sum((labels == 1) & baseline & ~predictions))
        precision = corrections / action_count if action_count else None
        stage2_gate = (
            action_count >= 2
            and precision >= 2 / 3
            and positive_vetoes <= 1
            and behavior["fpr"] < baseline_behavior["fpr"]
            and behavior["tpr"] >= baseline_behavior["tpr"] - 0.005
            and behavior["balanced_accuracy"] >= baseline_behavior["balanced_accuracy"]
            and behavior["mcc"] >= baseline_behavior["mcc"]
        )
        full_behavior = {
            "baseline": baseline_behavior,
            "cycle": behavior,
            "action_count": action_count,
            "false_positive_corrections": corrections,
            "true_positive_vetoes": positive_vetoes,
            "correction_precision": precision,
            "stage2_gate_passed": stage2_gate,
        }

    parse_count = sum(bool(row["grounding_boxes"]) for row in rows)
    named_count = sum(
        any(region["target_named"] for region in row["regions"]) for row in rows
    )
    positive_margin_count = sum(
        any(region["target_margin"] > 0 for region in row["regions"]) for row in rows
    )
    conflict_rejection_count = sum(
        any(region["directional_conflict"] is not None for region in row["regions"])
        for row in rows
    )
    payload = {
        "inputs": args.inputs,
        "source_inputs": args.source_inputs,
        "samples": len(rows),
        "label_counts": {"yes": len(true_rows), "no": len(false_rows)},
        "funnel": {
            "rows_with_valid_box": parse_count,
            "rows_with_target_free_target_name": named_count,
            "rows_with_positive_target_margin": positive_margin_count,
            "rows_with_directional_conflict": conflict_rejection_count,
            "rows_with_cycle_support": len(supported),
        },
        "stage1": {
            "supported_true_targets": len(supported_true),
            "false_supports_on_absent_targets": len(supported_false),
            "support_precision": (
                len(supported_true) / len(supported) if supported else None
            ),
            "remaining_false_positive_corrections": sum(
                row["label"] == "no" for row in actions
            ),
            "remaining_true_positive_vetoes": sum(
                row["label"] == "yes" for row in actions
            ),
            "gate_passed": stage1_gate,
        },
        "full_conflict_coverage": full_coverage,
        "full_behavior": full_behavior,
        "stage2_gate_passed": stage2_gate,
        "examples": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
