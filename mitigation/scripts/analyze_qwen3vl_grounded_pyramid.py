#!/usr/bin/env python3
"""Analyze the pre-registered multiscale alternative-identity certificate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mitigation.src.xev import lexical_equivalent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Grounded-pyramid JSONL files")
    parser.add_argument("--source_inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--minimum_views", type=int, default=2)
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


def view_alternatives(region: dict) -> list[dict]:
    alternatives = []
    for comparison in region["pairwise_comparisons"]:
        if (
            comparison.get("relation") == "C"
            and "open" in comparison.get("sources", [])
            and float(comparison["keep_margin"]) > 0
            and float(comparison["target_first_margin"]) < 0
            and float(comparison["alternative_first_margin"]) > 0
        ):
            alternatives.append(
                {
                    "candidate": comparison["candidate"],
                    "proposal_index": int(region["proposal_index"]),
                    "crop_scale": float(region["crop_scale"]),
                    "bbox": region["bbox"],
                    "target_first_margin": float(
                        comparison["target_first_margin"]
                    ),
                    "alternative_first_margin": float(
                        comparison["alternative_first_margin"]
                    ),
                    "target_preference": float(comparison["target_preference"]),
                }
            )
    return alternatives


def multiscale_certificates(row: dict, minimum_views: int) -> list[dict]:
    groups: list[dict] = []
    for region in row["regions"]:
        for evidence in view_alternatives(region):
            group = next(
                (
                    item
                    for item in groups
                    if item["proposal_index"] == evidence["proposal_index"]
                    and lexical_equivalent(item["candidate"], evidence["candidate"])
                ),
                None,
            )
            if group is None:
                group = {
                    "candidate": evidence["candidate"],
                    "proposal_index": evidence["proposal_index"],
                    "views": [],
                }
                groups.append(group)
            if evidence["bbox"] not in [view["bbox"] for view in group["views"]]:
                group["views"].append(evidence)

    certificates = [group for group in groups if len(group["views"]) >= minimum_views]
    return sorted(
        certificates,
        key=lambda item: (
            -len(item["views"]),
            sum(view["target_preference"] for view in item["views"]),
            item["candidate"],
        ),
    )


def main() -> None:
    args = parse_args()
    if args.minimum_views < 2:
        raise ValueError("--minimum_views must be at least 2")
    rows = read_rows(args.inputs)
    source_rows = read_rows(args.source_inputs)
    if not rows or not source_rows:
        raise ValueError("Pyramid and source inputs must both be non-empty")

    conflict_ids = {
        str(row["question_id"])
        for row in source_rows
        if row.get("global_support_verification", {}).get(
            "local_identity_proposals", []
        )
    }
    pyramid_by_id = {str(row["question_id"]): row for row in rows}
    if set(pyramid_by_id) != conflict_ids:
        missing = sorted(conflict_ids - set(pyramid_by_id), key=int)
        extra = sorted(set(pyramid_by_id) - conflict_ids, key=int)
        raise ValueError(f"Pyramid coverage mismatch; missing={missing}, extra={extra}")

    labels = np.asarray(
        [row["label"] == "yes" for row in source_rows], dtype=np.int64
    )
    baseline = np.asarray(
        [row["baseline_margin"] >= 0 for row in source_rows], dtype=bool
    )
    predictions = baseline.copy()
    row_diagnostics = []
    for index, source in enumerate(source_rows):
        question_id = str(source["question_id"])
        pyramid = pyramid_by_id.get(question_id)
        if pyramid is None:
            continue
        certificates = multiscale_certificates(pyramid, args.minimum_views)
        target_support = any(region["cycle_support"] for region in pyramid["regions"])
        action = bool(baseline[index] and certificates and not target_support)
        if action:
            predictions[index] = False
        row_diagnostics.append(
            {
                "question_id": question_id,
                "target": source["target"],
                "label": source["label"],
                "grounding_boxes": pyramid["grounding_boxes"],
                "view_count": len(pyramid["regions"]),
                "target_support": target_support,
                "certificates": certificates,
                "action": action,
            }
        )

    baseline_behavior = binary_metrics(labels, baseline)
    behavior = binary_metrics(labels, predictions)
    action_count = int(np.sum(baseline & ~predictions))
    corrections = int(np.sum((labels == 0) & baseline & ~predictions))
    positive_vetoes = int(np.sum((labels == 1) & baseline & ~predictions))
    precision = corrections / action_count if action_count else None
    gate = (
        action_count >= 2
        and precision >= 2 / 3
        and positive_vetoes <= 1
        and behavior["fpr"] < baseline_behavior["fpr"]
        and behavior["tpr"] >= baseline_behavior["tpr"] - 0.005
        and behavior["balanced_accuracy"] >= baseline_behavior["balanced_accuracy"]
        and behavior["mcc"] >= baseline_behavior["mcc"]
    )

    scale_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for region in row["regions"]:
            scale = f"{float(region['crop_scale']):g}"
            counts = scale_counts.setdefault(
                scale,
                {"views": 0, "target_supports": 0, "directional_conflicts": 0},
            )
            counts["views"] += 1
            counts["target_supports"] += int(region["cycle_support"])
            counts["directional_conflicts"] += int(
                bool(view_alternatives(region))
            )

    payload = {
        "inputs": args.inputs,
        "source_inputs": args.source_inputs,
        "samples": len(source_rows),
        "local_conflict_rows": len(conflict_ids),
        "pyramid_rows_with_valid_box": sum(
            bool(row["grounding_boxes"]) for row in rows
        ),
        "pyramid_rows_with_target_support": sum(
            any(region["cycle_support"] for region in row["regions"])
            for row in rows
        ),
        "pyramid_rows_with_multiscale_certificate": sum(
            bool(multiscale_certificates(row, args.minimum_views)) for row in rows
        ),
        "scale_diagnostics": scale_counts,
        "baseline": baseline_behavior,
        "pyramid": behavior,
        "action_count": action_count,
        "false_positive_corrections": corrections,
        "true_positive_vetoes": positive_vetoes,
        "correction_precision": precision,
        "development_gate_passed": gate,
        "minimum_views": args.minimum_views,
        "examples": row_diagnostics,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "examples"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
