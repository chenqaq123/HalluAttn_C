#!/usr/bin/env python3
"""Analyze replicated target-negative evidence over grounded crop pyramids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mitigation.scripts.analyze_qwen3vl_grounded_pyramid import (  # noqa: E402
    binary_metrics,
    read_rows,
    view_alternatives,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Grounded-pyramid JSONL files")
    parser.add_argument("--source_inputs", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--minimum_negative_views", type=int, default=2)
    parser.add_argument(
        "--evaluation",
        choices=("mechanism", "confirmatory"),
        default="mechanism",
    )
    return parser.parse_args()


def source_has_directional_conflict(row: dict) -> bool:
    return any(
        comparison.get("relation") == "C"
        and float(comparison["target_first_margin"]) < 0
        and float(comparison["alternative_first_margin"]) > 0
        for comparison in row.get("pairwise_verification", {}).get(
            "comparisons", []
        )
    )


def target_negative_certificates(row: dict, minimum_views: int) -> list[dict]:
    proposal_indices = sorted(
        {int(region["proposal_index"]) for region in row["regions"]}
    )
    certificates = []
    for proposal_index in proposal_indices:
        regions = [
            region
            for region in row["regions"]
            if int(region["proposal_index"]) == proposal_index
        ]
        negative_views = []
        alternatives = []
        for region in regions:
            if (
                float(region["target_margin"]) < 0
                and region["bbox"] not in [view["bbox"] for view in negative_views]
            ):
                negative_views.append(
                    {
                        "crop_scale": float(region["crop_scale"]),
                        "bbox": region["bbox"],
                        "target_margin": float(region["target_margin"]),
                    }
                )
            alternatives.extend(view_alternatives(region))
        if len(negative_views) >= minimum_views and alternatives:
            certificates.append(
                {
                    "proposal_index": proposal_index,
                    "negative_views": negative_views,
                    "alternatives": alternatives,
                }
            )
    return certificates


def main() -> None:
    args = parse_args()
    if args.minimum_negative_views < 2:
        raise ValueError("--minimum_negative_views must be at least 2")
    rows = read_rows(args.inputs)
    source_rows = read_rows(args.source_inputs)
    if not rows or not source_rows:
        raise ValueError("Pyramid and source inputs must both be non-empty")

    conflict_ids = {
        str(row["question_id"])
        for row in source_rows
        if source_has_directional_conflict(row)
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
    examples = []
    for index, source in enumerate(source_rows):
        question_id = str(source["question_id"])
        pyramid = pyramid_by_id.get(question_id)
        if pyramid is None:
            continue
        certificates = target_negative_certificates(
            pyramid, args.minimum_negative_views
        )
        target_support = any(region["cycle_support"] for region in pyramid["regions"])
        action = bool(baseline[index] and certificates and not target_support)
        if action:
            predictions[index] = False
        examples.append(
            {
                "question_id": question_id,
                "target": source["target"],
                "label": source["label"],
                "grounding_boxes": pyramid["grounding_boxes"],
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
    mechanism_gate = (
        corrections >= 2
        and positive_vetoes <= 1
        and precision is not None
        and precision >= 2 / 3
    )
    confirmatory_gate = (
        action_count >= 2
        and precision is not None
        and precision >= 2 / 3
        and positive_vetoes <= 1
        and behavior["fpr"] < baseline_behavior["fpr"]
        and behavior["tpr"] >= baseline_behavior["tpr"] - 0.01
        and behavior["balanced_accuracy"] >= baseline_behavior["balanced_accuracy"]
        and behavior["mcc"] >= baseline_behavior["mcc"]
    )
    gate = (
        confirmatory_gate if args.evaluation == "confirmatory" else mechanism_gate
    )

    payload = {
        "inputs": args.inputs,
        "source_inputs": args.source_inputs,
        "samples": len(source_rows),
        "strict_directional_conflict_rows": len(conflict_ids),
        "label_counts_on_conflicts": {
            "yes": sum(example["label"] == "yes" for example in examples),
            "no": sum(example["label"] == "no" for example in examples),
        },
        "rows_with_valid_box": sum(bool(row["grounding_boxes"]) for row in rows),
        "rows_with_target_support": sum(
            any(region["cycle_support"] for region in row["regions"])
            for row in rows
        ),
        "rows_with_replicated_negative_target": sum(
            any(
                sum(
                    int(
                        region["target_margin"] < 0
                        and int(region["proposal_index"]) == proposal_index
                    )
                    for region in row["regions"]
                )
                >= args.minimum_negative_views
                for proposal_index in {
                    int(region["proposal_index"]) for region in row["regions"]
                }
            )
            for row in rows
        ),
        "rows_with_open_directional_alternative": sum(
            any(view_alternatives(region) for region in row["regions"])
            for row in rows
        ),
        "rows_with_joint_certificate": sum(
            bool(target_negative_certificates(row, args.minimum_negative_views))
            for row in rows
        ),
        "baseline": baseline_behavior,
        "method": behavior,
        "action_count": action_count,
        "false_positive_corrections": corrections,
        "true_positive_vetoes": positive_vetoes,
        "correction_precision": precision,
        "evaluation": args.evaluation,
        "gate_passed": gate,
        "stage1_gate_passed": mechanism_gate,
        "confirmatory_gate_passed": (
            confirmatory_gate if args.evaluation == "confirmatory" else None
        ),
        "minimum_negative_views": args.minimum_negative_views,
        "examples": examples,
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
