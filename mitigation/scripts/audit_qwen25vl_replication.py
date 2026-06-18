#!/usr/bin/env python3
"""Audit Qwen2.5-VL POPE replication artifacts against saved metrics.

This script is intentionally narrow: it checks the second-model replication
used by the current paper draft, where each POPE split lives under its own
result root. It exits non-zero if any invariant needed for paper tables fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))

from src.evaluation import pope_metrics, read_jsonl  # noqa: E402


SPLITS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit Qwen2.5-VL all-splits replication")
    p.add_argument(
        "--result_root_template",
        default="mitigation/results/qwen25vl_pope_{split}_full",
        help="Template for split-specific result roots.",
    )
    p.add_argument(
        "--semantic_json_template",
        default="mitigation/results/semantic_neighbor_audit/qwen25vl_{split}_full_subset_eval/semantic_neighbor_subset_metrics.json",
        help="Template for semantic-neighbor subset metric JSON files.",
    )
    p.add_argument(
        "--fixed_metrics_json_template",
        default="mitigation/results/semantic_neighbor_audit/qwen25vl_fixed_hybrid_region_rule_{split}_full/fixed_hybrid_metrics.json",
        help="Template for fixed TDEV metric JSON files.",
    )
    p.add_argument(
        "--fixed_metrics_csv_template",
        default="mitigation/results/semantic_neighbor_audit/qwen25vl_fixed_hybrid_region_rule_{split}_full/fixed_hybrid_metrics.csv",
        help="Template for fixed TDEV metric CSV files.",
    )
    p.add_argument("--expected_rows", type=int, default=3000)
    p.add_argument("--expected_positive", type=int, default=1500)
    p.add_argument("--expected_negative", type=int, default=1500)
    p.add_argument("--tolerance", type=float, default=1e-12)
    p.add_argument("--output_json", default="")
    return p.parse_args()


def path_from_template(template: str, split: str) -> Path:
    return PROJECT_ROOT / template.format(split=split)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def almost_equal(a: float, b: float, tolerance: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)


def compare_metrics(saved: dict, recomputed: dict, tolerance: float, prefix: str, errors: list[str]) -> None:
    keys = [
        "samples",
        "total_rows",
        "invalid",
        "tp",
        "fp",
        "tn",
        "fn",
        "accuracy",
        "precision",
        "recall_tpr",
        "fpr",
        "tnr",
        "f1",
        "balanced_accuracy",
        "mcc",
        "yes_rate",
    ]
    for key in keys:
        if key not in saved:
            fail(errors, f"{prefix}: saved metrics missing {key}")
            continue
        if key not in recomputed:
            fail(errors, f"{prefix}: recomputed metrics missing {key}")
            continue
        if isinstance(saved[key], int):
            if int(saved[key]) != int(recomputed[key]):
                fail(errors, f"{prefix}: metric {key} mismatch saved={saved[key]} recomputed={recomputed[key]}")
        elif not almost_equal(float(saved[key]), float(recomputed[key]), tolerance):
            fail(errors, f"{prefix}: metric {key} mismatch saved={saved[key]} recomputed={recomputed[key]}")


def row_map(rows: list[dict]) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in rows}


def get_csv_subset(path: Path, subset: str) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["subset"] == subset:
                return row
    raise ValueError(f"Missing subset={subset!r} in {path}")


def audit_split(split: str, args: argparse.Namespace, errors: list[str]) -> dict:
    result_root = path_from_template(args.result_root_template, split)
    method_dir = result_root / "pope" / split / "vanilla"
    shard_path = method_dir / "shard0.jsonl"
    pred_path = method_dir / "predictions.jsonl"
    metrics_path = method_dir / "metrics.json"

    for path in (shard_path, pred_path, metrics_path):
        if not path.exists():
            fail(errors, f"{split}: missing {path}")
            return {"split": split, "missing": str(path)}

    shard_rows = read_jsonl(shard_path)
    rows = read_jsonl(pred_path)
    ids = [str(row["question_id"]) for row in rows]
    label_yes = sum(str(row["label"]).lower() == "yes" for row in rows)
    label_no = sum(str(row["label"]).lower() == "no" for row in rows)

    if len(rows) != args.expected_rows:
        fail(errors, f"{split}: rows={len(rows)} expected={args.expected_rows}")
    if len(shard_rows) != args.expected_rows:
        fail(errors, f"{split}: shard rows={len(shard_rows)} expected={args.expected_rows}")
    if len(set(ids)) != len(ids):
        fail(errors, f"{split}: duplicate question_id count={len(ids) - len(set(ids))}")
    if label_yes != args.expected_positive or label_no != args.expected_negative:
        fail(errors, f"{split}: label balance yes={label_yes} no={label_no}")
    if row_map(rows) != row_map(shard_rows):
        fail(errors, f"{split}: predictions.jsonl and shard0.jsonl differ by question_id content")

    recomputed = pope_metrics(rows, invalid_policy="error")
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    compare_metrics(saved, recomputed, args.tolerance, split, errors)

    semantic_path = path_from_template(args.semantic_json_template, split)
    if not semantic_path.exists():
        fail(errors, f"{split}: missing semantic metrics {semantic_path}")
        semantic_missing = None
        semantic_subsets = {}
    else:
        semantic_payload = json.loads(semantic_path.read_text(encoding="utf-8"))
        method_payload = semantic_payload["splits"][split]["vanilla"]
        semantic_missing = int(method_payload["missing_audit_rows"])
        if semantic_missing:
            fail(errors, f"{split}: semantic missing_audit_rows={semantic_missing}")
        semantic_subsets = method_payload["subsets"]

    fixed_json_path = path_from_template(args.fixed_metrics_json_template, split)
    fixed_csv_path = path_from_template(args.fixed_metrics_csv_template, split)
    fixed_payload = json.loads(fixed_json_path.read_text(encoding="utf-8")) if fixed_json_path.exists() else {}
    if not fixed_payload:
        fail(errors, f"{split}: missing fixed TDEV metrics {fixed_json_path}")
    elif int(fixed_payload.get("matched_rows", -1)) != args.expected_rows or int(fixed_payload.get("missing_base_rows", -1)) != 0:
        fail(
            errors,
            f"{split}: fixed TDEV match mismatch matched={fixed_payload.get('matched_rows')} "
            f"missing_base={fixed_payload.get('missing_base_rows')}",
        )

    if not fixed_csv_path.exists():
        fail(errors, f"{split}: missing fixed TDEV CSV {fixed_csv_path}")
        fixed_all = fixed_related = fixed_plain = {}
    else:
        fixed_all = get_csv_subset(fixed_csv_path, "all")
        fixed_related = get_csv_subset(fixed_csv_path, "negative_related_present")
        fixed_plain = get_csv_subset(fixed_csv_path, "negative_absent_plain")

    return {
        "split": split,
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "label_yes": label_yes,
        "label_no": label_no,
        "invalid": recomputed["invalid"],
        "semantic_missing": semantic_missing,
        "vanilla": {
            "accuracy": recomputed["accuracy"],
            "mcc": recomputed["mcc"],
            "recall_tpr": recomputed["recall_tpr"],
            "fpr": recomputed["fpr"],
            "yes_rate": recomputed["yes_rate"],
            "related_fpr": semantic_subsets.get("negative_related_present", {}).get("fpr"),
            "plain_fpr": semantic_subsets.get("negative_absent_plain", {}).get("fpr"),
        },
        "fixed_tdev": {
            "matched_rows": fixed_payload.get("matched_rows"),
            "missing_base_rows": fixed_payload.get("missing_base_rows"),
            "accuracy": float(fixed_all["accuracy"]) if fixed_all else None,
            "mcc": float(fixed_all["mcc"]) if fixed_all else None,
            "recall_tpr": float(fixed_all["recall_tpr"]) if fixed_all else None,
            "fpr": float(fixed_all["fpr"]) if fixed_all else None,
            "yes_rate": float(fixed_all["yes_rate"]) if fixed_all else None,
            "related_fpr": float(fixed_related["fpr"]) if fixed_related else None,
            "plain_fpr": float(fixed_plain["fpr"]) if fixed_plain else None,
        },
    }


def macro(rows: list[dict], method: str, key: str) -> float | None:
    values = [row.get(method, {}).get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return mean(float(value) for value in values)


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    split_rows = [audit_split(split, args, errors) for split in SPLITS]
    summary = {
        "splits": split_rows,
        "macro": {
            "vanilla": {
                "mcc": macro(split_rows, "vanilla", "mcc"),
                "recall_tpr": macro(split_rows, "vanilla", "recall_tpr"),
                "fpr": macro(split_rows, "vanilla", "fpr"),
                "yes_rate": macro(split_rows, "vanilla", "yes_rate"),
                "related_fpr": macro(split_rows, "vanilla", "related_fpr"),
                "plain_fpr": macro(split_rows, "vanilla", "plain_fpr"),
            },
            "fixed_tdev": {
                "mcc": macro(split_rows, "fixed_tdev", "mcc"),
                "recall_tpr": macro(split_rows, "fixed_tdev", "recall_tpr"),
                "fpr": macro(split_rows, "fixed_tdev", "fpr"),
                "yes_rate": macro(split_rows, "fixed_tdev", "yes_rate"),
                "related_fpr": macro(split_rows, "fixed_tdev", "related_fpr"),
                "plain_fpr": macro(split_rows, "fixed_tdev", "plain_fpr"),
            },
        },
        "errors": errors,
    }

    if args.output_json:
        output = PROJECT_ROOT / args.output_json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
