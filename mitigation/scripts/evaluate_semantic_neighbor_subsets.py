#!/usr/bin/env python3
"""Evaluate POPE predictions on semantic-neighbor negative subsets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join POPE predictions with semantic-neighbor audit labels")
    p.add_argument("--result_root", required=True, help="Root containing pope/<split>/<method>/predictions.jsonl")
    p.add_argument("--audit_csv", required=True, help="CSV from build_semantic_neighbor_audit.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--methods", default="vanilla,pai,clearsight,visattnsink")
    p.add_argument("--invalid_policy", choices=["error", "as_wrong", "drop"], default="error")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


def load_audit_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["split"], str(row["question_id"]))] = row
    return rows


def metrics(rows: list[dict], invalid_policy: str) -> dict:
    tp = fp = tn = fn = invalid = 0
    invalid_examples = []
    for row in rows:
        pred = normalize_yes_no(row.get("text", row.get("prediction", "")))
        gold = str(row["label"]).lower()
        if pred == "invalid":
            invalid += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({"question_id": row.get("question_id"), "text": row.get("text", "")})
            if invalid_policy == "error":
                continue
            if invalid_policy == "drop":
                continue
            if gold == "yes":
                fn += 1
            elif gold == "no":
                fp += 1
            continue
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1

    if invalid and invalid_policy == "error":
        raise ValueError(f"Found {invalid} invalid yes/no predictions: {invalid_examples}")

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
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
        "mcc": ((tp * tn - fp * fn) / denominator) if denominator else 0.0,
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


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = load_audit_rows(Path(args.audit_csv))
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    subsets = [
        "all",
        "positive",
        "negative",
        "negative_related_present",
        "negative_absent_plain",
        "negative_target_present_coco_label",
    ]

    summary_rows = []
    payload = {"result_root": str(result_root), "audit_csv": args.audit_csv, "splits": {}}
    for split in splits:
        payload["splits"][split] = {}
        for method in methods:
            pred_path = result_root / "pope" / split / method / "predictions.jsonl"
            predictions = read_jsonl(pred_path)
            joined = []
            missing = 0
            for row in predictions:
                key = (split, str(row["question_id"]))
                audit = audit_rows.get(key)
                if audit is None:
                    missing += 1
                    continue
                joined.append({**row, "negative_type": audit["negative_type"], "target": audit["target"]})

            method_payload = {"prediction_file": str(pred_path), "missing_audit_rows": missing, "subsets": {}}
            for subset in subsets:
                subset_metric = metrics(subset_rows(joined, subset), args.invalid_policy)
                method_payload["subsets"][subset] = subset_metric
                summary_rows.append(
                    {
                        "split": split,
                        "method": method,
                        "subset": subset,
                        **subset_metric,
                    }
                )
            payload["splits"][split][method] = method_payload

    json_path = output_dir / "semantic_neighbor_subset_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    csv_path = output_dir / "semantic_neighbor_subset_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
