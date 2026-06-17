"""Controlled evaluation for mitigation effects on POPE and CHAIR."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


_YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)


def normalize_yes_no(text: str) -> str:
    """Return the first explicit POPE answer: yes, no, or invalid."""
    match = _YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


def pope_metrics(rows: list[dict], invalid_policy: str = "error") -> dict:
    if invalid_policy not in {"error", "as_wrong", "drop"}:
        raise ValueError(f"Unknown invalid_policy={invalid_policy!r}")

    tp = fp = tn = fn = invalid = 0
    invalid_examples = []
    for row in rows:
        pred = normalize_yes_no(row.get("text", row.get("prediction", "")))
        gold = str(row["label"]).lower()
        if pred == "invalid":
            invalid += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({
                    "id": row.get("question_id", row.get("image_id")),
                    "label": gold,
                    "text": row.get("text", row.get("prediction", "")),
                })
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
        raise ValueError(f"Found {invalid} non yes/no POPE outputs; first examples: {invalid_examples}")

    n = tp + fp + tn + fn
    total = n + (invalid if invalid_policy == "error" else 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": n,
        "total_rows": total if invalid_policy == "error" else n + (invalid if invalid_policy == "drop" else 0),
        "invalid": invalid,
        "invalid_policy": invalid_policy,
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


def intervention_delta(method_metrics: dict, vanilla_metrics: dict) -> dict:
    return {
        "delta_accuracy": method_metrics["accuracy"] - vanilla_metrics["accuracy"],
        "delta_f1": method_metrics["f1"] - vanilla_metrics["f1"],
        "delta_yes_rate": method_metrics["yes_rate"] - vanilla_metrics["yes_rate"],
        "delta_tpr": method_metrics["recall_tpr"] - vanilla_metrics["recall_tpr"],
        "delta_fpr": method_metrics["fpr"] - vanilla_metrics["fpr"],
        "delta_tpr_minus_delta_fpr": (
            method_metrics["recall_tpr"]
            - vanilla_metrics["recall_tpr"]
            - method_metrics["fpr"]
            + vanilla_metrics["fpr"]
        ),
    }


def basic_caption_stats(rows: list[dict]) -> dict:
    lengths = [len(str(row.get("caption", "")).split()) for row in rows]
    return {
        "samples": len(rows),
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "median_words": statistics.median(lengths) if lengths else 0,
        "empty_captions": sum(length == 0 for length in lengths),
    }
