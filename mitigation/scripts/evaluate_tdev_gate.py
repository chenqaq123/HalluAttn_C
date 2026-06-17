#!/usr/bin/env python3
"""Evaluate TDEV margins as a gate on existing POPE predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate calibrated TDEV margin gates on POPE predictions")
    p.add_argument("--tdev_predictions_csv", required=True)
    p.add_argument("--result_root", required=True, help="Root containing pope/<split>/<method>/predictions.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--threshold_mode", choices=["zero", "calibrate_mcc"], default="calibrate_mcc")
    return p.parse_args()


def normalize_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = invalid = 0
    for row in rows:
        pred = row["prediction"]
        gold = row["label"]
        if pred == "invalid":
            invalid += 1
            continue
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
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
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
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


def load_tdev_rows(path: Path) -> dict[tuple[str, str], dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {(row["split"], str(row["question_id"])): row for row in csv.DictReader(f)}


def load_base_rows(result_root: Path, splits: list[str], method: str, tdev_rows: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for row in read_jsonl(pred_path):
            key = (split, str(row["question_id"]))
            tdev = tdev_rows[key]
            rows.append({
                "split": split,
                "question_id": str(row["question_id"]),
                "label": str(row["label"]).lower(),
                "base_prediction": normalize_yes_no(row.get("text", row.get("prediction", ""))),
                "tdev_margin": float(tdev["tdev_margin"]),
                "negative_type": tdev["negative_type"],
                "target": tdev["target"],
            })
    return rows


def apply_gate(rows: list[dict], threshold: float) -> list[dict]:
    gated = []
    for row in rows:
        pred = "yes" if row["base_prediction"] == "yes" and row["tdev_margin"] > threshold else "no"
        gated.append({**row, "prediction": pred})
    return gated


def choose_threshold(rows: list[dict]) -> tuple[float, dict]:
    margins = sorted({row["tdev_margin"] for row in rows})
    candidates = [margins[0] - 1e-6, margins[-1] + 1e-6]
    candidates.extend((a + b) / 2 for a, b in zip(margins, margins[1:]))
    best_threshold = 0.0
    best_metrics: dict | None = None
    for threshold in candidates:
        vals = metrics(apply_gate(rows, threshold))
        if best_metrics is None or vals["mcc"] > best_metrics["mcc"]:
            best_threshold = threshold
            best_metrics = vals
    assert best_metrics is not None
    return best_threshold, best_metrics


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    tdev_rows = load_tdev_rows(Path(args.tdev_predictions_csv))
    base_rows = load_base_rows(Path(args.result_root), splits, args.base_method, tdev_rows)

    if args.threshold_mode == "zero":
        threshold = 0.0
        calibration_metrics = metrics(apply_gate([row for row in base_rows if row["split"] == args.calibration_split], threshold))
    else:
        threshold, calibration_metrics = choose_threshold([row for row in base_rows if row["split"] == args.calibration_split])

    gated_rows = apply_gate(base_rows, threshold)
    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    summary_rows = []
    for split in splits + ["macro"]:
        split_rows = gated_rows if split == "macro" else [row for row in gated_rows if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    with (output_dir / "tdev_gate_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gated_rows)
    with (output_dir / "tdev_gate_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "threshold": threshold,
        "threshold_mode": args.threshold_mode,
        "calibration_split": args.calibration_split,
        "calibration_metrics": calibration_metrics,
        "base_method": args.base_method,
    }
    with (output_dir / "tdev_gate_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
