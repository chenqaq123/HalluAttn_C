#!/usr/bin/env python3
"""Evaluate a POPE score CSV on the rows it contains.

Supports direct scoring and gate-on-vanilla scoring for bounded subset runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a POPE score CSV on its available rows")
    p.add_argument("--score_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--score_field", default="tdev_margin")
    p.add_argument("--mode", choices=["direct", "gate"], default="direct")
    p.add_argument("--result_root", default="", help="Required for --mode gate; root containing pope/<split>/<method>/predictions.jsonl")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--threshold_mode", choices=["zero", "calibrate_mcc"], default="calibrate_mcc")
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


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


def attach_base_predictions(rows: list[dict], result_root: Path, method: str) -> list[dict]:
    splits = sorted({row["split"] for row in rows})
    base = {}
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for pred in read_jsonl(pred_path):
            base[(split, str(pred["question_id"]))] = normalize_yes_no(pred.get("text", pred.get("prediction", "")))
    out = []
    for row in rows:
        key = (row["split"], str(row["question_id"]))
        if key not in base:
            continue
        out.append({**row, "base_prediction": base[key]})
    return out


def apply_threshold(rows: list[dict], score_field: str, threshold: float, mode: str) -> list[dict]:
    out = []
    for row in rows:
        score_yes = float(row[score_field]) > threshold
        if mode == "direct":
            pred = "yes" if score_yes else "no"
        else:
            pred = "yes" if row.get("base_prediction") == "yes" and score_yes else "no"
        out.append({**row, "prediction": pred, "threshold": threshold})
    return out


def choose_threshold(rows: list[dict], score_field: str, mode: str) -> tuple[float, dict]:
    scores = sorted({float(row[score_field]) for row in rows})
    candidates = [scores[0] - 1e-8, scores[-1] + 1e-8]
    candidates.extend((a + b) / 2 for a, b in zip(scores, scores[1:]))
    best_threshold = candidates[0]
    best_metrics = metrics(apply_threshold(rows, score_field, best_threshold, mode))
    for threshold in candidates[1:]:
        vals = metrics(apply_threshold(rows, score_field, threshold, mode))
        if vals["mcc"] > best_metrics["mcc"]:
            best_threshold = threshold
            best_metrics = vals
    return best_threshold, best_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.score_csv))
    if not rows:
        raise ValueError(f"No rows in {args.score_csv}")
    if args.score_field not in rows[0]:
        raise KeyError(f"{args.score_field!r} is not a column in {args.score_csv}")
    rows = [{**row, "label": str(row["label"]).lower()} for row in rows]
    if args.mode == "gate":
        if not args.result_root:
            raise ValueError("--result_root is required for --mode gate")
        rows = attach_base_predictions(rows, Path(args.result_root), args.base_method)

    calibration_rows = [row for row in rows if row["split"] == args.calibration_split]
    if not calibration_rows:
        raise ValueError(f"No calibration rows for split={args.calibration_split}")
    if args.threshold_mode == "zero":
        threshold = 0.0
        calibration_metrics = metrics(apply_threshold(calibration_rows, args.score_field, threshold, args.mode))
    else:
        threshold, calibration_metrics = choose_threshold(calibration_rows, args.score_field, args.mode)
    predicted = apply_threshold(rows, args.score_field, threshold, args.mode)

    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain", "negative_target_present_coco_label"]
    splits = sorted({row["split"] for row in predicted})
    summary_rows = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    write_csv(output_dir / f"{args.mode}_predictions.csv", predicted)
    write_csv(output_dir / f"{args.mode}_metrics.csv", summary_rows)
    payload = {
        "score_csv": args.score_csv,
        "score_field": args.score_field,
        "mode": args.mode,
        "threshold": threshold,
        "threshold_mode": args.threshold_mode,
        "calibration_split": args.calibration_split,
        "calibration_metrics": calibration_metrics,
        "rows": len(predicted),
    }
    (output_dir / f"{args.mode}_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
