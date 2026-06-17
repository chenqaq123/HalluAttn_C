#!/usr/bin/env python3
"""Calibrate a target-minus-neighbor verifier score on POPE predictions.

The score family is:

    score = target_score - alpha * best_neighbor_score

`alpha=0` recovers an open-vocabulary target detector baseline, while
`alpha=1` recovers the current target-vs-neighbor margin. Searching alpha tests
whether a softer neighbor penalty can keep target recall while reducing
semantic-neighbor false positives.
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
    p = argparse.ArgumentParser(description="Evaluate calibrated target-minus-neighbor POPE scores")
    p.add_argument("--predictions_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["direct", "gate"], default="direct")
    p.add_argument("--result_root", default="", help="Required for --mode gate")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--alpha_min", type=float, default=0.0)
    p.add_argument("--alpha_max", type=float, default=1.5)
    p.add_argument("--alpha_step", type=float, default=0.05)
    return p.parse_args()


def normalize_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def alpha_values(alpha_min: float, alpha_max: float, alpha_step: float) -> list[float]:
    if alpha_step <= 0:
        raise ValueError("--alpha_step must be positive")
    values = []
    cur = alpha_min
    while cur <= alpha_max + 1e-12:
        values.append(round(cur, 10))
        cur += alpha_step
    return values


def attach_scores(rows: list[dict], alpha: float) -> list[dict]:
    scored = []
    for row in rows:
        target = float(row["target_score"])
        neighbor = float(row["best_neighbor_score"])
        if not math.isfinite(neighbor):
            neighbor = 0.0
        score = target - alpha * neighbor
        scored.append({**row, "neighbor_penalty_alpha": alpha, "neighbor_penalty_score": score})
    return scored


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


def apply_threshold(rows: list[dict], threshold: float, mode: str) -> list[dict]:
    predicted = []
    for row in rows:
        allowed = True if mode == "direct" else row["base_prediction"] == "yes"
        pred = "yes" if allowed and float(row["neighbor_penalty_score"]) > threshold else "no"
        predicted.append({**row, "prediction": pred})
    return predicted


def metric_from_counts(tp: int, fp: int, total_pos: int, total_neg: int) -> dict:
    fn = total_pos - tp
    tn = total_neg - fp
    n = total_pos + total_neg
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / total_pos if total_pos else 0.0
    fpr = fp / total_neg if total_neg else 0.0
    tnr = tn / total_neg if total_neg else 0.0
    denom = math.sqrt((tp + fp) * total_pos * total_neg * (tn + fn))
    return {
        "samples": n,
        "invalid": 0,
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


def best_threshold_for_alpha(rows: list[dict], alpha: float, mode: str) -> tuple[float, dict]:
    scored = attach_scores(rows, alpha)
    total_pos = sum(1 for row in scored if row["label"] == "yes")
    total_neg = sum(1 for row in scored if row["label"] == "no")
    eligible = [
        row for row in scored
        if mode == "direct" or row["base_prediction"] == "yes"
    ]
    eligible.sort(key=lambda row: float(row["neighbor_penalty_score"]), reverse=True)
    if not eligible:
        return 0.0, metric_from_counts(0, 0, total_pos, total_neg)

    best_threshold = float(eligible[0]["neighbor_penalty_score"]) + 1e-6
    best_metrics = metric_from_counts(0, 0, total_pos, total_neg)
    tp = fp = 0
    i = 0
    while i < len(eligible):
        score = float(eligible[i]["neighbor_penalty_score"])
        j = i
        while j < len(eligible) and float(eligible[j]["neighbor_penalty_score"]) == score:
            if eligible[j]["label"] == "yes":
                tp += 1
            else:
                fp += 1
            j += 1
        next_threshold = score - 1e-6 if j == len(eligible) else (score + float(eligible[j]["neighbor_penalty_score"])) / 2
        vals = metric_from_counts(tp, fp, total_pos, total_neg)
        if vals["mcc"] > best_metrics["mcc"]:
            best_threshold = next_threshold
            best_metrics = vals
        i = j
    return best_threshold, best_metrics


def choose_alpha_threshold(rows: list[dict], alphas: list[float], mode: str) -> tuple[float, float, dict]:
    best_alpha = 0.0
    best_threshold = 0.0
    best_metrics: dict | None = None
    for alpha in alphas:
        threshold, vals = best_threshold_for_alpha(rows, alpha, mode=mode)
        if best_metrics is None or vals["mcc"] > best_metrics["mcc"]:
            best_alpha = alpha
            best_threshold = threshold
            best_metrics = vals
    assert best_metrics is not None
    return best_alpha, best_threshold, best_metrics


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "positive":
        return [row for row in rows if row["label"] == "yes"]
    if subset == "negative":
        return [row for row in rows if row["label"] == "no"]
    return [row for row in rows if row.get("negative_type") == subset]


def attach_base_predictions(rows: list[dict], result_root: Path, splits: list[str], method: str) -> list[dict]:
    base_by_key = {}
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for row in read_jsonl(pred_path):
            base_by_key[(split, str(row["question_id"]))] = normalize_yes_no(row.get("text", row.get("prediction", "")))
    attached = []
    for row in rows:
        key = (row["split"], str(row["question_id"]))
        attached.append({**row, "base_prediction": base_by_key[key]})
    return attached


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    rows = read_csv(Path(args.predictions_csv))
    required = {"target_score", "best_neighbor_score", "split", "question_id", "label", "negative_type"}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Missing columns in {args.predictions_csv}: {sorted(missing)}")

    if args.mode == "gate":
        if not args.result_root:
            raise ValueError("--result_root is required for --mode gate")
        rows = attach_base_predictions(rows, Path(args.result_root), splits, args.base_method)
    else:
        rows = [{**row, "base_prediction": ""} for row in rows]

    calibration_rows = [row for row in rows if row["split"] == args.calibration_split]
    alphas = alpha_values(args.alpha_min, args.alpha_max, args.alpha_step)
    best_alpha, threshold, calibration_metrics = choose_alpha_threshold(calibration_rows, alphas, mode=args.mode)

    predicted = apply_threshold(attach_scores(rows, best_alpha), threshold, mode=args.mode)
    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    summary_rows = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            summary_rows.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})

    with (output_dir / "neighbor_penalty_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predicted[0].keys()))
        writer.writeheader()
        writer.writerows(predicted)
    with (output_dir / "neighbor_penalty_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "predictions_csv": args.predictions_csv,
        "mode": args.mode,
        "base_method": args.base_method if args.mode == "gate" else "",
        "calibration_split": args.calibration_split,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_step": args.alpha_step,
        "alpha": best_alpha,
        "threshold": threshold,
        "calibration_metrics": calibration_metrics,
    }
    with (output_dir / "neighbor_penalty_config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
