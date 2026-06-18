#!/usr/bin/env python3
"""Evaluate LH-Shape triage for POPE TDEV-region calls.

The script joins cached LH-Shape absent-risk scores with existing OWLv2/TDEV
hybrid predictions. It simulates calling TDEV only on high-risk rows and using
vanilla/base predictions elsewhere, then reports POPE metrics by split and
semantic-neighbor subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LH-Shape triage for TDEV-region POPE calls")
    p.add_argument("--lh_predictions_csv", required=True)
    p.add_argument("--tdev_predictions_csv", default="mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_predictions.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--score_names", default="lh_shape_pope_layers_22_31,lh_shape_pope_layers_31,prompt_token_pos,target_char_len")
    p.add_argument("--call_rates", default="0.10,0.25,0.50,0.75,1.00")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_floats(spec: str) -> list[float]:
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("call rate list cannot be empty")
    return values


def top_fraction_keys(rows: list[dict], rate: float, eligible_keys: set[tuple[str, str]] | None = None) -> set[tuple[str, str]]:
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"call rate must be in (0, 1], got {rate}")
    eligible = [row for row in rows if eligible_keys is None or (row["split"], row["question_id"]) in eligible_keys]
    if not eligible:
        return set()
    k = max(1, min(math.ceil(len(eligible) * rate), len(eligible)))
    ranked = sorted(eligible, key=lambda row: float(row["absent_score"]), reverse=True)
    return {(row["split"], row["question_id"]) for row in ranked[:k]}


def metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = row["prediction"]
        gold = row["label"]
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
        "balanced_accuracy": 0.5 * (tpr + tnr),
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


def summarize(predicted: list[dict], method: str, score_name: str, call_rate: float, calls: int) -> list[dict]:
    splits = sorted({row["split"] for row in predicted})
    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain", "negative_target_present_coco_label"]
    rows = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [row for row in predicted if row["split"] == split]
        for subset in subsets:
            items = subset_rows(split_rows, subset)
            if not items:
                continue
            rows.append({
                "method": method,
                "score": score_name,
                "call_rate": call_rate,
                "detector_calls": calls,
                "detector_call_savings": 1.0 - calls / len(predicted) if predicted else 0.0,
                "split": split,
                "subset": subset,
                **metrics(items),
            })
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lh_rows = read_csv(Path(args.lh_predictions_csv))
    tdev_rows = read_csv(Path(args.tdev_predictions_csv))
    tdev_by_key = {(row["split"], row["question_id"]): row for row in tdev_rows}
    score_names = parse_list(args.score_names)
    call_rates = parse_floats(args.call_rates)
    summary_rows = []
    prediction_rows = []

    for score_name in score_names:
        score_rows = [row for row in lh_rows if row["score"] == score_name]
        if not score_rows:
            continue
        keys = [(row["split"], row["question_id"]) for row in score_rows]
        joined = []
        for row, key in zip(score_rows, keys):
            if key not in tdev_by_key:
                raise KeyError(f"Missing TDEV row for {key}")
            tdev = tdev_by_key[key]
            joined.append({
                "split": row["split"],
                "question_id": row["question_id"],
                "target": row["target"],
                "label": tdev["label"],
                "negative_type": tdev["negative_type"],
                "base_prediction": tdev["base_prediction"],
                "tdev_prediction": tdev["prediction"],
                "absent_score": float(row["absent_score"]),
            })
        base_pred = [{**row, "prediction": row["base_prediction"]} for row in joined]
        full_tdev = [{**row, "prediction": row["tdev_prediction"]} for row in joined]
        summary_rows.extend(summarize(base_pred, "vanilla_base", score_name, 0.0, 0))
        summary_rows.extend(summarize(full_tdev, "full_tdev", score_name, 1.0, len(joined)))

        base_yes_keys = {(row["split"], row["question_id"]) for row in joined if row["base_prediction"] == "yes"}
        for rate in call_rates:
            for mode, eligible in (("all_selected", None), ("base_yes_selected", base_yes_keys)):
                selected = top_fraction_keys(
                    [{"split": row["split"], "question_id": row["question_id"], "absent_score": row["absent_score"]} for row in joined],
                    rate,
                    eligible_keys=eligible,
                )
                predicted = []
                for row in joined:
                    key = (row["split"], row["question_id"])
                    pred = row["tdev_prediction"] if key in selected else row["base_prediction"]
                    item = {**row, "prediction": pred, "called_tdev": int(key in selected), "mode": mode, "score": score_name, "call_rate": rate}
                    predicted.append(item)
                    prediction_rows.append(item)
                summary_rows.extend(summarize(predicted, mode, score_name, rate, len(selected)))

    with (output_dir / "pope_lh_tdev_cascade_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "pope_lh_tdev_cascade_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    payload = {
        "lh_predictions_csv": args.lh_predictions_csv,
        "tdev_predictions_csv": args.tdev_predictions_csv,
        "score_names": score_names,
        "call_rates": call_rates,
        "num_lh_prediction_rows": len(lh_rows),
        "metrics": summary_rows,
        "predictions_csv": str(output_dir / "pope_lh_tdev_cascade_predictions.csv"),
        "caveat": "Cascade uses cached LH-Shape OOF scores and existing OWLv2/TDEV predictions; it estimates detector-call savings from saved predictions, not a fresh detector runtime benchmark.",
    }
    with (output_dir / "pope_lh_tdev_cascade_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    best = max(
        [row for row in summary_rows if row["split"] == "macro" and row["subset"] == "all" and row["method"] not in {"vanilla_base", "full_tdev"}],
        key=lambda row: (row["mcc"], -row["detector_calls"]),
    )
    print(f"Wrote {output_dir / 'pope_lh_tdev_cascade_metrics.json'}")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
