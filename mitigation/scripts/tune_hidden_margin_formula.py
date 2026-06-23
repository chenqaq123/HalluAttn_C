#!/usr/bin/env python3
"""Tune low-dimensional D-lite hidden-margin formulas from cached predictions.

This does not run the VLM. It reads hidden_margin_tdev_predictions.csv, searches
small linear combinations of the saved training-free components on a calibration
split, then writes scored CSVs and metrics for selected operating points.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

COMPONENTS = [
    "hidden_align_margin",
    "hidden_cross_margin",
    "hidden_obj_separation",
    "hidden_vis_separation",
]
SUBSETS = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain", "negative_target_present_coco_label"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune D-lite hidden-margin formulas from cached scores")
    p.add_argument("--score_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--calibration_split", default="random")
    p.add_argument("--weight_grid", default="-1,-0.5,0,0.25,0.5,1,1.5,2")
    p.add_argument("--min_tprs", default="0.85,0.80,0.75,0.70,0.65")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--base_prediction_csv", default="", help="Optional existing gate prediction CSV with base_prediction column; not required")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


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
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "positive":
        return [r for r in rows if r["label"] == "yes"]
    if subset == "negative":
        return [r for r in rows if r["label"] == "no"]
    return [r for r in rows if r.get("negative_type") == subset]


def summarize(predicted: list[dict]) -> list[dict]:
    splits = sorted({r["split"] for r in predicted})
    out = []
    for split in splits + ["macro"]:
        split_rows = predicted if split == "macro" else [r for r in predicted if r["split"] == split]
        for subset in SUBSETS:
            out.append({"split": split, "subset": subset, **metrics(subset_rows(split_rows, subset))})
    return out


def score_row(row: dict, weights: tuple[float, ...]) -> float:
    return sum(w * float(row[c]) for w, c in zip(weights, COMPONENTS))


def candidate_thresholds(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [0.0]
    out = [unique[0] - 1e-8, unique[-1] + 1e-8]
    out.extend((a + b) / 2.0 for a, b in zip(unique, unique[1:]))
    return out


def apply(rows: list[dict], weights: tuple[float, ...], threshold: float, mode: str) -> list[dict]:
    out = []
    for row in rows:
        score = score_row(row, weights)
        score_yes = score > threshold
        if mode == "direct":
            pred = "yes" if score_yes else "no"
        else:
            base = row.get("base_prediction") or row.get("prediction")
            pred = "yes" if base == "yes" and score_yes else "no"
        out.append({**row, "prediction": pred, "formula_score": score})
    return out


def objective_key(m: dict, min_tpr: float, objective: str) -> tuple[float, ...]:
    if m["recall_tpr"] < min_tpr:
        return (-1.0,)
    if objective == "min_fpr":
        return (-m["fpr"], m["mcc"], m["recall_tpr"])
    return (m["mcc"], -m["fpr"], m["recall_tpr"])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.score_csv))
    has_base_predictions = False
    if args.base_prediction_csv:
        base_rows = read_csv(Path(args.base_prediction_csv))
        base = {(r["split"], r["question_id"]): r.get("base_prediction", r.get("prediction", "")) for r in base_rows}
        rows = [{**r, "base_prediction": base.get((r["split"], r["question_id"]), "")} for r in rows]
        has_base_predictions = True
    rows = [{**r, "label": r["label"].lower()} for r in rows]
    modes = ("direct", "gate") if has_base_predictions else ("direct",)
    cal_rows = [r for r in rows if r["split"] == args.calibration_split]
    if not cal_rows:
        raise ValueError(f"No calibration rows for split={args.calibration_split}")

    weights_values = parse_float_list(args.weight_grid)
    min_tprs = parse_float_list(args.min_tprs)
    candidates = []
    for weights in itertools.product(weights_values, repeat=len(COMPONENTS)):
        if all(w == 0.0 for w in weights):
            continue
        cal_scores = [score_row(r, weights) for r in cal_rows]
        for threshold in candidate_thresholds(cal_scores):
            for mode in modes:
                cal_pred = apply(cal_rows, weights, threshold, mode)
                cal_m = metrics(cal_pred)
                candidates.append({
                    "mode": mode,
                    "threshold": threshold,
                    "weights": weights,
                    "cal_mcc": cal_m["mcc"],
                    "cal_tpr": cal_m["recall_tpr"],
                    "cal_fpr": cal_m["fpr"],
                    "cal_yes_rate": cal_m["yes_rate"],
                })

    selected = []
    for mode in modes:
        mode_candidates = [c for c in candidates if c["mode"] == mode]
        selected.append((f"{mode}_best_cal_mcc", max(mode_candidates, key=lambda c: (c["cal_mcc"], -c["cal_fpr"], c["cal_tpr"]))))
        for min_tpr in min_tprs:
            valid = [c for c in mode_candidates if c["cal_tpr"] >= min_tpr]
            if valid:
                selected.append((f"{mode}_minfpr_cal_tpr{min_tpr:.2f}", min(valid, key=lambda c: (c["cal_fpr"], -c["cal_mcc"], -c["cal_tpr"]))))

    # Keep a short calibration leaderboard for diagnostics only.
    leaderboard = sorted(candidates, key=lambda c: (c["cal_mcc"], -c["cal_fpr"], c["cal_tpr"]), reverse=True)[: args.top_k]
    selected_rows = []
    seen = set()
    for name, cand in selected:
        key = (cand["mode"], cand["threshold"], cand["weights"])
        if key in seen:
            continue
        seen.add(key)
        pred = apply(rows, cand["weights"], cand["threshold"], cand["mode"])
        pred_rows = [{**r, "formula_name": name, "formula_weights": "|".join(f"{w:g}" for w in cand["weights"]), "formula_threshold": cand["threshold"], "formula_mode": cand["mode"]} for r in pred]
        metric_rows = summarize(pred_rows)
        macro_all = next(r for r in metric_rows if r["split"] == "macro" and r["subset"] == "all")
        rel = next(r for r in metric_rows if r["split"] == "macro" and r["subset"] == "negative_related_present")
        plain = next(r for r in metric_rows if r["split"] == "macro" and r["subset"] == "negative_absent_plain")
        write_csv(output_dir / f"{name}_predictions.csv", pred_rows)
        write_csv(output_dir / f"{name}_metrics.csv", metric_rows)
        selected_rows.append({
            "name": name,
            **cand,
            "weights": list(cand["weights"]),
            "macro_mcc": macro_all["mcc"],
            "macro_tpr": macro_all["recall_tpr"],
            "macro_fpr": macro_all["fpr"],
            "macro_yes_rate": macro_all["yes_rate"],
            "related_fpr": rel["fpr"],
            "plain_fpr": plain["fpr"],
        })

    summary_csv_rows = []
    for item in selected_rows:
        summary_csv_rows.append({**{k: v for k, v in item.items() if k != "weights"}, "weights": "|".join(f"{w:g}" for w in item["weights"])})
    write_csv(output_dir / "formula_sweep_selected.csv", summary_csv_rows)
    write_csv(output_dir / "formula_sweep_leaderboard.csv", [{**{k: v for k, v in c.items() if k != "weights"}, "weights": "|".join(f"{w:g}" for w in c["weights"])} for c in leaderboard])
    payload = {
        "score_csv": args.score_csv,
        "calibration_split": args.calibration_split,
        "components": COMPONENTS,
        "weight_grid": weights_values,
        "min_tprs": min_tprs,
        "modes": list(modes),
        "base_prediction_csv": args.base_prediction_csv,
        "num_candidates": len(candidates),
        "selected": selected_rows,
        "leaderboard_top_calibration_mcc": leaderboard,
        "caveat": "Selected operating points are calibrated on calibration_split only; leaderboard is diagnostic and should not be used as a held-out claim.",
    }
    (output_dir / "formula_sweep_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected_rows, "leaderboard": leaderboard[:5]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
