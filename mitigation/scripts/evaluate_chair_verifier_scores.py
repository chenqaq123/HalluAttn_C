#!/usr/bin/env python3
"""Evaluate CHAIR mention-level verifier score CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

COMPONENT_DEFAULT = [
    "hidden_align_margin",
    "hidden_cross_margin",
    "hidden_obj_separation",
    "hidden_vis_separation",
    "answer_support_score",
    "answer_contrast_margin",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CHAIR internal verifier scores")
    p.add_argument("--score_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--components", default=",".join(COMPONENT_DEFAULT))
    p.add_argument("--weight_grid", default="-1,-0.5,0,0.5,1")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--standardize", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def parse_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_float_list(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def auc(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)



def metric_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall_tpr": tpr,
        "fpr": fpr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "pred_rate": (tp + fp) / n if n else 0.0,
    }

def metrics(labels: list[int], preds: list[int]) -> dict:
    tp = fp = tn = fn = 0
    for y, p in zip(labels, preds):
        if p == 1 and y == 1:
            tp += 1
        elif p == 1 and y == 0:
            fp += 1
        elif p == 0 and y == 0:
            tn += 1
        elif p == 0 and y == 1:
            fn += 1
    return metric_counts(tp, fp, tn, fn)


def best_threshold(labels: list[int], scores: list[float], min_tpr: float | None = None) -> tuple[float, dict]:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if not pairs:
        return 0.0, metrics([], [])
    best = None
    best_t = pairs[0][0] + 1e-8
    tp = fp = 0

    def consider(threshold: float, tp_val: int, fp_val: int):
        nonlocal best, best_t
        tn = total_neg - fp_val
        fn = total_pos - tp_val
        m = metric_counts(tp_val, fp_val, tn, fn)
        if min_tpr is not None and m["recall_tpr"] < min_tpr:
            return
        key = (m["mcc"], -m["fpr"], m["recall_tpr"])
        if best is None or key > best[0]:
            best = (key, m)
            best_t = threshold

    consider(pairs[0][0] + 1e-8, 0, 0)
    i = 0
    n = len(pairs)
    while i < n:
        value = pairs[i][0]
        while i < n and pairs[i][0] == value:
            if pairs[i][1] == 1:
                tp += 1
            else:
                fp += 1
            i += 1
        next_value = pairs[i][0] if i < n else value - 1e-8
        threshold = (value + next_value) / 2.0 if i < n else value - 1e-8
        consider(threshold, tp, fp)
    if best is None:
        return pairs[0][0] + 1e-8, metric_counts(0, 0, total_neg, total_pos)
    return best_t, best[1]


def params(rows: list[dict], comps: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for c in comps:
        vals = [float(r[c]) for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) or 1.0
        if std < 1e-12:
            std = 1.0
        out[c] = (mean, std)
    return out


def apply_params(row: dict, comps: list[str], ps: dict[str, tuple[float, float]]) -> dict:
    rr = dict(row)
    for c in comps:
        mean, std = ps[c]
        rr[c] = (float(row[c]) - mean) / std
    return rr


def score(row: dict, comps: list[str], weights: tuple[float, ...]) -> float:
    # Larger score should mean more hallucinated. Components with support direction
    # are learned by the grid; no hard-coded sign is imposed here.
    return sum(w * float(row[c]) for w, c in zip(weights, comps))


def grouped_folds(rows: list[dict], k: int) -> list[list[dict]]:
    groups = {}
    for r in rows:
        groups.setdefault(str(r["image_id"]), []).append(r)
    folds = [[] for _ in range(k)]
    for idx, image_id in enumerate(sorted(groups, key=lambda x: int(x))):
        folds[idx % k].extend(groups[image_id])
    return folds


def eval_fixed_score(rows: list[dict], score_field: str) -> dict:
    labels = [int(r["label"]) for r in rows]
    scores = [float(r[score_field]) for r in rows]
    t, _ = best_threshold(labels, scores)
    return {"score": score_field, "threshold": t, "auroc": auc(labels, scores), **metrics(labels, [1 if s > t else 0 for s in scores])}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.score_csv))
    comps = parse_list(args.components)
    grid = parse_float_list(args.weight_grid)
    folds = grouped_folds(rows, args.folds)
    selected_rows = []
    pred_rows = []
    for fold_idx in range(args.folds):
        test = folds[fold_idx]
        train = [r for i, fold in enumerate(folds) if i != fold_idx for r in fold]
        if args.standardize:
            ps = params(train, comps)
            train_eval = [apply_params(r, comps, ps) for r in train]
            test_eval = [apply_params(r, comps, ps) for r in test]
        else:
            train_eval, test_eval = train, test
        best = None
        best_item = None
        for weights in __import__('itertools').product(grid, repeat=len(comps)):
            if all(w == 0 for w in weights):
                continue
            train_labels = [int(r["label"]) for r in train_eval]
            train_scores = [score(r, comps, weights) for r in train_eval]
            threshold, train_m = best_threshold(train_labels, train_scores)
            key = (train_m["mcc"], -train_m["fpr"], train_m["recall_tpr"])
            if best is None or key > best:
                best = key
                best_item = (weights, threshold, train_m)
        weights, threshold, train_m = best_item
        test_labels = [int(r["label"]) for r in test_eval]
        test_scores = [score(r, comps, weights) for r in test_eval]
        test_preds = [1 if s > threshold else 0 for s in test_scores]
        test_m = metrics(test_labels, test_preds)
        selected_rows.append({
            "fold": fold_idx,
            "weights": "|".join(f"{w:g}" for w in weights),
            "threshold": threshold,
            "train_mcc": train_m["mcc"],
            "train_tpr": train_m["recall_tpr"],
            "train_fpr": train_m["fpr"],
            "test_auroc": auc(test_labels, test_scores),
            "test_mcc": test_m["mcc"],
            "test_tpr": test_m["recall_tpr"],
            "test_fpr": test_m["fpr"],
            "test_pred_rate": test_m["pred_rate"],
        })
        for raw, s, p in zip(test, test_scores, test_preds):
            pred_rows.append({**raw, "verifier_score": s, "verifier_prediction": p, "fold": fold_idx})
    labels = [int(r["label"]) for r in pred_rows]
    scores = [float(r["verifier_score"]) for r in pred_rows]
    preds = [int(r["verifier_prediction"]) for r in pred_rows]
    macro = {"score": "grid_verifier", "auroc": auc(labels, scores), **metrics(labels, preds)}
    baselines = []
    for field in ["gen_pos", "hidden_margin_score", "answer_absence_score", "answer_neighbor_dominance"]:
        if field in rows[0]:
            baselines.append(eval_fixed_score(rows, field))
    write_csv(out_dir / "chair_verifier_folds.csv", selected_rows)
    write_csv(out_dir / "chair_verifier_predictions.csv", pred_rows)
    write_csv(out_dir / "chair_verifier_baselines.csv", baselines + [macro])
    payload = {
        "score_csv": args.score_csv,
        "components": comps,
        "weight_grid": grid,
        "folds": args.folds,
        "standardize": args.standardize,
        "rows": len(rows),
        "positives_hallucinated": sum(int(r["label"]) for r in rows),
        "macro": macro,
        "baselines": baselines,
    }
    (out_dir / "chair_verifier_metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
