#!/usr/bin/env python3
"""Stage 2: merge shards and run OOF logistic regression for supervised ceiling.

Usage:
  python merge_evaluate_hidden_contrast_probe.py \\
    --shard_dirs results/hidden_contrast_probe_full/shard0,shard1,shard2,shard3,shard4 \\
    --result_root results/coco_llava_7b_attention_only \\
    --output_dir results/hidden_contrast_probe_full

Reports the same metrics as the original evaluate_hidden_contrast_probe_pope.py
(bounded subset), now on the full 9000-row POPE dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)

SUBSETS = [
    "all", "positive", "negative",
    "negative_related_present", "negative_absent_plain",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard_dirs", required=True, help="Comma-separated shard output directories")
    p.add_argument("--result_root", default="", help="Root with pope/<split>/vanilla/predictions.jsonl")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--l2", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_yes_no(text: str) -> str:
    m = YES_NO_RE.search(text or "")
    return m.group(1).lower() if m else "invalid"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Logistic regression (same as evaluate_hidden_contrast_probe_pope.py)
# ---------------------------------------------------------------------------

def train_logreg(X: np.ndarray, y: np.ndarray, l2: float, lr: float, epochs: int) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        w -= lr * ((X.T @ g) / n + l2 * w / n)
        b -= lr * float(g.mean())
    return w, b


def image_folds(image_ids: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    unique = np.asarray(sorted(set(image_ids.tolist())))
    rng.shuffle(unique)
    fold_images = np.array_split(unique, folds)
    return [(~np.isin(image_ids, fi), np.isin(image_ids, fi)) for fi in fold_images]


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return 0.0, {}
    uniq = np.unique(finite)
    candidates = [float(uniq[0] - 1e-6), float(uniq[-1] + 1e-6)]
    candidates += [float((a + b) / 2) for a, b in zip(uniq, uniq[1:])]
    best_t, best_m = candidates[0], None
    for t in candidates:
        m = binary_metrics(labels, scores > t)
        if best_m is None or m["mcc"] > best_m["mcc"]:
            best_t, best_m = t, m
    return best_t, best_m or {}


def binary_metrics(labels: np.ndarray, pred_absent: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    pred_absent = np.asarray(pred_absent, dtype=bool)
    tp = int(((labels == 1) & pred_absent).sum())
    fp = int(((labels == 0) & pred_absent).sum())
    tn = int(((labels == 0) & ~pred_absent).sum())
    fn = int(((labels == 1) & ~pred_absent).sum())
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "absent_tpr": tp / (tp + fn) if tp + fn else 0.0,
            "present_fpr": fp / (fp + tn) if fp + tn else 0.0,
            "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0}


def roc_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int32)
    valid = np.isfinite(values)
    labels = labels[valid]
    values = values[valid]
    n_pos, n_neg = int(labels.sum()), int(labels.size - labels.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values)
    sv = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def oof_scores(X: np.ndarray, labels: np.ndarray, image_ids: np.ndarray, folds: int, seed: int, l2: float, lr: float, epochs: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(labels), dtype=np.float64)
    pred_absent = np.zeros(len(labels), dtype=bool)
    for train_mask, test_mask in image_folds(image_ids, folds, seed):
        mu = X[train_mask].mean(0)
        sd = X[train_mask].std(0) + 1e-8
        X_tr = (X[train_mask] - mu) / sd
        X_te = (X[test_mask] - mu) / sd
        w, b = train_logreg(X_tr, labels[train_mask], l2, lr, epochs)
        tr_scores = X_tr @ w + b
        threshold, _ = choose_threshold(labels[train_mask], tr_scores)
        te_scores = X_te @ w + b
        scores[test_mask] = te_scores
        pred_absent[test_mask] = te_scores > threshold
    return scores, pred_absent


# ---------------------------------------------------------------------------
# Metrics for gated POPE predictions
# ---------------------------------------------------------------------------

def gate_metrics(meta: list[dict], support_scores: np.ndarray, pred_absent: np.ndarray,
                 base_preds: dict[tuple[str, str], str]) -> list[dict]:
    """Convert OOF probe scores into gated POPE predictions and compute metrics."""
    splits = sorted({r["split"] for r in meta})
    out = []
    for split in splits + ["macro"]:
        sm = split == "macro" or True  # placeholder; build row-level predictions first
        _ = sm  # unused
    # Build gated prediction rows
    rows = []
    for i, m in enumerate(meta):
        bp = base_preds.get((m["split"], m["question_id"]), "yes")
        support = float(support_scores[i])
        absent = bool(pred_absent[i])
        # Gate: if vanilla=yes and probe says absent → flip to no
        if bp == "yes" and absent:
            pred = "no"
        else:
            pred = bp
        rows.append({**m, "prediction": pred, "support_score": support, "absent_pred": absent})
    return rows


def subset_stats(rows: list[dict], split: str, subset: str) -> dict | None:
    if split != "macro":
        rows = [r for r in rows if r["split"] == split]
    if subset == "positive":
        rows = [r for r in rows if r["label"] == "yes"]
    elif subset == "negative":
        rows = [r for r in rows if r["label"] == "no"]
    elif subset in ("negative_related_present", "negative_absent_plain"):
        rows = [r for r in rows if r.get("negative_type") == subset]
    if not rows:
        return None
    tp = sum(1 for r in rows if r["prediction"] == "yes" and r["label"] == "yes")
    fp = sum(1 for r in rows if r["prediction"] == "yes" and r["label"] == "no")
    tn = sum(1 for r in rows if r["prediction"] == "no" and r["label"] == "no")
    fn = sum(1 for r in rows if r["prediction"] == "no" and r["label"] == "yes")
    n = tp + fp + tn + fn
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {"split": split, "subset": subset, "samples": n, "mcc": mcc, "recall_tpr": tpr, "fpr": fpr, "yes_rate": (tp + fp) / n if n else 0.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    shard_dirs = [Path(d.strip()) for d in args.shard_dirs.split(",") if d.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge shards
    all_X, all_labels, all_meta = [], [], []
    for sd in shard_dirs:
        X = np.load(sd / "features.npy")
        y = np.load(sd / "labels.npy")
        meta = read_csv(sd / "meta.csv")
        assert len(meta) == len(X) == len(y), f"Shard {sd} size mismatch"
        all_X.append(X)
        all_labels.append(y)
        all_meta.extend(meta)
        print(f"Loaded shard {sd.name}: {len(meta)} rows, feature_dim={X.shape[1]}")

    X = np.concatenate(all_X, axis=0).astype(np.float64)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    print(f"Merged: {len(all_meta)} rows, feature_dim={X.shape[1]}")

    image_ids = np.array([int(m["image_id"]) for m in all_meta], dtype=np.int64)

    # OOF probe
    scores_absent, pred_absent = oof_scores(
        X, labels, image_ids, args.folds, args.seed, args.l2, args.lr, args.epochs
    )
    support_scores = -scores_absent

    # AUROCs
    overall_auroc = roc_auc(labels, scores_absent)
    print(f"OOF absent AUROC (overall): {overall_auroc:.4f}")

    # Load vanilla base predictions
    base_preds: dict[tuple[str, str], str] = {}
    if args.result_root:
        result_root = Path(args.result_root)
        splits_in_data = sorted({m["split"] for m in all_meta})
        for split in splits_in_data:
            pred_path = result_root / "pope" / split / args.base_method / "predictions.jsonl"
            if pred_path.exists():
                for pred in read_jsonl(pred_path):
                    key = (split, str(pred["question_id"]))
                    base_preds[key] = normalize_yes_no(pred.get("text", pred.get("prediction", "")))
    if not base_preds:
        # Fall back to absent prediction
        for m in all_meta:
            base_preds[(m["split"], m["question_id"])] = "yes" if m["label"] == "yes" else "yes"

    gated_rows = gate_metrics(all_meta, support_scores, pred_absent, base_preds)

    # Save predictions
    write_csv(out_dir / "hidden_contrast_probe_full_predictions.csv", gated_rows)

    # Compute summary metrics
    splits_list = sorted({r["split"] for r in gated_rows})
    metric_rows = []
    for split in splits_list + ["macro"]:
        for subset in SUBSETS:
            s = subset_stats(gated_rows, split, subset)
            if s:
                metric_rows.append(s)
    write_csv(out_dir / "hidden_contrast_probe_full_metrics.csv", metric_rows)

    # Summary
    macro_all = subset_stats(gated_rows, "macro", "all") or {}
    macro_rel = subset_stats(gated_rows, "macro", "negative_related_present") or {}
    macro_plain = subset_stats(gated_rows, "macro", "negative_absent_plain") or {}
    adv_rows = [r for r in gated_rows if r["split"] == "adversarial"]
    adv_rel = subset_stats(adv_rows, "macro", "negative_related_present") or {}

    summary = {
        "oof_absent_auroc": overall_auroc,
        "macro_mcc": macro_all.get("mcc"),
        "macro_tpr": macro_all.get("recall_tpr"),
        "macro_fpr": macro_all.get("fpr"),
        "macro_related_fpr": macro_rel.get("fpr"),
        "macro_plain_fpr": macro_plain.get("fpr"),
        "macro_gap": (macro_rel.get("fpr", 0.0) or 0.0) - (macro_plain.get("fpr", 0.0) or 0.0),
        "adv_related_fpr": adv_rel.get("fpr"),
        "feature_dim": int(X.shape[1]),
        "total_rows": len(all_meta),
        "folds": args.folds,
        "l2": args.l2,
    }
    (out_dir / "hidden_contrast_probe_full_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
