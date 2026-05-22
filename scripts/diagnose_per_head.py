#!/usr/bin/env python3
"""Diagnostic analysis: does a position-independent grounding signal survive in
PER-HEAD attention shape features?

Two tests, both position-controlled (numpy only, no sklearn):

  (A) Single-feature screen. For every (layer, head, feature), compute the
      within-bin AUROC (size-weighted over generation-position bins of width
      10). If even the best single per-head feature is ~0.50, the signal is
      absent; if some reach >=0.60, signal exists but was washed out by
      mean-over-heads aggregation.

  (B) Multivariate probe. L2-regularized logistic regression over ALL per-head
      features (L*H*F dims), evaluated with k-fold cross-validation; we then
      report the within-bin AUROC of the held-out predictions. This asks
      whether a *combination* of heads recovers position-independent signal.

Reference point: IC reaches 0.686 within-bin on this data; mean-over-heads
SinkDetect collapses to ~0.50.

Usage:
  python scripts/diagnose_per_head.py \
      --cache experiments/coco_llava_7b_rows/per_head_row_cache.npz \
      --generation_json experiments/coco_llava_7b/generation.json
Pass --cache_glob to merge shards, e.g. ".../per_head_row_cache_shard*.npz".
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


# ---------- metrics (numpy only) ----------

def roc_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = labels.astype(np.int32)
    values = values.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values)
    sv = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def within_bin_auroc(values, labels, gen_pos, bin_width=10, min_n=10) -> float | None:
    aucs, weights = [], []
    lo, hi = int(gen_pos.min()), int(gen_pos.max())
    for s in range(lo, hi + 1, bin_width):
        mask = (gen_pos >= s) & (gen_pos < s + bin_width)
        if mask.sum() < min_n:
            continue
        v = values[mask]
        valid = np.isfinite(v)
        if valid.sum() < min_n:
            continue
        auc = roc_auc(labels[mask][valid], v[valid])
        if auc is not None:
            aucs.append(auc)
            weights.append(int(valid.sum()))
    if not aucs:
        return None
    return float(np.average(aucs, weights=weights))


# ---------- numpy logistic regression ----------

def train_logreg(X, y, l2=1.0, lr=0.5, epochs=300):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    y = y.astype(np.float64)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


def kfold_scores(X, y, k=5, l2=1.0, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    preds = np.zeros(len(y))
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(k) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        w, b = train_logreg(Xtr, y[tr], l2=l2)
        preds[te] = Xte @ w + b
    return preds


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="experiments/coco_llava_7b_rows/per_head_row_cache.npz")
    p.add_argument("--cache_glob", default=None, help="Glob to merge shard caches instead of --cache.")
    p.add_argument("--generation_json", default="experiments/coco_llava_7b/generation.json")
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    if args.cache_glob:
        files = sorted(glob.glob(args.cache_glob))
        if not files:
            raise FileNotFoundError(args.cache_glob)
        parts = [np.load(f, allow_pickle=True) for f in files]
        feats = np.concatenate([z["feats"] for z in parts], axis=0)
        labels = np.concatenate([z["labels"] for z in parts], axis=0).astype(np.int32)
        token_pos = np.concatenate([z["token_pos"] for z in parts], axis=0).astype(np.int32)
        image_ids = np.concatenate([z["image_ids"] for z in parts], axis=0).astype(np.int64)
        layer_indices = parts[0]["layer_indices"]
        feature_names = [str(x) for x in parts[0]["feature_names"]]
    else:
        z = np.load(args.cache, allow_pickle=True)
        feats = z["feats"]
        labels = z["labels"].astype(np.int32)
        token_pos = z["token_pos"].astype(np.int32)
        image_ids = z["image_ids"].astype(np.int64)
        layer_indices = z["layer_indices"]
        feature_names = [str(x) for x in z["feature_names"]]

    N, L, H, F = feats.shape
    print(f"Loaded N={N} mentions, L={L} layers {list(map(int, layer_indices))}, "
          f"H={H} heads, F={F} features {feature_names}")
    print(f"Hallucinated={int(labels.sum())}  Grounded={int(len(labels)-labels.sum())}")

    gen = json.loads(Path(args.generation_json).read_text())
    prompt_end = {int(x["image_id"]): int(x["prompt_end_idx"]) for x in gen}
    gen_pos = np.array(
        [int(t) - prompt_end[int(i)] for t, i in zip(token_pos, image_ids)], dtype=np.int32
    )

    # ---- Test A: single-feature within-bin screen ----
    print("\n=== Test A: single (layer, head, feature) within-bin AUROC screen ===")
    rows = []
    for li in range(L):
        for h in range(H):
            for fi in range(F):
                v = feats[:, li, h, fi]
                wb = within_bin_auroc(v, labels, gen_pos, bin_width=args.bin_width)
                ov = roc_auc(labels, v)
                if wb is not None:
                    rows.append((int(layer_indices[li]), h, feature_names[fi], ov, wb))
    rows.sort(key=lambda r: r[4], reverse=True)
    print(f"{'layer':>5} {'head':>4} {'feature':<18} {'overall':>8} {'within-bin':>10}")
    for layer, head, fname, ov, wb in rows[: args.top]:
        print(f"{layer:>5} {head:>4} {fname:<18} {ov:8.4f} {wb:10.4f}")
    best = rows[0]
    print(f"\nBest single per-head feature within-bin AUROC = {best[4]:.4f} "
          f"(layer {best[0]}, head {best[1]}, {best[2]})")
    print("Reference: IC=0.686, mean-over-heads SinkDetect~0.50")

    # ---- Test B: multivariate probe (k-fold CV, position-controlled) ----
    print("\n=== Test B: multivariate logistic probe over all per-head features ===")
    X = feats.reshape(N, L * H * F)
    finite = np.isfinite(X).all(axis=1)
    Xg, yg, gpg = X[finite], labels[finite], gen_pos[finite]
    preds = kfold_scores(Xg, yg.astype(np.float64), k=5, l2=args.l2)
    overall = roc_auc(yg, preds)
    wb = within_bin_auroc(preds, yg, gpg, bin_width=args.bin_width)
    print(f"5-fold CV  overall AUROC = {overall:.4f}")
    print(f"5-fold CV  within-bin AUROC = {wb:.4f}   (IC reference 0.686)")

    print("\n--- VERDICT ---")
    verdict_signal = max(best[4], wb if wb else 0.0)
    if verdict_signal >= 0.60:
        print(f"Per-head signal SURVIVES position control ({verdict_signal:.3f} >= 0.60): "
              "grounding info is present in attention but washed out by head averaging. "
              "=> Route: build a per-head/position-controlled detector.")
    else:
        print(f"Per-head signal does NOT survive ({verdict_signal:.3f} < 0.60): "
              "attention shape carries no position-independent grounding signal even per-head. "
              "=> Route: constructive contribution should be position-calibrated fusion, not attention.")


if __name__ == "__main__":
    main()
