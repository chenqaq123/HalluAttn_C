"""Common metrics for baseline evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except ModuleNotFoundError:
    _sklearn_roc_auc_score = None

_EPS = 1e-12


def roc_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    labels = labels[valid]
    values = values[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(labels, values))

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_bin_auc(
    labels: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    bin_width: int = 10,
    min_n: int = 30,
    min_pos: int = 1,
    min_neg: int = 1,
) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int32)
    bins = []
    weights = []
    aucs = []
    if positions.size == 0:
        return {"auc": None, "bins": []}
    start = int(math.floor(positions.min() / bin_width) * bin_width)
    end = int(math.ceil((positions.max() + 1) / bin_width) * bin_width)
    for lo in range(start, end, bin_width):
        hi = lo + bin_width - 1
        mask = (positions >= lo) & (positions <= hi) & np.isfinite(values)
        n = int(mask.sum())
        n_pos = int(labels[mask].sum())
        n_neg = n - n_pos
        item = {
            "bin_start": lo,
            "bin_end": hi,
            "n": n,
            "hallucinated": n_pos,
            "auc": None,
        }
        if n >= min_n and n_pos >= min_pos and n_neg >= min_neg:
            item["auc"] = roc_auc(labels[mask], values[mask])
            if item["auc"] is not None:
                aucs.append(item["auc"])
                weights.append(n)
        bins.append(item)
    weighted = float(np.average(aucs, weights=weights)) if aucs else None
    return {"auc": weighted, "bins": bins}


def matched_pair_auc(
    labels: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    delta: int = 5,
) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int32)
    valid = np.isfinite(values)
    pos_idx = np.where(valid & (labels == 1))[0]
    neg_idx = np.where(valid & (labels == 0))[0]
    wins = 0.0
    pairs = 0
    for i in pos_idx:
        candidates = neg_idx[np.abs(positions[neg_idx] - positions[i]) <= delta]
        if candidates.size == 0:
            continue
        diff = values[i] - values[candidates]
        wins += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        pairs += int(candidates.size)
    return {"auc": (wins / pairs if pairs else None), "pairs": pairs}


def residualize_by_position(values: np.ndarray, positions: np.ndarray, bin_width: int = 10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int32)
    residual = values.copy()
    if values.size == 0:
        return residual
    start = int(math.floor(positions.min() / bin_width) * bin_width)
    end = int(math.ceil((positions.max() + 1) / bin_width) * bin_width)
    global_mean = np.nanmean(values)
    for lo in range(start, end, bin_width):
        hi = lo + bin_width - 1
        mask = (positions >= lo) & (positions <= hi) & np.isfinite(values)
        if mask.sum() == 0:
            continue
        residual[mask] = values[mask] - float(np.mean(values[mask]))
    residual[~np.isfinite(residual)] = global_mean
    return residual


def compute_metric_bundle(
    labels: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    bin_width: int = 10,
    matched_delta: int = 5,
) -> dict:
    within = within_bin_auc(labels, values, positions, bin_width=bin_width)
    matched = matched_pair_auc(labels, values, positions, delta=matched_delta)
    residual = residualize_by_position(values, positions, bin_width=bin_width)
    return {
        "overall_auroc": roc_auc(labels, values),
        "within_bin_auroc": within["auc"],
        "matched_pair_auroc": matched["auc"],
        "matched_pair_count": matched["pairs"],
        "residual_auroc": roc_auc(labels, residual),
        "finite_count": int(np.isfinite(values).sum()),
        "within_bins": within["bins"],
    }


def score_direction_note(score_names: Iterable[str]) -> dict[str, str]:
    return {name: "larger means more likely hallucinated" for name in score_names}

