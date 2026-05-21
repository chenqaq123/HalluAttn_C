#!/usr/bin/env python3
"""Ablate position-prior/RoPE-style debiasing from a saved shape cache.

This is a cache-only experiment: it does not recompute model attention.
For each branch, it estimates a per-layer visual position prior from the
instruction-null rows and then corrects object/null visual distributions by
either division or subtraction before recomputing CVG, concentration, and CLC.

The correction is a statistical proxy for RoPE/position bias removal, not a
strict no-RoPE forward pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except ModuleNotFoundError:
    _sklearn_roc_auc_score = None

_EPS = 1e-10


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(x.sum(axis=-1, keepdims=True), _EPS, None)


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, None)
    return -(p * np.log(p)).sum(axis=-1)


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    return (p * (np.log(p) - np.log(q))).sum(axis=-1)


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _roc_auc_score(labels: np.ndarray, values: np.ndarray) -> float:
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(labels, values))

    labels = labels.astype(np.int32)
    values = values.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    order = np.argsort(values)
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rank_sum_pos = ranks[labels == 1].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _compute_auroc(scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    roc_auc = {}
    for key, vals in scores.items():
        vals = np.asarray(vals, dtype=np.float64)
        valid = np.isfinite(vals)
        if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
            continue
        roc_auc[key] = round(float(_roc_auc_score(labels[valid], vals[valid])), 4)
    return dict(sorted(roc_auc.items(), key=lambda x: x[1], reverse=True))


def _strip_and_normalize(x: np.ndarray, sink_mask: np.ndarray, strip_sinks: bool) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if strip_sinks:
        x[sink_mask] = 0.0
    return _normalize(x)


def _position_prior(null_clean: np.ndarray, sink_mask: np.ndarray, strip_sinks: bool) -> np.ndarray:
    prior = null_clean.mean(axis=0, keepdims=True)
    if strip_sinks:
        prior = prior.copy()
        prior[sink_mask[:1]] = 0.0
    return _normalize(prior)


def _debias(
    p: np.ndarray,
    prior: np.ndarray,
    mode: str,
    alpha: float,
) -> np.ndarray:
    if mode == "divide":
        out = p / np.clip(prior, _EPS, None)
    elif mode == "subtract":
        out = np.maximum(p - alpha * prior, 0.0)
    else:
        raise ValueError(f"Unknown debias mode: {mode}")
    return _normalize(out)


def _append_shape_scores(
    scores: dict[str, np.ndarray],
    prefix: str,
    obj: np.ndarray,
    null: np.ndarray,
    sink_mask: np.ndarray,
    strip_sinks: bool,
    mode: str,
    alpha: float,
) -> None:
    obj_clean = _strip_and_normalize(obj, sink_mask, strip_sinks)
    null_clean = _strip_and_normalize(null, sink_mask, strip_sinks)
    prior = _position_prior(null_clean, sink_mask, strip_sinks)
    obj_deb = _debias(obj_clean, prior, mode=mode, alpha=alpha)
    null_deb = _debias(null_clean, prior, mode=mode, alpha=alpha)

    n, n_layers, n_v = obj_deb.shape
    for l in range(n_layers):
        scores[f"{prefix}rope_{mode}_cvg_kl_instr_layer_{l}"] = -_kl(
            obj_deb[:, l], null_deb[:, l]
        )
        scores[f"{prefix}rope_{mode}_cvg_jsd_instr_layer_{l}"] = -_jsd(
            obj_deb[:, l], null_deb[:, l]
        )
        uniform = np.ones((n, n_v), dtype=np.float32)
        if strip_sinks:
            uniform[sink_mask[:, l]] = 0.0
        uniform = _normalize(uniform)
        scores[f"{prefix}rope_{mode}_cvg_kl_uniform_layer_{l}"] = -_kl(
            obj_deb[:, l], uniform
        )

        ent = _entropy(obj_deb[:, l])
        sorted_vals = np.sort(obj_deb[:, l], axis=-1)[:, ::-1]
        scores[f"{prefix}rope_{mode}_conc_entropy_layer_{l}"] = ent
        scores[f"{prefix}rope_{mode}_conc_top1_mass_layer_{l}"] = -sorted_vals[:, 0]
        scores[f"{prefix}rope_{mode}_conc_top5_mass_layer_{l}"] = -sorted_vals[:, :5].sum(axis=-1)
        scores[f"{prefix}rope_{mode}_conc_top10_mass_layer_{l}"] = -sorted_vals[:, :10].sum(axis=-1)
        scores[f"{prefix}rope_{mode}_conc_max_over_mean_layer_{l}"] = -(
            obj_deb[:, l].max(axis=-1) / np.clip(obj_deb[:, l].mean(axis=-1), _EPS, None)
        )

    obj_avg = _normalize(obj_deb.mean(axis=1))
    null_avg = _normalize(null_deb.mean(axis=1))
    scores[f"{prefix}rope_{mode}_global_cvg_kl_instr"] = -_kl(obj_avg, null_avg)
    scores[f"{prefix}rope_{mode}_global_cvg_jsd_instr"] = -_jsd(obj_avg, null_avg)
    scores[f"{prefix}rope_{mode}_global_conc_entropy"] = _entropy(obj_avg)
    sorted_global = np.sort(obj_avg, axis=-1)[:, ::-1]
    scores[f"{prefix}rope_{mode}_global_conc_top5_mass"] = -sorted_global[:, :5].sum(axis=-1)
    scores[f"{prefix}rope_{mode}_global_conc_top10_mass"] = -sorted_global[:, :10].sum(axis=-1)
    scores[f"{prefix}rope_{mode}_global_conc_max_over_mean"] = -(
        obj_avg.max(axis=-1) / np.clip(obj_avg.mean(axis=-1), _EPS, None)
    )

    mean_dist = _normalize(obj_deb.mean(axis=1))
    scores[f"{prefix}rope_{mode}_clc_gen_jsd"] = (
        _entropy(mean_dist) - _entropy(obj_deb).mean(axis=1)
    )
    lo, hi = max(0, n_layers // 2 - 2), min(n_layers, n_layers - 4)
    if hi - lo >= 2:
        mid = obj_deb[:, lo:hi]
        mid_mean = _normalize(mid.mean(axis=1))
        scores[f"{prefix}rope_{mode}_clc_gen_jsd_midlate"] = (
            _entropy(mid_mean) - _entropy(mid).mean(axis=1)
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--variants", default="orig,purified",
                   help="Comma-separated cache variants: orig,sink_only,topmass_only,purified")
    p.add_argument("--mode", choices=["divide", "subtract"], default="divide")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Prior subtraction weight when --mode subtract.")
    args = p.parse_args()

    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir) if args.output_dir else cache_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(cache_path)
    labels = data["labels"].astype(np.int32)
    sink_mask = data["sink_mask"].astype(bool)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    scores: dict[str, np.ndarray] = {}
    for name in variants:
        prefix = "" if name == "orig" else f"{name}_"
        strip_sinks = name != "topmass_only"
        _append_shape_scores(
            scores=scores,
            prefix=prefix,
            obj=data[f"{name}_obj"].astype(np.float32),
            null=data[f"{name}_null"].astype(np.float32),
            sink_mask=sink_mask,
            strip_sinks=strip_sinks,
            mode=args.mode,
            alpha=args.alpha,
        )

    roc_auc = _compute_auroc(scores, labels)
    suffix = f"rope_{args.mode}"
    if args.mode == "subtract":
        suffix += f"_a{args.alpha:g}"

    raw_path = output_dir / f"{suffix}_shape_scores_from_cache.npz"
    np.savez(raw_path, labels=labels, **{k: v.astype(np.float32) for k, v in scores.items()})

    metrics = {
        "cache": str(cache_path),
        "debias_mode": args.mode,
        "alpha": args.alpha,
        "variants": variants,
        "objects": {
            "total": int(len(labels)),
            "hallucinated": int(labels.sum()),
            "non_hallucinated": int(len(labels) - labels.sum()),
        },
        "roc_auc": roc_auc,
    }
    metrics_path = output_dir / f"{suffix}_shape_metrics_from_cache.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Wrote {raw_path}")
    print(f"Wrote {metrics_path}")
    for i, (key, val) in enumerate(roc_auc.items()):
        if i >= 20:
            break
        print(f"{i + 1:02d} {key} {val:.4f}")


if __name__ == "__main__":
    main()
