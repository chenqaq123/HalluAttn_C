#!/usr/bin/env python3
"""Recompute CVG / concentration / CLC scores from a saved shape cache.

This avoids re-running LLaVA forward passes when changing score formulas,
signs, global aggregation, or fusion experiments. The cache is produced by:

    python scripts/detect.py ... --save_shape_cache
"""

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


def _strip_sinks(x: np.ndarray, sink_mask: np.ndarray, strip: bool) -> np.ndarray:
    if not strip:
        return x.astype(np.float32, copy=True)
    out = x.astype(np.float32, copy=True)
    out[sink_mask] = 0.0
    return out


def _clean_distributions(
    obj: np.ndarray,
    null: np.ndarray,
    sink_mask: np.ndarray,
    strip_sinks: bool,
) -> tuple[np.ndarray, np.ndarray]:
    obj_clean = _normalize(_strip_sinks(obj, sink_mask, strip_sinks))
    null_clean = _normalize(_strip_sinks(null, sink_mask, strip_sinks))
    return obj_clean, null_clean


def _mean_pairwise_jsd(P: np.ndarray) -> float:
    n = P.shape[0]
    if n < 2:
        return 0.0
    vals = []
    for i in range(n):
        vals.append(_jsd(P[i + 1:], P[i]).sum())
    return float(np.sum(vals) / (n * (n - 1) / 2))


def _append_scores(
    scores: dict[str, np.ndarray],
    prefix: str,
    obj: np.ndarray,
    null: np.ndarray,
    sink_mask: np.ndarray,
    strip_sinks: bool,
    include_pairwise_clc: bool,
) -> None:
    obj_clean, null_clean = _clean_distributions(obj, null, sink_mask, strip_sinks)
    n, n_layers, n_v = obj_clean.shape

    # Per-layer CVG. Negative divergence: close to null = higher hallucination score.
    scores.update({
        f"{prefix}cvg_kl_instr_layer_{l}": -_kl(obj_clean[:, l], null_clean[:, l])
        for l in range(n_layers)
    })
    scores.update({
        f"{prefix}cvg_jsd_instr_layer_{l}": -_jsd(obj_clean[:, l], null_clean[:, l])
        for l in range(n_layers)
    })

    uniform = np.ones_like(obj_clean, dtype=np.float32)
    if strip_sinks:
        uniform[sink_mask] = 0.0
    uniform = _normalize(uniform)
    scores.update({
        f"{prefix}cvg_kl_uniform_layer_{l}": -_kl(obj_clean[:, l], uniform[:, l])
        for l in range(n_layers)
    })

    # Concentration.
    ent = _entropy(obj_clean)
    sorted_vals = np.sort(obj_clean, axis=-1)[:, :, ::-1]
    scores.update({f"{prefix}conc_entropy_layer_{l}": ent[:, l] for l in range(n_layers)})
    scores.update({f"{prefix}conc_top1_mass_layer_{l}": -sorted_vals[:, l, 0] for l in range(n_layers)})
    scores.update({f"{prefix}conc_top5_mass_layer_{l}": -sorted_vals[:, l, :5].sum(axis=-1) for l in range(n_layers)})
    scores.update({f"{prefix}conc_top10_mass_layer_{l}": -sorted_vals[:, l, :10].sum(axis=-1) for l in range(n_layers)})
    scores.update({
        f"{prefix}conc_max_over_mean_layer_{l}": -(
            obj_clean[:, l].max(axis=-1) / np.clip(obj_clean[:, l].mean(axis=-1), _EPS, None)
        )
        for l in range(n_layers)
    })

    # Global versions average distributions first, then score.
    obj_avg = _normalize(obj_clean.mean(axis=1))
    null_avg = _normalize(null_clean.mean(axis=1))
    scores[f"{prefix}global_cvg_kl_instr"] = -_kl(obj_avg, null_avg)
    scores[f"{prefix}global_cvg_jsd_instr"] = -_jsd(obj_avg, null_avg)
    scores[f"{prefix}global_conc_entropy"] = _entropy(obj_avg)
    sorted_global = np.sort(obj_avg, axis=-1)[:, ::-1]
    scores[f"{prefix}global_conc_top5_mass"] = -sorted_global[:, :5].sum(axis=-1)
    scores[f"{prefix}global_conc_top10_mass"] = -sorted_global[:, :10].sum(axis=-1)
    scores[f"{prefix}global_conc_max_over_mean"] = -(
        obj_avg.max(axis=-1) / np.clip(obj_avg.mean(axis=-1), _EPS, None)
    )

    # CLC.
    mean_dist = _normalize(obj_clean.mean(axis=1))
    scores[f"{prefix}clc_gen_jsd"] = _entropy(mean_dist) - _entropy(obj_clean).mean(axis=1)
    lo, hi = max(0, n_layers // 2 - 2), min(n_layers, n_layers - 4)
    if hi - lo >= 2:
        mid = obj_clean[:, lo:hi]
        mid_mean = _normalize(mid.mean(axis=1))
        scores[f"{prefix}clc_gen_jsd_midlate"] = _entropy(mid_mean) - _entropy(mid).mean(axis=1)

    argmaxes = obj_clean.argmax(axis=-1)
    agree = np.zeros(n, dtype=np.float32)
    denom = n_layers * (n_layers - 1) / 2
    for i in range(n):
        counts = np.bincount(argmaxes[i], minlength=n_v)
        agree[i] = (np.sum(counts * (counts - 1) / 2) / denom) if denom else 0.0
    scores[f"{prefix}clc_argmax_agree"] = -agree

    if include_pairwise_clc:
        scores[f"{prefix}clc_mean_pairwise_jsd"] = np.array(
            [_mean_pairwise_jsd(obj_clean[i]) for i in range(n)],
            dtype=np.float32,
        )


def _compute_auroc(scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    roc_auc = {}
    for key, vals in scores.items():
        vals = np.asarray(vals, dtype=np.float64)
        valid = np.isfinite(vals)
        if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
            continue
        try:
            roc_auc[key] = round(float(_roc_auc_score(labels[valid], vals[valid])), 4)
        except ValueError:
            pass
    return dict(sorted(roc_auc.items(), key=lambda x: x[1], reverse=True))


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
        # Average 1-based rank for ties.
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rank_sum_pos = ranks[labels == 1].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="shape_cache.npz path")
    p.add_argument("--output_dir", default=None, help="Where to write recomputed outputs")
    p.add_argument("--include_pairwise_clc", action="store_true",
                   help="Also recompute mean pairwise CLC; slower on full COCO runs.")
    args = p.parse_args()

    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir) if args.output_dir else cache_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(cache_path)
    labels = data["labels"].astype(np.int32)
    sink_mask = data["sink_mask"].astype(bool)

    scores: dict[str, np.ndarray] = {}
    variants = {
        "": ("orig", True),
        "sink_only_": ("sink_only", True),
        "topmass_only_": ("topmass_only", False),
        "purified_": ("purified", True),
        "no_rope_": ("no_rope", True),
        "no_rope_sink_only_": ("no_rope_sink_only", True),
        "no_rope_topmass_only_": ("no_rope_topmass_only", False),
        "no_rope_purified_": ("no_rope_purified", True),
    }
    for prefix, (name, strip_sinks) in variants.items():
        if f"{name}_obj" not in data or f"{name}_null" not in data:
            continue
        _append_scores(
            scores=scores,
            prefix=prefix,
            obj=data[f"{name}_obj"].astype(np.float32),
            null=data[f"{name}_null"].astype(np.float32),
            sink_mask=sink_mask,
            strip_sinks=strip_sinks,
            include_pairwise_clc=args.include_pairwise_clc,
        )

    roc_auc = _compute_auroc(scores, labels)

    raw_path = output_dir / "shape_scores_from_cache.npz"
    np.savez(raw_path, labels=labels, **{k: v.astype(np.float32) for k, v in scores.items()})

    metrics = {
        "cache": str(cache_path),
        "objects": {
            "total": int(len(labels)),
            "hallucinated": int(labels.sum()),
            "non_hallucinated": int(len(labels) - labels.sum()),
        },
        "include_pairwise_clc": bool(args.include_pairwise_clc),
        "roc_auc": roc_auc,
    }
    metrics_path = output_dir / "shape_metrics_from_cache.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote {raw_path}")
    print(f"Wrote {metrics_path}")
    if roc_auc:
        best_key = next(iter(roc_auc))
        print(f"Best AUROC: {roc_auc[best_key]:.4f} ({best_key})")


if __name__ == "__main__":
    main()
