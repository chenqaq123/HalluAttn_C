#!/usr/bin/env python3
"""Stage 2b: recompute shape metrics from compact attention-row cache."""

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


def _top_mass_mask(x: np.ndarray, ratio: float) -> np.ndarray:
    order = np.argsort(x, axis=-1)[:, :, ::-1]
    sorted_vals = np.take_along_axis(x, order, axis=-1)
    cumsum = np.cumsum(sorted_vals, axis=-1)
    total = sorted_vals.sum(axis=-1, keepdims=True)
    keep_sorted = cumsum <= ratio * total
    keep_sorted[:, :, 0] = True
    mask = np.zeros_like(x, dtype=bool)
    np.put_along_axis(mask, order, keep_sorted, axis=-1)
    return mask


def _prepare_branch(
    obj: np.ndarray,
    instr_null: np.ndarray,
    local_null: np.ndarray,
    sink_mask: np.ndarray,
    remove_sinks: bool,
    apply_top_mass: bool,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj = obj.astype(np.float32, copy=True)
    instr_null = instr_null.astype(np.float32, copy=True)
    local_null = local_null.astype(np.float32, copy=True)
    if remove_sinks:
        obj[sink_mask] = 0.0
        instr_null[sink_mask] = 0.0
        local_null[sink_mask] = 0.0
    if apply_top_mass:
        mask = _top_mass_mask(obj, ratio)
        obj[~mask] = 0.0
        instr_null[~mask] = 0.0
        local_null[~mask] = 0.0
    return _normalize(obj), _normalize(instr_null), _normalize(local_null)


def _append_scores(
    scores: dict[str, np.ndarray],
    prefix: str,
    obj: np.ndarray,
    instr_null: np.ndarray,
    local_null: np.ndarray,
    sink_mask: np.ndarray,
    valid_local: np.ndarray,
    layer_indices: np.ndarray,
    remove_sinks: bool,
    apply_top_mass: bool,
    ratio: float,
) -> None:
    obj, instr_null, local_null = _prepare_branch(
        obj,
        instr_null,
        local_null,
        sink_mask,
        remove_sinks=remove_sinks,
        apply_top_mass=apply_top_mass,
        ratio=ratio,
    )
    n, n_layers, n_v = obj.shape
    uniform = np.ones_like(obj, dtype=np.float32)
    if remove_sinks:
        uniform[sink_mask] = 0.0
    uniform = _normalize(uniform)

    for l in range(n_layers):
        layer_id = int(layer_indices[l])
        scores[f"{prefix}cvg_kl_instr_layer_{layer_id}"] = -_kl(obj[:, l], instr_null[:, l])
        scores[f"{prefix}cvg_jsd_instr_layer_{layer_id}"] = -_jsd(obj[:, l], instr_null[:, l])
        scores[f"{prefix}cvg_kl_uniform_layer_{layer_id}"] = -_kl(obj[:, l], uniform[:, l])
        local_kl = -_kl(obj[:, l], local_null[:, l])
        local_jsd = -_jsd(obj[:, l], local_null[:, l])
        local_kl = np.where(valid_local, local_kl, np.nan)
        local_jsd = np.where(valid_local, local_jsd, np.nan)
        scores[f"{prefix}cvg_kl_local_nonobj_layer_{layer_id}"] = local_kl
        scores[f"{prefix}cvg_jsd_local_nonobj_layer_{layer_id}"] = local_jsd

        ent = _entropy(obj[:, l])
        sorted_vals = np.sort(obj[:, l], axis=-1)[:, ::-1]
        scores[f"{prefix}conc_entropy_layer_{layer_id}"] = ent
        scores[f"{prefix}conc_top1_mass_layer_{layer_id}"] = -sorted_vals[:, 0]
        scores[f"{prefix}conc_top5_mass_layer_{layer_id}"] = -sorted_vals[:, :5].sum(axis=-1)
        scores[f"{prefix}conc_top10_mass_layer_{layer_id}"] = -sorted_vals[:, :10].sum(axis=-1)
        scores[f"{prefix}conc_max_over_mean_layer_{layer_id}"] = -(
            obj[:, l].max(axis=-1) / np.clip(obj[:, l].mean(axis=-1), _EPS, None)
        )

    obj_avg = _normalize(obj.mean(axis=1))
    instr_avg = _normalize(instr_null.mean(axis=1))
    local_avg = _normalize(local_null.mean(axis=1))
    scores[f"{prefix}global_cvg_kl_instr"] = -_kl(obj_avg, instr_avg)
    scores[f"{prefix}global_cvg_jsd_instr"] = -_jsd(obj_avg, instr_avg)
    scores[f"{prefix}global_cvg_kl_local_nonobj"] = np.where(
        valid_local, -_kl(obj_avg, local_avg), np.nan
    )
    scores[f"{prefix}global_cvg_jsd_local_nonobj"] = np.where(
        valid_local, -_jsd(obj_avg, local_avg), np.nan
    )
    scores[f"{prefix}global_conc_entropy"] = _entropy(obj_avg)
    sorted_global = np.sort(obj_avg, axis=-1)[:, ::-1]
    scores[f"{prefix}global_conc_top5_mass"] = -sorted_global[:, :5].sum(axis=-1)
    scores[f"{prefix}global_conc_top10_mass"] = -sorted_global[:, :10].sum(axis=-1)
    scores[f"{prefix}global_conc_max_over_mean"] = -(
        obj_avg.max(axis=-1) / np.clip(obj_avg.mean(axis=-1), _EPS, None)
    )

    mean_dist = _normalize(obj.mean(axis=1))
    scores[f"{prefix}clc_gen_jsd"] = _entropy(mean_dist) - _entropy(obj).mean(axis=1)
    lo, hi = max(0, n_layers // 2 - 2), min(n_layers, n_layers - 1)
    if hi - lo >= 2:
        mid = obj[:, lo:hi]
        mid_mean = _normalize(mid.mean(axis=1))
        scores[f"{prefix}clc_gen_jsd_midlate"] = _entropy(mid_mean) - _entropy(mid).mean(axis=1)
    argmaxes = obj.argmax(axis=-1)
    agree = np.zeros(n, dtype=np.float32)
    denom = n_layers * (n_layers - 1) / 2
    for i in range(n):
        counts = np.bincount(argmaxes[i], minlength=n_v)
        agree[i] = np.sum(counts * (counts - 1) / 2) / denom if denom else 0.0
    scores[f"{prefix}clc_argmax_agree"] = -agree


def _roc_auc_score(labels: np.ndarray, values: np.ndarray) -> float:
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(labels, values))
    labels = labels.astype(np.int32)
    values = values.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
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
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _compute_auroc(scores: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    out = {}
    for k, vals in scores.items():
        vals = vals.astype(np.float64)
        valid = np.isfinite(vals)
        if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
            continue
        out[k] = round(float(_roc_auc_score(labels[valid], vals[valid])), 4)
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ratio", type=float, default=0.5)
    args = p.parse_args()

    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir) if args.output_dir else cache_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(cache_path)
    labels = data["labels"].astype(np.int32)
    sink_mask = data["sink_mask"].astype(bool)
    valid_local = data["valid_local_null"].astype(bool)
    layer_indices = data["layer_indices"].astype(np.int32)

    scores: dict[str, np.ndarray] = {}
    base_prefixes = [
        ("", "orig"),
        ("no_rope_", "no_rope"),
    ]
    branch_specs = [
        ("", False, False),
        ("sink_only_", True, False),
        ("topmass_only_", False, True),
        ("purified_", True, True),
    ]
    for base_prefix, base_name in base_prefixes:
        for branch_prefix, remove_sinks, apply_top_mass in branch_specs:
            prefix = base_prefix + branch_prefix
            _append_scores(
                scores,
                prefix,
                data[f"{base_name}_obj"],
                data[f"{base_name}_instr_null"],
                data[f"{base_name}_local_null"],
                sink_mask,
                valid_local,
                layer_indices,
                remove_sinks=remove_sinks,
                apply_top_mass=apply_top_mass,
                ratio=args.ratio,
            )

    roc_auc = _compute_auroc(scores, labels)
    raw_path = output_dir / "row_cache_scores.npz"
    np.savez(raw_path, labels=labels, **{k: v.astype(np.float32) for k, v in scores.items()})
    metrics = {
        "cache": str(cache_path),
        "ratio": args.ratio,
        "layer_indices": layer_indices.astype(int).tolist(),
        "objects": {
            "total": int(len(labels)),
            "hallucinated": int(labels.sum()),
            "non_hallucinated": int(len(labels) - labels.sum()),
        },
        "roc_auc": roc_auc,
    }
    metrics_path = output_dir / "row_cache_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {raw_path}")
    print(f"Wrote {metrics_path}")
    for i, (k, v) in enumerate(roc_auc.items()):
        if i >= 20:
            break
        print(f"{i + 1:02d} {k} {v:.4f}")


if __name__ == "__main__":
    main()
