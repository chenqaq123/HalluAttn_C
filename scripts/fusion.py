#!/usr/bin/env python3
"""
Logistic regression fusion of hallucination detection signals.

Reads raw_scores.npz from a SinkDetect experiment directory, trains a
logistic regression classifier with 5-fold cross-validation, and reports
AUROC for various signal combinations (PAS-only, non-PAS, per-head,
combined). Also outputs the learned coefficients for interpretability.

Usage:
    python scripts/fusion.py --exp_dir experiments/coco_llava_7b
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))


def _load_data(exp_dir: Path):
    """Load raw_scores.npz and return (dict_of_arrays, labels)."""
    raw_path = exp_dir / "raw_scores.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_scores.npz not found in {exp_dir}")
    data = dict(np.load(raw_path))
    labels = data.pop("labels").astype(int)
    return data, labels


def _classify_keys(keys):
    """Sort score keys into families."""
    pas_keys = [k for k in keys if k.startswith(("orig_prelim_attn", "orig_image_attn",
                                                   "orig_bos_attn", "global_orig_"))]
    purified_pas_keys = [k for k in keys if k.startswith(("purified_prelim_attn", "purified_image_attn",
                                                           "purified_bos_attn", "global_purified_"))]
    sink_only_pas_keys = [
        k for k in keys
        if k.startswith(("sink_only_prelim_attn",
                         "sink_only_image_attn",
                         "sink_only_bos_attn",
                         "global_sink_only_"))
    ]
    topmass_only_pas_keys = [
        k for k in keys
        if k.startswith(("topmass_only_prelim_attn",
                         "topmass_only_image_attn",
                         "topmass_only_bos_attn",
                         "global_topmass_only_"))
    ]
    sink_keys = [k for k in keys if k.startswith(("sink_attn_mass", "sink_count",
                                                   "global_sink_"))]
    cvg_keys = [k for k in keys if k.startswith("cvg_") or k.startswith("global_cvg_")]
    conc_keys = [k for k in keys if k.startswith("conc_") or k.startswith("global_conc_")]
    clc_keys = [k for k in keys if k.startswith("clc_")]
    purified_cvg_keys = [k for k in keys if k.startswith("purified_cvg_") or k.startswith("purified_global_cvg_")]
    purified_conc_keys = [k for k in keys if k.startswith("purified_conc_") or k.startswith("purified_global_conc_")]
    purified_clc_keys = [k for k in keys if k.startswith("purified_clc_")]
    sink_only_cvg_keys = [k for k in keys if k.startswith("sink_only_cvg_") or k.startswith("sink_only_global_cvg_")]
    sink_only_conc_keys = [k for k in keys if k.startswith("sink_only_conc_") or k.startswith("sink_only_global_conc_")]
    sink_only_clc_keys = [k for k in keys if k.startswith("sink_only_clc_")]
    topmass_only_cvg_keys = [k for k in keys if k.startswith("topmass_only_cvg_") or k.startswith("topmass_only_global_cvg_")]
    topmass_only_conc_keys = [k for k in keys if k.startswith("topmass_only_conc_") or k.startswith("topmass_only_global_conc_")]
    topmass_only_clc_keys = [k for k in keys if k.startswith("topmass_only_clc_")]
    ph_keys = [k for k in keys if k.startswith("ph_")]
    shift_keys = [k for k in keys if k.startswith(("attn_shift_", "global_attn_shift_"))]
    return {
        "pas": pas_keys,
        "purified_pas": purified_pas_keys,
        "sink_only_pas": sink_only_pas_keys,
        "topmass_only_pas": topmass_only_pas_keys,
        "sink": sink_keys,
        "cvg": cvg_keys,
        "concentration": conc_keys,
        "clc": clc_keys,
        "purified_cvg": purified_cvg_keys,
        "purified_concentration": purified_conc_keys,
        "purified_clc": purified_clc_keys,
        "sink_only_cvg": sink_only_cvg_keys,
        "sink_only_concentration": sink_only_conc_keys,
        "sink_only_clc": sink_only_clc_keys,
        "topmass_only_cvg": topmass_only_cvg_keys,
        "topmass_only_concentration": topmass_only_conc_keys,
        "topmass_only_clc": topmass_only_clc_keys,
        "per_head": ph_keys,
        "shift": shift_keys,
    }


def _build_X(data, keys):
    """Build feature matrix from selected keys, dropping any with NaN/Inf."""
    arrays = []
    valid_keys = []
    for k in keys:
        if k not in data:
            continue
        arr = data[k].astype(np.float32)
        if np.isfinite(arr).all():
            arrays.append(arr)
            valid_keys.append(k)
    if not arrays:
        return None, []
    return np.column_stack(arrays), valid_keys


def _eval_fusion(X, y, cv=5):
    """5-fold CV AUROC with standardised logistic regression."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean(), scores.std()


def main():
    parser = argparse.ArgumentParser(description="Logistic regression fusion of detection signals")
    parser.add_argument("--exp_dir", type=str,
                        default=str(ROOT_DIR / "experiments" / "coco_llava_7b"),
                        help="Experiment directory containing raw_scores.npz")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: exp_dir/fusion_results.json)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    data, labels = _load_data(exp_dir)
    families = _classify_keys(list(data.keys()))

    print(f"Loaded {len(data)} signal arrays, {len(labels)} labels "
          f"({labels.sum()} hallucinated, {(1-labels).sum()} non-hallucinated)")
    print()

    results = {}

    # ── Evaluate individual families ──────────────────────────────────────
    print("=" * 70)
    print("Family-level fusion results (5-fold CV AUROC)")
    print("=" * 70)

    for family_name, keys in families.items():
        if not keys:
            continue
        X, valid_keys = _build_X(data, keys)
        if X is None:
            continue
        mean, std = _eval_fusion(X, labels, cv=args.cv)
        results[family_name] = {
            "auroc_mean": round(mean, 4),
            "auroc_std": round(std, 4),
            "n_features": len(valid_keys),
            "features": valid_keys,
        }
        print(f"  {family_name:20s}: AUROC = {mean:.4f} +/- {std:.4f}  ({len(valid_keys)} features)")

    # ── Combined subsets ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Combined subsets")
    print("=" * 70)

    combos = {
        "pas + purified_pas": families["pas"] + families["purified_pas"],
        "pas + sink_only_pas": families["pas"] + families["sink_only_pas"],
        "pas + topmass_only_pas": families["pas"] + families["topmass_only_pas"],
        "all_pas_family": (families["pas"] + families["purified_pas"]
                           + families["sink_only_pas"] + families["topmass_only_pas"]
                           + families["sink"] + families["shift"]),
        "non_pas": (families["cvg"] + families["concentration"]
                    + families["clc"]
                    + families["purified_cvg"] + families["purified_concentration"]
                    + families["purified_clc"]
                    + families["sink_only_cvg"]
                    + families["sink_only_concentration"]
                    + families["sink_only_clc"]
                    + families["topmass_only_cvg"]
                    + families["topmass_only_concentration"]
                    + families["topmass_only_clc"]
                    + families["per_head"]),
        "sink_only_shape": (
            families["sink_only_cvg"]
            + families["sink_only_concentration"]
            + families["sink_only_clc"]
        ),
        "topmass_only_shape": (
            families["topmass_only_cvg"]
            + families["topmass_only_concentration"]
            + families["topmass_only_clc"]
        ),
        "purified_shape": (families["purified_cvg"] + families["purified_concentration"]
                           + families["purified_clc"]),
        "all_signals": list(data.keys()),
    }

    for combo_name, keys in combos.items():
        if not keys:
            continue
        X, valid_keys = _build_X(data, keys)
        if X is None:
            continue
        mean, std = _eval_fusion(X, labels, cv=args.cv)
        results[combo_name] = {
            "auroc_mean": round(mean, 4),
            "auroc_std": round(std, 4),
            "n_features": len(valid_keys),
        }
        print(f"  {combo_name:30s}: AUROC = {mean:.4f} +/- {std:.4f}  ({len(valid_keys)} features)")

    # ── Best single-signal baselines ──────────────────────────────────────
    print()
    print("=" * 70)
    print("Top 10 single-signal baselines")
    print("=" * 70)

    from sklearn.metrics import roc_auc_score
    single_aurocs = {}
    for k, v in data.items():
        arr = v.astype(np.float32)
        if np.isfinite(arr).all() and len(set(labels)) >= 2:
            try:
                auc = roc_auc_score(labels, arr)
                single_aurocs[k] = round(auc, 4)
            except ValueError:
                pass
    single_sorted = sorted(single_aurocs.items(), key=lambda x: x[1], reverse=True)
    for k, v in single_sorted[:10]:
        print(f"  {k:50s}: {v:.4f}")
    results["top_single_signals"] = single_sorted[:10]

    # ── Top layer analysis for PAS ────────────────────────────────────────
    print()
    print("=" * 70)
    print("Per-layer PAS signal strength (orig_prelim_attn)")
    print("=" * 70)

    layer_aurocs = {}
    for k, v in single_aurocs.items():
        if k.startswith("orig_prelim_attn_layer_"):
            layer_aurocs[k] = v
    for k in sorted(layer_aurocs.keys(), key=lambda x: int(x.split("_")[-1])):
        print(f"  {k}: {layer_aurocs[k]:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else exp_dir / "fusion_results.json"
    # Convert tuples to lists for JSON serialization
    json_results = {}
    for k, v in results.items():
        if isinstance(v, list):
            json_results[k] = [[a, b] for a, b in v]
        else:
            json_results[k] = v
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
