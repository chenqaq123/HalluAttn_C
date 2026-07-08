#!/usr/bin/env python
"""Offline scoring/evaluation for the FIXED SVAR / Beyond-ADS / Beyond-CGC scores.

Reads the per-layer feature shards from extract_beyond_svar_features.py and
computes, against the same position/length-controlled metrics as the detection
baselines (overall / within-bin / matched-pair / residual AUROC):

  - SVAR (fixed)   : -(sum over layers 5..18 of image_mass)   [Jiang et al.]
  - SVAR (layer-0) : -image_mass[:,0]   (sanity vs. the old 0.834 number)
  - ADS (fixed)    : per-layer sweep + a fixed mid layer       [Beyond, arXiv:2604.04863]
  - CGC (fixed)    : per-layer sweep + a fixed mid layer
  - ADS+CGC (sup.) : image-grouped OOF gradient-boosting over the 64-dim
                     [per-layer ADS || per-layer CGC] feature vector (the paper's
                     combined detector; paper uses XGBoost, we use sklearn HGB).

Sign convention: larger score => more likely hallucinated.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from detection.baselines.metrics import compute_metric_bundle  # noqa: E402


def _fmt(b: dict) -> str:
    return (f"overall={b['overall_auroc']:.4f}  within={b['within_bin_auroc']:.4f}  "
            f"matched={b['matched_pair_auroc']:.4f}  residual={b['residual_auroc']:.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, help="glob for shard npz files")
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--matched_delta", type=int, default=5)
    p.add_argument("--svar_lo", type=int, default=5)
    p.add_argument("--svar_hi", type=int, default=18)
    p.add_argument("--fixed_layer", type=int, default=16, help="a-priori mid layer for standalone ADS/CGC")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    files = sorted(glob.glob(args.shards))
    if not files:
        raise SystemExit(f"no shards match {args.shards}")
    parts = [np.load(f) for f in files]
    cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
    labels = cat("label").astype(int)
    pos = cat("gen_pos").astype(int)
    image_ids = cat("image_id").astype(int)
    image_mass = cat("image_mass").astype(np.float64)  # [N, L]
    ads = cat("ads").astype(np.float64)                 # [N, L]
    cgc = cat("cgc").astype(np.float64)                 # [N, L]
    N, L = ads.shape
    print(f"loaded {N} mentions ({labels.sum()} hallucinated), {L} layers, from {len(files)} shards\n")

    def bundle(values):
        return compute_metric_bundle(labels, np.asarray(values, float), pos,
                                     bin_width=args.bin_width, matched_delta=args.matched_delta)

    results = {}

    # --- SVAR ---
    svar_fixed = -image_mass[:, args.svar_lo:args.svar_hi + 1].sum(1)  # low SVAR => hallu
    results["svar_fixed_5_18"] = bundle(svar_fixed)
    results["svar_layer0"] = bundle(-image_mass[:, 0])
    print(f"SVAR (fixed, layers {args.svar_lo}-{args.svar_hi}): {_fmt(results['svar_fixed_5_18'])}")
    print(f"SVAR (layer-0, sanity)              : {_fmt(results['svar_layer0'])}\n")

    # --- ADS per-layer (higher ADS => hallu) ---
    ads_layers = {l: bundle(ads[:, l]) for l in range(L)}
    best_ads = max(ads_layers, key=lambda l: ads_layers[l]["within_bin_auroc"])
    results["ads_best_layer"] = {"layer": int(best_ads), **ads_layers[best_ads]}
    results["ads_fixed_layer"] = {"layer": int(args.fixed_layer), **ads_layers[args.fixed_layer]}
    print(f"ADS (best within-bin @layer {best_ads:2d})   : {_fmt(ads_layers[best_ads])}")
    print(f"ADS (fixed @layer {args.fixed_layer})            : {_fmt(ads_layers[args.fixed_layer])}\n")

    # --- CGC per-layer (higher cgc => grounded, so hallu = -cgc) ---
    cgc_layers = {l: bundle(-cgc[:, l]) for l in range(L)}
    best_cgc = max(cgc_layers, key=lambda l: cgc_layers[l]["within_bin_auroc"])
    results["cgc_best_layer"] = {"layer": int(best_cgc), **cgc_layers[best_cgc]}
    results["cgc_fixed_layer"] = {"layer": int(args.fixed_layer), **cgc_layers[args.fixed_layer]}
    print(f"CGC (best within-bin @layer {best_cgc:2d})   : {_fmt(cgc_layers[best_cgc])}")
    print(f"CGC (fixed @layer {args.fixed_layer})            : {_fmt(cgc_layers[args.fixed_layer])}\n")

    # --- ADS+CGC supervised (image-grouped OOF) over 64-dim per-layer features ---
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([ads, cgc], axis=1)  # [N, 2L]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    gkf = GroupKFold(n_splits=args.folds)
    for name, mk in [("hgb", lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=300, random_state=0)),
                     ("logreg", lambda: LogisticRegression(max_iter=2000, C=1.0))]:
        oof = np.zeros(N)
        for tr, te in gkf.split(X, labels, groups=image_ids):
            if name == "logreg":
                sc = StandardScaler().fit(X[tr])
                clf = mk().fit(sc.transform(X[tr]), labels[tr])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            else:
                clf = mk().fit(X[tr], labels[tr])
                oof[te] = clf.predict_proba(X[te])[:, 1]
        results[f"adscgc_supervised_{name}"] = bundle(oof)
        print(f"ADS+CGC supervised ({name:6s}, OOF) : {_fmt(results[f'adscgc_supervised_{name}'])}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n": int(N), "hallucinated": int(labels.sum()), "results": results},
                  open(args.out_json, "w"), indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
