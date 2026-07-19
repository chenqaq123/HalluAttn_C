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
    sae_R = cat("sae_R").astype(np.float64)             # [N, L] seg-free skeleton (dispersion)
    lascd_raw = cat("lascd_raw").astype(np.float64)     # [N, L] higher energy = more fragmented
    lascd_norm = cat("lascd_norm").astype(np.float64)   # [N, L] mass-normalized
    has_seg = all("sae_seg" in p.files for p in parts)
    sae_seg = cat("sae_seg").astype(np.float64) if has_seg else None  # [N, L] true SAE (Mask2Former)
    has_tver = all("tver" in p.files for p in parts)
    tver = cat("tver").astype(np.float64) if has_tver else None       # [N, L] H(text)/H(vis), higher=text-prior
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

    # --- ADS single-layer (DIAGNOSTIC ONLY; the paper's ADS is the supervised per-layer
    #     VECTOR below, never a single decoder layer). Higher ADS => hallu. ---
    ads_layers = {l: bundle(ads[:, l]) for l in range(L)}
    best_ads = max(ads_layers, key=lambda l: ads_layers[l]["within_bin_auroc"])
    results["ads_best_layer"] = {"layer": int(best_ads), **ads_layers[best_ads]}
    results["ads_fixed_layer"] = {"layer": int(args.fixed_layer), **ads_layers[args.fixed_layer]}
    print(f"ADS 1-layer diag (best @layer {best_ads:2d})  : {_fmt(ads_layers[best_ads])}")
    print(f"ADS 1-layer diag (fixed @layer {args.fixed_layer})   : {_fmt(ads_layers[args.fixed_layer])}\n")

    # --- CGC per-layer (higher cgc => grounded, so hallu = -cgc) ---
    cgc_layers = {l: bundle(-cgc[:, l]) for l in range(L)}
    best_cgc = max(cgc_layers, key=lambda l: cgc_layers[l]["within_bin_auroc"])
    results["cgc_best_layer"] = {"layer": int(best_cgc), **cgc_layers[best_cgc]}
    results["cgc_fixed_layer"] = {"layer": int(args.fixed_layer), **cgc_layers[args.fixed_layer]}
    print(f"CGC (best within-bin @layer {best_cgc:2d})   : {_fmt(cgc_layers[best_cgc])}")
    print(f"CGC (fixed @layer {args.fixed_layer})            : {_fmt(cgc_layers[args.fixed_layer])}\n")

    # --- SAE R-SKELETON, segmentation-free (raw-patch entropy) — NOT the paper's SAE ---
    # The paper's SAE takes entropy over Mask2Former segmentation regions; we use raw-576
    # patch entropy, so this measures attention DISPERSION, not semantic-space entropy.
    # Window = VAR middle layers 5-18 (Jiang et al.), the referenced middle-layer stage.
    mid = list(range(args.svar_lo, 19))
    results["sae_raw_midmean"] = bundle(-sae_R[:, mid].mean(1))
    sae_layers = {l: bundle(-sae_R[:, l]) for l in range(L)}
    best_sae = max(sae_layers, key=lambda l: sae_layers[l]["within_bin_auroc"])
    results["sae_raw_best_layer"] = {"layer": int(best_sae), **sae_layers[best_sae]}
    print(f"SAE-skeleton seg-free (mid-mean L{args.svar_lo}-18): {_fmt(results['sae_raw_midmean'])}")
    print(f"SAE-skeleton seg-free (best @layer {best_sae:2d})   : {_fmt(sae_layers[best_sae])}")

    # --- SAE (TRUE, arXiv:2603.16558): Mask2Former segmentation-region entropy. score = -R ---
    if sae_seg is not None:
        results["sae_seg_midmean"] = bundle(-sae_seg[:, mid].mean(1))
        seg_layers = {l: bundle(-sae_seg[:, l]) for l in range(L)}
        best_seg = max(seg_layers, key=lambda l: seg_layers[l]["within_bin_auroc"])
        results["sae_seg_best_layer"] = {"layer": int(best_seg), **seg_layers[best_seg]}
        print(f"SAE TRUE seg (mid-mean L{args.svar_lo}-18)      : {_fmt(results['sae_seg_midmean'])}")
        print(f"SAE TRUE seg (best @layer {best_seg:2d})         : {_fmt(seg_layers[best_seg])}")
    print()

    # --- LaSCD (arXiv:2605.11559): the energy is a WITHIN-token cross-LAYER selector (argmax=
    #     premature/argmin=GT, feeding contrastive decoding), never a cross-token detector — the
    #     repo/paper report no per-sample AUROC. The implemented RAW energy is mass-weighted (Eq. 3),
    #     so cross-token it degenerates into a visual-mass proxy that ANTI-correlates with hallu;
    #     Theorem 1 (fixed mass) is about the MASS-NORMALIZED shape energy. Fair test = normalized
    #     shape energy over LaSCD's own band L11-28, higher = hallucinated. ---
    Lb = list(range(11, 29))
    results["lascd_shape_meanband"] = bundle(lascd_norm[:, Lb].mean(1))
    results["lascd_shape_peakgap"] = bundle(lascd_norm[:, Lb].max(1) - lascd_norm[:, Lb].min(1))
    ln_layers = {l: bundle(lascd_norm[:, l]) for l in Lb}
    best_ln = max(ln_layers, key=lambda l: ln_layers[l]["within_bin_auroc"])
    results["lascd_shape_best_layer"] = {"layer": int(best_ln), **ln_layers[best_ln]}
    results["lascd_raw_massproxy"] = bundle(lascd_raw[:, Lb].mean(1))  # anti-correlated mass proxy
    print(f"LaSCD shape-energy (norm, mean L11-28) : {_fmt(results['lascd_shape_meanband'])}")
    print(f"LaSCD shape-energy (norm, peak-min gap): {_fmt(results['lascd_shape_peakgap'])}")
    print(f"LaSCD shape-energy (norm, best @L{best_ln:2d})   : {_fmt(ln_layers[best_ln])}")
    print(f"LaSCD raw energy (mass proxy, mean band): {_fmt(results['lascd_raw_massproxy'])}\n")

    # --- TVER (ONLY, arXiv:2507.00898): H(text attn)/H(visual attn). ONLY uses layer 0
    #     (enhance_layer_index=0). Higher TVER = text-prioritizing = hallucination-prone (their
    #     framing; sign checked empirically here). Report layer-0 (faithful) + best-layer sweep. ---
    if tver is not None:
        results["tver_layer0"] = bundle(tver[:, 0])
        tver_layers = {l: bundle(tver[:, l]) for l in range(L)}
        best_tver = max(tver_layers, key=lambda l: tver_layers[l]["within_bin_auroc"])
        results["tver_best_layer"] = {"layer": int(best_tver), **tver_layers[best_tver]}
        print(f"TVER (layer 0, faithful ONLY)     : {_fmt(results['tver_layer0'])}")
        print(f"TVER (best within-bin @layer {best_tver:2d})  : {_fmt(tver_layers[best_tver])}\n")

    # --- Supervised per-layer-VECTOR detectors (image-grouped OOF), the paper's actual
    #     ADS / CGC / ADS+CGC form. ADS-only is the faithful test of whether the paper's
    #     multi-layer ATTENTION-dispersion detector survives length control (Prong 1). ---
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=args.folds)

    def supervised(X, key, label):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
            results[f"{key}_{name}"] = bundle(oof)
            print(f"{label:24s} ({name:6s}, OOF): {_fmt(results[f'{key}_{name}'])}")

    supervised(ads, "ads_sup", "ADS-only sup (attention)")   # paper's ADS-alone; Prong-1 test
    supervised(cgc, "cgc_sup", "CGC-only sup (repr)")        # paper's CGC-alone
    supervised(np.concatenate([ads, cgc], 1), "adscgc_supervised", "ADS+CGC sup")
    print()

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n": int(N), "hallucinated": int(labels.sum()), "results": results},
                  open(args.out_json, "w"), indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
