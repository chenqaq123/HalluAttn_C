#!/usr/bin/env python
"""Position-dependence experiment for supervised attention-based hallucination detectors.

Two setups on the same feature matrix, to show a supervised classifier leans on
generation-position even when its features are attention statistics:

  A. Position-agnostic (normal reproduction): image-grouped OOF over ALL positions.
  B. Position-constrained: train ONLY on a narrow MID gen_pos band (train-image half),
     test on LOW and HIGH bands (held-out test-image half). MID/LOW/HIGH are gapped.

Decisive readout in both: among TRULY-GROUNDED objects, does the predicted
hallucination score rise with gen_pos? A position-invariant grounding detector
keeps grounded objects near 0 regardless of position; a position-dependent one
flags late grounded objects. Also reports the fixed-threshold yes-rate transfer.

Feature matrix is built by concatenating per-mention npz keys (e.g. --features ads,cgc).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from detection.baselines.metrics import roc_auc, within_bin_auc  # noqa: E402


def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=300, random_state=0)


def _wb(y, s, p, bw=10):
    return within_bin_auc(y, s, p, bin_width=bw)["auc"]


def _grounded_drift(score, y, pos, low_m, high_m):
    """Mean predicted score among truly-grounded (and hallu) objects, LOW vs HIGH + slope."""
    g = y == 0
    gl = score[g & low_m].mean() if (g & low_m).any() else float("nan")
    gh = score[g & high_m].mean() if (g & high_m).any() else float("nan")
    h = y == 1
    hl = score[h & low_m].mean() if (h & low_m).any() else float("nan")
    hh = score[h & high_m].mean() if (h & high_m).any() else float("nan")
    # slope of score vs gen_pos within grounded, over the union of low+high
    u = (low_m | high_m) & g
    slope = np.polyfit(pos[u], score[u], 1)[0] * 100 if u.sum() > 2 else float("nan")
    return gl, gh, hl, hh, slope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--features", required=True, help="comma list of npz keys, e.g. ads,cgc")
    ap.add_argument("--mid_lo", type=int, default=30, help="MID train band gen_pos >= mid_lo")
    ap.add_argument("--mid_hi", type=int, default=42, help="MID train band gen_pos < mid_hi")
    ap.add_argument("--low_hi", type=int, default=15, help="LOW test band gen_pos < low_hi")
    ap.add_argument("--high_lo", type=int, default=70, help="HIGH test band gen_pos >= high_lo")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(args.shards))
    parts = [np.load(f) for f in files]
    cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
    keys = [k.strip() for k in args.features.split(",")]
    X = np.concatenate([cat(k).astype(np.float64).reshape(len(cat("label")), -1) for k in keys], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = cat("label").astype(int)
    pos = cat("gen_pos").astype(int)
    img = cat("image_id").astype(int)
    low_m, high_m = pos < args.low_hi, pos >= args.high_lo
    print(f"features={keys} dim={X.shape[1]}  N={len(y)} hallu={y.sum()}")
    print(f"bands: LOW gen_pos<{args.low_hi} | MID[{args.mid_lo},{args.mid_hi}) | HIGH>={args.high_lo}")
    for nm, m in [("LOW", low_m), ("MID", (pos >= args.mid_lo) & (pos < args.mid_hi)), ("HIGH", high_m)]:
        print(f"  {nm}: n={int(m.sum())} true-rate={y[m].mean():.3f} mean-pos={pos[m].mean():.1f}")

    # ---- Setup A: position-agnostic image-grouped OOF over ALL positions ----
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=args.folds).split(X, y, groups=img):
        oof[te] = _hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    gl, gh, hl, hh, slope = _grounded_drift(oof, y, pos, low_m, high_m)
    print("\n=== SETUP A: position-agnostic OOF (normal reproduction) ===")
    print(f"  overall AUROC={roc_auc(y, oof):.4f}  within-bin={_wb(y, oof, pos):.4f}")
    for nm, m in [("LOW", low_m), ("HIGH", high_m)]:
        print(f"  on {nm:>4} band: AUROC={roc_auc(y[m], oof[m]):.4f}  within-bin={_wb(y[m], oof[m], pos[m]):.4f}")
    print(f"  grounded mean-score  LOW={gl:.4f}  HIGH={gh:.4f}  drift=+{gh-gl:.4f}")
    print(f"  hallu    mean-score  LOW={hl:.4f}  HIGH={hh:.4f}  drift=+{hh-hl:.4f}")
    print(f"  slope(score~gen_pos | grounded) = {slope:+.4f} / 100 tok")

    # ---- Setup B: train on narrow MID (train images), test LOW/HIGH (test images) ----
    uimg = np.unique(img)
    is_tr = np.isin(img, uimg[uimg % 2 == 0])
    trm = (pos >= args.mid_lo) & (pos < args.mid_hi) & is_tr
    clf = _hgb().fit(X[trm], y[trm])
    thr = np.quantile(clf.predict_proba(X[trm])[:, 1], 1 - y[trm].mean())
    print(f"\n=== SETUP B: train on MID/train-images (n={int(trm.sum())}), thr={thr:.3f} ===")
    print(f"  {'range':>5} {'n':>5} {'trueR':>6} {'predYes':>7} {'AUROC':>6} {'within':>6} {'g-score':>8} {'h-score':>8}")
    sc_all = clf.predict_proba(X)[:, 1]
    for nm, rng in [("LOW", low_m), ("HIGH", high_m)]:
        m = rng & (~is_tr)
        sc, yy, pp = sc_all[m], y[m], pos[m]
        print(f"  {nm:>5} {int(m.sum()):>5} {yy.mean():>6.3f} {(sc>=thr).mean():>7.3f} "
              f"{roc_auc(yy, sc):>6.3f} {_wb(yy, sc, pp):>6.3f} {sc[yy==0].mean():>8.4f} {sc[yy==1].mean():>8.4f}")
    te = ~is_tr
    gl, gh, hl, hh, slope = _grounded_drift(sc_all, y, pos, low_m & te, high_m & te)
    print(f"  DECISIVE — grounded mean-score LOW={gl:.4f} HIGH={gh:.4f} drift=+{gh-gl:.4f}"
          f"  (~{gh/gl:.1f}x)" if gl > 0 else "")
    print(f"  slope(score~gen_pos | grounded, test) = {slope:+.4f} / 100 tok")


if __name__ == "__main__":
    main()
