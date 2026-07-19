#!/usr/bin/env python
"""Position-dependence experiment for PUBLISHED attention-based supervised detectors.

Reproduces two accepted detectors from the saved DHCP attention tensor
(dhcp_feat: [N, 32, 32, 576] = per-(layer,head) attention over 576 visual tokens):

  - DHCP  (ACM MM 2025, btzyd/DHCP): MLP over the full [32,32,576] tensor (589,824-d).
  - VAR   (Devils in Middle Layers, CVPR 2025): MLP over per-(layer,head) Visual
          Attention Ratio = sum over the 576 visual tokens, layers 5-18 -> [14,32]=448-d.

Classifier faithful to DHCP: Linear(D->128)->ReLU->Linear(128->2), CrossEntropy,
Adam 1e-3, class-balanced sampling, batch 1024.

Two setups per detector:
  A. position-agnostic image-grouped OOF (normal reproduction).
  B. train on a narrow MID gen_pos band (train-image half), test LOW & HIGH tails
     (held-out test-image half). MID/LOW/HIGH are gapped.

Decisive readout: among TRULY-GROUNDED objects, does the predicted hallucination
score rise from LOW to HIGH position? (position-invariant grounding => flat.)
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from detection.baselines.metrics import roc_auc, within_bin_auc  # noqa: E402


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, dev, epochs=30, bs=1024, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = MLP(d).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    # class-balanced sampler (DHCP WeightedRandomSampler)
    cls_cnt = np.bincount(ytr, minlength=2).astype(np.float64)
    w = (1.0 / np.clip(cls_cnt, 1, None))[ytr]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w), num_samples=len(ytr), replacement=True)
    Xt = torch.from_numpy(Xtr)  # fp16 on CPU
    yt = torch.from_numpy(ytr.astype(np.int64))
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt), batch_size=bs, sampler=sampler)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb.float().to(dev)), yb.to(dev))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, X, dev, bs=4096):
    model.eval()
    out = np.empty(len(X), np.float64)
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(dev)
        out[i:i + bs] = torch.softmax(model(xb), -1)[:, 1].cpu().numpy()
    return out


def _wb(y, s, p):
    return within_bin_auc(y, s, p, bin_width=10)["auc"]


def report(tag, sc, y, pos, low_m, high_m, extra=""):
    print(f"[{tag}] overall={roc_auc(y, sc):.4f} within={_wb(y, sc, pos):.4f}{extra}")
    for nm, m in [("LOW", low_m), ("HIGH", high_m)]:
        if m.sum() > 10:
            print(f"    on {nm:>4}: AUROC={roc_auc(y[m], sc[m]):.4f} within={_wb(y[m], sc[m], pos[m]):.4f}"
                  f"  grounded-score={sc[m & (y == 0)].mean():.4f}")
    gl = sc[low_m & (y == 0)].mean(); gh = sc[high_m & (y == 0)].mean()
    u = (low_m | high_m) & (y == 0)
    slope = np.polyfit(pos[u], sc[u], 1)[0] * 100
    print(f"    DECISIVE grounded drift LOW={gl:.4f} -> HIGH={gh:.4f} (+{gh-gl:.4f}, ~{gh/max(gl,1e-6):.1f}x)"
          f"  slope={slope:+.3f}/100tok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="experiments/dhcp_feats/shard*.npz")
    ap.add_argument("--features", choices=["dhcp", "var"], required=True)
    ap.add_argument("--mid_lo", type=int, default=30)
    ap.add_argument("--mid_hi", type=int, default=42)
    ap.add_argument("--low_hi", type=int, default=15)
    ap.add_argument("--high_lo", type=int, default=70)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    parts = [np.load(f) for f in sorted(glob.glob(args.shards))]
    cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
    y = cat("label").astype(int); pos = cat("gen_pos").astype(int); img = cat("image_id").astype(int)
    feat = cat("dhcp_feat")  # [N,32,32,576] fp16
    N = len(y)
    if args.features == "dhcp":
        X = feat.reshape(N, -1)                                   # [N,589824] fp16
    else:  # VAR: sum over 576 visual tokens -> per (layer,head), layers 5-18
        X = feat.astype(np.float32).sum(axis=3)[:, 5:19, :].reshape(N, -1).astype(np.float16)  # [N,448]
    print(f"features={args.features} dim={X.shape[1]} N={N} hallu={y.sum()}")
    low_m, high_m = pos < args.low_hi, pos >= args.high_lo
    for nm, m in [("LOW", low_m), ("MID", (pos >= args.mid_lo) & (pos < args.mid_hi)), ("HIGH", high_m)]:
        print(f"  {nm}: n={int(m.sum())} true-rate={y[m].mean():.3f}")

    # ---- Same 50/50 image split; vary ONLY the training positions, test on fixed LOW/HIGH ----
    uimg = np.unique(img)
    is_tr = np.isin(img, uimg[uimg % 2 == 0])
    te = ~is_tr
    all_idx = np.where(is_tr)[0]                             # regime A: all positions of train images
    mid_idx = np.where(is_tr & (pos >= args.mid_lo) & (pos < args.mid_hi))[0]  # regime B: MID band only
    # size-matched control: all-positions subsampled to the SAME count as MID-only
    rng = np.random.default_rng(0)
    allmatch_idx = rng.choice(all_idx, size=len(mid_idx), replace=False)
    print(f"\ntrain images: all-position n={len(all_idx)} | all-pos size-matched n={len(allmatch_idx)} | MID-only n={len(mid_idx)}")

    regimes = []
    for name, idx in [("all-positions", all_idx),
                      (f"all-pos (n={len(mid_idx)})", allmatch_idx),
                      (f"MID[{args.mid_lo},{args.mid_hi})", mid_idx)]:
        model = train_mlp(X[idx], y[idx], dev, epochs=args.epochs)
        regimes.append((name, predict(model, X, dev)))

    print("\n=== TRAINING-REGIME COMPARISON (same test images; vary only TRAINING positions) ===")
    print("decision = P(hallu) >= 0.5 (balanced-sampler MLP). yes-rate is the predicted hallucination rate.\n")
    for rnm, rng in [("LOW", low_m), ("HIGH", high_m)]:
        m = rng & te
        print(f"--- test region {rnm}  (n={int(m.sum())}, TRUE hallu-rate={y[m].mean():.3f}) ---")
        print(f"    {'trained on':<16} {'AUROC':>7} {'pred-yes':>9} {'mean-score':>11} {'score:grnd':>11} {'score:hallu':>12}")
        for name, sc in regimes:
            s = sc[m]
            print(f"    {name:<16} {roc_auc(y[m], s):>7.4f} {(s>=0.5).mean():>9.3f} {s.mean():>11.4f} "
                  f"{s[y[m]==0].mean():>11.4f} {s[y[m]==1].mean():>12.4f}")
        print()


if __name__ == "__main__":
    main()
