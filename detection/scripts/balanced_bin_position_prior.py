#!/usr/bin/env python
"""Balanced-bin test for a learned POSITION PRIOR in supervised attention detectors.

Construct a test set where EVERY generation-position bin is balanced 50/50
(hallucinated vs grounded), so the TRUE hallucination rate is constant across bins.
Then measure each classifier's predicted hallucination rate per bin:

  - position-INVARIANT classifier -> flat predicted rate across bins.
  - classifier with a POSITION PRIOR -> predicted rate CLIMBS with position,
    despite the constant true rate. The climb IS the prior.

Three classifiers on the SAME balanced bins (all trained on the disjoint train-image half):
  normal        : label-balanced sampler (natural position->label correlation)
  decorrelated  : (position-bin x label)-balanced sampler (position _|_ label in training)
  gen_pos-only  : logistic on generation position (pure position prior reference)
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

BINS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 80), (80, 300)]


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, w, dev, epochs=30, bs=1024, seed=0):
    torch.manual_seed(seed)
    m = MLP(X.shape[1]).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w), num_samples=len(y), replacement=True)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))),
        batch_size=bs, sampler=sampler)
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); lf(m(xb.float().to(dev)), yb.to(dev)).backward(); opt.step()
    return m


@torch.no_grad()
def predict(m, X, dev, bs=4096):
    m.eval()
    out = np.empty(len(X), np.float64)
    for i in range(0, len(X), bs):
        out[i:i + bs] = torch.softmax(m(torch.from_numpy(X[i:i + bs]).float().to(dev)), -1)[:, 1].cpu().numpy()
    return out


def decorr_w(y, pos, binw=15):
    key = (pos // binw).astype(int) * 2 + y
    cnt = np.bincount(key)
    return 1.0 / np.clip(cnt[key], 1, None).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="")
    ap.add_argument("--features", choices=["dhcp", "var", "ads"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    shards = args.shards or ("experiments/beyond_svar_v3/shard*.npz" if args.features == "ads"
                             else "experiments/dhcp_feats/shard*.npz")
    parts = [np.load(f) for f in sorted(glob.glob(shards))]
    cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
    y = cat("label").astype(int); pos = cat("gen_pos").astype(int); img = cat("image_id").astype(int)
    N = len(y)
    if args.features == "ads":
        # Beyond ADS-alone supervised: classifier over the per-layer (32-d) ADS vector.
        X = cat("ads").astype(np.float32).reshape(N, -1)
    elif args.features == "dhcp":
        X = cat("dhcp_feat").reshape(N, -1)
    else:
        X = cat("dhcp_feat").astype(np.float32).sum(axis=3)[:, 5:19, :].reshape(N, -1).astype(np.float16)
    print(f"features={args.features} dim={X.shape[1]} N={N}")

    uimg = np.unique(img); is_tr = np.isin(img, uimg[uimg % 2 == 0]); te = np.where(~is_tr)[0]
    tr = np.where(is_tr)[0]

    # classifiers (trained on disjoint train images)
    scores = {}
    lb = (1.0 / np.clip(np.bincount(y[tr], minlength=2), 1, None).astype(np.float64))[y[tr]]
    scores["normal"] = predict(train_mlp(X[tr], y[tr], lb, dev, args.epochs), X, dev)
    scores["decorrelated"] = predict(train_mlp(X[tr], y[tr], decorr_w(y[tr], pos[tr]), dev, args.epochs), X, dev)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(pos[tr].reshape(-1, 1), y[tr])
    scores["gen_pos-only"] = lr.predict_proba(pos.reshape(-1, 1))[:, 1]

    # build balanced (50/50) test bins from test images
    rng = np.random.default_rng(0)
    print("\n=== predicted hallucination rate per BALANCED position bin (true rate = 0.500 in every bin) ===")
    print(f"{'bin':>10} {'n(=2k)':>7} | " + "  ".join(f"{k:>12}" for k in scores) + "   (yes-rate@0.5 | mean-score)")
    for lo, hi in BINS:
        m = te[(pos[te] >= lo) & (pos[te] < hi)]
        p_idx = m[y[m] == 1]; n_idx = m[y[m] == 0]
        k = min(len(p_idx), len(n_idx))
        if k < 20:
            print(f"{f'[{lo},{hi})':>10} {2*k:>7} | (too few positives, skipped)")
            continue
        bal = np.concatenate([rng.choice(p_idx, k, replace=False), rng.choice(n_idx, k, replace=False)])
        cells = []
        for key in scores:
            s = scores[key][bal]
            cells.append(f"{(s>=0.5).mean():.2f}|{s.mean():.2f}")
        print(f"{f'[{lo},{hi})':>10} {2*k:>7} | " + "  ".join(f"{c:>12}" for c in cells))
    print("\nRead: normal climbing while true rate stays 0.50 => learned position prior; "
          "decorrelated flat => the climb is the prior (not the features); gen_pos-only = pure prior.")


if __name__ == "__main__":
    main()
