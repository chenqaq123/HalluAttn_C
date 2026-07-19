#!/usr/bin/env python
"""Does a supervised attention-based detector learn a POSITION PRIOR? Causal attribution.

Proof design (per detector, fixed 50/50 image split, evaluated on TRULY-GROUNDED test
objects — which should score ~0 regardless of position):

  1. normal        : label-balanced sampler (natural position->label correlation present)
                     -> grounded score-drift = learned-prior + feature-intrinsic
  2. decorrelated  : sampler balances (position-bin x label) so position _|_ label in
                     training, all positions/features in-distribution
                     -> grounded score-drift = feature-intrinsic ONLY
  3. gen_pos-only  : logistic regression on generation position alone
                     -> the pure position prior (reference drift)

Conclusion "the classifier learned a large position prior" is supported iff
  normal drift  >>  decorrelated drift   AND   normal drift ~= gen_pos-only drift.
learned position prior magnitude = (normal - decorrelated) grounded drift.
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


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, w, dev, epochs=30, bs=1024, seed=0):
    torch.manual_seed(seed)
    model = MLP(X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w), num_samples=len(y), replacement=True)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            lossf(model(xb.float().to(dev)), yb.to(dev)).backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, X, dev, bs=4096):
    model.eval()
    out = np.empty(len(X), np.float64)
    for i in range(0, len(X), bs):
        out[i:i + bs] = torch.softmax(model(torch.from_numpy(X[i:i + bs]).float().to(dev)), -1)[:, 1].cpu().numpy()
    return out


def label_balanced_w(y):
    c = np.bincount(y, minlength=2).astype(np.float64)
    return (1.0 / np.clip(c, 1, None))[y]


def decorrelated_w(y, pos, binw):
    """Balance (position-bin x label) so position carries no label info in the stream."""
    b = (pos // binw).astype(int)
    key = b * 2 + y
    cnt = np.bincount(key)
    return 1.0 / np.clip(cnt[key], 1, None).astype(np.float64)


def grounded_drift(sc, y, pos, low_hi, high_lo, binw=15):
    """Mean predicted score among truly-grounded objects, by position; drift LOW->HIGH + slope."""
    g = y == 0
    lo = g & (pos < low_hi); hi = g & (pos >= high_lo)
    gl, gh = sc[lo].mean(), sc[hi].mean()
    slope = np.polyfit(pos[g], sc[g], 1)[0] * 100
    # curve (for figure): mean grounded score per position bin
    bins = np.arange(0, pos[g].max() + binw, binw)
    curve = [(int(b), float(sc[g & (pos >= b) & (pos < b + binw)].mean()) if (g & (pos >= b) & (pos < b + binw)).any() else float("nan"))
             for b in bins]
    return gl, gh, gh - gl, slope, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="experiments/dhcp_feats/shard*.npz")
    ap.add_argument("--features", choices=["dhcp", "var"], required=True)
    ap.add_argument("--low_hi", type=int, default=15)
    ap.add_argument("--high_lo", type=int, default=70)
    ap.add_argument("--binw", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    parts = [np.load(f) for f in sorted(glob.glob(args.shards))]
    cat = lambda k: np.concatenate([p[k] for p in parts], axis=0)
    y = cat("label").astype(int); pos = cat("gen_pos").astype(int); img = cat("image_id").astype(int)
    feat = cat("dhcp_feat"); N = len(y)
    if args.features == "dhcp":
        X = feat.reshape(N, -1)
    else:
        X = feat.astype(np.float32).sum(axis=3)[:, 5:19, :].reshape(N, -1).astype(np.float16)
    print(f"features={args.features} dim={X.shape[1]} N={N} hallu={y.sum()}")

    uimg = np.unique(img); is_tr = np.isin(img, uimg[uimg % 2 == 0]); te = ~is_tr
    tr = np.where(is_tr)[0]

    # attention-feature regimes
    results = {}
    for name, w in [("normal (label-bal)", label_balanced_w(y[tr])),
                    ("decorrelated (pos_|_label)", decorrelated_w(y[tr], pos[tr], args.binw))]:
        m = train_mlp(X[tr], y[tr], w, dev, epochs=args.epochs)
        sc = predict(m, X, dev)
        results[name] = grounded_drift(sc[te], y[te], pos[te], args.low_hi, args.high_lo, args.binw)

    # gen_pos-only logistic reference (the pure position prior)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(pos[tr].reshape(-1, 1), y[tr])
    scp = lr.predict_proba(pos.reshape(-1, 1))[:, 1]
    results["gen_pos-only (reference)"] = grounded_drift(scp[te], y[te], pos[te], args.low_hi, args.high_lo, args.binw)

    print(f"\n=== POSITION-PRIOR ATTRIBUTION — TRULY-GROUNDED test objects "
          f"(LOW pos<{args.low_hi}, HIGH pos>={args.high_lo}) ===")
    print(f"{'regime':<28} {'g-LOW':>7} {'g-HIGH':>7} {'drift':>7} {'slope/100':>10}")
    for k, (gl, gh, d, sl, _) in results.items():
        print(f"{k:<28} {gl:>7.4f} {gh:>7.4f} {d:>+7.4f} {sl:>+10.3f}")
    n = results["normal (label-bal)"][2]; dc = results["decorrelated (pos_|_label)"][2]
    ref = results["gen_pos-only (reference)"][2]
    print(f"\nLEARNED POSITION PRIOR (normal - decorrelated grounded drift) = {n - dc:+.4f}")
    print(f"  vs pure gen_pos-only prior drift = {ref:+.4f}  -> normal/reference ratio = {n/ref:.2f}")
    print("\ngrounded score-vs-position curves (bin_start: mean grounded score):")
    for k, (_, _, _, _, curve) in results.items():
        print(f"  {k}: " + " ".join(f"{b}:{v:.2f}" for b, v in curve if v == v))


if __name__ == "__main__":
    main()
