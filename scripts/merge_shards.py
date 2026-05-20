#!/usr/bin/env python3
"""Merge per-shard outputs from a parallel SinkDetect run.

Modes:
  caption — concatenates generation_shard{i}.json files (one per shard) into
            a single generation.json, sorted by image_id.
  detect  — concatenates raw_scores_shard{i}.npz files (one per shard) and
            recomputes AUROC on the pooled data; writes metrics.json +
            raw_scores.npz.

Usage:
  python scripts/merge_shards.py --mode caption --output_dir <exp_dir> --num_shards 4
  python scripts/merge_shards.py --mode detect  --output_dir <exp_dir> --num_shards 4
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge_captions(output_dir: Path, num_shards: int) -> None:
    merged: list[dict] = []
    for i in range(num_shards):
        p = output_dir / f"generation_shard{i}.json"
        if not p.exists():
            logger.warning("Missing shard file: %s — skipping", p)
            continue
        with open(p) as f:
            merged.extend(json.load(f))
    merged.sort(key=lambda x: x["image_id"])
    out = output_dir / "generation.json"
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Merged %d captions → %s", len(merged), out)


def merge_detect(output_dir: Path, num_shards: int) -> None:
    shard_data = []
    chair_shards = []
    for i in range(num_shards):
        npz_p = output_dir / f"raw_scores_shard{i}.npz"
        if not npz_p.exists():
            logger.warning("Missing shard npz: %s — skipping", npz_p)
            continue
        shard_data.append(dict(np.load(npz_p)))
        m_p = output_dir / f"metrics_shard{i}.json"
        if m_p.exists():
            with open(m_p) as f:
                chair_shards.append(json.load(f))

    if not shard_data:
        raise RuntimeError("No shard npz files found in %s" % output_dir)

    # Take the intersection of keys (shards should agree, but be defensive)
    common_keys = set(shard_data[0].keys())
    for s in shard_data[1:]:
        common_keys &= set(s.keys())
    if "labels" not in common_keys:
        raise RuntimeError("labels missing in shard outputs")

    merged = {k: np.concatenate([s[k] for s in shard_data]) for k in common_keys}
    labels = merged.pop("labels").astype(np.int32)

    # Recompute AUROC on the pooled data
    roc_auc: dict[str, float] = {}
    for k, v in merged.items():
        v = np.asarray(v, dtype=np.float64)
        valid = np.isfinite(v)
        if valid.sum() == 0:
            continue
        y = labels[valid]
        if len(np.unique(y)) < 2:
            continue
        try:
            auc = roc_auc_score(y, v[valid])
            roc_auc[k] = round(float(auc), 4)
        except ValueError:
            continue
    roc_auc = dict(sorted(roc_auc.items(), key=lambda x: x[1], reverse=True))

    # Aggregate CHAIRi / CHAIRs from per-shard metrics if available
    chair_agg = None
    if chair_shards:
        n_total = sum(s["objects"]["total"] for s in chair_shards)
        n_hallu = sum(s["objects"]["hallucinated"] for s in chair_shards)
        chair_agg = {
            "CHAIRi": (n_hallu / n_total) if n_total else 0.0,
            "shards": [s.get("chair_metrics", {}) for s in chair_shards],
        }

    n_total = int(len(labels))
    n_hallu = int(labels.sum())
    metrics = {
        "objects": {
            "total": n_total,
            "hallucinated": n_hallu,
            "non_hallucinated": n_total - n_hallu,
        },
        "chair_metrics": chair_agg,
        "roc_auc": roc_auc,
        "num_shards": num_shards,
    }

    # Save
    np.savez(
        output_dir / "raw_scores.npz",
        labels=labels,
        **{k: v.astype(np.float32) for k, v in merged.items()},
    )
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Merged %d shards → %s (%d objects, %d hallu)",
        num_shards, output_dir / "metrics.json", n_total, n_hallu,
    )
    if roc_auc:
        best_k = next(iter(roc_auc))
        logger.info("Best AUROC: %.4f (%s)", roc_auc[best_k], best_k)
        # Show top 10
        for i, (k, v) in enumerate(list(roc_auc.items())[:10]):
            logger.info("  %2d. %s: %.4f", i + 1, k, v)

    merge_shape_cache_if_present(output_dir, num_shards)


def merge_shape_cache_if_present(output_dir: Path, num_shards: int) -> None:
    shard_data = []
    for i in range(num_shards):
        p = output_dir / f"shape_cache_shard{i}.npz"
        if p.exists():
            shard_data.append(dict(np.load(p)))

    if not shard_data:
        return

    common_keys = set(shard_data[0].keys())
    for s in shard_data[1:]:
        common_keys &= set(s.keys())
    merged = {k: np.concatenate([s[k] for s in shard_data]) for k in common_keys}
    out = output_dir / "shape_cache.npz"
    np.savez_compressed(out, **merged)
    logger.info("Merged %d shape-cache shards → %s", len(shard_data), out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["caption", "detect"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_shards", type=int, required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    if args.mode == "caption":
        merge_captions(output_dir, args.num_shards)
    else:
        merge_detect(output_dir, args.num_shards)


if __name__ == "__main__":
    main()
