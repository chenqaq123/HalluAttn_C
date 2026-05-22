#!/usr/bin/env python3
"""Analyze how token-level detection changes with object mention position."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
except ModuleNotFoundError:
    _sklearn_roc_auc_score = None


def _roc_auc_score(labels: np.ndarray, values: np.ndarray) -> float:
    if _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(labels, values))
    labels = labels.astype(np.int32)
    values = values.astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative samples")
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


def _compute_auroc(labels: np.ndarray, values: np.ndarray) -> float | None:
    valid = np.isfinite(values)
    if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
        return None
    return _roc_auc_score(labels[valid], values[valid])


def _parse_metrics(spec: str | None, scores: np.lib.npyio.NpzFile, top_k: int, metrics_json: Path | None) -> list[str]:
    if spec:
        return [x.strip() for x in spec.split(",") if x.strip()]
    if metrics_json and metrics_json.exists():
        metrics = json.loads(metrics_json.read_text())
        return [k for k in metrics["roc_auc"].keys() if k in scores.files][:top_k]
    candidates = [k for k in scores.files if k != "labels"]
    labels = scores["labels"].astype(np.int32)
    ranked = []
    for key in candidates:
        auc = _compute_auroc(labels, scores[key])
        if auc is not None:
            ranked.append((key, auc))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in ranked[:top_k]]


def _make_bins(values: np.ndarray, bin_width: int | None, num_bins: int) -> list[tuple[int, int]]:
    lo = int(np.nanmin(values))
    hi = int(np.nanmax(values))
    if bin_width and bin_width > 0:
        starts = list(range((lo // bin_width) * bin_width, hi + 1, bin_width))
        return [(s, s + bin_width - 1) for s in starts]
    edges = np.quantile(values, np.linspace(0, 1, num_bins + 1))
    edges = np.unique(np.rint(edges).astype(int))
    if len(edges) < 2:
        return [(lo, hi)]
    bins = []
    for i in range(len(edges) - 1):
        start = int(edges[i])
        end = int(edges[i + 1])
        if i < len(edges) - 2:
            end -= 1
        bins.append((start, end))
    return bins


def _load_prompt_end_by_image(generation_json: Path) -> dict[int, int]:
    data = json.loads(generation_json.read_text())
    return {int(x["image_id"]): int(x["prompt_end_idx"]) for x in data}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="experiments/coco_llava_7b_rows/attention_row_cache.npz")
    p.add_argument("--scores", default="experiments/coco_llava_7b_rows/row_cache_scores.npz")
    p.add_argument("--generation_json", default="experiments/coco_llava_7b/generation.json")
    p.add_argument("--metrics_json", default="experiments/coco_llava_7b_rows/row_cache_metrics.json")
    p.add_argument("--output_csv", default="experiments/coco_llava_7b_rows/position_effect.csv")
    p.add_argument("--metrics", default=None, help="Comma-separated score keys. Defaults to top-k metrics.")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--bin_width", type=int, default=10, help="Generated-token position bin width. Use 0 for quantile bins.")
    p.add_argument("--num_bins", type=int, default=10, help="Used only when --bin_width 0.")
    args = p.parse_args()

    cache = np.load(args.cache)
    scores = np.load(args.scores)
    labels = cache["labels"].astype(np.int32)
    if not np.array_equal(labels, scores["labels"].astype(np.int32)):
        raise RuntimeError("cache labels and score labels do not match")

    prompt_end = _load_prompt_end_by_image(Path(args.generation_json))
    image_ids = cache["image_ids"].astype(np.int64)
    token_pos = cache["token_pos"].astype(np.int32)
    gen_pos = np.asarray(
        [int(pos) - prompt_end[int(image_id)] for pos, image_id in zip(token_pos, image_ids)],
        dtype=np.int32,
    )

    metrics = _parse_metrics(
        args.metrics,
        scores,
        top_k=args.top_k,
        metrics_json=Path(args.metrics_json) if args.metrics_json else None,
    )
    bins = _make_bins(gen_pos, args.bin_width, args.num_bins)

    rows = []
    for start, end in bins:
        mask = (gen_pos >= start) & (gen_pos <= end)
        if mask.sum() == 0:
            continue
        row = {
            "bin_start": start,
            "bin_end": end,
            "n": int(mask.sum()),
            "hallucinated": int(labels[mask].sum()),
            "hallucination_rate": float(labels[mask].mean()),
            "mean_gen_pos": float(gen_pos[mask].mean()),
        }
        for key in metrics:
            auc = _compute_auroc(labels[mask], scores[key][mask])
            row[key] = "" if auc is None else round(float(auc), 4)
        rows.append(row)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bin_start", "bin_end", "n", "hallucinated", "hallucination_rate", "mean_gen_pos", *metrics]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")
    print("metrics:")
    for key in metrics:
        print(f"  {key}")
    print()
    print("bin_start bin_end n hallu_rate " + " ".join(metrics[:4]))
    for row in rows:
        vals = [str(row.get(k, "")) for k in metrics[:4]]
        print(
            f"{row['bin_start']:>3} {row['bin_end']:>3} {row['n']:>5} "
            f"{row['hallucination_rate']:.3f} " + " ".join(vals)
        )


if __name__ == "__main__":
    main()
