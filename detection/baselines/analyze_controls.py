#!/usr/bin/env python3
"""Additional position/category/image controls for object-level detectors.

This script is intentionally post-hoc: it reads the unified baseline outputs
and does not reload the VLM. It is meant for reviewer-facing robustness checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_ROOT = PROJECT_ROOT / "detection"
for path in (PROJECT_ROOT, DETECTION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detection.baselines.data.build_object_cache import read_jsonl
from detection.baselines.metrics import compute_metric_bundle, residualize_by_position, roc_auc


def _safe_float(value: float | None) -> float | str:
    return "" if value is None or not np.isfinite(value) else float(value)


def _retained(controlled: float | None, overall: float | None) -> float | None:
    if controlled is None or overall is None:
        return None
    denom = overall - 0.5
    if abs(denom) < 1e-12:
        return None
    return float((controlled - 0.5) / denom)


def _poly_residual(values: np.ndarray, positions: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    residual = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(positions)
    if valid.sum() <= degree + 1:
        return residual
    centered = positions[valid] - positions[valid].mean()
    scale = centered.std()
    if scale > 0:
        centered = centered / scale
    design = np.vstack([centered ** power for power in range(degree + 1)]).T
    coef, *_ = np.linalg.lstsq(design, values[valid], rcond=None)
    residual[valid] = values[valid] - design @ coef
    return residual


def _quantile_residual(values: np.ndarray, positions: np.ndarray, num_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    residual = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(positions)
    if valid.sum() == 0:
        return residual
    quantiles = np.linspace(0, 1, num_bins + 1)
    edges = np.unique(np.quantile(positions[valid], quantiles))
    if edges.size <= 1:
        residual[valid] = values[valid] - values[valid].mean()
        return residual
    for idx in range(edges.size - 1):
        lo, hi = edges[idx], edges[idx + 1]
        if idx == edges.size - 2:
            mask = valid & (positions >= lo) & (positions <= hi)
        else:
            mask = valid & (positions >= lo) & (positions < hi)
        if mask.any():
            residual[mask] = values[mask] - float(values[mask].mean())
    return residual


def _matched_auc(
    labels: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    records: list[dict[str, Any]],
    delta: int,
    group_keys: Iterable[str] = (),
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int32)
    group_keys = tuple(group_keys)
    valid = np.isfinite(values)
    neg_by_group: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    pos_items: list[tuple[int, tuple[Any, ...]]] = []
    for idx, record in enumerate(records):
        group = tuple(record.get(key) for key in group_keys)
        if not valid[idx]:
            continue
        if labels[idx] == 1:
            pos_items.append((idx, group))
        else:
            neg_by_group[group].append(idx)

    wins = 0.0
    pairs = 0
    positives_with_match = 0
    for pos_idx, group in pos_items:
        candidates = np.asarray(neg_by_group.get(group, []), dtype=np.int64)
        if candidates.size == 0:
            continue
        candidates = candidates[np.abs(positions[candidates] - positions[pos_idx]) <= delta]
        if candidates.size == 0:
            continue
        diff = values[pos_idx] - values[candidates]
        wins += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        pairs += int(candidates.size)
        positives_with_match += 1
    return {
        "auc": (wins / pairs if pairs else None),
        "pairs": pairs,
        "positives_with_match": positives_with_match,
        "group_keys": list(group_keys),
    }


def _position_drift_rows(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    positions: np.ndarray,
    scores: dict[str, np.ndarray],
    bin_width: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if positions.size == 0:
        return rows
    start = int(math.floor(positions.min() / bin_width) * bin_width)
    end = int(math.ceil((positions.max() + 1) / bin_width) * bin_width)
    for lo in range(start, end, bin_width):
        hi = lo + bin_width - 1
        mask = (positions >= lo) & (positions <= hi)
        if not mask.any():
            continue
        row: dict[str, Any] = {
            "bin_start": lo,
            "bin_end": hi,
            "n": int(mask.sum()),
            "hallucinated": int(labels[mask].sum()),
            "hallucination_rate": float(labels[mask].mean()),
            "unique_images": len({records[i]["image_id"] for i in np.where(mask)[0]}),
            "unique_words": len({records[i]["word"] for i in np.where(mask)[0]}),
        }
        for name, values in scores.items():
            valid = mask & np.isfinite(values)
            row[f"{name}_mean"] = float(values[valid].mean()) if valid.any() else ""
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute stronger post-hoc controls for baseline detector scores")
    p.add_argument("--result_dir", default=str(PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_baselines"))
    p.add_argument("--output_dir", default="")
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--matched_delta", type=int, default=5)
    p.add_argument("--quantile_bins", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / "controlled_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(result_dir / "object_cache.jsonl")
    data = np.load(result_dir / "baseline_scores.npz")
    labels = data["labels"].astype(np.int32)
    positions = data["gen_pos"].astype(np.int32)
    scores = {
        key: data[key].astype(np.float64)
        for key in data.files
        if key not in {"labels", "gen_pos"}
    }
    if len(records) != labels.size:
        raise ValueError(f"object_cache rows ({len(records)}) != score rows ({labels.size})")

    metrics: dict[str, Any] = {
        "objects": {
            "total": int(labels.size),
            "hallucinated": int(labels.sum()),
            "non_hallucinated": int(labels.size - labels.sum()),
            "unique_images": len({row["image_id"] for row in records}),
            "unique_words": len({row["word"] for row in records}),
        },
        "settings": {
            "bin_width": args.bin_width,
            "matched_delta": args.matched_delta,
            "quantile_bins": args.quantile_bins,
        },
        "position_only_auroc": roc_auc(labels, positions.astype(np.float64)),
        "scores": {},
    }

    summary_rows = []
    for name, values in scores.items():
        base = compute_metric_bundle(labels, values, positions, args.bin_width, args.matched_delta)
        bin_residual = residualize_by_position(values, positions, bin_width=args.bin_width)
        linear_residual = _poly_residual(values, positions, degree=1)
        cubic_residual = _poly_residual(values, positions, degree=3)
        quantile_residual = _quantile_residual(values, positions, num_bins=args.quantile_bins)
        same_image = _matched_auc(labels, values, positions, records, args.matched_delta, ("image_id",))
        same_word = _matched_auc(labels, values, positions, records, args.matched_delta, ("word",))
        same_image_word = _matched_auc(labels, values, positions, records, args.matched_delta, ("image_id", "word"))
        score_metrics = {
            **base,
            "fixed_bin_residual_auroc": roc_auc(labels, bin_residual),
            "linear_residual_auroc": roc_auc(labels, linear_residual),
            "cubic_residual_auroc": roc_auc(labels, cubic_residual),
            "quantile_residual_auroc": roc_auc(labels, quantile_residual),
            "matched_same_image": same_image,
            "matched_same_word": same_word,
            "matched_same_image_word": same_image_word,
            "retained_within_vs_overall": _retained(base["within_bin_auroc"], base["overall_auroc"]),
            "retained_same_word_vs_overall": _retained(same_word["auc"], base["overall_auroc"]),
        }
        metrics["scores"][name] = score_metrics
        summary_rows.append({
            "score": name,
            "overall_auroc": _safe_float(base["overall_auroc"]),
            "within_bin_auroc": _safe_float(base["within_bin_auroc"]),
            "matched_pair_auroc": _safe_float(base["matched_pair_auroc"]),
            "fixed_bin_residual_auroc": _safe_float(score_metrics["fixed_bin_residual_auroc"]),
            "linear_residual_auroc": _safe_float(score_metrics["linear_residual_auroc"]),
            "cubic_residual_auroc": _safe_float(score_metrics["cubic_residual_auroc"]),
            "quantile_residual_auroc": _safe_float(score_metrics["quantile_residual_auroc"]),
            "same_image_matched_auroc": _safe_float(same_image["auc"]),
            "same_image_pairs": same_image["pairs"],
            "same_word_matched_auroc": _safe_float(same_word["auc"]),
            "same_word_pairs": same_word["pairs"],
            "same_image_word_matched_auroc": _safe_float(same_image_word["auc"]),
            "same_image_word_pairs": same_image_word["pairs"],
            "retained_within_vs_overall": _safe_float(score_metrics["retained_within_vs_overall"]),
            "retained_same_word_vs_overall": _safe_float(score_metrics["retained_same_word_vs_overall"]),
        })

    (output_dir / "controlled_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "controlled_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    drift_rows = _position_drift_rows(records, labels, positions, scores, args.bin_width)
    with (output_dir / "position_drift.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(drift_rows[0].keys()) if drift_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(drift_rows)

    print(f"Wrote {output_dir / 'controlled_metrics.json'}")
    print(f"Wrote {output_dir / 'controlled_summary.csv'}")
    print(f"Wrote {output_dir / 'position_drift.csv'}")


if __name__ == "__main__":
    main()
