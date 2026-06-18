#!/usr/bin/env python3
"""Evaluate LURE-style statistical factors on CHAIR object mentions.

This is an analysis baseline inspired by LURE's reported hallucination factors:
co-occurrence, decoding uncertainty, and object position. It does not implement
LURE's caption revisor. Instead, it asks whether these cheap factors explain
the same CHAIR hallucination labels as TDEV region evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import compute_metric_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LURE-style CHAIR detection factors")
    p.add_argument(
        "--object_cache",
        default="detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl",
    )
    p.add_argument(
        "--baseline_scores",
        default="detection/baselines/results/coco_llava_7b_baselines/baseline_scores.npz",
    )
    p.add_argument(
        "--neighbors_json",
        default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json",
    )
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/lure_style_detection",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_neighbor_scores(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        target: {item["object"]: float(item.get("jaccard", 0.0)) for item in items}
        for target, items in raw.items()
    }


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    out = (values - mean) / std
    out[~np.isfinite(out)] = 0.0
    return out


def build_cooccurrence_support(records: list[dict], neighbor_scores: dict[str, dict[str, float]]) -> np.ndarray:
    words_by_image: dict[int, list[str]] = defaultdict(list)
    for record in records:
        words_by_image[int(record["image_id"])].append(str(record["word"]))

    values = np.zeros(len(records), dtype=np.float64)
    for idx, record in enumerate(records):
        target = str(record["word"])
        image_words = words_by_image[int(record["image_id"])]
        target_neighbors = neighbor_scores.get(target, {})
        supports = [
            target_neighbors.get(other, 0.0)
            for other in image_words
            if other != target
        ]
        values[idx] = max(supports) if supports else 0.0
    return values


def metric_row(name: str, values: np.ndarray, labels: np.ndarray, positions: np.ndarray) -> dict:
    metrics = compute_metric_bundle(labels, values, positions)
    return {
        "score": name,
        "overall_auroc": metrics["overall_auroc"],
        "within_bin_auroc": metrics["within_bin_auroc"],
        "matched_pair_auroc": metrics["matched_pair_auroc"],
        "matched_pair_count": metrics["matched_pair_count"],
        "residual_auroc": metrics["residual_auroc"],
        "finite_count": metrics["finite_count"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(Path(args.object_cache))
    score_npz = np.load(args.baseline_scores)
    labels = score_npz["labels"].astype(np.int32)
    positions = score_npz["gen_pos"].astype(np.int32)
    if len(records) != labels.shape[0]:
        raise ValueError(f"Object cache rows {len(records)} != score rows {labels.shape[0]}")

    neighbor_scores = load_neighbor_scores(Path(args.neighbors_json))
    cooccur = build_cooccurrence_support(records, neighbor_scores)
    nll = score_npz["nll_hallu_score"].astype(np.float64)
    entropy = score_npz["entropy_hallu_score"].astype(np.float64)
    position = positions.astype(np.float64)

    score_arrays = {
        "lure_position": position,
        "lure_uncertainty_nll": nll,
        "lure_uncertainty_entropy": entropy,
        "lure_cooccurrence_support": cooccur,
        "lure_uncertainty_mean": 0.5 * (zscore(nll) + zscore(entropy)),
        "lure_position_uncertainty": 0.5 * (zscore(position) + zscore(entropy)),
        "lure_position_cooccurrence": 0.5 * (zscore(position) + zscore(cooccur)),
        "lure_cooccurrence_uncertainty": 0.5 * (zscore(cooccur) + zscore(entropy)),
        "lure_style_all_factors": (zscore(position) + zscore(entropy) + zscore(cooccur)) / 3.0,
        "lure_style_all_factors_nll": (zscore(position) + zscore(nll) + zscore(cooccur)) / 3.0,
    }

    metric_rows = [
        metric_row(name, values, labels, positions)
        for name, values in score_arrays.items()
    ]
    metric_rows.sort(key=lambda row: row["within_bin_auroc"] or 0.0, reverse=True)
    write_csv(output_dir / "lure_style_detection_metrics.csv", metric_rows)
    np.savez(output_dir / "lure_style_detection_scores.npz", labels=labels, gen_pos=positions, **score_arrays)

    payload = {
        "object_cache": args.object_cache,
        "baseline_scores": args.baseline_scores,
        "neighbors_json": args.neighbors_json,
        "samples": int(labels.shape[0]),
        "hallucinated": int(labels.sum()),
        "score_direction": "larger score means more likely hallucinated",
        "metrics": {row["score"]: row for row in metric_rows},
    }
    with (output_dir / "lure_style_detection_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'lure_style_detection_metrics.csv'}")


if __name__ == "__main__":
    main()
