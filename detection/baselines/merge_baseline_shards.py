#!/usr/bin/env python3
"""Merge per-shard baseline outputs into one object-level result directory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_ROOT = PROJECT_ROOT / "detection"
for path in (PROJECT_ROOT, DETECTION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detection.baselines.data.build_object_cache import read_jsonl, write_jsonl
from detection.baselines.metrics import compute_metric_bundle, roc_auc, score_direction_note


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_scores_npz(scores: dict[str, np.ndarray], labels: np.ndarray, gen_pos: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, labels=labels.astype(np.int32), gen_pos=gen_pos.astype(np.int32), **scores)


def _write_scores_csv(records: list[dict[str, Any]], scores: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["object_id", "image_id", "word", "label", "token_pos", "gen_pos", *scores.keys()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_idx, record in enumerate(records):
            row = {key: record.get(key, "") for key in fields[:6]}
            for key, values in scores.items():
                value = values[row_idx] if row_idx < len(values) else np.nan
                row[key] = "" if not np.isfinite(value) else float(value)
            writer.writerow(row)


def _compute_all_metrics(
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    gen_pos: np.ndarray,
    bin_width: int,
    matched_delta: int,
) -> dict:
    metrics = {
        "objects": {
            "total": int(labels.size),
            "hallucinated": int(labels.sum()),
            "non_hallucinated": int(labels.size - labels.sum()),
        },
        "position_only": {
            "overall_auroc": roc_auc(labels, gen_pos.astype(np.float64)),
        },
        "scores": {},
        "score_direction": score_direction_note(scores.keys()),
    }
    for key, values in scores.items():
        metrics["scores"][key] = compute_metric_bundle(
            labels,
            values,
            gen_pos,
            bin_width=bin_width,
            matched_delta=matched_delta,
        )
    return metrics


def _load_shard(shard_dir: Path) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    object_cache = shard_dir / "object_cache.jsonl"
    score_npz = shard_dir / "baseline_scores.npz"
    run_config = shard_dir / "run_config.json"
    if not object_cache.exists():
        raise FileNotFoundError(f"Missing shard object cache: {object_cache}")
    if not score_npz.exists():
        raise FileNotFoundError(f"Missing shard score file: {score_npz}")

    records = read_jsonl(object_cache)
    data = np.load(score_npz)
    scores = {key: data[key].astype(np.float32) for key in data.files if key not in {"labels", "gen_pos"}}
    config = json.loads(run_config.read_text(encoding="utf-8")) if run_config.exists() else {}
    return records, scores, config


def _merge_scores(
    shard_payloads: list[tuple[Path, list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    row_items: list[tuple[int, dict[str, Any]]] = []
    score_items: dict[str, list[tuple[int, float]]] = {}
    for _shard_dir, records, scores, _config in shard_payloads:
        object_ids = [int(record["object_id"]) for record in records]
        row_items.extend((object_id, record) for object_id, record in zip(object_ids, records))
        for key, values in scores.items():
            score_items.setdefault(key, [])
            for object_id, value in zip(object_ids, values):
                score_items[key].append((object_id, float(value)))

    seen: set[int] = set()
    duplicate_ids: list[int] = []
    for object_id, _record in row_items:
        if object_id in seen:
            duplicate_ids.append(object_id)
        seen.add(object_id)
    if duplicate_ids:
        raise ValueError(f"Duplicate object_id values across shards, first few: {duplicate_ids[:10]}")

    sorted_items = sorted(row_items, key=lambda x: x[0])
    object_ids_sorted = [object_id for object_id, _record in sorted_items]
    id_to_row = {object_id: i for i, object_id in enumerate(object_ids_sorted)}
    records_sorted = [record for _object_id, record in sorted_items]

    merged_scores: dict[str, np.ndarray] = {}
    for key, pairs in score_items.items():
        values = np.full(len(records_sorted), np.nan, dtype=np.float32)
        for object_id, value in pairs:
            if object_id in id_to_row:
                values[id_to_row[object_id]] = value
        merged_scores[key] = values
    return records_sorted, merged_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge baseline shard result directories")
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_parallel"))
    p.add_argument("--num_shards", type=int, default=4)
    p.add_argument("--shard_prefix", default="shard")
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--matched_delta", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    shard_payloads = []
    for shard_idx in range(args.num_shards):
        shard_dir = output_dir / f"{args.shard_prefix}{shard_idx}"
        records, scores, config = _load_shard(shard_dir)
        shard_payloads.append((shard_dir, records, scores, config))

    records, scores = _merge_scores(shard_payloads)
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int32)
    gen_pos = np.asarray([int(r["gen_pos"]) for r in records], dtype=np.int32)

    write_jsonl(records, output_dir / "object_cache.jsonl")
    _write_scores_npz(scores, labels, gen_pos, output_dir / "baseline_scores.npz")
    _write_scores_csv(records, scores, output_dir / "baseline_scores.csv")
    metrics = _compute_all_metrics(scores, labels, gen_pos, args.bin_width, args.matched_delta)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_config = {
        "git_commit": _git_commit(),
        "merged": True,
        "num_shards": args.num_shards,
        "shards": [
            {
                "path": str(shard_dir),
                "objects": len(records_i),
                "run_config": config,
            }
            for shard_dir, records_i, _scores_i, config in shard_payloads
        ],
        "bin_width": args.bin_width,
        "matched_delta": args.matched_delta,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Merged {len(records)} object mentions from {args.num_shards} shards")
    print(f"Wrote {output_dir / 'object_cache.jsonl'}")
    print(f"Wrote {output_dir / 'baseline_scores.npz'}")
    print(f"Wrote {output_dir / 'baseline_scores.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
