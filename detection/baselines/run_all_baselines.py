#!/usr/bin/env python3
"""Run controlled/adapted baselines on the current SinkDetect COCO setting."""

from __future__ import annotations

import argparse
import csv
import json
import os
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

from detection.baselines.data.build_object_cache import build_object_cache, write_jsonl
from detection.baselines.metrics import compute_metric_bundle, roc_auc, score_direction_note
from detection.baselines.methods.extract_model_signals import compute_model_baselines
from detection.baselines.our_method.collect_sinkdetect_scores import collect_sinkdetect_scores


def _read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _resolve_path(value: str | None, default: Path) -> str:
    if not value:
        return str(default)
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


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
    fields = [
        "object_id",
        "image_id",
        "word",
        "label",
        "token_pos",
        "gen_pos",
        *scores.keys(),
    ]
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


def parse_args() -> argparse.Namespace:
    env = _read_env(PROJECT_ROOT / ".env")
    p = argparse.ArgumentParser(description="Run controlled/adapted baselines on current COCO/LLaVA setting")
    p.add_argument("--generation_json", default=str(PROJECT_ROOT / "experiments/coco_llava_7b/generation.json"))
    p.add_argument("--row_cache", default=str(PROJECT_ROOT / "experiments/coco_llava_7b_rows/attention_row_cache.npz"))
    p.add_argument("--row_scores", default=str(PROJECT_ROOT / "experiments/coco_llava_7b_rows/row_cache_scores.npz"))
    p.add_argument("--output_dir", default=str(PROJECT_ROOT / "detection/baselines/results/coco_llava_7b"))
    p.add_argument("--model_path", default=env.get("MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--coco_path", default=_resolve_path(env.get("COCO_PATH"), Path("/data/common_dataset/coco-2014-dataset/")))
    p.add_argument("--chair_pkl", default=_resolve_path(env.get("CHAIR_PKL"), PROJECT_ROOT.parent / "pas/data/chair_coco.pkl"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="Limit number of object mentions; 0 means all.")
    p.add_argument("--skip_model_baselines", action="store_true", help="Only build cache and collect existing SinkDetect scores.")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_by", choices=["image", "object"], default="image")
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--matched_delta", type=int, default=5)
    p.add_argument("--glsim_top_k", type=int, default=32)
    p.add_argument("--glsim_w", type=float, default=0.6)
    p.add_argument("--text_layer", type=int, default=31)
    p.add_argument("--image_layer", type=int, default=32)
    p.add_argument("--beyond_layer", type=int, default=1)
    p.add_argument("--alignment_policy", choices=["error", "warn", "skip"], default="error")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_object_cache(
        generation_json=args.generation_json,
        row_cache=args.row_cache,
        chair_pkl=args.chair_pkl,
        model_path=args.model_path,
        limit=args.limit,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        shard_by=args.shard_by,
    )
    write_jsonl(records, output_dir / "object_cache.jsonl")
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int32)
    gen_pos = np.asarray([int(r["gen_pos"]) for r in records], dtype=np.int32)

    scores: dict[str, np.ndarray] = {}
    if not args.skip_model_baselines:
        scores.update(
            compute_model_baselines(
                records=records,
                generation_json=args.generation_json,
                model_path=args.model_path,
                coco_path=args.coco_path,
                device_index=args.device,
                glsim_top_k=args.glsim_top_k,
                glsim_w=args.glsim_w,
                text_layer=args.text_layer,
                image_layer=args.image_layer,
                beyond_layer=args.beyond_layer,
                alignment_policy=args.alignment_policy,
            )
        )

    object_ids = [int(r["object_id"]) for r in records]
    scores.update(collect_sinkdetect_scores(args.row_scores, num_objects=len(records), object_ids=object_ids))

    _write_scores_npz(scores, labels, gen_pos, output_dir / "baseline_scores.npz")
    _write_scores_csv(records, scores, output_dir / "baseline_scores.csv")
    metrics = _compute_all_metrics(scores, labels, gen_pos, args.bin_width, args.matched_delta)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_config = {
        "git_commit": _git_commit(),
        "generation_json": str(args.generation_json),
        "row_cache": str(args.row_cache),
        "row_scores": str(args.row_scores),
        "model_path": str(args.model_path),
        "coco_path": str(args.coco_path),
        "chair_pkl": str(args.chair_pkl),
        "limit": args.limit,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "shard_by": args.shard_by,
        "bin_width": args.bin_width,
        "matched_delta": args.matched_delta,
        "alignment_policy": args.alignment_policy,
        "glsim": {
            "source": "adapted_from_official_deeplearning_wisc_glsim",
            "note": "Ported to this project's teacher-forced object-level cache; not a bit-level official rerun.",
            "top_k": args.glsim_top_k,
            "w": args.glsim_w,
            "text_layer": args.text_layer,
            "image_layer": args.image_layer,
        },
        "beyond_global_scores": {
            "source": "paper_reimplementation",
            "note": "No official GitHub repository was found during implementation; ADS+CGC is a deterministic paper-level reimplementation.",
            "ads_attention_layer": args.beyond_layer,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'object_cache.jsonl'}")
    print(f"Wrote {output_dir / 'baseline_scores.npz'}")
    print(f"Wrote {output_dir / 'baseline_scores.csv'}")
    print(f"Wrote {output_dir / 'metrics.json'}")
    print(f"Wrote {output_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
