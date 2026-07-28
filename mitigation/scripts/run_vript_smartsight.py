#!/usr/bin/env python3
"""Run the registered SmartSight disclosed-core reimplementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.open_caption_models import (
    OpenCaptionModel,
    realized_vision_shapes,
)
from mitigation.src.smartsight import (
    TemporalAttentionCapture,
    frame_groups,
    motion_segments,
    score_trace,
    video_token_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["llava_next_video", "qwen25", "qwen3"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--vav_alpha", type=float, default=1.2)
    parser.add_argument("--vav_window", type=int, default=10)
    return parser.parse_args()


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [
            json.loads(line) for line in handle if line.strip()
        ]


def _completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open() as handle:
        return {
            str(json.loads(line)["clip_id"])
            for line in handle
            if line.strip()
        }


def _candidate_seed(base: int, clip_id: str, candidate: int) -> int:
    digest = hashlib.sha256(
        f"{base}:{clip_id}:{candidate}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little")


def _atomic_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("Invalid shard")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    records = [
        item
        for index, item in enumerate(_jsonl(args.manifest))
        if index % args.num_shards == args.shard_idx
    ]
    if args.limit is not None:
        records = records[: args.limit]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [
        item for item in records if str(item["clip_id"]) not in completed
    ]

    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    metadata = {
        "model": captioner.metadata(),
        "method": "smartsight-disclosed-core",
        "manifest": str(Path(args.manifest).resolve()),
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "registered_samples_in_shard": len(records),
        "seed": args.seed,
        "candidates": args.candidates,
        "temperature": args.temperature,
        "vav_alpha": args.vav_alpha,
        "vav_window": args.vav_window,
        "segment_threshold_completion": "within-clip median",
        "efficiency_note": (
            "All candidates are completed, then TAC is truncated at each VAV "
            "point. This is selection-equivalent but does not claim the "
            "paper's unpublished parallel early-termination implementation."
        ),
    }

    with output_path.open("a") as output_handle:
        for record in tqdm(pending, desc=f"smartsight-{args.shard_idx}"):
            started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(captioner.device)
            inputs, indices = captioner.prepare_video(
                record["video_path"],
                record["prompt"],
                system=record.get("system"),
            )
            if indices != record["frame_indices"]:
                raise ValueError(
                    f"Frame-index mismatch for {record['clip_id']}"
                )
            prompt_length = int(inputs["input_ids"].shape[-1])
            positions = captioner.visual_positions(
                inputs, media_type="video"
            )
            groups = frame_groups(
                captioner.kind, inputs, positions, frames=32
            )
            features = video_token_features(
                captioner, inputs, frames=32
            )
            segments, motion_distances, segment_threshold = (
                motion_segments(features)
            )
            del features

            candidates = []
            for candidate_index in range(args.candidates):
                seed = _candidate_seed(
                    args.seed, record["clip_id"], candidate_index
                )
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with TemporalAttentionCapture(
                    captioner.decoder.modeling_module,
                    captioner.decoder.attention_classes,
                    groups,
                    prompt_length,
                ) as capture:
                    full_ids = captioner.generate(
                        inputs,
                        do_sample=True,
                        temperature=args.temperature,
                    )
                trace = capture.trace()
                score = score_trace(
                    trace,
                    segments,
                    alpha=args.vav_alpha,
                    window=args.vav_window,
                )
                caption = captioner.decode_continuation(
                    full_ids, prompt_length
                )
                candidates.append(
                    {
                        "index": candidate_index,
                        "seed": seed,
                        "caption": caption,
                        "generated_tokens": int(
                            full_ids.shape[-1] - prompt_length
                        ),
                        **score,
                    }
                )
                del full_ids, trace

            selected_index = int(
                np.argmax([item["tac"] for item in candidates])
            )
            selected = candidates[selected_index]
            result = {
                "clip_id": record["clip_id"],
                "benchmark": "vript-hal",
                "caption": selected["caption"],
                "method": "smartsight-disclosed-core",
                "selected_index": selected_index,
                "generated_tokens": selected["generated_tokens"],
                "words": len(selected["caption"].split()),
                "premature_eos": (
                    selected["generated_tokens"] < args.max_new_tokens
                ),
                "frame_indices": indices,
                "segments": segments,
                "segment_count": len(segments),
                "motion_distances": motion_distances,
                "segment_threshold": segment_threshold,
                "candidates": candidates,
                "wall_seconds": time.perf_counter() - started,
                "peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(captioner.device))
                    if torch.cuda.is_available()
                    else None
                ),
                "visual_positions": {
                    "count": int(positions.numel()),
                    "start": int(positions.min()),
                    "end_inclusive": int(positions.max()),
                },
                "realized_vision_shapes": realized_vision_shapes(inputs),
            }
            output_handle.write(json.dumps(result) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())

    metadata["completed_samples"] = len(_completed(output_path))
    _atomic_metadata(
        output_path.with_suffix(output_path.suffix + ".meta.json"),
        metadata,
    )


if __name__ == "__main__":
    main()
