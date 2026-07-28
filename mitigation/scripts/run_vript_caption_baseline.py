#!/usr/bin/env python3
"""Generate one vanilla or VCD shard on the frozen Vript-HAL manifest."""

from __future__ import annotations

import argparse
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

from mitigation.src.formula_ports import generate_vcd_greedy_multimodal
from mitigation.src.open_caption_models import (
    OpenCaptionModel,
    realized_vision_shapes,
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
    parser.add_argument(
        "--method", choices=["vanilla", "vcd"], required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument(
        "--vcd_identity",
        action="store_true",
        help="Use alpha=0 for the registered no-op comparison.",
    )
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
        "method": args.method,
        "manifest": str(Path(args.manifest).resolve()),
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "registered_samples_in_shard": len(records),
        "seed": args.seed,
        "frames": 32,
        "frame_size": 392,
        "vcd": {
            "alpha": 0.0 if args.vcd_identity else 0.5,
            "beta": 0.1,
            "noise_step": 500,
        }
        if args.method == "vcd"
        else None,
    }

    with output_path.open("a") as output_handle:
        for record in tqdm(pending, desc=f"{args.method}-{args.shard_idx}"):
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
            decode_metadata: dict[str, Any] = {}
            if args.method == "vanilla":
                full_ids = captioner.generate(inputs)
            else:
                full_ids, decode_metadata = (
                    generate_vcd_greedy_multimodal(
                        captioner.model,
                        captioner.tokenizer,
                        inputs,
                        max_new_tokens=args.max_new_tokens,
                        alpha=0.0 if args.vcd_identity else 0.5,
                        beta=0.1,
                        noise_step=500,
                        seed=args.seed,
                    )
                )
            caption = captioner.decode_continuation(
                full_ids, prompt_length
            )
            generated_tokens = int(full_ids.shape[-1] - prompt_length)
            result = {
                "clip_id": record["clip_id"],
                "benchmark": "vript-hal",
                "caption": caption,
                "method": args.method,
                "prompt": record["prompt"],
                "system": record.get("system"),
                "frame_indices": indices,
                "generated_tokens": generated_tokens,
                "words": len(caption.split()),
                "premature_eos": generated_tokens < args.max_new_tokens,
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
                "decode": decode_metadata,
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
