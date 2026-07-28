#!/usr/bin/env python3
"""Validate Vript-HAL and freeze the registered 32-frame clip manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.open_caption_models import (
    VIDEO_FRAMES,
    VRIPT_SYSTEM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="/tmp/sinkdetect_datasets/VRIPT-HAL"
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, records: list[dict]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    annotation_path = root / "HAL_annotations.jsonl"
    annotations = [
        json.loads(line)
        for line in annotation_path.read_text().splitlines()
        if line.strip()
    ]
    if len(annotations) != 122:
        raise ValueError(
            f"Expected 122 Vript-HAL annotations, got {len(annotations)}"
        )
    clip_ids = [str(item["clip_id"]) for item in annotations]
    if len(set(clip_ids)) != 122:
        raise ValueError("Duplicate Vript-HAL clip IDs")

    from decord import VideoReader, cpu

    records = []
    total_bytes = 0
    for annotation in sorted(annotations, key=lambda item: item["clip_id"]):
        clip_id = str(annotation["clip_id"])
        video_path = root / "HAL_scenes" / f"{clip_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        reader = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(reader)
        if frame_count <= 0:
            raise ValueError(f"No decodable frames: {video_path}")
        indices = np.rint(
            np.linspace(0, frame_count - 1, VIDEO_FRAMES)
        ).astype(np.int64)
        decoded = reader.get_batch(indices).asnumpy()
        if decoded.shape[0] != VIDEO_FRAMES:
            raise ValueError(f"Could not decode 32 frames: {video_path}")
        fps = float(reader.get_avg_fps())
        size = video_path.stat().st_size
        total_bytes += size
        records.append(
            {
                "benchmark": "vript-hal",
                "clip_id": clip_id,
                "video_path": str(video_path.resolve()),
                "prompt": "Describe this video in detail.",
                "system": VRIPT_SYSTEM,
                "source_frames": frame_count,
                "source_fps": fps,
                "duration_seconds": (
                    float(frame_count / fps) if fps > 0 else None
                ),
                "frame_indices": indices.tolist(),
                "video_bytes": size,
                "video_sha256": sha256_file(video_path),
            }
        )

    frame_manifest = root / "frame_manifest.jsonl"
    manifest_sha = _atomic_jsonl(frame_manifest, records)
    metadata = {
        "benchmark": "Vript-HAL",
        "clips": len(records),
        "frames_per_clip": VIDEO_FRAMES,
        "annotation_path": str(annotation_path.resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "frame_manifest_path": str(frame_manifest.resolve()),
        "frame_manifest_sha256": manifest_sha,
        "total_video_bytes": total_bytes,
        "missing_or_bad": [],
    }
    metadata_path = root / "dataset_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
