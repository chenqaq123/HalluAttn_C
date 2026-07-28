#!/usr/bin/env python3
"""Replay a frozen vanilla subset and fail on any saved-output mismatch."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.generation_audit import continuation_audit
from mitigation.src.open_caption_models import OpenCaptionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=["llava_next", "llava_next_video", "qwen25", "qwen3"],
        required=True,
    )
    parser.add_argument("--media", choices=["image", "video"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--saved",
        action="append",
        required=True,
        help="Saved vanilla JSONL path or glob; repeat for shards.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def _saved(patterns: list[str], id_key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for pattern in patterns:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        for path in paths:
            for row in _jsonl(Path(path)):
                key = str(row[id_key])
                if key in result:
                    raise ValueError(f"Duplicate saved ID {key}")
                if row.get("method") != "vanilla":
                    raise ValueError(f"{path}: expected vanilla row")
                result[key] = row
    return result


def _sha256(ids: torch.Tensor) -> str:
    values = ids.detach().cpu().to(torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    id_key = "sample_id" if args.media == "image" else "clip_id"
    manifest = _jsonl(Path(args.manifest))
    saved = _saved(args.saved, id_key)
    records = manifest[: args.limit]
    missing = [
        str(record[id_key])
        for record in records
        if str(record[id_key]) not in saved
    ]
    if missing:
        raise ValueError(f"Saved vanilla is missing replay IDs: {missing}")

    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    rows = []
    for record in tqdm(records, desc=f"vanilla-replay-{args.kind}"):
        if args.media == "image":
            with Image.open(record["image_path"]) as source:
                inputs = captioner.prepare_image(
                    source.convert("RGB"), record["prompt"]
                )
            frame_indices = None
        else:
            inputs, frame_indices = captioner.prepare_video(
                record["video_path"],
                record["prompt"],
                system=record.get("system"),
            )
            if frame_indices != record["frame_indices"]:
                raise ValueError(
                    f"Frame mismatch for {record[id_key]}"
                )
        prompt_length = int(inputs["input_ids"].shape[-1])
        full_ids = captioner.generate(inputs)
        caption = captioner.decode_continuation(full_ids, prompt_length)
        termination = continuation_audit(
            full_ids,
            prompt_length,
            captioner.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        original = saved[str(record[id_key])]
        fields = (
            "caption",
            "generated_tokens",
            "content_tokens",
            "terminated_by_eos",
            "hit_max_new_tokens",
            "premature_eos",
        )
        replay = {"caption": caption, **termination}
        differences = {
            field: {
                "saved": original.get(field),
                "replay": replay.get(field),
            }
            for field in fields
            if original.get(field) != replay.get(field)
        }
        rows.append(
            {
                id_key: record[id_key],
                "identical": not differences,
                "differences": differences,
                "replay_token_sha256": _sha256(full_ids),
                "frame_indices": frame_indices,
            }
        )
        del inputs, full_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "kind": args.kind,
        "media": args.media,
        "model_path": str(Path(args.model_path).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(rows),
        "identical": sum(row["identical"] for row in rows),
        "comparison_fields": list(fields),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "samples": payload["samples"],
                "identical": payload["identical"],
            },
            indent=2,
        )
    )
    if payload["identical"] != payload["samples"]:
        raise RuntimeError("Saved vanilla replay audit failed")


if __name__ == "__main__":
    main()
