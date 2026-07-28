#!/usr/bin/env python3
"""Merge registered caption shards into official evaluator input schemas."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


EXPECTED_COUNTS = {
    "chair": 500,
    "amber-generative": 1004,
    "throne-objects365": 5110,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="JSONL paths or glob patterns for one complete result cell.",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(EXPECTED_COUNTS),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Permit fewer than the registered full-split count for smoke use.",
    )
    return parser.parse_args()


def _paths(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        paths.extend(Path(match) for match in matches)
    unique = sorted(set(paths))
    if not unique:
        raise ValueError("No input shards")
    return unique


def _rows(paths: list[Path], benchmark: str) -> list[dict]:
    by_id: dict[int, dict] = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("benchmark") != benchmark:
                    raise ValueError(
                        f"{path}: expected {benchmark}, got "
                        f"{row.get('benchmark')}"
                    )
                sample_id = int(row["sample_id"])
                if sample_id in by_id:
                    if by_id[sample_id]["caption"] != row["caption"]:
                        raise ValueError(
                            f"Conflicting duplicate sample {sample_id}"
                        )
                    continue
                by_id[sample_id] = row
    return [by_id[key] for key in sorted(by_id)]


def _payload(benchmark: str, rows: list[dict]):
    if benchmark == "chair":
        return [
            {
                "image_id": int(row["sample_id"]),
                "caption": row["caption"],
            }
            for row in rows
        ]
    if benchmark == "amber-generative":
        return [
            {
                "id": int(row["sample_id"]),
                "response": row["caption"],
            }
            for row in rows
        ]
    if benchmark == "throne-objects365":
        return {
            "prompts": ["Describe this image in detail."],
            "responses": [
                [0, int(row["sample_id"]), row["caption"]]
                for row in rows
            ],
        }
    raise ValueError(benchmark)


def main() -> None:
    args = parse_args()
    paths = _paths(args.inputs)
    rows = _rows(paths, args.benchmark)
    expected = EXPECTED_COUNTS[args.benchmark]
    if not args.allow_partial and len(rows) != expected:
        raise ValueError(
            f"{args.benchmark}: expected {expected}, got {len(rows)}"
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(_payload(args.benchmark, rows)) + "\n")
    os.replace(temporary, output)
    metadata = {
        "benchmark": args.benchmark,
        "registered_count": expected,
        "exported_count": len(rows),
        "complete": len(rows) == expected,
        "input_shards": [str(path.resolve()) for path in paths],
        "output": str(output.resolve()),
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(meta_tmp, meta_path)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
