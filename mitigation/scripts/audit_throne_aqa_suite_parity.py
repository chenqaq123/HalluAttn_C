#!/usr/bin/env python3
"""Fail unless a refactored THRONE AQA suite exactly matches a frozen run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    expected: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if set(expected) != set(candidate):
        raise ValueError("AQA image-ID sets differ")
    image_ids = sorted(expected, key=int)
    answer_counts = {
        image_id: (len(expected[image_id]), len(candidate[image_id]))
        for image_id in image_ids
    }
    if any(
        first != 1095 or second != 1095
        for first, second in answer_counts.values()
    ):
        raise ValueError("Every parity row must contain exactly 1,095 answers")
    mismatches = sum(
        first != second
        for image_id in image_ids
        for first, second in zip(expected[image_id], candidate[image_id])
    )
    return {
        "images": len(image_ids),
        "answers_per_image": 1095,
        "answers_compared": len(image_ids) * 1095,
        "answer_mismatches": mismatches,
        "exact_match": mismatches == 0,
    }


def main() -> None:
    args = parse_args()
    expected_path = Path(args.expected)
    candidate_path = Path(args.candidate)
    comparison = compare(
        json.loads(expected_path.read_text()),
        json.loads(candidate_path.read_text()),
    )
    payload = {
        "schema": "sinkdetect-throne-aqa-suite-parity-v1",
        "expected": str(expected_path.resolve()),
        "expected_sha256": _sha256(expected_path),
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": _sha256(candidate_path),
        **comparison,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2))
    if not payload["exact_match"]:
        raise RuntimeError("THRONE AQA suite parity failed")


if __name__ == "__main__":
    main()
