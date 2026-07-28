#!/usr/bin/env python3
"""Run multiple registered formula-baseline shards with one model load."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.scripts.run_open_caption_baseline import (
    run_registered_shard,
)
from mitigation.src.open_caption_models import OpenCaptionModel


METHODS = {"vanilla", "pai", "clearsight", "diml", "allpath"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["llava_next", "qwen25", "qwen3"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--job",
        action="append",
        nargs=3,
        required=True,
        metavar=("METHOD", "MANIFEST", "OUTPUT"),
        help="Repeat for each registered method/manifest/output shard.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument("--allpath_heads")
    parser.add_argument(
        "--pai_gate",
        help=(
            "Required for a PAI job. JSON from audit_pai_generation.py; "
            "must cover 20 active samples and exercise every attention port."
        ),
    )
    return parser.parse_args()


def validate_jobs(args: argparse.Namespace) -> None:
    methods = [job[0] for job in args.job]
    invalid = sorted(set(methods) - METHODS)
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}")
    outputs = [str(Path(job[2]).resolve()) for job in args.job]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Every suite job must have a distinct output")
    if "allpath" in methods and not args.allpath_heads:
        raise ValueError("--allpath_heads is required for AllPath")
    if "pai" not in methods:
        return
    if not args.pai_gate:
        raise ValueError("--pai_gate is required before a formal PAI job")
    gate = json.loads(Path(args.pai_gate).read_text())
    if gate.get("kind") != args.kind:
        raise ValueError(
            f"PAI gate kind {gate.get('kind')!r} != {args.kind!r}"
        )
    if int(gate.get("samples", 0)) != 20:
        raise ValueError("PAI gate must contain exactly 20 samples")
    summary = gate.get("summary", {})
    if not summary.get("all_attention_ports_called"):
        raise ValueError("PAI gate did not exercise every attention port")
    if int(summary.get("changed_outputs", 0)) == 0:
        raise ValueError("Active PAI gate did not change any output")


def main() -> None:
    args = parse_args()
    validate_jobs(args)
    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    for method, manifest, output in args.job:
        shard_args = SimpleNamespace(
            kind=args.kind,
            model_path=args.model_path,
            manifest=manifest,
            method=method,
            output=output,
            device=args.device,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            allpath_heads=args.allpath_heads,
            identity=False,
        )
        run_registered_shard(shard_args, captioner=captioner)


if __name__ == "__main__":
    main()
