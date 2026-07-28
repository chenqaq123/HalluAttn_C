#!/usr/bin/env python3
"""Evaluate multiple THRONE caption rows with one Flan-T5 model load."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.scripts.run_registered_throne_aqa import (
    run_registered_aqa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--coco_ann", required=True)
    parser.add_argument("--evaluator_model_path", required=True)
    parser.add_argument(
        "--job",
        action="append",
        nargs=2,
        required=True,
        metavar=("CAPTION_GLOB", "OUTPUT"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def validate_jobs(args: argparse.Namespace) -> None:
    outputs = [str(Path(job[1]).resolve()) for job in args.job]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Every AQA suite job must have a distinct output")
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("Invalid shard")


def main() -> None:
    args = parse_args()
    validate_jobs(args)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(args.evaluator_model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for caption, output in args.job:
        job_args = SimpleNamespace(
            manifest=args.manifest,
            caption=[caption],
            coco_ann=args.coco_ann,
            evaluator_model_path=args.evaluator_model_path,
            output=output,
            device=args.device,
            batch_size=args.batch_size,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            limit=args.limit,
        )
        run_registered_aqa(
            job_args,
            tokenizer=tokenizer,
            model=model,
        )


if __name__ == "__main__":
    main()
