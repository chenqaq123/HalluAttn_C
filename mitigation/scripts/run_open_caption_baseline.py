#!/usr/bin/env python3
"""Generate one registered image-caption model/method/benchmark shard."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.formula_ports import (
    FormulaAttentionPort,
    FormulaPortConfig,
    generate_pai_greedy,
)
from mitigation.src.open_caption_models import (
    OpenCaptionModel,
    realized_vision_shapes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["llava_next", "qwen25", "qwen3"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=["vanilla", "pai", "clearsight", "diml", "allpath"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=927)
    parser.add_argument("--allpath_heads")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="Set intervention coefficients to identity for no-op testing.",
    )
    return parser.parse_args()


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _completed(path: Path) -> set[int | str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            completed.add(json.loads(line)["sample_id"])
    return completed


def _load_allpath_heads(
    path: str | None,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[str, Any]]:
    if not path:
        raise ValueError("--allpath_heads is required for AllPath")
    source = Path(path)
    payload = json.loads(source.read_text())

    def normalize(value) -> dict[int, list[int]]:
        if isinstance(value, dict):
            return {
                int(layer): sorted({int(head) for head in heads})
                for layer, heads in value.items()
            }
        result: dict[int, list[int]] = {}
        for item in value:
            result.setdefault(int(item["layer"]), []).append(int(item["head"]))
        return {
            layer: sorted(set(heads)) for layer, heads in result.items()
        }

    return (
        normalize(payload["good_heads"]),
        normalize(payload.get("hallu_heads", [])),
        payload,
    )


def _port_config(
    method: str,
    layers: int,
    positions: torch.Tensor,
    *,
    identity: bool,
    allpath_path: str | None,
) -> tuple[FormulaPortConfig, dict[str, Any] | None]:
    position_list = [int(value) for value in positions.detach().cpu()]
    head_metadata = None
    if method == "pai":
        return (
            FormulaPortConfig(
                method=method,
                start_layer=2,
                end_layer=layers,
                visual_positions=position_list,
                pai_alpha=0.0 if identity else 0.5,
            ),
            None,
        )
    if method == "clearsight":
        return (
            FormulaPortConfig(
                method=method,
                start_layer=math.floor(9 * layers / 32),
                end_layer=math.ceil(15 * layers / 32),
                visual_positions=position_list,
                clearsight_enhance=1.0 if identity else 1.15,
                clearsight_suppress=1.0 if identity else 0.95,
            ),
            None,
        )
    if method == "diml":
        return (
            FormulaPortConfig(
                method=method,
                start_layer=math.floor(5 * layers / 32),
                end_layer=math.ceil(18 * layers / 32),
                visual_positions=position_list,
                diml_alpha=0.0 if identity else 0.5,
            ),
            None,
        )
    if method == "allpath":
        good, hallu, head_metadata = _load_allpath_heads(allpath_path)
        return (
            FormulaPortConfig(
                method=method,
                start_layer=0,
                end_layer=layers,
                visual_positions=position_list,
                allpath_good_heads=good,
                allpath_hallu_heads=hallu,
                allpath_in_scale=1.0 if identity else 2.0,
                allpath_de_scale=1.0 if identity else 0.0,
            ),
            head_metadata,
        )
    raise ValueError(method)


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
    torch.use_deterministic_algorithms(False)

    records = _load_jsonl(args.manifest)
    records = [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_idx
    ]
    if args.limit is not None:
        records = records[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path)
    pending = [
        record for record in records if record["sample_id"] not in completed
    ]
    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    metadata: dict[str, Any] = {
        "model": captioner.metadata(),
        "method": args.method,
        "identity": bool(args.identity),
        "manifest": str(Path(args.manifest).resolve()),
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "registered_samples_in_shard": len(records),
        "seed": args.seed,
        "formula_port": "mitigation/src/formula_ports.py",
        "allpath_heads": args.allpath_heads,
    }
    head_metadata = None

    with output_path.open("a") as output_handle:
        for record in tqdm(pending, desc=f"{args.method}-{args.shard_idx}"):
            started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(captioner.device)
            with Image.open(record["image_path"]) as source:
                image = source.convert("RGB")
                inputs = captioner.prepare_image(image, record["prompt"])
            prompt_length = int(inputs["input_ids"].shape[-1])
            positions = captioner.visual_positions(
                inputs, media_type="image"
            )
            audit = None
            decode_metadata: dict[str, Any] = {}
            if args.method == "vanilla":
                full_ids = captioner.generate(inputs)
            else:
                config, head_metadata = _port_config(
                    args.method,
                    captioner.decoder.layers,
                    positions,
                    identity=args.identity,
                    allpath_path=args.allpath_heads,
                )
                with FormulaAttentionPort(
                    captioner.decoder.modeling_module,
                    captioner.decoder.attention_classes,
                    config,
                ) as port:
                    if args.method == "pai" and not args.identity:
                        unconditional = captioner.prepare_text_only(
                            record["prompt"]
                        )
                        full_ids, decode_metadata = generate_pai_greedy(
                            captioner.model,
                            captioner.tokenizer,
                            inputs,
                            unconditional,
                            port,
                            max_new_tokens=args.max_new_tokens,
                            guidance_scale=2.0,
                            plausibility_cutoff=0.1,
                        )
                    else:
                        full_ids = captioner.generate(inputs)
                    audit = port.audit.to_dict()

            caption = captioner.decode_continuation(
                full_ids, prompt_length
            )
            generated_tokens = int(full_ids.shape[-1] - prompt_length)
            result = {
                "sample_id": record["sample_id"],
                "benchmark": record["benchmark"],
                "caption": caption,
                "method": args.method,
                "prompt": record["prompt"],
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
                "audit": audit,
                "decode": decode_metadata,
            }
            output_handle.write(json.dumps(result) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())

    metadata["completed_samples"] = len(_completed(output_path))
    if head_metadata is not None:
        metadata["allpath_head_metadata"] = head_metadata
    _atomic_metadata(
        output_path.with_suffix(output_path.suffix + ".meta.json"), metadata
    )


if __name__ == "__main__":
    main()
