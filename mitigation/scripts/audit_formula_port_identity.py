#!/usr/bin/env python3
"""Check that every image formula port is token-identical at identity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.formula_ports import (
    FormulaAttentionPort,
    FormulaPortConfig,
)
from mitigation.src.open_caption_models import OpenCaptionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["llava_next", "qwen25", "qwen3"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def records(path: str | Path, limit: int) -> list[dict]:
    with Path(path).open() as handle:
        return [
            json.loads(line)
            for index, line in enumerate(handle)
            if line.strip() and index < limit
        ]


def config(
    method: str,
    layers: int,
    positions: torch.Tensor,
) -> FormulaPortConfig:
    visual = [int(value) for value in positions.detach().cpu()]
    if method == "pai":
        return FormulaPortConfig(
            method=method,
            start_layer=2,
            end_layer=layers,
            visual_positions=visual,
            pai_alpha=0.0,
        )
    if method == "clearsight":
        return FormulaPortConfig(
            method=method,
            start_layer=math.floor(9 * layers / 32),
            end_layer=math.ceil(15 * layers / 32),
            visual_positions=visual,
            clearsight_enhance=1.0,
            clearsight_suppress=1.0,
        )
    if method == "diml":
        return FormulaPortConfig(
            method=method,
            start_layer=math.floor(5 * layers / 32),
            end_layer=math.ceil(18 * layers / 32),
            visual_positions=visual,
            diml_alpha=0.0,
        )
    if method == "allpath":
        return FormulaPortConfig(
            method=method,
            start_layer=0,
            end_layer=layers,
            visual_positions=visual,
            allpath_good_heads={0: [0]},
            allpath_hallu_heads={1: [0]},
            allpath_in_scale=1.0,
            allpath_de_scale=1.0,
        )
    raise ValueError(method)


def main() -> None:
    args = parse_args()
    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    rows = []
    methods = ("pai", "clearsight", "diml", "allpath")
    for record in tqdm(
        records(args.manifest, args.limit), desc=f"identity-{args.kind}"
    ):
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
            inputs = captioner.prepare_image(
                image, record["prompt"]
            )
        vanilla = captioner.generate(inputs)
        positions = captioner.visual_positions(
            inputs, media_type="image"
        )
        method_rows = {}
        for method in methods:
            with FormulaAttentionPort(
                captioner.decoder.modeling_module,
                captioner.decoder.attention_classes,
                config(method, captioner.decoder.layers, positions),
            ) as port:
                candidate = captioner.generate(inputs)
            method_rows[method] = {
                "token_identical": bool(torch.equal(vanilla, candidate)),
                "audit": port.audit.to_dict(),
            }
        rows.append(
            {
                "sample_id": record["sample_id"],
                "vanilla_tokens": int(
                    vanilla.shape[-1] - inputs["input_ids"].shape[-1]
                ),
                "methods": method_rows,
            }
        )
    summary = {
        "kind": args.kind,
        "model_path": str(Path(args.model_path).resolve()),
        "samples": len(rows),
        "methods": {
            method: {
                "identical": int(
                    sum(
                        row["methods"][method]["token_identical"]
                        for row in rows
                    )
                ),
                "total": len(rows),
            }
            for method in methods
        },
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["methods"], indent=2))
    if any(
        value["identical"] != value["total"]
        for value in summary["methods"].values()
    ):
        raise RuntimeError("At least one formula identity audit failed")


if __name__ == "__main__":
    main()
