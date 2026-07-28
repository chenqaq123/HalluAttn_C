#!/usr/bin/env python3
"""Verify that registered VCD is token-identical when alpha is zero."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.formula_ports import generate_vcd_greedy_multimodal
from mitigation.src.open_caption_models import OpenCaptionModel


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
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def _records(path: str | Path, limit: int) -> list[dict]:
    with Path(path).open() as handle:
        return [
            json.loads(line)
            for index, line in enumerate(handle)
            if line.strip() and index < limit
        ]


def _sha256(ids: torch.Tensor) -> str:
    values = ids.detach().cpu().to(torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    rows = []
    for record in tqdm(
        _records(args.manifest, args.limit),
        desc=f"vcd-identity-{args.kind}",
    ):
        inputs, indices = captioner.prepare_video(
            record["video_path"],
            record["prompt"],
            system=record.get("system"),
        )
        if indices != record["frame_indices"]:
            raise ValueError(
                f"Frame-index mismatch for {record['clip_id']}"
            )
        vanilla = captioner.generate(inputs)
        identity, audit = generate_vcd_greedy_multimodal(
            captioner.model,
            captioner.tokenizer,
            inputs,
            max_new_tokens=args.max_new_tokens,
            alpha=0.0,
            beta=0.1,
            noise_step=500,
            seed=args.seed,
        )
        rows.append(
            {
                "clip_id": record["clip_id"],
                "token_identical": bool(torch.equal(vanilla, identity)),
                "vanilla_sha256": _sha256(vanilla),
                "identity_sha256": _sha256(identity),
                "vanilla_tokens": int(
                    vanilla.shape[-1] - inputs["input_ids"].shape[-1]
                ),
                "identity_tokens": int(
                    identity.shape[-1] - inputs["input_ids"].shape[-1]
                ),
                "audit": audit,
            }
        )
        del inputs, vanilla, identity
        torch.cuda.empty_cache()

    payload = {
        "kind": args.kind,
        "model_path": str(Path(args.model_path).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(rows),
        "identical": sum(row["token_identical"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "kind": args.kind,
                "identical": payload["identical"],
                "samples": payload["samples"],
            },
            indent=2,
        )
    )
    if payload["identical"] != payload["samples"]:
        raise RuntimeError("VCD identity audit failed")


if __name__ == "__main__":
    main()
