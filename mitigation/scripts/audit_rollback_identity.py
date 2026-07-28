#!/usr/bin/env python3
"""Verify that registered rollback with zero allowed actions is a token no-op."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.scripts.run_registered_qwen_rollback import (
    RollbackRunner,
    _records,
)
from mitigation.scripts.audit_registered_vanilla_replay import _saved
from mitigation.src.generation_audit import continuation_audit


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
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--sensor_mode", choices=["mean", "oof"], default="mean")
    parser.add_argument("--thresholds", required=True)
    parser.add_argument(
        "--saved_vanilla",
        action="append",
        help=(
            "Shared vanilla JSONL path/glob. When provided, the rollback "
            "runner's source generation must also match this control."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def _sha256(ids: torch.Tensor) -> str:
    array = ids.detach().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # RollbackRunner reads this value; zero is the registered identity setting.
    args.max_actions = 0
    runner = RollbackRunner(args)
    records = _records(args.manifest)
    ordered_ids = sorted(
        str(record["sample_id" if args.media == "image" else "clip_id"])
        for record in records
    )
    index_by_id = {
        identifier: index for index, identifier in enumerate(ordered_ids)
    }
    id_key = "sample_id" if args.media == "image" else "clip_id"
    shared = (
        _saved(args.saved_vanilla, id_key)
        if args.saved_vanilla
        else None
    )
    results = []
    for record in tqdm(records[: args.limit], desc=f"rollback-identity-{args.kind}"):
        if args.media == "image":
            with Image.open(record["image_path"]) as source:
                inputs = runner.captioner.prompt_inputs(
                    source.convert("RGB"), record["prompt"]
                )
        else:
            inputs = runner.captioner.video_prompt_inputs(
                record["video_path"],
                record["prompt"],
                system=record.get("system"),
            )
        sample_index = index_by_id[str(record[id_key])]
        full_ids = runner.captioner.generate_ids(inputs)
        vanilla = runner.analyze(sample_index, inputs, full_ids)
        identity = runner.intervene(sample_index, inputs, vanilla)
        identical = torch.equal(vanilla["full_ids"], identity["full_ids"])
        shared_differences = {}
        if shared is not None:
            key = str(record[id_key])
            if key not in shared:
                raise ValueError(f"Shared vanilla is missing {id_key} {key}")
            termination = continuation_audit(
                vanilla["full_ids"],
                int(inputs["input_ids"].shape[-1]),
                runner.captioner.tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            replay = {"caption": vanilla["caption"], **termination}
            for field in (
                "caption",
                "generated_tokens",
                "content_tokens",
                "terminated_by_eos",
                "hit_max_new_tokens",
                "premature_eos",
            ):
                if shared[key].get(field) != replay.get(field):
                    shared_differences[field] = {
                        "saved": shared[key].get(field),
                        "runner": replay.get(field),
                    }
        results.append(
            {
                id_key: record[id_key],
                "token_identical": bool(identical),
                "caption_identical": vanilla["caption"] == identity["caption"],
                "shared_control_identical": (
                    not shared_differences
                    if shared is not None
                    else None
                ),
                "shared_control_differences": shared_differences,
                "vanilla_sha256": _sha256(vanilla["full_ids"]),
                "identity_sha256": _sha256(identity["full_ids"]),
                "actions": len(identity["actions"]),
            }
        )
        del inputs, full_ids, vanilla, identity
        torch.cuda.empty_cache()
    payload = {
        "kind": args.kind,
        "media": args.media,
        "identity_setting": {"max_actions": 0},
        "samples": len(results),
        "all_token_identical": all(row["token_identical"] for row in results),
        "all_caption_identical": all(
            row["caption_identical"] for row in results
        ),
        "shared_control_checked": shared is not None,
        "all_shared_control_identical": (
            all(row["shared_control_identical"] for row in results)
            if shared is not None
            else None
        ),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    if not payload["all_token_identical"]:
        raise RuntimeError("Rollback identity audit failed")
    if (
        payload["shared_control_checked"]
        and not payload["all_shared_control_identical"]
    ):
        raise RuntimeError("Rollback source differs from shared vanilla")


if __name__ == "__main__":
    main()
