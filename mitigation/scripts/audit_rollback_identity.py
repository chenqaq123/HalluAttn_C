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
        results.append(
            {
                id_key: record[id_key],
                "token_identical": bool(identical),
                "caption_identical": vanilla["caption"] == identity["caption"],
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


if __name__ == "__main__":
    main()
