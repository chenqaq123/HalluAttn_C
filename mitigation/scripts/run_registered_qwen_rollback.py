#!/usr/bin/env python3
"""Run detector-gated one-slot rollback for registered Qwen caption cells."""

from __future__ import annotations

import argparse
import json
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

from mitigation.src.probe_sensor import MeanProbeSensor, OOFProbeSensor
from mitigation.src.generation_audit import continuation_audit
from mitigation.src.registered_llava_probe import RegisteredLlavaCaptioner
from mitigation.src.registered_qwen_probe import (
    RegisteredQwen25Captioner,
    RegisteredQwen3Captioner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=[
            "llava_next",
            "llava_next_video",
            "qwen25",
            "qwen3",
        ],
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
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_actions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def _records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [
            json.loads(line) for line in handle if line.strip()
        ]


def _completed(path: Path, key: str) -> set[str]:
    if not path.is_file():
        return set()
    with path.open() as handle:
        return {
            str(json.loads(line)[key])
            for line in handle
            if line.strip()
        }


def _atomic_metadata(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


class RollbackRunner:
    def __init__(self, args: argparse.Namespace):
        import spacy

        self.args = args
        self.device = torch.device(args.device)
        if args.kind == "qwen25":
            self.captioner = RegisteredQwen25Captioner(
                args.model_path,
                self.device,
                max_new_tokens=args.max_new_tokens,
            )
        elif args.kind == "qwen3":
            self.captioner = RegisteredQwen3Captioner(
                args.model_path,
                self.device,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            self.captioner = RegisteredLlavaCaptioner(
                args.model_path,
                self.device,
                kind=args.kind,
                max_new_tokens=args.max_new_tokens,
            )
        self.nlp = spacy.load("en_core_web_lg")
        sensor_class = (
            MeanProbeSensor
            if args.sensor_mode == "mean"
            else OOFProbeSensor
        )
        self.sensor = sensor_class(args.sensor, self.device)
        self.threshold = float(
            json.loads(Path(args.thresholds).read_text())["tau_risk"]
        )

    def _score(
        self,
        sample_index: int,
        post_feature: torch.Tensor,
    ) -> float:
        if self.args.sensor_mode == "oof":
            return self.sensor.score(sample_index, post_feature)
        return self.sensor.score(post_feature)

    def analyze(
        self,
        sample_index: int,
        inputs: dict[str, Any],
        full_ids: torch.Tensor,
    ) -> dict[str, Any]:
        prompt_length = int(inputs["input_ids"].shape[-1])
        caption = self.captioner.caption_text(full_ids, prompt_length)
        generated_ids = full_ids[0, prompt_length:].tolist()
        raw = self.captioner.noun_mentions(self.nlp, generated_ids)
        anchors = [
            prompt_length + int(mention["rel_anchor"])
            for mention in raw
        ]
        keep = [
            anchor + 1 < full_ids.shape[-1] for anchor in anchors
        ]
        raw = [
            mention for mention, selected in zip(raw, keep) if selected
        ]
        anchors = [
            anchor for anchor, selected in zip(anchors, keep) if selected
        ]
        mentions = []
        if raw:
            features, first_logp = self.captioner.replay_features(
                inputs, full_ids, anchors
            )
            post = torch.from_numpy(
                features[:, :, 1, :].reshape(features.shape[0], -1)
            )
            for index, mention in enumerate(raw):
                mentions.append(
                    {
                        "word": mention["word"],
                        "surface": mention["surface"],
                        "token_pos": int(anchors[index]),
                        "gen_pos": int(mention["rel_anchor"]),
                        "risk": self._score(
                            sample_index, post[index]
                        ),
                        "nll": float(-first_logp[index]),
                        "first_id": int(
                            full_ids[0, anchors[index] + 1]
                        ),
                    }
                )
        return {
            "caption": caption,
            "mentions": mentions,
            "full_ids": full_ids,
        }

    def intervene(
        self,
        sample_index: int,
        inputs: dict[str, Any],
        vanilla: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_length = int(inputs["input_ids"].shape[-1])
        current = vanilla
        actions = []
        used_slots: set[int] = set()
        for _ in range(self.args.max_actions):
            candidate = next(
                (
                    mention
                    for mention in current["mentions"]
                    if mention["token_pos"] not in used_slots
                    and mention["risk"] >= self.threshold
                ),
                None,
            )
            if candidate is None:
                break
            slot = int(candidate["token_pos"])
            used_slots.add(slot)
            prefix = current["full_ids"][:, : slot + 1]
            regenerated = self.captioner.rollback_once(
                inputs,
                prefix,
                {int(candidate["first_id"])},
                prompt_length=prompt_length,
            )
            next_state = self.analyze(
                sample_index, inputs, regenerated
            )
            replacement = next(
                (
                    {
                        "word": mention["word"],
                        "surface": mention["surface"],
                    }
                    for mention in next_state["mentions"]
                    if mention["token_pos"] == slot
                ),
                None,
            )
            actions.append(
                {
                    "slot": slot,
                    "word": candidate["word"],
                    "surface": candidate["surface"],
                    "risk": candidate["risk"],
                    "nll": candidate["nll"],
                    "banned_first_id": candidate["first_id"],
                    "replacement": replacement,
                }
            )
            current = next_state
        unresolved = [
            {
                "word": mention["word"],
                "token_pos": mention["token_pos"],
                "risk": mention["risk"],
            }
            for mention in current["mentions"]
            if mention["risk"] >= self.threshold
            and mention["token_pos"] not in used_slots
        ]
        return {
            "caption": current["caption"],
            "full_ids": current["full_ids"],
            "mentions": [
                {
                    key: mention[key]
                    for key in (
                        "word",
                        "surface",
                        "token_pos",
                        "gen_pos",
                        "risk",
                        "nll",
                    )
                }
                for mention in current["mentions"]
            ],
            "actions": actions,
            "unresolved_triggers": unresolved,
        }


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("Invalid shard")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    records = [
        record
        for index, record in enumerate(_records(args.manifest))
        if index % args.num_shards == args.shard_idx
    ]
    if args.limit is not None:
        records = records[: args.limit]
    id_key = "sample_id" if args.media == "image" else "clip_id"
    ordered_ids = sorted(str(record[id_key]) for record in _records(args.manifest))
    index_by_id = {
        identifier: index for index, identifier in enumerate(ordered_ids)
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output_path, id_key)
    pending = [
        record
        for record in records
        if str(record[id_key]) not in completed
    ]
    runner = RollbackRunner(args)
    metadata = {
        "kind": args.kind,
        "media": args.media,
        "model_path": str(Path(args.model_path).resolve()),
        "sensor": str(Path(args.sensor).resolve()),
        "sensor_mode": args.sensor_mode,
        "thresholds": str(Path(args.thresholds).resolve()),
        "tau_risk": runner.threshold,
        "manifest": str(Path(args.manifest).resolve()),
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "registered_samples_in_shard": len(records),
        "max_new_tokens": args.max_new_tokens,
        "max_actions": args.max_actions,
        "rollback_rule": "one first-token ban at one prefix; no repeated slot",
        "seed": args.seed,
    }

    with output_path.open("a") as output_handle:
        for record in tqdm(pending, desc=f"ours-{args.shard_idx}"):
            started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(runner.device)
            if args.media == "image":
                with Image.open(record["image_path"]) as source:
                    image = source.convert("RGB")
                    inputs = runner.captioner.prompt_inputs(
                        image, record["prompt"]
                    )
                frame_indices = None
            else:
                inputs = runner.captioner.video_prompt_inputs(
                    record["video_path"],
                    record["prompt"],
                    system=record.get("system"),
                )
                frame_indices = record["frame_indices"]
            sample_index = index_by_id[str(record[id_key])]
            full_ids = runner.captioner.generate_ids(inputs)
            vanilla = runner.analyze(
                sample_index, inputs, full_ids
            )
            ours = runner.intervene(
                sample_index, inputs, vanilla
            )
            termination = continuation_audit(
                ours["full_ids"],
                int(inputs["input_ids"].shape[-1]),
                runner.captioner.tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            row = {
                id_key: record[id_key],
                "benchmark": record["benchmark"],
                "method": "ours",
                "source_vanilla": {
                    "caption": vanilla["caption"],
                    "mentions": [
                        {
                            key: mention[key]
                            for key in (
                                "word",
                                "surface",
                                "token_pos",
                                "gen_pos",
                                "risk",
                                "nll",
                            )
                        }
                        for mention in vanilla["mentions"]
                    ],
                },
                "caption": ours["caption"],
                "mentions": ours["mentions"],
                "actions": ours["actions"],
                "unresolved_triggers": ours["unresolved_triggers"],
                "frame_indices": frame_indices,
                **termination,
                "words": len(ours["caption"].split()),
                "wall_seconds": time.perf_counter() - started,
                "peak_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(runner.device))
                    if torch.cuda.is_available()
                    else None
                ),
            }
            output_handle.write(json.dumps(row) + "\n")
            output_handle.flush()
            os.fsync(output_handle.fileno())
            del inputs, full_ids, vanilla, ours
            torch.cuda.empty_cache()

    metadata["completed_samples"] = len(_completed(output_path, id_key))
    _atomic_metadata(
        output_path.with_suffix(output_path.suffix + ".meta.json"),
        metadata,
    )


if __name__ == "__main__":
    main()
