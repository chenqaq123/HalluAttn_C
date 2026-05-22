"""Build an object-level evaluation cache for baseline reproduction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT_DIR
SRC_DIR = PROJECT_ROOT / "src"
PAS_SRC = PROJECT_ROOT.parent / "pas" / "src"
for path in (SRC_DIR, PAS_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_generation(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _caption_by_image(generation: list[dict[str, Any]]) -> dict[int, str]:
    return {int(x["image_id"]): str(x["caption"]) for x in generation}


def _prompt_end_by_image(generation: list[dict[str, Any]]) -> dict[int, int]:
    return {int(x["image_id"]): int(x["prompt_end_idx"]) for x in generation}


def _from_row_cache(row_cache_path: Path, generation_json: Path, limit: int = 0) -> list[dict[str, Any]]:
    data = np.load(row_cache_path)
    generation = _load_generation(generation_json)
    captions = _caption_by_image(generation)
    prompt_ends = _prompt_end_by_image(generation)

    labels = data["labels"].astype(np.int32)
    image_ids = data["image_ids"].astype(np.int64)
    token_pos = data["token_pos"].astype(np.int32)
    words = data["words"].astype(str)
    n = len(labels) if limit <= 0 else min(limit, len(labels))
    records = []
    for i in range(n):
        image_id = int(image_ids[i])
        prompt_end_idx = int(prompt_ends[image_id])
        records.append(
            {
                "object_id": i,
                "image_id": image_id,
                "caption": captions[image_id],
                "word": str(words[i]),
                "label": int(labels[i]),
                "token_pos": int(token_pos[i]),
                "gen_pos": int(token_pos[i] - prompt_end_idx),
                "prompt_end_idx": prompt_end_idx,
            }
        )
    return records


def _from_chair(
    generation_json: Path,
    chair_pkl: Path,
    model_path: str,
    limit: int = 0,
) -> list[dict[str, Any]]:
    from transformers import LlavaProcessor

    from sinkdetect.chair import add_token_labels, evaluate_chair, find_first_mentions, load_chair_evaluator

    generation = _load_generation(generation_json)
    evaluator = load_chair_evaluator(str(chair_pkl))
    chair_data = [{"image_id": x["image_id"], "caption": x["caption"]} for x in generation]
    eval_dicts, _ = evaluate_chair(evaluator, data=chair_data)
    processor = LlavaProcessor.from_pretrained(model_path)
    sequences = [
        processor.tokenizer.encode(entry["caption"], add_special_tokens=False)
        for entry in generation
    ]
    eval_dicts = add_token_labels(eval_dicts, sequences, processor, evaluator)

    records = []
    object_id = 0
    for entry, eval_dict in zip(generation, eval_dicts):
        prompt_end_idx = int(entry["prompt_end_idx"])
        for mention in find_first_mentions(eval_dict):
            token_pos = int(mention["pos"] + prompt_end_idx)
            records.append(
                {
                    "object_id": object_id,
                    "image_id": int(entry["image_id"]),
                    "caption": str(entry["caption"]),
                    "word": str(mention["word"]),
                    "label": 1 if mention["hallucinated"] else 0,
                    "token_pos": token_pos,
                    "gen_pos": int(token_pos - prompt_end_idx),
                    "prompt_end_idx": prompt_end_idx,
                }
            )
            object_id += 1
            if limit > 0 and object_id >= limit:
                return records
    return records


def _filter_shard(records: list[dict[str, Any]], shard_idx: int, num_shards: int, shard_by: str) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return records
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx={shard_idx} must be in [0, {num_shards})")
    if shard_by == "object":
        return [r for r in records if int(r["object_id"]) % num_shards == shard_idx]
    if shard_by != "image":
        raise ValueError(f"Unknown shard_by={shard_by!r}; expected 'image' or 'object'")

    image_to_shard: dict[int, int] = {}
    for record in records:
        image_id = int(record["image_id"])
        if image_id not in image_to_shard:
            image_to_shard[image_id] = len(image_to_shard) % num_shards
    return [r for r in records if image_to_shard[int(r["image_id"])] == shard_idx]


def build_object_cache(
    generation_json: str | Path,
    row_cache: str | Path | None,
    chair_pkl: str | Path,
    model_path: str,
    limit: int = 0,
    shard_idx: int = 0,
    num_shards: int = 1,
    shard_by: str = "image",
) -> list[dict[str, Any]]:
    generation_json = Path(generation_json)
    row_cache_path = Path(row_cache) if row_cache else None
    if row_cache_path and row_cache_path.exists():
        records = _from_row_cache(row_cache_path, generation_json, limit=limit)
    else:
        records = _from_chair(generation_json, Path(chair_pkl), model_path, limit=limit)
    return _filter_shard(records, shard_idx=shard_idx, num_shards=num_shards, shard_by=shard_by)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
