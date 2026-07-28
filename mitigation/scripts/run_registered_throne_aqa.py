#!/usr/bin/env python3
"""Run the registered THRONE AQA parser with atomic per-image resume."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Exact first three question formats and first instruction from THRONE commit
# 45badbb71c2b3e8d6619f6b45cccdac24dd20b46.
QUESTION_FORMATS = (
    (
        "Please answer yes or no. Is there {article} {name} in this image?",
        "Please answer yes or no. Are there {name} in this image?",
    ),
    (
        "Is there {article} {name} in this image?",
        "Are there {name} in this image?",
    ),
    (
        "Please answer yes or no. Does the text imply {article} {name} "
        "is in the image?",
        "Please answer yes or no. Does the text imply {name} are in "
        "the image?",
    ),
)
INSTRUCTION = "Read the text and answer the question."
PROMPT_FORMAT = "Text: {context}\n\n" + INSTRUCTION + "\n\n{question}"
RESPONSE_TRUNCATION_CHARS = 1024
MAX_LENGTH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--caption",
        action="append",
        required=True,
        help="Caption JSONL path or glob; repeat for multiple generation shards.",
    )
    parser.add_argument("--coco_ann", required=True)
    parser.add_argument("--evaluator_model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def _paths(patterns: list[str]) -> list[Path]:
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(pattern)
        result.extend(Path(path) for path in matches)
    return sorted(set(result))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def throne_questions(categories: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(categories, key=lambda category: int(category["id"]))
    result = []
    for singular, plural in QUESTION_FORMATS:
        for category in ordered:
            raw_name = str(category["name"])
            if raw_name in {"skis", "scissors"}:
                result.append(plural.format(name=raw_name))
                continue
            name = "necktie" if raw_name == "tie" else raw_name
            # Preserve the release's case-sensitive check exactly. Objects365
            # names are title-cased, so this intentionally emits e.g.
            # "a Awning" rather than silently correcting the official prompt.
            article = "an" if name[0] in "aeiou" else "a"
            result.append(
                singular.format(article=article, name=name)
            )
    return result


def _captions(
    paths: list[Path], expected_ids: set[int]
) -> dict[int, str]:
    result: dict[int, str] = {}
    for path in paths:
        for row in _jsonl(path):
            if row.get("benchmark") != "throne-objects365":
                raise ValueError(
                    f"{path}: unexpected benchmark {row.get('benchmark')}"
                )
            image_id = int(row["sample_id"])
            if image_id in result:
                raise ValueError(f"Duplicate caption image ID {image_id}")
            result[image_id] = str(row["caption"])
    actual_ids = set(result)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Caption ID mismatch: missing={len(expected_ids-actual_ids)}, "
            f"extra={len(actual_ids-expected_ids)}"
        )
    return result


def _completed(path: Path, expected_answers: int) -> dict[int, list[str]]:
    if not path.is_file():
        return {}
    result: dict[int, list[str]] = {}
    for row in _jsonl(path):
        image_id = int(row["image_id"])
        answers = [str(answer) for answer in row["answers"]]
        if len(answers) != expected_answers:
            raise ValueError(
                f"Checkpoint image {image_id} has {len(answers)} answers"
            )
        if image_id in result:
            raise ValueError(f"Duplicate checkpoint image ID {image_id}")
        result[image_id] = answers
    return result


def _fingerprint(path: Path) -> dict[str, Any]:
    files = []
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "spiece.model",
        "model.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        candidate = path / name
        if candidate.is_file():
            files.append(
                {
                    "name": name,
                    "bytes": candidate.stat().st_size,
                    "sha256": (
                        _sha256(candidate)
                        if candidate.stat().st_size < 64 * 1024 * 1024
                        else None
                    ),
                }
            )
    return {"path": str(path.resolve()), "files": files}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, path)


def run_registered_aqa(
    args: argparse.Namespace,
    *,
    tokenizer=None,
    model=None,
) -> dict[str, Any]:
    """Run one caption-row/evaluator shard with optional model reuse."""

    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("Invalid shard")
    manifest_path = Path(args.manifest)
    manifest = _jsonl(manifest_path)
    ordered_ids = [int(row["sample_id"]) for row in manifest]
    if len(ordered_ids) != 5110 or len(set(ordered_ids)) != 5110:
        raise ValueError(
            f"Registered THRONE manifest must contain 5,110 unique rows, "
            f"found {len(ordered_ids)}/{len(set(ordered_ids))}"
        )
    caption_paths = _paths(args.caption)
    captions = _captions(caption_paths, set(ordered_ids))
    selected_ids = [
        image_id
        for index, image_id in enumerate(ordered_ids)
        if index % args.num_shards == args.shard_idx
    ]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    with open(os.devnull, "w") as sink:
        old_stdout = sys.stdout
        try:
            sys.stdout = sink
            coco = COCO(args.coco_ann)
        finally:
            sys.stdout = old_stdout
    questions = throne_questions(list(coco.cats.values()))
    if len(questions) != 1095:
        raise ValueError(
            f"Expected 1,095 registered questions, found {len(questions)}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path.with_suffix(".jsonl")
    completed = _completed(checkpoint, len(questions))
    extra = set(completed) - set(selected_ids)
    if extra:
        raise ValueError(
            f"Checkpoint contains {len(extra)} IDs outside this shard"
        )
    pending = [image_id for image_id in selected_ids if image_id not in completed]

    model_path = Path(args.evaluator_model_path)
    if tokenizer is None or model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

    started = time.perf_counter()
    with checkpoint.open("a") as handle, torch.inference_mode():
        for image_id in tqdm(pending, desc="throne-aqa"):
            context = captions[image_id][:RESPONSE_TRUNCATION_CHARS]
            prompts = [
                PROMPT_FORMAT.format(context=context, question=question)
                for question in questions
            ]
            answers = []
            for start in range(0, len(prompts), args.batch_size):
                batch = prompts[start : start + args.batch_size]
                # Deliberately omit attention_mask to preserve the released
                # evaluator's exact inference call.
                input_ids = tokenizer(
                    batch, padding=True, return_tensors="pt"
                ).input_ids.to(model.device)
                output_ids = model.generate(
                    input_ids,
                    use_cache=True,
                    max_length=MAX_LENGTH,
                    do_sample=False,
                )
                answers.extend(
                    tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )
                )
            if len(answers) != len(questions):
                raise RuntimeError(
                    f"Image {image_id}: expected {len(questions)} answers, "
                    f"got {len(answers)}"
                )
            handle.write(
                json.dumps(
                    {"image_id": image_id, "answers": answers}
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            completed[image_id] = answers

    if set(completed) != set(selected_ids):
        raise RuntimeError("AQA checkpoint is incomplete after generation")
    ordered_payload = {
        str(image_id): completed[image_id] for image_id in selected_ids
    }
    _atomic_json(output_path, ordered_payload)
    normalized = [
        str(answer).lower().strip()
        for answers in ordered_payload.values()
        for answer in answers
    ]
    metadata = {
        "protocol": "docs/iteration_protocols/image_caption_benchmark_matrix.md",
        "official_commit": "45badbb71c2b3e8d6619f6b45cccdac24dd20b46",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "caption_sources": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in caption_paths
        ],
        "evaluator": _fingerprint(model_path),
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
        "images": len(ordered_payload),
        "questions_per_image": len(questions),
        "question_formats": 3,
        "categories": 365,
        "instruction": INSTRUCTION,
        "response_truncation_chars": RESPONSE_TRUNCATION_CHARS,
        "max_length": MAX_LENGTH,
        "batch_size": args.batch_size,
        "attention_mask_omitted_to_match_release": True,
        "valid_answer_rate": (
            sum(answer in {"yes", "no"} for answer in normalized)
            / len(normalized)
        ),
        "yes_rate": (
            sum(answer == "yes" for answer in normalized)
            / len(normalized)
        ),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "output_sha256": _sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    _atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    run_registered_aqa(parse_args())


if __name__ == "__main__":
    main()
