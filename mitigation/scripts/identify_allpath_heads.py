#!/usr/bin/env python3
"""Identify AllPath format/image heads on disjoint ResampledCHAIR images."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "detection" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "detection" / "src"))

from sinkdetect.chair import evaluate_chair

from mitigation.src.allpath_identification import (
    aggregate_head_scores,
    allpath_scores,
    capture_allpath_replay,
    first_chair_mentions,
    selected_head_payload,
    visual_instance_mask,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--logit_chunk_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--resampled_source",
        default="/tmp/sinkdetect-allpath-audit/more_benchmarks.py",
    )
    parser.add_argument(
        "--evaluation_source",
        default="/tmp/sinkdetect-allpath-audit/playground/benchmarks.py",
    )
    parser.add_argument(
        "--coco_root",
        default="/home/chenguanxu/common_dataset/coco-2014-dataset",
    )
    parser.add_argument(
        "--chair_pkl",
        default="/home/chenguanxu/nas/VLM/Hallucination/pas/data/chair_coco.pkl",
    )
    parser.add_argument(
        "--pas_src",
        default="/home/chenguanxu/nas/VLM/Hallucination/pas/src",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_ids(path: str | Path, class_name: str) -> list[int]:
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "fixed_500"
                for target in statement.targets
            ):
                continue
            values = [int(value) for value in ast.literal_eval(statement.value)]
            if len(values) != 500 or len(set(values)) != 500:
                raise ValueError(f"Invalid {class_name}.fixed_500")
            return values
    raise ValueError(f"Could not find {class_name}.fixed_500 in {path}")


def completed(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with path.open() as handle:
        return {
            int(json.loads(line)["image_id"])
            for line in handle
            if line.strip()
        }


def instance_mask(
    coco: COCO, category_by_name: dict[str, int], image_id: int, word: str
) -> np.ndarray | None:
    category = category_by_name.get(word)
    if category is None:
        return None
    annotations = coco.loadAnns(
        coco.getAnnIds(imgIds=[int(image_id)], catIds=[int(category)])
    )
    result = None
    for annotation in annotations:
        value = coco.annToMask(annotation).astype(bool)
        result = value if result is None else np.logical_or(result, value)
    return result


def zero_summary(layers: int, heads: int) -> dict[str, Any]:
    zeros = np.zeros((layers, heads), dtype=np.float64).tolist()
    return {
        "format_safe_sum": zeros,
        "format_hallu_sum": zeros,
        "image_safe_sum": zeros,
        "image_hallu_sum": zeros,
        "format_safe_count": 0,
        "format_hallu_count": 0,
        "image_safe_count": 0,
        "image_hallu_count": 0,
    }


def category_summary(
    format_values: np.ndarray,
    format_safe: list[bool],
    image_values: np.ndarray,
    image_safe: list[bool],
) -> dict[str, Any]:
    shape = format_values.shape[1:]

    def total(values: np.ndarray, labels: list[bool], selected: bool):
        indices = [
            index for index, label in enumerate(labels)
            if bool(label) == selected
        ]
        if not indices:
            return np.zeros(shape, dtype=np.float64)
        return values[indices].astype(np.float64).sum(axis=0)

    return {
        "format_safe_sum": total(
            format_values, format_safe, True
        ).tolist(),
        "format_hallu_sum": total(
            format_values, format_safe, False
        ).tolist(),
        "image_safe_sum": total(
            image_values, image_safe, True
        ).tolist(),
        "image_hallu_sum": total(
            image_values, image_safe, False
        ).tolist(),
        "format_safe_count": int(sum(format_safe)),
        "format_hallu_count": int(len(format_safe) - sum(format_safe)),
        "image_safe_count": int(sum(image_safe)),
        "image_hallu_count": int(len(image_safe) - sum(image_safe)),
    }


def aggregate(details_path: Path) -> dict[str, Any]:
    sums = {
        key: None
        for key in (
            "format_safe_sum",
            "format_hallu_sum",
            "image_safe_sum",
            "image_hallu_sum",
        )
    }
    counts = {
        key: 0
        for key in (
            "format_safe_count",
            "format_hallu_count",
            "image_safe_count",
            "image_hallu_count",
        )
    }
    images = 0
    mentions = 0
    with details_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            images += 1
            mentions += len(row["mentions"])
            for key in sums:
                value = np.asarray(row[key], dtype=np.float64)
                sums[key] = value if sums[key] is None else sums[key] + value
            for key in counts:
                counts[key] += int(row[key])
    if images == 0 or any(value is None for value in sums.values()):
        raise ValueError("No AllPath identification details to aggregate")
    return {
        **sums,
        **counts,
        "images": images,
        "mentions": mentions,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, args.pas_src)
    evaluator = pickle.load(open(args.chair_pkl, "rb"))

    development_ids = fixed_ids(
        args.resampled_source, "ResampledCHAIR"
    )[: args.limit]
    evaluation_ids = set(fixed_ids(args.evaluation_source, "CHAIR"))
    overlap = sorted(set(development_ids) & evaluation_ids)
    if overlap:
        raise RuntimeError(
            f"AllPath development/evaluation leakage: {overlap[:10]}"
        )

    coco_root = Path(args.coco_root)
    coco_annotation = coco_root / "annotations" / "instances_val2014.json"
    coco = COCO(str(coco_annotation))
    category_by_name = {
        str(category["name"]): int(category["id"])
        for category in coco.loadCats(coco.getCatIds())
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path = output_path.with_suffix(".details.jsonl")
    done = completed(details_path)
    pending = [
        image_id for image_id in development_ids if image_id not in done
    ]
    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"{args.kind}: {len(pending)} pending of {len(development_ids)} "
        f"disjoint ResampledCHAIR images"
    )

    with details_path.open("a") as output:
        for image_id in tqdm(pending, desc=f"allpath-{args.kind}"):
            started = time.perf_counter()
            image_path = (
                coco_root
                / "val2014"
                / f"COCO_val2014_{image_id:012d}.jpg"
            )
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                inputs = captioner.prepare_image(
                    image, "Please describe this image in detail."
                )
            prompt_length = int(inputs["input_ids"].shape[-1])
            full_ids = captioner.generate(inputs)
            generated_ids = full_ids[0, prompt_length:].tolist()
            caption = captioner.decode_continuation(
                full_ids, prompt_length
            )
            try:
                evaluations, _ = evaluate_chair(
                    evaluator,
                    image_ids=[image_id],
                    captions=[caption],
                )
                mentions = first_chair_mentions(
                    captioner.tokenizer,
                    evaluator,
                    evaluations[0],
                    generated_ids,
                )
            except ZeroDivisionError:
                mentions = []
            anchors = [
                prompt_length + int(mention["rel_anchor"])
                for mention in mentions
            ]
            keep = [
                anchor + 1 < full_ids.shape[-1] for anchor in anchors
            ]
            mentions = [
                mention
                for mention, selected in zip(mentions, keep)
                if selected
            ]
            anchors = [
                anchor
                for anchor, selected in zip(anchors, keep)
                if selected
            ]

            summary = zero_summary(
                captioner.decoder.layers, captioner.decoder.heads
            )
            if mentions:
                visual_positions = captioner.visual_positions(
                    {"input_ids": full_ids}, media_type="image"
                )
                raw_masks = [
                    instance_mask(
                        coco,
                        category_by_name,
                        image_id,
                        mention["word"],
                    )
                    for mention in mentions
                ]
                token_masks = [
                    (
                        None
                        if mask is None
                        else visual_instance_mask(
                            args.kind,
                            captioner.model,
                            inputs,
                            mask,
                            int(visual_positions.numel()),
                        )
                    )
                    for mask in raw_masks
                ]
                with capture_allpath_replay(
                    captioner,
                    full_ids,
                    inputs,
                    anchors,
                    visual_positions,
                ) as captured:
                    pass
                target_ids = [
                    int(full_ids[0, anchor + 1]) for anchor in anchors
                ]
                format_values, image_values, image_safe = allpath_scores(
                    captured["replay"],
                    captioner.model.lm_head,
                    target_ids,
                    token_masks,
                    logit_chunk_size=args.logit_chunk_size,
                )
                format_safe = [
                    not bool(mention["hallucinated"])
                    for mention in mentions
                ]
                summary = category_summary(
                    format_values,
                    format_safe,
                    image_values,
                    image_safe,
                )
            row = {
                "image_id": int(image_id),
                "caption": caption,
                "mentions": [
                    {
                        "word": mention["word"],
                        "surface": mention["surface"],
                        "rel_anchor": int(mention["rel_anchor"]),
                        "format_safe": not bool(
                            mention["hallucinated"]
                        ),
                    }
                    for mention in mentions
                ],
                **summary,
                "wall_seconds": time.perf_counter() - started,
            }
            output.write(json.dumps(row) + "\n")
            output.flush()
            os.fsync(output.fileno())
            del inputs, full_ids
            torch.cuda.empty_cache()

    totals = aggregate(details_path)
    format_scores = aggregate_head_scores(
        totals["format_safe_sum"],
        totals["format_safe_count"],
        totals["format_hallu_sum"],
        totals["format_hallu_count"],
    )
    image_scores = aggregate_head_scores(
        totals["image_safe_sum"],
        totals["image_safe_count"],
        totals["image_hallu_sum"],
        totals["image_hallu_count"],
    )
    payload = selected_head_payload(format_scores, image_scores)
    payload.update(
        {
            "schema": "sinkdetect-allpath-heads-v1",
            "model": captioner.metadata(),
            "identification": {
                "rule": "official AllPath ResampledCHAIR T2T/I2T",
                "formula_port": (
                    "teacher-forced residual + single projected head; "
                    "exact target log-probability"
                ),
                "image_rule": (
                    "instance-mask attention for segmented mentions; "
                    "whole visual block otherwise"
                ),
                "development_images": totals["images"],
                "mentions": totals["mentions"],
                "format_safe": totals["format_safe_count"],
                "format_hallucinated": totals["format_hallu_count"],
                "image_safe": totals["image_safe_count"],
                "image_hallucinated": totals["image_hallu_count"],
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "logit_chunk_size": args.logit_chunk_size,
                "resampled_source": str(
                    Path(args.resampled_source).resolve()
                ),
                "resampled_source_sha256": sha256_file(
                    args.resampled_source
                ),
                "evaluation_source": str(
                    Path(args.evaluation_source).resolve()
                ),
                "evaluation_source_sha256": sha256_file(
                    args.evaluation_source
                ),
                "development_evaluation_overlap": 0,
                "coco_instances_sha256": sha256_file(coco_annotation),
                "details": str(details_path.resolve()),
            },
        }
    )
    atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "images": totals["images"],
                "mentions": totals["mentions"],
                "counts": payload["counts"],
                "unique_good_heads": sum(
                    len(value) for value in payload["good_heads"].values()
                ),
                "unique_hallu_heads": sum(
                    len(value) for value in payload["hallu_heads"].values()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
