#!/usr/bin/env python3
"""Aggregate registered THRONE AQA rows and paired image bootstraps."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO


METRIC_NAMES = (
    "classwise_F1",
    "classwise_F0.5",
    "classwise_P",
    "classwise_R",
    "micro_F1",
    "micro_F0.5",
    "micro_P",
    "micro_R",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--coco_ann", required=True)
    parser.add_argument(
        "--caption_row",
        action="append",
        required=True,
        metavar="METHOD=GLOB",
    )
    parser.add_argument(
        "--aqa_row",
        action="append",
        required=True,
        metavar="METHOD=GLOB",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_images", type=int, default=5110)
    parser.add_argument("--expected_evaluators", type=int, default=3)
    parser.add_argument("--expected_prompts", type=int, default=3)
    parser.add_argument("--voting_threshold", type=int, default=9)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap_chunk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _specs(values: list[str], *, require_vanilla: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=GLOB, got {value!r}")
        method, pattern = value.split("=", 1)
        if not method or not pattern or method in result:
            raise ValueError(f"Invalid or duplicate row {value!r}")
        result[method] = pattern
    if require_vanilla and "vanilla" not in result:
        raise ValueError("A vanilla row is required")
    return result


def _load_caption_row(
    pattern: str,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No caption files match {pattern!r}")
    by_id: dict[int, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            image_id = int(row["sample_id"])
            if image_id in by_id:
                raise ValueError(f"Duplicate image {image_id} in {pattern}")
            by_id[image_id] = row
    return by_id, [str(path.resolve()) for path in paths]


def _f_scores(
    precision: np.ndarray, recall: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2.0 * precision * recall / (precision + recall)
        f_half = (
            1.25 * precision * recall / (0.25 * precision + recall)
        )
    return f1, f_half


def metric_vector(
    predicted: np.ndarray,
    correct: np.ndarray,
    gt_valid: np.ndarray,
) -> np.ndarray:
    """Match the released classwise and micro aggregation exactly."""

    class_predicted = predicted.sum(axis=0, dtype=np.float64)
    class_correct = correct.sum(axis=0, dtype=np.float64)
    class_gt = gt_valid.sum(axis=0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_precision = class_correct / class_predicted
        class_recall = class_correct / class_gt
    class_f1, class_f_half = _f_scores(
        class_precision, class_recall
    )

    micro_precision = np.zeros(predicted.shape[0], dtype=np.float64)
    micro_recall = np.zeros(predicted.shape[0], dtype=np.float64)
    pred_count = predicted.sum(axis=1)
    gt_count = gt_valid.sum(axis=1)
    intersection = correct.sum(axis=1)
    usable = (pred_count > 0) & (gt_count > 0)
    micro_precision[usable] = intersection[usable] / pred_count[usable]
    micro_recall[usable] = intersection[usable] / gt_count[usable]
    micro_f1, micro_f_half = _f_scores(
        micro_precision, micro_recall
    )
    return np.asarray(
        [
            np.nanmean(class_f1),
            np.nanmean(class_f_half),
            np.nanmean(class_precision),
            np.nanmean(class_recall),
            np.nanmean(micro_f1),
            np.nanmean(micro_f_half),
            np.nanmean(micro_precision),
            np.nanmean(micro_recall),
        ],
        dtype=np.float64,
    )


def _weighted_metric_vectors(
    predicted: np.ndarray,
    correct: np.ndarray,
    gt_valid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Vectorized bootstrap statistics for one method and a weights chunk."""

    predicted_f = predicted.astype(np.float32, copy=False)
    correct_f = correct.astype(np.float32, copy=False)
    gt_f = gt_valid.astype(np.float32, copy=False)
    class_predicted = weights @ predicted_f
    class_correct = weights @ correct_f
    class_gt = weights @ gt_f
    with np.errstate(divide="ignore", invalid="ignore"):
        class_precision = class_correct / class_predicted
        class_recall = class_correct / class_gt
    class_f1, class_f_half = _f_scores(
        class_precision, class_recall
    )

    pred_count = predicted.sum(axis=1)
    gt_count = gt_valid.sum(axis=1)
    intersection = correct.sum(axis=1)
    micro_precision = np.zeros(predicted.shape[0], dtype=np.float32)
    micro_recall = np.zeros(predicted.shape[0], dtype=np.float32)
    usable = (pred_count > 0) & (gt_count > 0)
    micro_precision[usable] = intersection[usable] / pred_count[usable]
    micro_recall[usable] = intersection[usable] / gt_count[usable]
    micro_f1, micro_f_half = _f_scores(
        micro_precision, micro_recall
    )
    per_image = (
        micro_f1,
        micro_f_half,
        micro_precision,
        micro_recall,
    )
    micro_results = []
    for values in per_image:
        finite = np.isfinite(values).astype(np.float32)
        numerator = weights @ np.nan_to_num(values).astype(np.float32)
        denominator = weights @ finite
        micro_results.append(numerator / denominator)
    return np.stack(
        [
            np.nanmean(class_f1, axis=1),
            np.nanmean(class_f_half, axis=1),
            np.nanmean(class_precision, axis=1),
            np.nanmean(class_recall, axis=1),
            *micro_results,
        ],
        axis=1,
    )


def paired_bootstrap(
    candidate: tuple[np.ndarray, np.ndarray, np.ndarray],
    vanilla: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    repeats: int,
    chunk: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    observed = metric_vector(*candidate) - metric_vector(*vanilla)
    images = candidate[0].shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, len(METRIC_NAMES)), dtype=np.float64)
    probabilities = np.full(images, 1.0 / images)
    for start in range(0, repeats, chunk):
        size = min(chunk, repeats - start)
        weights = rng.multinomial(
            images, probabilities, size=size
        ).astype(np.float32)
        draws[start : start + size] = (
            _weighted_metric_vectors(*candidate, weights)
            - _weighted_metric_vectors(*vanilla, weights)
        )
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        name: {
            "delta_points": 100.0 * float(observed[index]),
            "ci95_low": 100.0 * float(low[index]),
            "ci95_high": 100.0 * float(high[index]),
        }
        for index, name in enumerate(METRIC_NAMES)
    }


def _load_aqa_votes(
    pattern: str,
    ordered_ids: list[int],
    *,
    categories: int,
    evaluators: int,
    prompts: int,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if len(paths) != evaluators:
        raise ValueError(
            f"Expected {evaluators} evaluator files for {pattern!r}, "
            f"found {len(paths)}"
        )
    votes = np.zeros((len(ordered_ids), categories), dtype=np.uint8)
    expected_ids = set(ordered_ids)
    audits = {}
    for path in paths:
        raw = json.loads(path.read_text())
        actual_ids = {int(key) for key in raw}
        if actual_ids != expected_ids:
            raise ValueError(
                f"{path}: AQA ID mismatch; missing="
                f"{sorted(expected_ids-actual_ids)}, extra="
                f"{sorted(actual_ids-expected_ids)}"
            )
        valid_answers = 0
        yes_answers = 0
        total_answers = 0
        prompt_agree = 0
        for image_index, image_id in enumerate(ordered_ids):
            answers = raw[str(image_id)] if str(image_id) in raw else raw[image_id]
            if len(answers) != prompts * categories:
                raise ValueError(
                    f"{path}: image {image_id} has {len(answers)} answers"
                )
            normalized = [str(answer).lower().strip() for answer in answers]
            yes = np.asarray(
                [answer == "yes" for answer in normalized],
                dtype=np.uint8,
            ).reshape(prompts, categories)
            valid_answers += sum(
                answer in {"yes", "no"} for answer in normalized
            )
            yes_answers += int(yes.sum())
            total_answers += len(normalized)
            prompt_agree += int(
                np.all(yes == yes[0:1], axis=0).sum()
            )
            votes[image_index] += yes.sum(axis=0).astype(np.uint8)
        audits[path.name] = {
            "valid_answer_rate": valid_answers / total_answers,
            "yes_rate": yes_answers / total_answers,
            "within_evaluator_prompt_agreement": (
                prompt_agree / (len(ordered_ids) * categories)
            ),
            "answers": total_answers,
        }
    return votes, audits, [str(path.resolve()) for path in paths]


def _ground_truth(
    coco: COCO,
    ordered_ids: list[int],
    category_id_to_index: dict[int, int],
) -> np.ndarray:
    result = np.zeros(
        (len(ordered_ids), len(category_id_to_index)), dtype=bool
    )
    for image_index, image_id in enumerate(ordered_ids):
        for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
            result[
                image_index,
                category_id_to_index[int(annotation["category_id"])],
            ] = True
    return result


def _mean(values: list[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def _repetition(caption: str) -> float:
    words = re.findall(r"\w+", caption.lower())
    return 1.0 - len(set(words)) / len(words) if words else 0.0


def _summarize(
    caption_rows: dict[int, dict[str, Any]],
    predicted: np.ndarray,
    correct: np.ndarray,
    gt_valid: np.ndarray,
    disqualified: np.ndarray,
    parser_audits: dict[str, Any],
) -> dict[str, Any]:
    scores = metric_vector(predicted, correct, gt_valid)
    rows = [caption_rows[image_id] for image_id in sorted(caption_rows)]
    return {
        **{
            name: 100.0 * float(scores[index])
            for index, name in enumerate(METRIC_NAMES)
        },
        "official_rounded": {
            name: round(100.0 * float(scores[index]), 3)
            for index, name in enumerate(METRIC_NAMES)
        },
        "images": len(rows),
        "valid_vote_coverage": 1.0 - float(disqualified.mean()),
        "disqualified_image_class_pairs": int(disqualified.sum()),
        "parser_audits": parser_audits,
        "mean_generated_tokens": _mean(
            [row.get("generated_tokens") for row in rows]
        ),
        "mean_content_tokens": _mean(
            [row.get("content_tokens") for row in rows]
        ),
        "mean_words": _mean([row.get("words") for row in rows]),
        "mean_token_repetition": float(
            np.mean([_repetition(str(row["caption"])) for row in rows])
        ),
        "premature_eos_rate": _mean(
            [int(bool(row.get("premature_eos", False))) for row in rows]
        ),
        "eos_termination_rate": _mean(
            [
                int(bool(row.get("terminated_by_eos", False)))
                for row in rows
            ]
        ),
        "max_token_termination_rate": _mean(
            [
                int(bool(row.get("hit_max_new_tokens", False)))
                for row in rows
            ]
        ),
        "mean_wall_seconds": _mean(
            [row.get("wall_seconds") for row in rows]
        ),
        "max_peak_memory_bytes": max(
            (
                int(row["peak_memory_bytes"])
                for row in rows
                if row.get("peak_memory_bytes") is not None
            ),
            default=None,
        ),
        "triggers": sum(len(row.get("actions", [])) for row in rows),
        "completed_rollbacks": sum(
            len(row.get("actions", [])) for row in rows
        ),
        "unresolved_triggers": sum(
            len(row.get("unresolved_triggers", [])) for row in rows
        ),
    }


def score(
    manifest_path: Path,
    coco_ann_path: Path,
    caption_patterns: dict[str, str],
    aqa_patterns: dict[str, str],
    *,
    expected_images: int,
    expected_evaluators: int,
    expected_prompts: int,
    voting_threshold: int,
    bootstrap: int,
    bootstrap_chunk: int,
    seed: int,
) -> dict[str, Any]:
    if set(caption_patterns) != set(aqa_patterns):
        raise ValueError("Caption and AQA method sets must match")
    manifest = _jsonl(manifest_path)
    ordered_ids = sorted(int(row["sample_id"]) for row in manifest)
    if len(manifest) != expected_images or len(set(ordered_ids)) != expected_images:
        raise ValueError(
            f"Expected {expected_images} unique manifest images"
        )
    coco = COCO(str(coco_ann_path))
    sorted_categories = sorted(coco.cats.values(), key=lambda row: row["id"])
    category_id_to_index = {
        int(category["id"]): index
        for index, category in enumerate(sorted_categories)
    }
    gt = _ground_truth(coco, ordered_ids, category_id_to_index)
    required_votes = expected_evaluators * expected_prompts
    if voting_threshold != required_votes:
        raise ValueError(
            "Registered THRONE uses unanimous threshold "
            f"{required_votes}, got {voting_threshold}"
        )

    captions: dict[str, dict[int, dict[str, Any]]] = {}
    caption_sources: dict[str, list[str]] = {}
    aqa_sources: dict[str, list[str]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    votes_by_method: dict[str, np.ndarray] = {}
    for method in caption_patterns:
        captions[method], caption_sources[method] = _load_caption_row(
            caption_patterns[method]
        )
        if set(captions[method]) != set(ordered_ids):
            raise ValueError(f"{method}: incomplete caption row")
        votes, parser_audits, aqa_sources[method] = _load_aqa_votes(
            aqa_patterns[method],
            ordered_ids,
            categories=len(sorted_categories),
            evaluators=expected_evaluators,
            prompts=expected_prompts,
        )
        votes_by_method[method] = votes
        predicted = votes >= voting_threshold
        disqualified = (votes > 0) & (votes < voting_threshold)
        gt_valid = gt & ~disqualified
        correct = predicted & gt
        arrays[method] = (predicted, correct, gt_valid)
        summaries[method] = _summarize(
            captions[method],
            predicted,
            correct,
            gt_valid,
            disqualified,
            parser_audits,
        )

    # Attribute evaluator-visible removals by the shared vanilla-vs-ours
    # class-set difference, avoiding unsupported word-to-class heuristics.
    if "ours" in summaries:
        removed = (
            votes_by_method["vanilla"] >= voting_threshold
        ) & ~(votes_by_method["ours"] >= voting_threshold)
        removed_hallucinated = removed & ~gt
        removed_grounded = removed & gt
        count_h = int(removed_hallucinated.sum())
        count_g = int(removed_grounded.sum())
        summaries["ours"].update(
            {
                "evaluator_removed_classes": count_h + count_g,
                "evaluator_matched_hallucinated_removals": count_h,
                "evaluator_matched_grounded_removals": count_g,
                "removal_precision": (
                    count_h / (count_h + count_g)
                    if count_h + count_g
                    else None
                ),
            }
        )

    paired = {
        method: paired_bootstrap(
            arrays[method],
            arrays["vanilla"],
            repeats=bootstrap,
            chunk=bootstrap_chunk,
            seed=seed,
        )
        for method in arrays
        if method != "vanilla"
    }
    ours_gate = None
    if "ours" in paired:
        ours_gate = {
            "classwise_F0.5_ci_excludes_zero_positive": (
                paired["ours"]["classwise_F0.5"]["ci95_low"] > 0.0
            ),
            "classwise_recall_ci_not_exclusively_negative": (
                paired["ours"]["classwise_R"]["ci95_high"] >= 0.0
            ),
        }
    return {
        "protocol": "docs/iteration_protocols/image_caption_benchmark_matrix.md",
        "benchmark": "THRONE-Objects365",
        "official_commit": "45badbb71c2b3e8d6619f6b45cccdac24dd20b46",
        "manifest": str(manifest_path.resolve()),
        "coco_ann": str(coco_ann_path.resolve()),
        "images": expected_images,
        "categories": len(sorted_categories),
        "evaluator_models": expected_evaluators,
        "prompt_formats": expected_prompts,
        "voting_threshold": voting_threshold,
        "caption_sources": caption_sources,
        "aqa_sources": aqa_sources,
        "rows": summaries,
        "paired_vs_vanilla": paired,
        "ours_gate_components": ours_gate,
        "bootstrap": {
            "unit": "image",
            "paired": True,
            "repeats": bootstrap,
            "seed": seed,
        },
    }


def main() -> None:
    args = parse_args()
    payload = score(
        Path(args.manifest),
        Path(args.coco_ann),
        _specs(args.caption_row, require_vanilla=True),
        _specs(args.aqa_row, require_vanilla=True),
        expected_images=args.expected_images,
        expected_evaluators=args.expected_evaluators,
        expected_prompts=args.expected_prompts,
        voting_threshold=args.voting_threshold,
        bootstrap=args.bootstrap,
        bootstrap_chunk=args.bootstrap_chunk,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                method: {
                    key: row[key]
                    for key in (
                        "classwise_F0.5",
                        "classwise_P",
                        "classwise_R",
                        "micro_F0.5",
                        "mean_words",
                    )
                }
                for method, row in payload["rows"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
