#!/usr/bin/env python3
"""Score complete registered CHAIR rows with paired image bootstrap audits."""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "detection" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "detection" / "src"))

from sinkdetect.chair import evaluate_chair


METRIC_NAMES = ("CHAIRs", "CHAIRi", "Recall", "Precision", "F1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chair_pkl", required=True)
    parser.add_argument("--pas_src", required=True)
    parser.add_argument(
        "--row", action="append", required=True, metavar="METHOD=GLOB"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_images", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_specs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected METHOD=GLOB, got {value!r}")
        method, pattern = value.split("=", 1)
        if not method or not pattern or method in result:
            raise ValueError(f"Invalid or duplicate row {value!r}")
        result[method] = pattern
    if "vanilla" not in result:
        raise ValueError("A vanilla row is required")
    return result


def _load_row(pattern: str) -> tuple[dict[int, dict[str, Any]], list[str]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No files match {pattern!r}")
    by_id: dict[int, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            image_id = int(row["sample_id"])
            if image_id in by_id:
                raise ValueError(f"Duplicate image {image_id} in {pattern}")
            by_id[image_id] = row
    return by_id, [str(path.resolve()) for path in paths]


def _components(sentences: list[dict[str, Any]]) -> np.ndarray:
    """Return additive [chairs_n/d, chairi_n/d, recall_n/d, precision_n/d]."""

    result = np.zeros((len(sentences), 8), dtype=np.float64)
    for index, sentence in enumerate(sentences):
        generated = list(sentence["mscoco_generated_words"])
        ground = set(sentence["mscoco_gt_words"])
        hallucinated = list(sentence["mscoco_hallucinated_words"])
        recalled = ground.intersection(generated)
        result[index] = (
            int(bool(hallucinated)),
            1,
            len(hallucinated),
            len(generated),
            len(recalled),
            len(ground),
            len(recalled),
            len(set(generated)),
        )
    return result


def metric_vector(components: np.ndarray) -> np.ndarray:
    totals = components.sum(axis=0)

    def ratio(numerator: int, denominator: int) -> float:
        return (
            float(totals[numerator] / totals[denominator])
            if totals[denominator] > 0
            else 0.0
        )

    chairs = ratio(0, 1)
    chairi = ratio(2, 3)
    recall = ratio(4, 5)
    precision = ratio(6, 7)
    f1 = (
        2.0 * recall * precision / (recall + precision)
        if recall + precision > 0
        else 0.0
    )
    return np.asarray(
        [chairs, chairi, recall, precision, f1], dtype=np.float64
    )


def paired_bootstrap(
    candidate: np.ndarray,
    vanilla: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if candidate.shape != vanilla.shape:
        raise ValueError("Paired CHAIR components are not aligned")
    observed = metric_vector(candidate) - metric_vector(vanilla)
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, len(METRIC_NAMES)), dtype=np.float64)
    for start in range(0, repeats, 500):
        size = min(500, repeats - start)
        indices = rng.integers(
            0, candidate.shape[0], size=(size, candidate.shape[0])
        )
        for offset, sample in enumerate(indices):
            draws[start + offset] = (
                metric_vector(candidate[sample])
                - metric_vector(vanilla[sample])
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


def _mean(values: list[float | int | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def _repetition(caption: str) -> float:
    words = re.findall(r"\w+", caption.lower())
    return 1.0 - len(set(words)) / len(words) if words else 0.0


def _canonical_action(evaluator: Any, word: str) -> str | None:
    _, node_words, _, _ = evaluator.caption_to_words(word)
    return node_words[0] if node_words else None


def _summarize(
    generated: dict[int, dict[str, Any]],
    sentences: list[dict[str, Any]],
    evaluator: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    ordered_sentences = sorted(sentences, key=lambda row: int(row["image_id"]))
    components = _components(ordered_sentences)
    metric_values = metric_vector(components)
    sentence_by_id = {
        int(sentence["image_id"]): sentence for sentence in ordered_sentences
    }
    action_records = []
    hallucinated_removals = 0
    grounded_removals = 0
    unscorable_removals = 0
    triggers = 0
    unresolved = 0
    for image_id in sorted(generated):
        row = generated[image_id]
        actions = list(row.get("actions", []))
        triggers += len(actions)
        unresolved += len(row.get("unresolved_triggers", []))
        ground = set(sentence_by_id[image_id]["mscoco_gt_words"])
        for action in actions:
            canonical = _canonical_action(evaluator, str(action["word"]))
            if canonical is None:
                status = "outside_chair_vocabulary"
                unscorable_removals += 1
            elif canonical in ground:
                status = "grounded"
                grounded_removals += 1
            else:
                status = "hallucinated"
                hallucinated_removals += 1
            action_records.append(
                {
                    "image_id": image_id,
                    "word": action["word"],
                    "canonical": canonical,
                    "status": status,
                    "replacement": action.get("replacement"),
                }
            )
    scored_actions = hallucinated_removals + grounded_removals
    rows = [generated[image_id] for image_id in sorted(generated)]
    summary = {
        **{
            name: 100.0 * float(metric_values[index])
            for index, name in enumerate(METRIC_NAMES)
        },
        "images": len(rows),
        "mean_generated_tokens": _mean(
            [row.get("generated_tokens") for row in rows]
        ),
        "mean_content_tokens": _mean(
            [row.get("content_tokens") for row in rows]
        ),
        "mean_words": _mean([row.get("words") for row in rows]),
        "mean_unique_coco_objects": float(
            np.mean(
                [
                    len(set(sentence_by_id[image_id]["mscoco_generated_words"]))
                    for image_id in sorted(generated)
                ]
            )
        ),
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
        "triggers": triggers,
        "completed_rollbacks": len(action_records),
        "unresolved_triggers": unresolved,
        "evaluator_scored_removals": scored_actions,
        "evaluator_unscorable_removals": unscorable_removals,
        "evaluator_matched_hallucinated_removals": hallucinated_removals,
        "evaluator_matched_grounded_removals": grounded_removals,
        "removal_precision": (
            hallucinated_removals / scored_actions
            if scored_actions
            else None
        ),
        "action_records": action_records,
    }
    return summary, components


def score(
    manifest_path: Path,
    chair_pkl_path: Path,
    pas_src: Path,
    row_patterns: dict[str, str],
    *,
    expected_images: int,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(pas_src))
    with chair_pkl_path.open("rb") as handle:
        evaluator = pickle.load(handle)
    manifest = _jsonl(manifest_path)
    expected_ids = {int(row["sample_id"]) for row in manifest}
    if len(manifest) != expected_images or len(expected_ids) != expected_images:
        raise ValueError(
            f"Expected {expected_images} unique manifest rows, "
            f"found {len(manifest)}/{len(expected_ids)}"
        )
    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    sources: dict[str, list[str]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    components: dict[str, np.ndarray] = {}
    for method, pattern in row_patterns.items():
        loaded[method], sources[method] = _load_row(pattern)
        actual_ids = set(loaded[method])
        if actual_ids != expected_ids:
            raise ValueError(
                f"{method} ID mismatch: missing={sorted(expected_ids-actual_ids)}, "
                f"extra={sorted(actual_ids-expected_ids)}"
            )
        ordered_ids = sorted(expected_ids)
        sentences, official = evaluate_chair(
            evaluator,
            image_ids=ordered_ids,
            captions=[
                str(loaded[method][image_id]["caption"])
                for image_id in ordered_ids
            ],
        )
        summaries[method], components[method] = _summarize(
            loaded[method], sentences, evaluator
        )
        calculated = metric_vector(components[method])
        for index, name in enumerate(METRIC_NAMES):
            if not np.isclose(float(official[name]), calculated[index]):
                raise ValueError(
                    f"{method} {name} disagrees with official evaluator: "
                    f"{official[name]} vs {calculated[index]}"
                )

    paired = {
        method: paired_bootstrap(
            components[method],
            components["vanilla"],
            repeats=bootstrap,
            seed=seed,
        )
        for method in loaded
        if method != "vanilla"
    }
    ours_gate = None
    if "ours" in paired:
        ours_gate = {
            "CHAIRs_ci_excludes_zero_improvement": (
                paired["ours"]["CHAIRs"]["ci95_high"] < 0.0
            ),
            "CHAIRi_ci_excludes_zero_improvement": (
                paired["ours"]["CHAIRi"]["ci95_high"] < 0.0
            ),
            "recall_ci_not_exclusively_negative": (
                paired["ours"]["Recall"]["ci95_high"] >= 0.0
            ),
        }
    return {
        "protocol": "docs/iteration_protocols/image_caption_benchmark_matrix.md",
        "benchmark": "CHAIR (COCO 2014)",
        "manifest": str(manifest_path.resolve()),
        "chair_pkl": str(chair_pkl_path.resolve()),
        "images": expected_images,
        "sources": sources,
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
        Path(args.chair_pkl),
        Path(args.pas_src),
        _row_specs(args.row),
        expected_images=args.expected_images,
        bootstrap=args.bootstrap,
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
                    for key in ("CHAIRs", "CHAIRi", "Recall", "mean_words")
                }
                for method, row in payload["rows"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
