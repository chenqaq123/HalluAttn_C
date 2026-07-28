#!/usr/bin/env python3
"""Score complete registered AMBER-generative rows with paired bootstrap."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import spacy
from nltk.stem import WordNetLemmatizer


METRIC_NAMES = ("CHAIR", "Cover", "Hal", "Cog")
EPSILON = 0.001  # exact released metrics.txt initialization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--word_association", required=True)
    parser.add_argument("--safe_words", required=True)
    parser.add_argument(
        "--row", action="append", required=True, metavar="METHOD=GLOB"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_images", type=int, default=1004)
    parser.add_argument("--similarity_score", type=float, default=0.8)
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


class AmberGenerativeMetric:
    """Per-image form of AMBER commit 534bab's released generative scorer."""

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        association: dict[str, list[str]],
        global_safe_words: set[str],
        *,
        similarity_score: float,
    ) -> None:
        self.annotations = {
            int(row["id"]): row
            for row in annotations
            if row["type"] == "generative"
        }
        self.association = association
        self.global_safe_words = global_safe_words
        self.similarity_score = similarity_score
        self.nlp = spacy.load("en_core_web_lg")
        self.lemmatizer = WordNetLemmatizer()
        self.hallucination_words = {
            value
            for word, related in association.items()
            for value in [word, *related]
        }

    def extract_nouns(self, text: str) -> list[str]:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return [
            self.lemmatizer.lemmatize(word)
            for word, pos in tagged
            if pos.startswith("NN")
        ]

    def similar(self, first: str, second: str) -> bool:
        return (
            self.nlp(first).similarity(self.nlp(second))
            > self.similarity_score
        )

    @staticmethod
    def _mark_exact(
        noun: str,
        words: list[str],
        indices: list[int],
        base_length: int,
    ) -> bool:
        for position, candidate in enumerate(words):
            if noun != candidate:
                continue
            if position < len(indices) - base_length:
                indices[indices[position] + len(indices) - base_length] = 1
            else:
                indices[position] = 1
            return True
        return False

    @staticmethod
    def _mark_position(
        position: int, indices: list[int], base_length: int
    ) -> None:
        if position < len(indices) - base_length:
            indices[indices[position] + len(indices) - base_length] = 1
        else:
            indices[position] = 1

    def evaluate(self, image_id: int, caption: str) -> dict[str, Any]:
        annotation = self.annotations[image_id]
        nouns = self.extract_nouns(caption)
        selected_nouns = [
            noun for noun in nouns if noun in self.hallucination_words
        ]
        safe_words: list[str] = []
        safe_indices: list[int] = []
        for index, word in enumerate(annotation["truth"]):
            safe_words.extend(self.association[word])
            safe_indices.extend([index] * len(self.association[word]))
        hallu_words: list[str] = []
        hallu_indices: list[int] = []
        for index, word in enumerate(annotation["hallu"]):
            hallu_words.extend(self.association[word])
            hallu_indices.extend([index] * len(self.association[word]))
        safe_words.extend(annotation["truth"])
        safe_length = len(annotation["truth"])
        safe_indices.extend([0] * safe_length)
        hallu_words.extend(annotation["hallu"])
        hallu_length = len(annotation["hallu"])
        hallu_indices.extend([0] * hallu_length)
        unsafe_flags = [0] * len(selected_nouns)

        for noun_index, noun in enumerate(selected_nouns):
            if noun in self.global_safe_words:
                continue
            if self._mark_exact(
                noun, safe_words, safe_indices, safe_length
            ):
                continue
            self._mark_exact(
                noun, hallu_words, hallu_indices, hallu_length
            )
            for position, candidate in enumerate(hallu_words):
                if self.similar(noun, candidate):
                    self._mark_position(
                        position, hallu_indices, hallu_length
                    )
                    break
            safe_match = False
            for position, candidate in enumerate(safe_words):
                if self.similar(noun, candidate):
                    safe_match = True
                    self._mark_position(
                        position, safe_indices, safe_length
                    )
                    break
            if not safe_match:
                unsafe_flags[noun_index] = 1

        return {
            "image_id": image_id,
            "nouns": nouns,
            "selected_nouns": selected_nouns,
            "chair_n": sum(unsafe_flags),
            "chair_d": len(unsafe_flags),
            "cover_n": sum(safe_indices[-safe_length:]),
            "cover_d": safe_length,
            "cog_n": sum(hallu_indices[-hallu_length:]),
            "cog_d": hallu_length,
            "non_hallu_n": int(sum(unsafe_flags) == 0),
            "non_hallu_d": 1,
        }

    def classify_action(self, image_id: int, word: str) -> dict[str, Any]:
        annotation = self.annotations[image_id]
        nouns = self.extract_nouns(word)
        noun = nouns[0] if nouns else self.lemmatizer.lemmatize(word)
        if noun not in self.hallucination_words:
            status = "outside_amber_vocabulary"
        elif noun in self.global_safe_words:
            status = "global_safe"
        else:
            safe_words = [
                value
                for truth in annotation["truth"]
                for value in [*self.association[truth], truth]
            ]
            hallu_words = [
                value
                for hallu in annotation["hallu"]
                for value in [*self.association[hallu], hallu]
            ]
            if noun in safe_words or any(
                self.similar(noun, candidate) for candidate in safe_words
            ):
                status = "grounded"
            elif noun in hallu_words or any(
                self.similar(noun, candidate) for candidate in hallu_words
            ):
                status = "hallucinated"
            else:
                # This is exactly AMBER's safe_flag fallback for a selected noun.
                status = "hallucinated"
        return {"noun": noun, "status": status}


def components(details: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [
                row["chair_n"],
                row["chair_d"],
                row["cover_n"],
                row["cover_d"],
                row["cog_n"],
                row["cog_d"],
                row["non_hallu_n"],
                row["non_hallu_d"],
            ]
            for row in details
        ],
        dtype=np.float64,
    )


def metric_vector(values: np.ndarray, *, epsilon: float = EPSILON) -> np.ndarray:
    totals = values.sum(axis=0)

    def ratio(numerator: int, denominator: int) -> float:
        return float(totals[numerator] / (totals[denominator] + epsilon))

    return np.asarray(
        [
            ratio(0, 1),
            ratio(2, 3),
            1.0 - ratio(6, 7),
            ratio(4, 5),
        ],
        dtype=np.float64,
    )


def paired_bootstrap(
    candidate: np.ndarray,
    vanilla: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if candidate.shape != vanilla.shape:
        raise ValueError("Paired AMBER components are not aligned")
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


def summarize(
    rows: dict[int, dict[str, Any]],
    details: list[dict[str, Any]],
    metric: AmberGenerativeMetric,
) -> tuple[dict[str, Any], np.ndarray]:
    values = components(details)
    scores = metric_vector(values)
    actions = []
    hallucinated = grounded = unscorable = triggers = unresolved = 0
    for image_id in sorted(rows):
        row = rows[image_id]
        triggers += len(row.get("actions", []))
        unresolved += len(row.get("unresolved_triggers", []))
        for action in row.get("actions", []):
            classification = metric.classify_action(
                image_id, str(action["word"])
            )
            status = classification["status"]
            if status == "hallucinated":
                hallucinated += 1
            elif status == "grounded":
                grounded += 1
            else:
                unscorable += 1
            actions.append(
                {
                    "image_id": image_id,
                    "word": action["word"],
                    **classification,
                    "replacement": action.get("replacement"),
                }
            )
    scored = hallucinated + grounded
    generated = [rows[image_id] for image_id in sorted(rows)]
    summary = {
        **{
            name: 100.0 * float(scores[index])
            for index, name in enumerate(METRIC_NAMES)
        },
        "official_rounded": {
            name: round(100.0 * float(scores[index]), 1)
            for index, name in enumerate(METRIC_NAMES)
        },
        "images": len(generated),
        "mean_generated_tokens": _mean(
            [row.get("generated_tokens") for row in generated]
        ),
        "mean_content_tokens": _mean(
            [row.get("content_tokens") for row in generated]
        ),
        "mean_words": _mean([row.get("words") for row in generated]),
        "mean_selected_object_nouns": float(
            np.mean([len(row["selected_nouns"]) for row in details])
        ),
        "mean_token_repetition": float(
            np.mean([_repetition(str(row["caption"])) for row in generated])
        ),
        "premature_eos_rate": _mean(
            [
                int(bool(row.get("premature_eos", False)))
                for row in generated
            ]
        ),
        "eos_termination_rate": _mean(
            [
                int(bool(row.get("terminated_by_eos", False)))
                for row in generated
            ]
        ),
        "max_token_termination_rate": _mean(
            [
                int(bool(row.get("hit_max_new_tokens", False)))
                for row in generated
            ]
        ),
        "mean_wall_seconds": _mean(
            [row.get("wall_seconds") for row in generated]
        ),
        "max_peak_memory_bytes": max(
            (
                int(row["peak_memory_bytes"])
                for row in generated
                if row.get("peak_memory_bytes") is not None
            ),
            default=None,
        ),
        "triggers": triggers,
        "completed_rollbacks": len(actions),
        "unresolved_triggers": unresolved,
        "evaluator_scored_removals": scored,
        "evaluator_unscorable_removals": unscorable,
        "evaluator_matched_hallucinated_removals": hallucinated,
        "evaluator_matched_grounded_removals": grounded,
        "removal_precision": hallucinated / scored if scored else None,
        "action_records": actions,
    }
    return summary, values


def score(
    manifest_path: Path,
    annotation_path: Path,
    association_path: Path,
    safe_words_path: Path,
    row_patterns: dict[str, str],
    *,
    expected_images: int,
    similarity_score: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    annotations = json.loads(annotation_path.read_text())
    association = json.loads(association_path.read_text())
    safe_words = {
        line.strip()
        for line in safe_words_path.read_text().splitlines()
        if line.strip()
    }
    metric = AmberGenerativeMetric(
        annotations,
        association,
        safe_words,
        similarity_score=similarity_score,
    )
    manifest = _jsonl(manifest_path)
    expected_ids = {int(row["sample_id"]) for row in manifest}
    if len(manifest) != expected_images or len(expected_ids) != expected_images:
        raise ValueError(
            f"Expected {expected_images} unique manifest rows, "
            f"found {len(manifest)}/{len(expected_ids)}"
        )
    if expected_ids != set(metric.annotations):
        raise ValueError("Manifest IDs do not equal AMBER generative IDs")

    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    sources: dict[str, list[str]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    values: dict[str, np.ndarray] = {}
    for method, pattern in row_patterns.items():
        loaded[method], sources[method] = _load_row(pattern)
        actual_ids = set(loaded[method])
        if actual_ids != expected_ids:
            raise ValueError(
                f"{method} ID mismatch: missing={sorted(expected_ids-actual_ids)}, "
                f"extra={sorted(actual_ids-expected_ids)}"
            )
        details = [
            metric.evaluate(
                image_id, str(loaded[method][image_id]["caption"])
            )
            for image_id in sorted(expected_ids)
        ]
        summaries[method], values[method] = summarize(
            loaded[method], details, metric
        )

    paired = {
        method: paired_bootstrap(
            values[method],
            values["vanilla"],
            repeats=bootstrap,
            seed=seed,
        )
        for method in loaded
        if method != "vanilla"
    }
    ours_gate = None
    if "ours" in paired:
        ours_gate = {
            "CHAIR_ci_excludes_zero_improvement": (
                paired["ours"]["CHAIR"]["ci95_high"] < 0.0
            ),
            "Hal_ci_excludes_zero_improvement": (
                paired["ours"]["Hal"]["ci95_high"] < 0.0
            ),
            "Cover_ci_not_exclusively_negative": (
                paired["ours"]["Cover"]["ci95_high"] >= 0.0
            ),
        }
    return {
        "protocol": "docs/iteration_protocols/image_caption_benchmark_matrix.md",
        "benchmark": "AMBER-generative",
        "official_commit": "534bab",
        "manifest": str(manifest_path.resolve()),
        "annotation": str(annotation_path.resolve()),
        "association": str(association_path.resolve()),
        "safe_words": str(safe_words_path.resolve()),
        "images": expected_images,
        "similarity_score": similarity_score,
        "epsilon_initialization": EPSILON,
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
        Path(args.annotation),
        Path(args.word_association),
        Path(args.safe_words),
        _row_specs(args.row),
        expected_images=args.expected_images,
        similarity_score=args.similarity_score,
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
                    for key in ("CHAIR", "Cover", "Hal", "Cog", "mean_words")
                }
                for method, row in payload["rows"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
