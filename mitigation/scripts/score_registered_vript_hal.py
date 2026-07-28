#!/usr/bin/env python3
"""Score the frozen registered Vript-HAL model/method matrix.

The metric intentionally follows Vript-HAL's released implementation:
spaCy VERB/NOUN/PROPN lemmas are matched to dense-reference lemmas with
all-mpnet-base-v2 cosine similarity > 0.5.  In addition to the official
per-clip macro P/R/F1, this script validates matrix completeness and frame
identity, reports registered companion audits, and computes paired clip
bootstrap intervals against vanilla.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class HalMetric(Protocol):
    def gt_words(self, contents: list[str]) -> list[str]: ...

    def precision_recall(
        self, gt_words: list[str], prediction: str
    ) -> tuple[float, float, list[str], list[bool]]: ...

    def word_matches_gt(self, word: str, gt_words: list[str]) -> bool: ...


class OfficialHalMetric:
    """Vript-HAL HallucinationScorer semantics."""

    def __init__(self) -> None:
        import spacy
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.nlp = spacy.load("en_core_web_lg")
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        self._gt_embedding_cache: dict[tuple[str, ...], Any] = {}

    def _gt_embedding(self, gt_words: list[str]):
        key = tuple(gt_words)
        if key not in self._gt_embedding_cache:
            self._gt_embedding_cache[key] = self.sentence_model.encode(
                gt_words, convert_to_tensor=True
            )
        return self._gt_embedding_cache[key]

    def extract_verbs_and_nouns(self, text: str) -> list[str]:
        doc = self.nlp(text)
        words = [token.text for token in doc if token.pos_ == "VERB"]
        words.extend(token.text for token in doc if token.pos_ == "NOUN")
        words.extend(token.text for token in doc if token.pos_ == "PROPN")
        return list({self.nlp(word)[0].lemma_ for word in words})

    def remove_stopword_and_stemming(self, words: list[str]) -> list[str]:
        kept = [word for word in words if word not in self.nlp.Defaults.stop_words]
        return [self.nlp(word)[0].lemma_ for word in kept]

    def gt_words(self, contents: list[str]) -> list[str]:
        words: list[str] = []
        for content in contents:
            words.extend(self.extract_verbs_and_nouns(content))
        return list(set(self.remove_stopword_and_stemming(words)))

    def precision_recall(
        self, gt_words: list[str], prediction: str
    ) -> tuple[float, float, list[str], list[bool]]:
        pred_words = list(
            set(
                self.remove_stopword_and_stemming(
                    self.extract_verbs_and_nouns(prediction)
                )
            )
        )
        if not pred_words or not gt_words:
            return 0.0, 0.0, pred_words, [False] * len(pred_words)
        gt_embedding = self._gt_embedding(gt_words)
        pred_embedding = self.sentence_model.encode(
            pred_words, convert_to_tensor=True
        )
        matched = []
        for embedding in pred_embedding:
            cosine = self.util.cos_sim(gt_embedding, embedding).squeeze(-1)
            matched.append(bool((cosine > 0.5).sum() > 0))
        correct = sum(matched)
        return (
            correct / len(pred_words),
            correct / len(gt_words),
            pred_words,
            matched,
        )

    def word_matches_gt(self, word: str, gt_words: list[str]) -> bool:
        if not gt_words:
            return False
        lemmas = self.remove_stopword_and_stemming([word])
        if not lemmas:
            return False
        gt_embedding = self._gt_embedding(gt_words)
        embedding = self.sentence_model.encode(
            [lemmas[0]], convert_to_tensor=True
        )[0]
        cosine = self.util.cos_sim(gt_embedding, embedding).squeeze(-1)
        return bool((cosine > 0.5).sum() > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        metavar="METHOD=GLOB",
        help="Repeat once per row; one row must be named vanilla.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_clips", type=int, default=122)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_rows(specifications: list[str]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected METHOD=GLOB, got {specification!r}")
        method, pattern = specification.split("=", 1)
        if not method or not pattern or method in rows:
            raise ValueError(f"Invalid or duplicate row: {specification!r}")
        rows[method] = pattern
    if "vanilla" not in rows:
        raise ValueError("A vanilla row is required for paired comparisons")
    return rows


def load_row(pattern: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern!r}")
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            clip_id = str(row["clip_id"])
            if clip_id in by_id:
                raise ValueError(f"Duplicate clip_id {clip_id} in {pattern}")
            by_id[clip_id] = row
    return by_id, [str(path.resolve()) for path in paths]


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


def _mean(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _token_repetition(caption: str) -> float:
    tokens = re.findall(r"\w+", caption.lower())
    return 1.0 - len(set(tokens)) / len(tokens) if tokens else 0.0


def _paired_bootstrap(
    candidate: np.ndarray,
    vanilla: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    if candidate.shape != vanilla.shape or candidate.ndim != 1:
        raise ValueError("Paired arrays must be one-dimensional and aligned")
    delta = candidate - vanilla
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 1_000):
        size = min(1_000, repeats - start)
        indices = rng.integers(0, delta.size, size=(size, delta.size))
        means[start : start + size] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _evaluate_row(
    rows: dict[str, dict[str, Any]],
    ordered_ids: list[str],
    gt_cache: dict[str, list[str]],
    metric: HalMetric,
) -> tuple[dict[str, Any], np.ndarray]:
    clip_metrics = np.empty((len(ordered_ids), 3), dtype=np.float64)
    object_counts: list[int] = []
    action_records: list[dict[str, Any]] = []
    grounded_removals = 0
    hallucinated_removals = 0
    unresolved = 0
    triggers = 0
    for index, clip_id in enumerate(ordered_ids):
        row = rows[clip_id]
        caption = str(row["caption"])
        precision, recall, pred_words, _ = metric.precision_recall(
            gt_cache[clip_id], caption
        )
        clip_metrics[index] = (precision, recall, _f1(precision, recall))
        object_counts.append(len(pred_words))
        actions = list(row.get("actions", []))
        triggers += len(actions)
        unresolved += len(row.get("unresolved_triggers", []))
        for action in actions:
            grounded = metric.word_matches_gt(
                str(action["word"]), gt_cache[clip_id]
            )
            grounded_removals += int(grounded)
            hallucinated_removals += int(not grounded)
            action_records.append(
                {
                    "clip_id": clip_id,
                    "word": action["word"],
                    "grounded": grounded,
                    "replacement": action.get("replacement"),
                }
            )

    generated = [rows[clip_id] for clip_id in ordered_ids]
    actions_total = grounded_removals + hallucinated_removals
    summary = {
        "clips": len(generated),
        "precision": 100.0 * float(clip_metrics[:, 0].mean()),
        "recall": 100.0 * float(clip_metrics[:, 1].mean()),
        "f1": 100.0 * float(clip_metrics[:, 2].mean()),
        "mean_generated_tokens": _mean(
            [row.get("generated_tokens") for row in generated]
        ),
        "mean_words": _mean([row.get("words") for row in generated]),
        "mean_object_mentions": float(np.mean(object_counts)),
        "mean_token_repetition": float(
            np.mean([_token_repetition(str(row["caption"])) for row in generated])
        ),
        "premature_eos_rate": _mean(
            [
                int(bool(row["premature_eos"]))
                for row in generated
                if "premature_eos" in row
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
        "candidate_count": _mean(
            [
                len(row["candidates"])
                for row in generated
                if "candidates" in row
            ]
        ),
        "mean_segment_count": _mean(
            [row.get("segment_count") for row in generated]
        ),
        "selected_candidate_histogram": {
            str(candidate): sum(
                row.get("selected_index") == candidate for row in generated
            )
            for candidate in sorted(
                {
                    int(row["selected_index"])
                    for row in generated
                    if "selected_index" in row
                }
            )
        },
        "triggers": triggers,
        "completed_rollbacks": actions_total,
        "unresolved_triggers": unresolved,
        "hallucinated_removals": hallucinated_removals,
        "grounded_removals": grounded_removals,
        "removal_precision": (
            hallucinated_removals / actions_total if actions_total else None
        ),
        "action_records": action_records,
    }
    return summary, clip_metrics


def score(
    annotations_path: Path,
    row_patterns: dict[str, str],
    *,
    expected_clips: int,
    bootstrap: int,
    seed: int,
    metric: HalMetric,
) -> dict[str, Any]:
    annotations = {
        str(row["clip_id"]): row for row in _read_jsonl(annotations_path)
    }
    if len(annotations) != expected_clips:
        raise ValueError(
            f"Expected {expected_clips} annotations, found {len(annotations)}"
        )
    ordered_ids = sorted(annotations)
    expected_ids = set(ordered_ids)
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    sources: dict[str, list[str]] = {}
    for method, pattern in row_patterns.items():
        loaded[method], sources[method] = load_row(pattern)
        actual_ids = set(loaded[method])
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                f"{method}: ID mismatch, missing={missing}, extra={extra}"
            )

    frame_mismatches: list[dict[str, Any]] = []
    for clip_id in ordered_ids:
        frames = {
            method: row[clip_id].get("frame_indices")
            for method, row in loaded.items()
        }
        present = {method: value for method, value in frames.items() if value is not None}
        if present and len({tuple(value) for value in present.values()}) != 1:
            frame_mismatches.append({"clip_id": clip_id, "frames": present})
    if frame_mismatches:
        raise ValueError(
            f"Frame mismatch on {len(frame_mismatches)} clips; first="
            f"{frame_mismatches[0]}"
        )

    gt_cache = {
        clip_id: metric.gt_words(list(annotations[clip_id]["content"]))
        for clip_id in ordered_ids
    }
    summaries: dict[str, dict[str, Any]] = {}
    per_clip: dict[str, np.ndarray] = {}
    for method, rows in loaded.items():
        summaries[method], per_clip[method] = _evaluate_row(
            rows, ordered_ids, gt_cache, metric
        )

    paired: dict[str, dict[str, dict[str, float]]] = {}
    for method in loaded:
        if method == "vanilla":
            continue
        paired[method] = {}
        for metric_index, metric_name in enumerate(("precision", "recall", "f1")):
            comparison = _paired_bootstrap(
                per_clip[method][:, metric_index],
                per_clip["vanilla"][:, metric_index],
                repeats=bootstrap,
                seed=seed + metric_index,
            )
            paired[method][metric_name] = {
                key: 100.0 * value for key, value in comparison.items()
            }

    ours_gate = None
    if "ours" in paired:
        ours_gate = {
            "precision_ci_excludes_zero_positive": (
                paired["ours"]["precision"]["ci95_low"] > 0.0
            ),
            "f1_ci_not_exclusively_negative": (
                paired["ours"]["f1"]["ci95_high"] >= 0.0
            ),
        }
        ours_gate["model_supports_h2"] = all(ours_gate.values())

    return {
        "protocol": "docs/iteration_protocols/video_caption_benchmark_matrix.md",
        "metric": {
            "name": "Vript-HAL released HallucinationScorer",
            "pos": ["VERB", "NOUN", "PROPN"],
            "embedding": "all-mpnet-base-v2",
            "cosine_threshold": 0.5,
            "aggregation": "mean of per-clip P/R/F1",
        },
        "annotations": str(annotations_path.resolve()),
        "clips": len(ordered_ids),
        "sources": sources,
        "frame_identity_pass": not frame_mismatches,
        "rows": summaries,
        "paired_vs_vanilla": paired,
        "h2_gate": ours_gate,
        "bootstrap": {
            "unit": "clip",
            "paired": True,
            "repeats": bootstrap,
            "seed": seed,
        },
    }


def main() -> None:
    args = parse_args()
    payload = score(
        Path(args.annotations),
        parse_rows(args.row),
        expected_clips=args.expected_clips,
        bootstrap=args.bootstrap,
        seed=args.seed,
        metric=OfficialHalMetric(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    compact = {
        method: {
            key: value
            for key, value in row.items()
            if key in {"clips", "precision", "recall", "f1", "mean_words"}
        }
        for method, row in payload["rows"].items()
    }
    print(json.dumps({"rows": compact, "h2_gate": payload["h2_gate"]}, indent=2))


if __name__ == "__main__":
    main()
