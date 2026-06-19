#!/usr/bin/env python3
"""Offline multi-image preflight for TDEV decode-gate candidates.

This script does not run a VLM or OWLv2. It uses cached caption-mention TDEV
scores to estimate whether a multi-image decode-gate run is worth spending GPU
on: how many image/object pairs would be denied, how often those pairs are CHAIR
hallucinations, and how wide the narrow alias phrase list would be.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
if str(DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DETECTION_SRC))

from sinkdetect.decode_gate import build_object_phrase_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--mention_matches_csv",
        default="detection/baselines/results/tdev_decode_gate_feasibility/mention_token_matches.csv",
    )
    p.add_argument("--variant_aliases_json", default="detection/config/object_variant_aliases.json")
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_multi_image_prefilter",
    )
    p.add_argument("--score", default="hybrid_positive_branch_absence")
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--max_images", type=int, default=100, help="0 means all images with selected mentions.")
    p.add_argument("--per_image_max_denied", type=int, default=0, help="0 keeps all selected words per chosen image.")
    p.add_argument("--alias_top_k", type=int, default=3)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    p.add_argument("--example_limit", type=int, default=40)
    p.add_argument(
        "--tokenizer_path",
        default="",
        help="Optional local tokenizer/model path. If omitted, token-sequence counts are skipped.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dedupe(items: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(item for item in items if item))


def pluralize_last_token(phrase: str) -> str:
    parts = phrase.split()
    if not parts:
        return phrase
    last = parts[-1]
    if last.endswith(("s", "x", "ch", "sh")):
        parts[-1] = f"{last}es"
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = f"{last[:-1]}ies"
    else:
        parts[-1] = f"{last}s"
    return " ".join(parts)


def observed_surface_aliases(path: Path, top_k: int) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in read_csv(path):
        if row.get("near_gen_pos_match") == "1" and row.get("matched_text", "").strip():
            word = row["word"].strip().lower()
            text = row["matched_text"].strip().lower()
            counts[word][text] += 1
    return {
        word: [item for item, _ in counter.most_common(top_k)]
        for word, counter in counts.items()
    }


def read_variant_aliases(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(word).strip().lower(): [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
        for word, aliases in raw.items()
    }


def narrow_aliases(
    word: str,
    observed_aliases: dict[str, list[str]],
    variant_aliases: dict[str, list[str]],
) -> list[str]:
    phrases = [word, pluralize_last_token(word)]
    phrases.extend(observed_aliases.get(word, []))
    phrases.extend(variant_aliases.get(word, []))
    return dedupe([phrase.lower() for phrase in phrases])


def two_stage_score(target: np.ndarray, margin: np.ndarray, low: float, high: float, margin_threshold: float) -> np.ndarray:
    high_branch = target - high
    medium_branch = np.minimum(target - low, margin - margin_threshold)
    return np.maximum(high_branch, medium_branch)


def build_scores(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    target = np.asarray([float(row["target_score"]) for row in rows], dtype=np.float64)
    neighbor = np.asarray([float(row["best_neighbor_score"]) for row in rows], dtype=np.float64)
    margin = np.asarray([float(row["tdev_margin"]) for row in rows], dtype=np.float64)
    dominance = np.maximum(neighbor - target, 0.0)
    return {
        "target_absence": -target,
        "margin_absence": -margin,
        "hybrid_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_margin,
        ),
        f"target_absence_plus_neighbor_dominance_{args.neighbor_dominance_alpha:g}": (
            -target + args.neighbor_dominance_alpha * dominance
        ),
    }


def top_fraction_mask(values: np.ndarray, frac: float) -> np.ndarray:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"top_frac must be in (0,1): {frac}")
    n_keep = max(1, int(round(values.size * frac)))
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def choose_images(candidates: list[dict[str, Any]], max_images: int) -> list[int]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_image[int(row["image_id"])].append(row)
    ranked = sorted(
        by_image,
        key=lambda image_id: (
            max(float(row["risk_score"]) for row in by_image[image_id]),
            len(by_image[image_id]),
            -image_id,
        ),
        reverse=True,
    )
    return ranked if max_images <= 0 else ranked[:max_images]


def load_tokenizer(path: str) -> Any | None:
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "p95": float(ordered[p95_idx]),
        "max": int(max(values)),
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.scores_csv))
    scores_by_name = build_scores(rows, args)
    if args.score not in scores_by_name:
        raise KeyError(f"Unknown score {args.score!r}; available: {sorted(scores_by_name)}")
    scores = scores_by_name[args.score]
    selected_mask = top_fraction_mask(scores, args.top_frac)
    observed_aliases = observed_surface_aliases(Path(args.mention_matches_csv), args.alias_top_k)
    variant_aliases = read_variant_aliases(Path(args.variant_aliases_json))
    tokenizer = load_tokenizer(args.tokenizer_path)

    all_candidates: list[dict[str, Any]] = []
    for idx, selected in enumerate(selected_mask):
        if not selected:
            continue
        row = rows[idx]
        all_candidates.append(
            {
                "row_index": idx,
                "object_id": int(row["object_id"]),
                "image_id": int(row["image_id"]),
                "word": row["word"].strip().lower(),
                "label": int(row["label"]),
                "risk_score": float(scores[idx]),
                "target_score": float(row["target_score"]),
                "best_neighbor": row.get("best_neighbor", ""),
                "best_neighbor_score": float(row["best_neighbor_score"]),
                "tdev_margin": float(row["tdev_margin"]),
                "caption": row["caption"],
            }
        )

    chosen_images = set(choose_images(all_candidates, args.max_images))
    selected_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in all_candidates:
        if int(row["image_id"]) in chosen_images:
            selected_by_image[int(row["image_id"])].append(row)
    for image_id, image_rows in selected_by_image.items():
        image_rows.sort(key=lambda item: (-float(item["risk_score"]), item["word"], item["object_id"]))
        if args.per_image_max_denied > 0:
            selected_by_image[image_id] = image_rows[: args.per_image_max_denied]

    selected_rows = [row for image_id in sorted(selected_by_image) for row in selected_by_image[image_id]]
    unique_pairs: dict[tuple[int, str], dict[str, Any]] = {}
    for row in selected_rows:
        key = (int(row["image_id"]), str(row["word"]))
        if key not in unique_pairs or float(row["risk_score"]) > float(unique_pairs[key]["risk_score"]):
            unique_pairs[key] = row

    rows_by_image: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_image[int(row["image_id"])].append(row)
    chosen_all_rows = [row for image_id in chosen_images for row in rows_by_image.get(image_id, [])]
    chosen_hallucinated = sum(int(row["label"]) for row in chosen_all_rows)
    selected_hallucinated = sum(int(row["label"]) for row in selected_rows)
    selected_grounded = len(selected_rows) - selected_hallucinated

    image_summaries: list[dict[str, Any]] = []
    phrase_counts: list[int] = []
    sequence_counts: list[int] = []
    single_token_sequence_counts: list[int] = []
    multi_token_sequence_counts: list[int] = []
    for image_id in sorted(selected_by_image):
        image_rows = selected_by_image[image_id]
        pair_rows = [row for key, row in unique_pairs.items() if key[0] == image_id]
        denied_words = [row["word"] for row in pair_rows]
        aliases_by_word: dict[str, list[str]] = {}
        seq_count = 0
        single_seq_count = 0
        multi_seq_count = 0
        for word in denied_words:
            aliases = narrow_aliases(word, observed_aliases, variant_aliases)
            aliases_by_word[word] = aliases
            phrase_counts.append(len(aliases))
            if tokenizer is not None:
                seqs = build_object_phrase_sequences(tokenizer, aliases)
                seq_count += len(seqs)
                single_seq_count += sum(1 for seq in seqs if len(seq) == 1)
                multi_seq_count += sum(1 for seq in seqs if len(seq) > 1)
        if tokenizer is not None:
            sequence_counts.append(seq_count)
            single_token_sequence_counts.append(single_seq_count)
            multi_token_sequence_counts.append(multi_seq_count)
        image_summaries.append(
            {
                "image_id": image_id,
                "num_selected_mentions": len(image_rows),
                "num_unique_denied_words": len(denied_words),
                "num_selected_hallucinated_mentions": sum(int(row["label"]) for row in image_rows),
                "num_selected_grounded_mentions": sum(1 - int(row["label"]) for row in image_rows),
                "max_risk_score": max(float(row["risk_score"]) for row in image_rows),
                "denied_words": denied_words,
                "aliases_by_word": aliases_by_word,
                "num_denied_token_sequences": seq_count if tokenizer is not None else None,
                "caption": image_rows[0]["caption"] if image_rows else "",
            }
        )

    top_words = Counter(str(row["word"]) for row in selected_rows).most_common(20)
    metrics = {
        "scores_csv": args.scores_csv,
        "score": args.score,
        "top_frac": args.top_frac,
        "max_images": args.max_images,
        "per_image_max_denied": args.per_image_max_denied,
        "num_rows": len(rows),
        "num_images_in_scores": len(rows_by_image),
        "num_global_selected_mentions": int(selected_mask.sum()),
        "num_chosen_images": len(chosen_images),
        "num_selected_mentions_on_chosen_images": len(selected_rows),
        "num_unique_image_word_pairs": len(unique_pairs),
        "selected_hallucinated_mentions": selected_hallucinated,
        "selected_grounded_mentions": selected_grounded,
        "selected_precision_hallucinated": selected_hallucinated / len(selected_rows) if selected_rows else 0.0,
        "chosen_image_hallucinated_mentions": chosen_hallucinated,
        "chosen_image_hallucination_coverage": selected_hallucinated / chosen_hallucinated if chosen_hallucinated else 0.0,
        "unique_denied_words_per_image": summarize_numbers([row["num_unique_denied_words"] for row in image_summaries]),
        "selected_mentions_per_image": summarize_numbers([row["num_selected_mentions"] for row in image_summaries]),
        "phrases_per_denied_word": summarize_numbers(phrase_counts),
        "tokenizer_path": args.tokenizer_path,
        "token_sequences_per_image": summarize_numbers(sequence_counts) if tokenizer is not None else None,
        "single_token_sequences_per_image": summarize_numbers(single_token_sequence_counts) if tokenizer is not None else None,
        "multi_token_sequences_per_image": summarize_numbers(multi_token_sequence_counts) if tokenizer is not None else None,
        "top_selected_words": top_words,
        "scope_note": (
            "Offline preflight over cached caption-mention scores. It estimates deny-list width "
            "and CHAIR-label precision for candidate multi-image decode-gate runs; it does not "
            "evaluate generated captions or all absent COCO objects."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "multi_image_prefilter_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "multi_image_prefilter_examples.json").write_text(
        json.dumps(image_summaries[: args.example_limit], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| chosen images | {metrics['num_chosen_images']} |",
        f"| selected mentions on chosen images | {metrics['num_selected_mentions_on_chosen_images']} |",
        f"| unique image-word pairs | {metrics['num_unique_image_word_pairs']} |",
        f"| selected hallucinated mentions | {metrics['selected_hallucinated_mentions']} |",
        f"| selected grounded mentions | {metrics['selected_grounded_mentions']} |",
        f"| selected hallucination precision | {metrics['selected_precision_hallucinated']:.4f} |",
        f"| chosen-image hallucination coverage | {metrics['chosen_image_hallucination_coverage']:.4f} |",
        f"| mean unique denied words / image | {metrics['unique_denied_words_per_image']['mean']:.2f} |",
        f"| p95 unique denied words / image | {metrics['unique_denied_words_per_image']['p95']:.0f} |",
        f"| mean phrases / denied word | {metrics['phrases_per_denied_word']['mean']:.2f} |",
    ]
    if metrics["token_sequences_per_image"] is not None:
        lines.extend(
            [
                f"| mean token sequences / image | {metrics['token_sequences_per_image']['mean']:.2f} |",
                f"| p95 token sequences / image | {metrics['token_sequences_per_image']['p95']:.0f} |",
            ]
        )
    (output_dir / "multi_image_prefilter_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
