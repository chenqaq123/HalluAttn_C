#!/usr/bin/env python3
"""Evaluate a sentence/clause-level TDEV object-claim gate for captions.

This is a deterministic offline prototype for the proposed decoding-time object
claim gate. Instead of replacing a high-risk object mention with a generic noun,
it suppresses the local sentence or clause that makes the unsupported object
claim. The goal is to test whether TDEV-selected target-vs-neighbor evidence is
useful at the claim-unit level, while keeping the output more grammatical than
word deletion. It is still not fluent regeneration and should be reported as a
bounded proof-of-concept only.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EDIT_SCRIPT = REPO_ROOT / "detection" / "scripts" / "evaluate_tdev_caption_edit.py"

CLAUSE_DELIMITERS = {",", ";", ":"}
SENTENCE_END_RE = re.compile(r"[.!?]")


def load_edit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tdev_caption_edit", EDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TDEV sentence/clause object-claim gate")
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--chair_source",
        default="../pas/src/pas/evaluate/chair.py",
        help="PAS CHAIR source file used only to parse synonyms_txt.",
    )
    p.add_argument("--output_dir", default="detection/baselines/results/tdev_caption_sentence_gate_top5")
    p.add_argument("--score", default="hybrid_positive_branch_absence")
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    p.add_argument("--gate_unit", choices=("sentence", "clause"), default="sentence")
    p.add_argument("--min_sentence_words", type=int, default=4)
    p.add_argument("--max_clause_words", type=int, default=18)
    p.add_argument("--run_chair", action="store_true", help="Rerun official PAS CHAIR on original and gated captions.")
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sentence_span(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for match in SENTENCE_END_RE.finditer(text, 0, pos):
        start = match.end()
    while start < len(text) and text[start].isspace():
        start += 1
    end_match = SENTENCE_END_RE.search(text, pos)
    end = end_match.end() if end_match else len(text)
    return start, end


def clause_span(sentence: str, rel_start: int, rel_end: int, max_clause_words: int) -> tuple[int, int] | None:
    starts = [0]
    ends: list[int] = []
    for idx, char in enumerate(sentence):
        if char in CLAUSE_DELIMITERS:
            ends.append(idx)
            starts.append(idx + 1)
    ends.append(len(sentence))
    best: tuple[int, int] | None = None
    for start, end in zip(starts, ends):
        if start <= rel_start and rel_end <= end:
            while start < end and sentence[start].isspace():
                start += 1
            while end > start and sentence[end - 1].isspace():
                end -= 1
            words = sentence[start:end].split()
            if 0 < len(words) <= max_clause_words:
                best = (start, end)
            break
    return best


def normalize_after_gate(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"(?:^|\s+)[,;:]\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def remove_span(text: str, start: int, end: int) -> str:
    left = text[:start].rstrip()
    right = text[end:].lstrip()
    return normalize_after_gate((left + " " + right).strip())


def gate_one_claim(caption: str, patterns: list[tuple[str, object]], edit, gate_unit: str, min_sentence_words: int, max_clause_words: int) -> tuple[str, str | None, str, str]:
    for phrase, pattern in patterns:
        match = pattern.search(caption)
        if not match:
            continue
        sent_start, sent_end = sentence_span(caption, match.start())
        sentence = caption[sent_start:sent_end]
        rel_start = match.start() - sent_start
        rel_end = match.end() - sent_start
        gated_unit = "sentence"
        local_start, local_end = sent_start, sent_end
        if gate_unit == "clause":
            local = clause_span(sentence, rel_start, rel_end, max_clause_words)
            if local is not None:
                clause_words = sentence[local[0]:local[1]].split()
                sentence_words = sentence.split()
                remove_from_start = local[0] == 0 and local[1] < len(sentence)
                if (
                    not remove_from_start
                    and len(clause_words) <= max_clause_words
                    and len(sentence_words) - len(clause_words) >= min_sentence_words
                ):
                    gated_unit = "clause"
                    local_start = sent_start + local[0]
                    local_end = sent_start + local[1]
                    # Remove an immediately preceding comma/semicolon when dropping a clause.
                    while local_start > 0 and caption[local_start - 1].isspace():
                        local_start -= 1
                    if local_start > 0 and caption[local_start - 1] in CLAUSE_DELIMITERS:
                        local_start -= 1
        gated = remove_span(caption, local_start, local_end)
        return edit.clean_caption(gated), phrase, gated_unit, caption[local_start:local_end].strip()
    return caption, None, "", ""


def image_hallucination_rate(rows: list[dict[str, str]], keep_mask: np.ndarray) -> float:
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    unique_images = np.unique(image_ids)
    if unique_images.size == 0:
        return 0.0
    hallucinated_images = 0
    for image_id in unique_images:
        mask = (image_ids == image_id) & keep_mask
        hallucinated_images += int(labels[mask].sum() > 0)
    return hallucinated_images / unique_images.size


def main() -> None:
    args = parse_args()
    edit = load_edit_module()

    rows = edit.read_rows(Path(args.scores_csv))
    scores = edit.build_scores(rows, args)
    if args.score not in scores:
        raise KeyError(f"Unknown score {args.score!r}; available: {sorted(scores)}")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    values = scores[args.score]
    selected_mask = edit.top_fraction_mask(values, args.top_frac)

    synonyms = edit.load_synonyms(Path(args.chair_source))
    pattern_cache = {word: edit.compile_patterns(phrases) for word, phrases in synonyms.items()}

    captions: OrderedDict[int, str] = OrderedDict()
    for row in rows:
        image_id = int(row["image_id"])
        captions.setdefault(image_id, row["caption"])
    original_captions = dict(captions)

    gated_mask = np.zeros(len(rows), dtype=bool)
    gate_rows: list[dict[str, object]] = []
    selected_indices = np.flatnonzero(selected_mask)
    selected_indices = selected_indices[np.argsort(-values[selected_indices], kind="mergesort")]
    for idx in selected_indices:
        row = rows[int(idx)]
        image_id = int(row["image_id"])
        word = row["word"].strip().lower()
        patterns = pattern_cache.get(word, edit.compile_patterns([word]))
        gated, phrase, gated_unit, removed_text = gate_one_claim(
            captions[image_id],
            patterns,
            edit,
            args.gate_unit,
            args.min_sentence_words,
            args.max_clause_words,
        )
        changed = phrase is not None and gated != captions[image_id]
        if changed:
            captions[image_id] = gated
            gated_mask[int(idx)] = True
        gate_rows.append(
            {
                "object_id": int(row["object_id"]),
                "image_id": image_id,
                "word": row["word"],
                "label": int(row["label"]),
                "score": float(values[int(idx)]),
                "target_score": float(row["target_score"]),
                "best_neighbor": row["best_neighbor"],
                "best_neighbor_score": float(row["best_neighbor_score"]),
                "tdev_margin": float(row["tdev_margin"]),
                "matched_phrase": phrase or "",
                "gated_unit": gated_unit,
                "removed_text": removed_text,
                "gated": int(changed),
            }
        )

    mentions = int(labels.size)
    hallucinated = int(labels.sum())
    grounded = mentions - hallucinated
    selected = int(selected_mask.sum())
    selected_hallucinated = int(labels[selected_mask].sum())
    selected_grounded = selected - selected_hallucinated
    gated = int(gated_mask.sum())
    gated_hallucinated = int(labels[gated_mask].sum())
    gated_grounded = gated - gated_hallucinated
    keep_mask = ~gated_mask
    remaining = int(keep_mask.sum())
    remaining_hallucinated = int(labels[keep_mask].sum())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gated_captions = [
        {"image_id": image_id, "caption": caption}
        for image_id, caption in captions.items()
    ]
    gated_pairs = [
        {
            "image_id": image_id,
            "caption": captions[image_id],
            "original_caption": original_captions[image_id],
        }
        for image_id in captions
    ]
    with (output_dir / "gated_captions.json").open("w", encoding="utf-8") as f:
        json.dump(gated_captions, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with (output_dir / "gated_caption_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(gated_pairs, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_csv(output_dir / "gated_mentions.csv", gate_rows)

    metrics = {
        "scores_csv": args.scores_csv,
        "score": args.score,
        "score_direction": "larger score means higher risk and is selected first",
        "top_frac": args.top_frac,
        "gate_unit": args.gate_unit,
        "min_sentence_words": args.min_sentence_words,
        "max_clause_words": args.max_clause_words,
        "mentions": mentions,
        "hallucinated_mentions": hallucinated,
        "grounded_mentions": grounded,
        "selected_mentions": selected,
        "selected_hallucinated": selected_hallucinated,
        "selected_grounded": selected_grounded,
        "gated_mentions": gated,
        "gated_hallucinated": gated_hallucinated,
        "gated_grounded": gated_grounded,
        "gate_success_rate": gated / selected if selected else 0.0,
        "gate_precision": gated_hallucinated / gated if gated else 0.0,
        "hallucinated_reduction_proxy": gated_hallucinated / hallucinated if hallucinated else 0.0,
        "grounded_gate_rate": gated_grounded / grounded if grounded else 0.0,
        "remaining_mentions_proxy": remaining,
        "remaining_hallucinated_proxy": remaining_hallucinated,
        "remaining_mention_hallucination_rate_proxy": remaining_hallucinated / remaining if remaining else 0.0,
        "remaining_image_hallucination_rate_proxy": image_hallucination_rate(rows, keep_mask),
        "chair_rerun_status": "not_run_use_--run_chair",
    }

    if args.run_chair:
        vanilla_data = [
            {"image_id": image_id, "caption": original_captions[image_id]}
            for image_id in captions
        ]
        official_chair = {
            "chair_pkl": args.chair_pkl,
            "sample_scope": "images with CHAIR object mentions in the OWLv2 score CSV",
            "vanilla": edit.chair_summary(vanilla_data, args.chair_pkl, output_dir, "vanilla"),
            "gated": edit.chair_summary(gated_captions, args.chair_pkl, output_dir, "gated"),
        }
        official_chair["delta"] = edit.metric_delta(official_chair["gated"], official_chair["vanilla"])
        with (output_dir / "chair_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(official_chair, f, indent=2, sort_keys=True)
            f.write("\n")
        metrics["chair_rerun_status"] = "run"
        metrics["chair_metrics_file"] = str(output_dir / "chair_metrics.json")
    with (output_dir / "caption_clause_gate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'caption_clause_gate_metrics.json'}")


if __name__ == "__main__":
    main()
