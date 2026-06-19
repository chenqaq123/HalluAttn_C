#!/usr/bin/env python3
"""Audit whether TDEV-selected object mentions are gateable during decoding.

The caption-edit prototypes operate after a caption is complete. This audit
checks the lower-level prerequisite for a decoding-time object gate: can the
object mention be located as a tokenizer span, and would a first-token or
phrase-prefix LogitsProcessor be able to intercept it?
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
EDIT_SCRIPT = REPO_ROOT / "detection" / "scripts" / "evaluate_tdev_caption_edit.py"


def load_edit_module() -> Any:
    spec = importlib.util.spec_from_file_location("tdev_caption_edit", EDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--model_path",
        default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    )
    p.add_argument("--chair_source", default="../pas/src/pas/evaluate/chair.py")
    p.add_argument("--output_dir", default="detection/baselines/results/tdev_decode_gate_feasibility")
    p.add_argument("--score", default="hybrid_positive_branch_absence")
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--window", type=int, default=4, help="Token-position matching window around gen_pos.")
    p.add_argument("--max_collision_examples", type=int, default=25)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dedupe(items: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(item for item in items if item))


def object_phrases(word: str, synonyms: dict[str, list[str]], edit: Any) -> list[str]:
    phrases: list[str] = []
    for phrase in synonyms.get(word, [word]):
        phrases.extend(edit.phrase_variants(phrase.lower()))
        phrases.append(phrase.lower())
    return dedupe(phrases)


def phrase_tokenizations(tokenizer: Any, phrases: list[str]) -> dict[str, list[list[int]]]:
    tokenizations: dict[str, list[list[int]]] = {}
    for phrase in phrases:
        forms = [
            phrase,
            f" {phrase}",
            phrase.capitalize(),
            f" {phrase.capitalize()}",
        ]
        seqs = []
        for form in dedupe(forms):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if ids:
                seqs.append(ids)
        tokenizations[phrase] = dedupe_token_seqs(seqs)
    return tokenizations


def dedupe_token_seqs(seqs: list[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for seq in seqs:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            out.append(seq)
    return out


def match_near(tokens: list[int], seqs: list[list[int]], pos: int, window: int) -> tuple[bool, int | None, list[int] | None]:
    starts = range(max(0, pos - window), min(len(tokens), pos + window + 1))
    for start in starts:
        for seq in seqs:
            if tokens[start : start + len(seq)] == seq:
                return True, start, seq
    return False, None, None


def match_anywhere(tokens: list[int], seqs: list[list[int]]) -> tuple[bool, int | None, list[int] | None]:
    for start in range(len(tokens)):
        for seq in seqs:
            if tokens[start : start + len(seq)] == seq:
                return True, start, seq
    return False, None, None


def action_token(tokenizer: Any, seq: list[int]) -> int | None:
    """Return the first non-whitespace token a decoder gate can act on."""
    for token_id in seq:
        if tokenizer.decode([token_id]).strip():
            return token_id
    return seq[0] if seq else None


def summarize_mask(name: str, rows: list[dict[str, str]], mask: np.ndarray, matched: np.ndarray, anywhere: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    n = int(mask.sum())
    hallucinated = int(labels[mask].sum())
    grounded = n - hallucinated
    return {
        "name": name,
        "mentions": n,
        "hallucinated_mentions": hallucinated,
        "grounded_mentions": grounded,
        "near_gen_pos_matches": int(matched[mask].sum()),
        "near_gen_pos_match_rate": float(matched[mask].mean()) if n else 0.0,
        "caption_anywhere_matches": int(anywhere[mask].sum()),
        "caption_anywhere_match_rate": float(anywhere[mask].mean()) if n else 0.0,
        "hallucinated_near_gen_pos_match_rate": float(matched[mask & (labels == 1)].mean()) if hallucinated else 0.0,
        "grounded_near_gen_pos_match_rate": float(matched[mask & (labels == 0)].mean()) if grounded else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    edit = load_edit_module()
    rows = read_rows(Path(args.scores_csv))
    scores = edit.build_scores(rows, args)
    if args.score not in scores:
        raise KeyError(f"Unknown score {args.score!r}; available: {sorted(scores)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    synonyms = edit.load_synonyms(Path(args.chair_source))

    word_to_phrases = {word: object_phrases(word, synonyms, edit) for word in sorted({row["word"].lower() for row in rows})}
    word_to_tokenizations = {
        word: phrase_tokenizations(tokenizer, phrases)
        for word, phrases in word_to_phrases.items()
    }
    word_to_seqs = {
        word: dedupe_token_seqs([seq for seqs in phrase_map.values() for seq in seqs])
        for word, phrase_map in word_to_tokenizations.items()
    }

    action_token_to_words: dict[int, set[str]] = defaultdict(set)
    vocab_rows: list[dict[str, Any]] = []
    for word, seqs in word_to_seqs.items():
        for seq in seqs:
            gate_token = action_token(tokenizer, seq)
            if gate_token is not None:
                action_token_to_words[gate_token].add(word)
        lengths = Counter(len(seq) for seq in seqs)
        action_tokens = {action_token(tokenizer, seq) for seq in seqs}
        action_tokens.discard(None)
        vocab_rows.append(
            {
                "word": word,
                "num_phrases": len(word_to_phrases[word]),
                "num_tokenizations": len(seqs),
                "single_token_tokenizations": lengths.get(1, 0),
                "single_token_fraction": lengths.get(1, 0) / len(seqs) if seqs else 0.0,
                "unique_action_tokens": len(action_tokens),
                "max_tokenization_len": max((len(seq) for seq in seqs), default=0),
            }
        )

    collision_rows: list[dict[str, Any]] = []
    for token_id, words in sorted(action_token_to_words.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(words) <= 1:
            continue
        collision_rows.append(
            {
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "num_words": len(words),
                "words": "|".join(sorted(words)),
            }
        )

    values = scores[args.score]
    selected_mask = edit.top_fraction_mask(values, args.top_frac)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    matched = np.zeros(len(rows), dtype=bool)
    anywhere = np.zeros(len(rows), dtype=bool)
    mention_rows: list[dict[str, Any]] = []
    token_cache: dict[str, list[int]] = {}

    for idx, row in enumerate(rows):
        caption = row["caption"]
        tokens = token_cache.get(caption)
        if tokens is None:
            tokens = tokenizer.encode(caption, add_special_tokens=False)
            token_cache[caption] = tokens
        word = row["word"].lower()
        seqs = word_to_seqs.get(word, [])
        near_ok, near_start, near_seq = match_near(tokens, seqs, int(row["gen_pos"]), args.window)
        any_ok, any_start, any_seq = match_anywhere(tokens, seqs)
        matched[idx] = near_ok
        anywhere[idx] = any_ok
        if selected_mask[idx] or not near_ok:
            chosen_seq = near_seq or any_seq or []
            mention_rows.append(
                {
                    "object_id": row["object_id"],
                    "image_id": row["image_id"],
                    "word": row["word"],
                    "label": row["label"],
                    "score": float(values[idx]),
                    "selected": int(selected_mask[idx]),
                    "gen_pos": row["gen_pos"],
                    "near_gen_pos_match": int(near_ok),
                    "near_match_start": "" if near_start is None else near_start,
                    "caption_anywhere_match": int(any_ok),
                    "anywhere_match_start": "" if any_start is None else any_start,
                    "matched_token_len": len(chosen_seq),
                    "matched_tokens": " ".join(str(item) for item in chosen_seq),
                    "matched_text": tokenizer.decode(chosen_seq) if chosen_seq else "",
                }
            )

    all_mask = np.ones(len(rows), dtype=bool)
    selected_hallucinated_mask = selected_mask & (labels == 1)
    selected_grounded_mask = selected_mask & (labels == 0)
    summaries = [
        summarize_mask("all_mentions", rows, all_mask, matched, anywhere),
        summarize_mask("hallucinated_mentions", rows, labels == 1, matched, anywhere),
        summarize_mask("grounded_mentions", rows, labels == 0, matched, anywhere),
        summarize_mask("tdev_selected_top_frac", rows, selected_mask, matched, anywhere),
        summarize_mask("tdev_selected_hallucinated", rows, selected_hallucinated_mask, matched, anywhere),
        summarize_mask("tdev_selected_grounded", rows, selected_grounded_mask, matched, anywhere),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "object_vocab_tokenization.csv", vocab_rows)
    write_csv(output_dir / "first_token_collisions.csv", collision_rows[: args.max_collision_examples])
    write_csv(output_dir / "mention_token_matches.csv", mention_rows)

    metrics = {
        "scores_csv": args.scores_csv,
        "model_path": args.model_path,
        "score": args.score,
        "top_frac": args.top_frac,
        "window": args.window,
        "num_mentions": len(rows),
        "num_unique_captions": len(token_cache),
        "num_object_words": len(word_to_seqs),
        "vocab_single_token_word_fraction": sum(1 for row in vocab_rows if row["single_token_tokenizations"] > 0) / len(vocab_rows),
        "vocab_all_words_have_action_token": all(row["unique_action_tokens"] > 0 for row in vocab_rows),
        "action_token_collision_count": len(collision_rows),
        "max_action_token_collision_words": max((row["num_words"] for row in collision_rows), default=1),
        "summaries": summaries,
        "interpretation": (
            "Near-gen-pos matches estimate direct decode-time gateability. "
            "Action tokens skip tokenizer whitespace tokens, so multi-token phrases require a prefix-state gate."
        ),
    }
    with (output_dir / "decode_gate_feasibility_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
