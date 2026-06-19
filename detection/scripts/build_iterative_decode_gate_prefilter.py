#!/usr/bin/env python3
"""Build a second-pass decode-gate prefilter from generated-caption audits.

The script implements a bounded iterative-deny-list prototype: if a previous
prefilter-gated generation routed a denied object to a new COCO object that TDEV
judges absent, add that substitute object to the same image's deny-list and write
a new prefilter examples file that can be consumed by
``run_tdev_decode_gate_caption_smoke.py --deny_phrase_source prefilter``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
if str(DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DETECTION_SRC))

from sinkdetect.decode_gate import build_object_phrase_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base_examples_json",
        default="detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_examples.json",
    )
    p.add_argument(
        "--audit_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_audit/closed_loop_example_audit.json",
    )
    p.add_argument(
        "--mention_matches_csv",
        default="detection/baselines/results/tdev_decode_gate_feasibility/mention_token_matches.csv",
    )
    p.add_argument("--variant_aliases_json", default="detection/config/object_variant_aliases.json")
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_prefilter",
    )
    p.add_argument("--alias_top_k", type=int, default=3)
    p.add_argument("--max_added_per_image", type=int, default=3)
    p.add_argument("--tokenizer_path", default="")
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
            counts[row["word"].strip().lower()][row["matched_text"].strip().lower()] += 1
    return {word: [item for item, _ in counter.most_common(top_k)] for word, counter in counts.items()}


def read_variant_aliases(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw = read_json(path)
    return {
        str(word).strip().lower(): [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
        for word, aliases in raw.items()
    }


def aliases_for_word(
    word: str,
    observed_aliases: dict[str, list[str]],
    variant_aliases: dict[str, list[str]],
) -> list[str]:
    phrases = [word, pluralize_last_token(word)]
    phrases.extend(observed_aliases.get(word, []))
    phrases.extend(variant_aliases.get(word, []))
    return dedupe([phrase.lower() for phrase in phrases])


def absent_introduced_words(example: dict[str, Any]) -> list[str]:
    words: list[str] = []
    claim_scores = example.get("introduced_claim_scores", {})
    for word in example.get("introduced_words", []):
        score = claim_scores.get(word, {})
        if score and int(score.get("two_stage_present", 1)) == 0:
            words.append(str(word).strip().lower())
    return dedupe(words)


def load_tokenizer(path: str) -> Any | None:
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def main() -> None:
    args = parse_args()
    base_examples = read_json(Path(args.base_examples_json))
    audit = read_json(Path(args.audit_json))
    observed_aliases = observed_surface_aliases(Path(args.mention_matches_csv), args.alias_top_k)
    variant_aliases = read_variant_aliases(Path(args.variant_aliases_json))
    tokenizer = load_tokenizer(args.tokenizer_path)

    additions_by_image: dict[int, list[str]] = {}
    for example in audit.get("examples", []):
        additions = absent_introduced_words(example)
        if args.max_added_per_image > 0:
            additions = additions[: args.max_added_per_image]
        if additions:
            additions_by_image[int(example["image_id"])] = additions

    updated = []
    added_rows = []
    for example in base_examples:
        row = dict(example)
        image_id = int(row["image_id"])
        denied_words = [str(word).strip().lower() for word in row.get("denied_words", [])]
        aliases_by_word = {
            str(word).strip().lower(): [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
            for word, aliases in row.get("aliases_by_word", {}).items()
        }
        for word in additions_by_image.get(image_id, []):
            if word in denied_words:
                continue
            denied_words.append(word)
            aliases_by_word[word] = aliases_for_word(word, observed_aliases, variant_aliases)
            added_rows.append({"image_id": image_id, "word": word, "aliases": aliases_by_word[word]})
        row["denied_words"] = denied_words
        row["aliases_by_word"] = aliases_by_word
        row["num_unique_denied_words"] = len(denied_words)
        row["iterative_added_words"] = additions_by_image.get(image_id, [])
        if tokenizer is not None:
            seq_count = 0
            for word in denied_words:
                seq_count += len(build_object_phrase_sequences(tokenizer, aliases_by_word.get(word, [word])))
            row["num_denied_token_sequences"] = seq_count
        updated.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "iterative_prefilter_examples.json"
    examples_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "base_examples_json": args.base_examples_json,
        "audit_json": args.audit_json,
        "num_images": len(updated),
        "num_images_with_additions": len(additions_by_image),
        "num_added_words": len(added_rows),
        "added_words": added_rows,
        "tokenizer_path": args.tokenizer_path,
        "output_examples_json": str(examples_path),
        "scope_note": (
            "Bounded second-pass deny-list expansion from generated-caption audits. "
            "Only introduced COCO words with TDEV two_stage_present=0 are added."
        ),
    }
    (output_dir / "iterative_prefilter_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| images | {summary['num_images']} |",
        f"| images with additions | {summary['num_images_with_additions']} |",
        f"| added words | {summary['num_added_words']} |",
    ]
    for item in added_rows:
        lines.append(f"| add image {item['image_id']} | {item['word']} |")
    (output_dir / "iterative_prefilter_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
