#!/usr/bin/env python3
"""Reparse saved atomic-detail raw generations with different filters."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=\s|$)")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "there",
    "this",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atomic_examples_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_gen_100/atomic_detail_generation_examples.json",
    )
    parser.add_argument(
        "--audit_template_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_gen_100/atomic_detail_generation_audit_examples.json",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_additions", type=int, default=2)
    parser.add_argument("--min_sentence_words", type=int, default=5)
    parser.add_argument("--max_sentence_words", type=int, default=16)
    parser.add_argument("--max_overlap", type=float, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(str(text).split())


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]*", text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def overlap(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def sentence_chunks(text: str) -> list[str]:
    stripped = str(text).strip()
    chunks: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(stripped):
        end = match.end()
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    tail = stripped[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def ensure_period(text: str) -> str:
    text = text.strip(" \t\n-*:;")
    if not text:
        return text
    if re.search(r"[.!?][\"')\]]*$", text):
        return text
    return text + "."


def parse_additions(raw_text: str, repaired_caption: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    raw = str(raw_text).strip()
    if not raw or re.search(r"\bnone\b", raw.lower()):
        return []
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    candidates = []
    for item in lines or [raw]:
        candidates.extend(sentence_chunks(item))

    additions: list[dict[str, Any]] = []
    repaired_sentences = sentence_chunks(repaired_caption)
    seen: set[str] = set()
    for sentence in candidates:
        sentence = ensure_period(sentence)
        norm = re.sub(r"\s+", " ", sentence.lower()).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        words = word_count(sentence)
        if words < args.min_sentence_words or words > args.max_sentence_words:
            continue
        best_overlap = max((overlap(sentence, base) for base in repaired_sentences), default=0.0)
        if best_overlap > args.max_overlap:
            continue
        additions.append({"text": sentence, "words": words, "best_overlap": best_overlap})
        if len(additions) >= args.max_additions:
            break
    return additions


def normalize_caption(chunks: list[str]) -> str:
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


def main() -> None:
    args = parse_args()
    rows = read_json(Path(args.atomic_examples_json))
    audit_templates = {int(row["image_id"]): row for row in read_json(Path(args.audit_template_json))}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    audit_examples = []
    for row in rows:
        image_id = int(row["image_id"])
        repaired = str(row.get("repaired_caption", "")).strip()
        additions = parse_additions(str(row.get("raw_atomic_detail_text", "")), repaired, args)
        augmented = normalize_caption([repaired] + [item["text"] for item in additions])
        out = dict(row)
        out["atomic_detail_additions"] = additions
        out["augmented_caption"] = augmented
        out["augmented_words"] = word_count(augmented)
        out["num_additions"] = len(additions)
        out["reparse_config"] = {
            "max_additions": args.max_additions,
            "min_sentence_words": args.min_sentence_words,
            "max_sentence_words": args.max_sentence_words,
            "max_overlap": args.max_overlap,
        }
        examples.append(out)
        template = audit_templates[image_id]
        audit_examples.append(
            {
                "image_id": image_id,
                "image_path": row["image_path"],
                "denied_items": template.get("denied_items", []),
                "num_denied_sequences": template.get("num_denied_sequences", 0),
                "vanilla_source": "claim_local_repair",
                "vanilla_caption": repaired,
                "gated_caption": augmented,
                "caption_differs_from_reference": int(repaired != augmented),
                "reference_comparison_note": f"atomic detail reparse o{args.max_overlap:g} vs claim-local repair",
                "gate_events": [],
            }
        )

    metrics = {
        "source_atomic_examples_json": args.atomic_examples_json,
        "num_examples": len(examples),
        "max_overlap": args.max_overlap,
        "max_additions": args.max_additions,
        "min_sentence_words": args.min_sentence_words,
        "max_sentence_words": args.max_sentence_words,
        "images_with_additions": sum(int(row["num_additions"] > 0) for row in examples),
        "total_additions": sum(row["num_additions"] for row in examples),
        "mean_words": {
            "repaired": sum(word_count(row["repaired_caption"]) for row in examples) / len(examples),
            "augmented": sum(word_count(row["augmented_caption"]) for row in examples) / len(examples),
        },
    }
    (output_dir / "atomic_detail_generation_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "atomic_detail_generation_audit_examples.json").write_text(
        json.dumps(audit_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "atomic_detail_reparse_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
