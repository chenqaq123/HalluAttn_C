#!/usr/bin/env python3
"""Append verified local detail sentences to repaired captions.

This prototype avoids whole-caption replacement. It keeps the claim-local repair
as the base caption, then appends non-overlapping sentences from a verified
detail-regeneration candidate. Selection uses only the repaired-to-candidate
closed-loop audit; CHAIR is used after selection for evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
PAS_SRC = REPO_ROOT.parent / "pas" / "src"
for import_path in (DETECTION_SRC, PAS_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sinkdetect.chair import evaluate_chair, load_chair_evaluator

GENERIC_OR_NONOBJECT_CLAIMS = {
    "additionally",
    "also",
    "appears",
    "displayed",
    "displays",
    "featuring",
    "including",
    "nearby",
    "observing",
    "passing",
    "possibly",
    "providing",
    "seem",
    "seen",
    "shows",
    "situated",
    "standing",
    "towards",
    "using",
    "watching",
}
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
        "--regen_examples_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_examples.json",
    )
    parser.add_argument(
        "--audit_json",
        default="detection/baselines/results/tdev_caption_verified_expansion_100_audit/closed_loop_example_audit.json",
    )
    parser.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    parser.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_caption_verified_local_additions_100",
    )
    parser.add_argument("--max_added_sentences", type=int, default=2)
    parser.add_argument("--min_sentence_words", type=int, default=8)
    parser.add_argument("--max_overlap", type=float, default=0.55)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(str(text).split())


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


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
    return chunks


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


def is_nonobject_claim(claim: str) -> bool:
    tokens = re.findall(r"[a-z0-9-]+", str(claim).strip().lower())
    return not tokens or all(token in GENERIC_OR_NONOBJECT_CLAIMS for token in tokens)


def unsafe_claims(audit_row: dict[str, Any], candidate_caption: str) -> dict[str, list[str]]:
    candidate_lower = candidate_caption.lower()
    denied_residual = [
        str(word)
        for word in audit_row.get("denied_words", [])
        if phrase_pattern(str(word)).search(candidate_lower)
    ]
    absent_introduced = [
        str(claim)
        for claim, score in audit_row.get("introduced_claim_scores", {}).items()
        if int(score.get("two_stage_present", 0)) == 0
    ]
    open_vocab = [
        str(claim)
        for claim in audit_row.get("unsupported_open_vocab_claims", [])
        if not is_nonobject_claim(str(claim))
    ]
    variant_leaks = [str(hit.get("alias", "")) for hit in audit_row.get("introduced_variant_leaks", [])]
    root_leaks = [str(hit.get("token", "")) for hit in audit_row.get("introduced_root_leaks", [])]
    return {
        "denied_residual": sorted({item for item in denied_residual if item}),
        "absent_introduced": sorted({item for item in absent_introduced if item}),
        "unsupported_open_vocab": sorted({item for item in open_vocab if item}),
        "variant_or_root_leaks": sorted({item for item in variant_leaks + root_leaks if item}),
    }


def has_unsafe(claims_by_source: dict[str, list[str]]) -> bool:
    return any(claims for claims in claims_by_source.values())


def choose_additions(repaired: str, candidate: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    repaired_sentences = sentence_chunks(repaired)
    additions: list[dict[str, Any]] = []
    for sentence in sentence_chunks(candidate):
        if word_count(sentence) < args.min_sentence_words:
            continue
        best_overlap = max((overlap(sentence, base) for base in repaired_sentences), default=0.0)
        if best_overlap > args.max_overlap:
            continue
        additions.append({"text": sentence, "best_overlap": best_overlap, "words": word_count(sentence)})
        if len(additions) >= args.max_added_sentences:
            break
    return additions


def normalize_caption(chunks: list[str]) -> str:
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


def chair_eval(rows: list[dict[str, Any]], chair_pkl: str, output_dir: Path, prefix: str) -> dict[str, Any]:
    evaluator = load_chair_evaluator(chair_pkl)
    per_sample, overall = evaluate_chair(
        evaluator,
        data=rows,
        json_path=str(output_dir / f"{prefix}_chair_input.json"),
    )
    details = []
    total_mentions = 0
    total_hallucinated = 0
    for sample in per_sample:
        generated = list(sample.get("mscoco_generated_words", []))
        grounded = set(sample.get("mscoco_gt_words", []))
        hallucinated = [word for word in generated if word not in grounded]
        total_mentions += len(generated)
        total_hallucinated += len(hallucinated)
        details.append(
            {
                "image_id": int(sample["image_id"]),
                "caption": sample.get("caption", ""),
                "generated_words": generated,
                "hallucinated_words": hallucinated,
            }
        )
    details_path = output_dir / f"{prefix}_chair_details.json"
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "overall": overall,
        "total_object_mentions": total_mentions,
        "total_hallucinated_mentions": total_hallucinated,
        "details_file": str(details_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regen_rows = read_json(Path(args.regen_examples_json))
    audit_by_image = {int(row["image_id"]): row for row in read_json(Path(args.audit_json))["examples"]}

    examples = []
    vanilla_rows = []
    gated_rows = []
    repaired_rows = []
    candidate_rows = []
    selected_rows = []
    for row in regen_rows:
        image_id = int(row["image_id"])
        repaired = str(row.get("repaired_caption", "")).strip()
        candidate = str(row.get("regenerated_caption", "")).strip()
        blocked = unsafe_claims(audit_by_image[image_id], candidate)
        additions = [] if has_unsafe(blocked) else choose_additions(repaired, candidate, args)
        selected = normalize_caption([repaired] + [item["text"] for item in additions])
        examples.append(
            {
                "image_id": image_id,
                "image_path": row["image_path"],
                "num_added_sentences": len(additions),
                "added_sentences": additions,
                "blocked_claims": blocked,
                "vanilla_caption": row.get("vanilla_caption", ""),
                "gated_caption": row.get("gated_caption", ""),
                "repaired_caption": repaired,
                "candidate_caption": candidate,
                "selected_caption": selected,
                "selected_words": word_count(selected),
            }
        )
        vanilla_rows.append({"image_id": image_id, "caption": row.get("vanilla_caption", "")})
        gated_rows.append({"image_id": image_id, "caption": row.get("gated_caption", "")})
        repaired_rows.append({"image_id": image_id, "caption": repaired})
        candidate_rows.append({"image_id": image_id, "caption": candidate})
        selected_rows.append({"image_id": image_id, "caption": selected})

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "repaired": chair_eval(repaired_rows, args.chair_pkl, output_dir, "repaired"),
        "candidate": chair_eval(candidate_rows, args.chair_pkl, output_dir, "candidate"),
        "selected": chair_eval(selected_rows, args.chair_pkl, output_dir, "selected"),
    }
    metrics = {
        "regen_examples_json": args.regen_examples_json,
        "audit_json": args.audit_json,
        "num_examples": len(examples),
        "max_added_sentences": args.max_added_sentences,
        "min_sentence_words": args.min_sentence_words,
        "max_overlap": args.max_overlap,
        "images_with_additions": sum(int(row["num_added_sentences"] > 0) for row in examples),
        "total_added_sentences": sum(row["num_added_sentences"] for row in examples),
        "mean_words": {
            "repaired": sum(word_count(row["repaired_caption"]) for row in examples) / len(examples),
            "candidate": sum(word_count(row["candidate_caption"]) for row in examples) / len(examples),
            "selected": sum(word_count(row["selected_caption"]) for row in examples) / len(examples),
        },
        "chair": chair,
        "scope_note": (
            "Verified local-addition prototype. It preserves the repaired caption and only appends "
            "low-overlap sentences from a candidate whose introduced claims pass closed-loop "
            "target-vs-neighbor verification."
        ),
    }
    (output_dir / "verified_local_addition_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "verified_local_addition_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Verified Caption Local Additions",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| images with additions | {metrics['images_with_additions']} |",
        f"| total added sentences | {metrics['total_added_sentences']} |",
        f"| selected mean words | {metrics['mean_words']['selected']:.2f} |",
        f"| selected CHAIRi | {chair['selected']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| repaired CHAIRi | {chair['repaired']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| selected hallucinated mentions | {chair['selected']['total_hallucinated_mentions']} |",
        f"| repaired hallucinated mentions | {chair['repaired']['total_hallucinated_mentions']} |",
    ]
    (output_dir / "verified_local_addition_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
