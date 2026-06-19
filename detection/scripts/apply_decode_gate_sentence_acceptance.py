#!/usr/bin/env python3
"""Offline sentence-level candidate acceptance for decode-gate smoke outputs.

This prototype tests a stricter policy than tail-only sentence repair: after a
caption is generated, reject any sentence/unfinished tail that contains a
residual denied claim or a newly introduced claim that the closed-loop audit
marks unsupported. It does not generate replacement text.
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

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=\s|$)")

# The open-vocabulary extractor intentionally over-generates. These tokens are
# useful for debugging detector prompts, but they should not drive sentence
# rejection by themselves.
GENERIC_UNSUPPORTED_CLAIMS = {
    "including",
    "nearby",
    "observing",
    "passing",
    "seem",
    "standing",
    "towards",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--examples_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96/gated_generation_examples.json",
    )
    p.add_argument(
        "--audit_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_audit_ov96/closed_loop_example_audit.json",
    )
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance",
    )
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def sentence_chunks(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    chunks: list[dict[str, Any]] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(stripped):
        end = match.end()
        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "complete": True})
        start = end
    tail = stripped[start:].strip()
    if tail:
        chunks.append({"text": tail, "complete": False})
    return chunks


def normalize_text(chunks: list[str]) -> str:
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()


def claim_hits(text: str, claims: list[str]) -> list[str]:
    return [claim for claim in claims if phrase_pattern(claim).search(text.lower())]


def unsupported_claims(audit_row: dict[str, Any], gated_caption: str) -> dict[str, list[str]]:
    gated_lower = gated_caption.lower()
    denied_residual = [
        word
        for word in audit_row.get("denied_words", [])
        if phrase_pattern(str(word)).search(gated_lower)
    ]
    introduced_hallucinated = [str(word) for word in audit_row.get("introduced_hallucinated_words", [])]
    open_vocab = []
    for claim in audit_row.get("unsupported_open_vocab_claims", []):
        claim = str(claim).strip().lower()
        if claim and claim not in GENERIC_UNSUPPORTED_CLAIMS:
            open_vocab.append(claim)
    variant_leaks = [str(hit.get("alias", "")) for hit in audit_row.get("introduced_variant_leaks", [])]
    root_leaks = [str(hit.get("token", "")) for hit in audit_row.get("introduced_root_leaks", [])]
    return {
        "denied_residual": sorted(set(denied_residual)),
        "introduced_hallucinated": sorted(set(introduced_hallucinated)),
        "unsupported_open_vocab": sorted(set(open_vocab)),
        "variant_or_root_leaks": sorted(set([item for item in variant_leaks + root_leaks if item])),
    }


def accept_caption(caption: str, claims_by_source: dict[str, list[str]]) -> tuple[str, dict[str, Any]]:
    all_claims = sorted({claim for claims in claims_by_source.values() for claim in claims})
    kept: list[str] = []
    removed: list[dict[str, Any]] = []
    for chunk in sentence_chunks(caption):
        hits = claim_hits(chunk["text"], all_claims)
        if hits:
            removed.append(
                {
                    "text": chunk["text"],
                    "complete": int(chunk["complete"]),
                    "claims": hits,
                    "reason": "unsupported_claim",
                }
            )
            continue
        if not chunk["complete"]:
            removed.append(
                {
                    "text": chunk["text"],
                    "complete": 0,
                    "claims": [],
                    "reason": "incomplete_tail",
                }
            )
            continue
        kept.append(chunk["text"])
    return normalize_text(kept), {"removed_chunks": removed, "claims_by_source": claims_by_source}


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


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    args = parse_args()
    examples = read_json(Path(args.examples_json))
    audit_payload = read_json(Path(args.audit_json))
    audit_by_image = {int(row["image_id"]): row for row in audit_payload["examples"]}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_examples = []
    vanilla_rows = []
    gated_rows = []
    accepted_rows = []
    for example in examples:
        image_id = int(example["image_id"])
        audit_row = audit_by_image[image_id]
        gated_caption = example.get("gated_caption", "")
        claims_by_source = unsupported_claims(audit_row, gated_caption)
        accepted_caption, acceptance_info = accept_caption(gated_caption, claims_by_source)
        vanilla_caption = example.get("vanilla_caption", "")
        accepted_examples.append(
            {
                "image_id": image_id,
                "image_path": example.get("image_path", ""),
                "denied_words": audit_row.get("denied_words", []),
                "vanilla_caption": vanilla_caption,
                "gated_caption": gated_caption,
                "accepted_caption": accepted_caption,
                "acceptance_info": acceptance_info,
                "vanilla_words": word_count(vanilla_caption),
                "gated_words": word_count(gated_caption),
                "accepted_words": word_count(accepted_caption),
                "removed_words_by_acceptance": word_count(gated_caption) - word_count(accepted_caption),
            }
        )
        vanilla_rows.append({"image_id": image_id, "caption": vanilla_caption})
        gated_rows.append({"image_id": image_id, "caption": gated_caption})
        accepted_rows.append({"image_id": image_id, "caption": accepted_caption})

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "accepted": chair_eval(accepted_rows, args.chair_pkl, output_dir, "accepted"),
    }
    num_changed = sum(int(row["accepted_caption"] != row["gated_caption"]) for row in accepted_examples)
    num_removed_chunks = sum(len(row["acceptance_info"]["removed_chunks"]) for row in accepted_examples)
    metrics = {
        "examples_json": args.examples_json,
        "audit_json": args.audit_json,
        "num_examples": len(accepted_examples),
        "num_changed": num_changed,
        "num_removed_chunks": num_removed_chunks,
        "mean_gated_words": sum(row["gated_words"] for row in accepted_examples) / len(accepted_examples) if accepted_examples else 0.0,
        "mean_accepted_words": sum(row["accepted_words"] for row in accepted_examples) / len(accepted_examples) if accepted_examples else 0.0,
        "mean_removed_words_by_acceptance": (
            sum(row["removed_words_by_acceptance"] for row in accepted_examples) / len(accepted_examples)
            if accepted_examples
            else 0.0
        ),
        "chair": chair,
        "scope_note": (
            "Offline sentence-level candidate acceptance proxy. It rejects generated sentences/tails "
            "containing residual denied claims or unsupported introduced claims from the closed-loop audit; "
            "it does not generate replacements."
        ),
    }

    (output_dir / "sentence_acceptance_examples.json").write_text(
        json.dumps(accepted_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sentence_acceptance_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| changed captions | {metrics['num_changed']} |",
        f"| removed chunks | {metrics['num_removed_chunks']} |",
        f"| mean gated words | {metrics['mean_gated_words']:.2f} |",
        f"| mean accepted words | {metrics['mean_accepted_words']:.2f} |",
        f"| mean removed words | {metrics['mean_removed_words_by_acceptance']:.2f} |",
        f"| gated CHAIRi | {chair['gated']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| accepted CHAIRi | {chair['accepted']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| gated hallucinated mentions | {chair['gated']['total_hallucinated_mentions']} |",
        f"| accepted hallucinated mentions | {chair['accepted']['total_hallucinated_mentions']} |",
    ]
    (output_dir / "sentence_acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
