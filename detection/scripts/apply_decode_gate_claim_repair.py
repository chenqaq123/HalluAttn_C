#!/usr/bin/env python3
"""Claim-local repair for decode-gate caption smoke outputs.

This is a conservative follow-up to sentence-level acceptance. Instead of
dropping every sentence that contains an unsupported object claim, it first tries
to remove a trailing speculative or enumerating clause that contains the
unsupported claim. If the remaining sentence is still complete and claim-clean,
it is kept; otherwise the whole chunk is rejected. The script does not generate
new text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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
CLAUSE_MARKERS = [
    ", possibly",
    " possibly",
    ", including",
    " including",
    ", such as",
    " such as",
    " with ",
]

GENERIC_UNSUPPORTED_CLAIMS = {
    "including",
    "nearby",
    "observing",
    "passing",
    "for",
    "providing",
    "seem",
    "standing",
    "towards",
    "watching",
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
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_claim_repair",
    )
    p.add_argument("--min_repaired_chunk_words", type=int, default=8)
    p.add_argument("--min_words_for_nonempty", type=int, default=8)
    p.add_argument("--min_object_mentions_for_nonempty", type=int, default=1)
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


def word_count(text: str) -> int:
    return len(text.split())


def claim_hits(text: str, claims: list[str]) -> list[str]:
    lowered = text.lower()
    return [claim for claim in claims if phrase_pattern(claim).search(lowered)]


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


def first_hit_start(text: str, claims: list[str]) -> int | None:
    lowered = text.lower()
    starts = [
        match.start()
        for claim in claims
        for match in [phrase_pattern(claim).search(lowered)]
        if match is not None
    ]
    return min(starts) if starts else None


def ensure_period(text: str) -> str:
    text = text.strip(" ,;:")
    if not text:
        return text
    if re.search(r"[.!?][\"')\]]*$", text):
        return text
    return text + "."


def local_clause_repair(chunk: str, hits: list[str], all_claims: list[str], min_words: int) -> tuple[str, str]:
    first_hit = first_hit_start(chunk, hits)
    if first_hit is None:
        return "", "no_hit"
    lowered = chunk.lower()
    candidates: list[tuple[int, str]] = []
    for marker in CLAUSE_MARKERS:
        start = 0
        while True:
            idx = lowered.find(marker, start)
            if idx < 0:
                break
            if idx < first_hit:
                candidates.append((idx, marker.strip()))
            start = idx + 1
    for idx, marker in sorted(candidates, reverse=True):
        candidate = ensure_period(chunk[:idx])
        if word_count(candidate) < min_words:
            continue
        if claim_hits(candidate, all_claims):
            continue
        return candidate, f"trim_clause_before_{marker}"
    return "", "no_safe_clause"


def repair_caption(
    caption: str,
    claims_by_source: dict[str, list[str]],
    min_repaired_chunk_words: int,
) -> tuple[str, dict[str, Any]]:
    all_claims = sorted({claim for claims in claims_by_source.values() for claim in claims})
    kept: list[str] = []
    actions: list[dict[str, Any]] = []
    for chunk in sentence_chunks(caption):
        hits = claim_hits(chunk["text"], all_claims)
        if hits:
            repaired, reason = local_clause_repair(
                chunk["text"],
                hits,
                all_claims,
                min_repaired_chunk_words,
            )
            if repaired:
                kept.append(repaired)
                actions.append(
                    {
                        "text": chunk["text"],
                        "kept_text": repaired,
                        "complete": int(chunk["complete"]),
                        "claims": hits,
                        "reason": reason,
                        "action": "local_repair",
                    }
                )
            else:
                actions.append(
                    {
                        "text": chunk["text"],
                        "kept_text": "",
                        "complete": int(chunk["complete"]),
                        "claims": hits,
                        "reason": "unsupported_claim",
                        "repair_attempt": reason,
                        "action": "drop_chunk",
                    }
                )
            continue
        if not chunk["complete"]:
            actions.append(
                {
                    "text": chunk["text"],
                    "kept_text": "",
                    "complete": 0,
                    "claims": [],
                    "reason": "incomplete_tail",
                    "action": "drop_chunk",
                }
            )
            continue
        kept.append(chunk["text"])
    return normalize_text(kept), {"actions": actions, "claims_by_source": claims_by_source}


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


def count_items(items: list[str]) -> Counter[str]:
    return Counter(str(item) for item in items)


def grounded_counter(row: dict[str, Any]) -> Counter[str]:
    generated = count_items(row.get("generated_words", []))
    hallucinated = count_items(row.get("hallucinated_words", []))
    grounded = Counter(generated)
    grounded.subtract(hallucinated)
    return Counter({key: value for key, value in grounded.items() if value > 0})


def overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum((left & right).values())


def total(counter: Counter[str]) -> int:
    return sum(counter.values())


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def preservation_summary(output_dir: Path, examples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    variants = {
        "vanilla": read_json(output_dir / "vanilla_chair_details.json"),
        "gated": read_json(output_dir / "gated_chair_details.json"),
        "repaired": read_json(output_dir / "repaired_chair_details.json"),
    }
    by_variant = {
        name: {int(row["image_id"]): row for row in rows}
        for name, rows in variants.items()
    }
    image_ids = sorted(set(by_variant["vanilla"]) & set(by_variant["gated"]) & set(by_variant["repaired"]))
    totals = Counter()
    per_image = []
    example_by_image = {int(row["image_id"]): row for row in examples}
    for image_id in image_ids:
        rows = {name: by_variant[name][image_id] for name in variants}
        grounded = {name: grounded_counter(row) for name, row in rows.items()}
        hallucinated = {name: count_items(row.get("hallucinated_words", [])) for name, row in rows.items()}
        generated = {name: count_items(row.get("generated_words", [])) for name, row in rows.items()}
        words = {name: word_count(rows[name].get("caption", "")) for name in rows}
        repaired_object_mentions = total(generated["repaired"])
        generic_or_empty = int(
            words["repaired"] < args.min_words_for_nonempty
            or repaired_object_mentions < args.min_object_mentions_for_nonempty
        )
        row = {
            "image_id": image_id,
            "vanilla_words": words["vanilla"],
            "gated_words": words["gated"],
            "repaired_words": words["repaired"],
            "vanilla_object_mentions": total(generated["vanilla"]),
            "gated_object_mentions": total(generated["gated"]),
            "repaired_object_mentions": repaired_object_mentions,
            "vanilla_hallucinated_mentions": total(hallucinated["vanilla"]),
            "gated_hallucinated_mentions": total(hallucinated["gated"]),
            "repaired_hallucinated_mentions": total(hallucinated["repaired"]),
            "vanilla_grounded_mentions": total(grounded["vanilla"]),
            "gated_grounded_mentions": total(grounded["gated"]),
            "repaired_grounded_mentions": total(grounded["repaired"]),
            "repaired_retained_vanilla_grounded_mentions": overlap_count(grounded["repaired"], grounded["vanilla"]),
            "repaired_retained_gated_grounded_mentions": overlap_count(grounded["repaired"], grounded["gated"]),
            "generic_or_empty_repaired": generic_or_empty,
            "local_repair_actions": sum(1 for action in example_by_image[image_id]["repair_info"]["actions"] if action["action"] == "local_repair"),
            "dropped_chunks": sum(1 for action in example_by_image[image_id]["repair_info"]["actions"] if action["action"] == "drop_chunk"),
        }
        per_image.append(row)
        for key, value in row.items():
            if key != "image_id":
                totals[key] += int(value)
    num_images = len(image_ids)
    summary = {
        "num_images": num_images,
        "mean_words": {
            name: rate(totals[f"{name}_words"], num_images) for name in ("vanilla", "gated", "repaired")
        },
        "object_mentions": {
            name: totals[f"{name}_object_mentions"] for name in ("vanilla", "gated", "repaired")
        },
        "grounded_mentions": {
            name: totals[f"{name}_grounded_mentions"] for name in ("vanilla", "gated", "repaired")
        },
        "hallucinated_mentions": {
            name: totals[f"{name}_hallucinated_mentions"] for name in ("vanilla", "gated", "repaired")
        },
        "repaired_retained_vanilla_grounded_rate": rate(
            totals["repaired_retained_vanilla_grounded_mentions"], totals["vanilla_grounded_mentions"]
        ),
        "repaired_retained_gated_grounded_rate": rate(
            totals["repaired_retained_gated_grounded_mentions"], totals["gated_grounded_mentions"]
        ),
        "repaired_hallucination_reduction_vs_gated": rate(
            totals["gated_hallucinated_mentions"] - totals["repaired_hallucinated_mentions"],
            totals["gated_hallucinated_mentions"],
        ),
        "repaired_hallucination_reduction_vs_vanilla": rate(
            totals["vanilla_hallucinated_mentions"] - totals["repaired_hallucinated_mentions"],
            totals["vanilla_hallucinated_mentions"],
        ),
        "repaired_object_mention_retention_vs_gated": rate(
            totals["repaired_object_mentions"], totals["gated_object_mentions"]
        ),
        "repaired_object_mention_retention_vs_vanilla": rate(
            totals["repaired_object_mentions"], totals["vanilla_object_mentions"]
        ),
        "generic_or_empty_repaired": totals["generic_or_empty_repaired"],
        "generic_or_empty_repaired_rate": rate(totals["generic_or_empty_repaired"], num_images),
        "local_repair_actions": totals["local_repair_actions"],
        "dropped_chunks": totals["dropped_chunks"],
    }
    return {"summary": summary, "per_image": per_image}


def main() -> None:
    args = parse_args()
    examples = read_json(Path(args.examples_json))
    audit_payload = read_json(Path(args.audit_json))
    audit_by_image = {int(row["image_id"]): row for row in audit_payload["examples"]}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repaired_examples = []
    vanilla_rows = []
    gated_rows = []
    repaired_rows = []
    for example in examples:
        image_id = int(example["image_id"])
        audit_row = audit_by_image[image_id]
        gated_caption = example.get("gated_caption", "")
        claims_by_source = unsupported_claims(audit_row, gated_caption)
        repaired_caption, repair_info = repair_caption(
            gated_caption,
            claims_by_source,
            min_repaired_chunk_words=args.min_repaired_chunk_words,
        )
        vanilla_caption = example.get("vanilla_caption", "")
        repaired_examples.append(
            {
                "image_id": image_id,
                "image_path": example.get("image_path", ""),
                "denied_words": audit_row.get("denied_words", []),
                "vanilla_caption": vanilla_caption,
                "gated_caption": gated_caption,
                "repaired_caption": repaired_caption,
                "repair_info": repair_info,
                "vanilla_words": word_count(vanilla_caption),
                "gated_words": word_count(gated_caption),
                "repaired_words": word_count(repaired_caption),
                "removed_words_by_repair": word_count(gated_caption) - word_count(repaired_caption),
            }
        )
        vanilla_rows.append({"image_id": image_id, "caption": vanilla_caption})
        gated_rows.append({"image_id": image_id, "caption": gated_caption})
        repaired_rows.append({"image_id": image_id, "caption": repaired_caption})

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "repaired": chair_eval(repaired_rows, args.chair_pkl, output_dir, "repaired"),
    }
    preservation = preservation_summary(output_dir, repaired_examples, args)
    num_changed = sum(int(row["repaired_caption"] != row["gated_caption"]) for row in repaired_examples)
    num_local_repairs = preservation["summary"]["local_repair_actions"]
    num_dropped = preservation["summary"]["dropped_chunks"]
    metrics = {
        "examples_json": args.examples_json,
        "audit_json": args.audit_json,
        "num_examples": len(repaired_examples),
        "num_changed": num_changed,
        "num_local_repair_actions": num_local_repairs,
        "num_dropped_chunks": num_dropped,
        "mean_gated_words": sum(row["gated_words"] for row in repaired_examples) / len(repaired_examples) if repaired_examples else 0.0,
        "mean_repaired_words": sum(row["repaired_words"] for row in repaired_examples) / len(repaired_examples) if repaired_examples else 0.0,
        "mean_removed_words_by_repair": (
            sum(row["removed_words_by_repair"] for row in repaired_examples) / len(repaired_examples)
            if repaired_examples
            else 0.0
        ),
        "chair": chair,
        "preservation": preservation,
        "scope_note": (
            "Offline claim-local repair proxy. It rejects unsupported claims from the closed-loop audit, "
            "but first tries to preserve a complete sentence prefix before speculative or enumerating "
            "clauses. It does not generate replacement text."
        ),
    }

    (output_dir / "claim_repair_examples.json").write_text(
        json.dumps(repaired_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "claim_repair_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = preservation["summary"]
    lines = [
        "# TDEV Claim-Local Repair Smoke",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| changed captions | {metrics['num_changed']} |",
        f"| local repair actions | {metrics['num_local_repair_actions']} |",
        f"| dropped chunks | {metrics['num_dropped_chunks']} |",
        f"| mean gated words | {metrics['mean_gated_words']:.2f} |",
        f"| mean repaired words | {metrics['mean_repaired_words']:.2f} |",
        f"| mean removed words | {metrics['mean_removed_words_by_repair']:.2f} |",
        f"| gated CHAIRi | {chair['gated']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| repaired CHAIRi | {chair['repaired']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| gated hallucinated mentions | {chair['gated']['total_hallucinated_mentions']} |",
        f"| repaired hallucinated mentions | {chair['repaired']['total_hallucinated_mentions']} |",
        f"| retained vanilla grounded mentions | {summary['repaired_retained_vanilla_grounded_rate']:.2%} |",
        f"| hallucination reduction vs gated | {summary['repaired_hallucination_reduction_vs_gated']:.2%} |",
        f"| object mention retention vs gated | {summary['repaired_object_mention_retention_vs_gated']:.2%} |",
        f"| generic/empty repaired captions | {summary['generic_or_empty_repaired']} |",
    ]
    (output_dir / "claim_repair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
