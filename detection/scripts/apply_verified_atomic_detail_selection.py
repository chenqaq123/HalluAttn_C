#!/usr/bin/env python3
"""Select atomic detail additions after closed-loop verification."""

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
    "made",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atomic_examples_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_gen_100/atomic_detail_generation_examples.json",
    )
    parser.add_argument(
        "--audit_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_gen_100_audit/closed_loop_example_audit.json",
    )
    parser.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    parser.add_argument("--output_dir", default="detection/baselines/results/tdev_caption_atomic_detail_select_100")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(str(text).split())


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


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
    atomic_rows = read_json(Path(args.atomic_examples_json))
    audit_by_image = {int(row["image_id"]): row for row in read_json(Path(args.audit_json))["examples"]}

    examples = []
    vanilla_rows = []
    gated_rows = []
    repaired_rows = []
    augmented_rows = []
    selected_rows = []
    for row in atomic_rows:
        image_id = int(row["image_id"])
        repaired = str(row.get("repaired_caption", "")).strip()
        augmented = str(row.get("augmented_caption", "")).strip()
        blocked = unsafe_claims(audit_by_image[image_id], augmented)
        accept = bool(row.get("atomic_detail_additions", [])) and not has_unsafe(blocked)
        selected = augmented if accept else repaired
        examples.append(
            {
                "image_id": image_id,
                "image_path": row.get("image_path", ""),
                "selected_source": "verified_atomic_additions" if accept else "claim_local_repair",
                "blocked_claims": blocked,
                "atomic_detail_additions": row.get("atomic_detail_additions", []),
                "repaired_caption": repaired,
                "augmented_caption": augmented,
                "selected_caption": selected,
                "selected_words": word_count(selected),
            }
        )
        vanilla_rows.append({"image_id": image_id, "caption": row.get("vanilla_caption", "")})
        gated_rows.append({"image_id": image_id, "caption": row.get("gated_caption", "")})
        repaired_rows.append({"image_id": image_id, "caption": repaired})
        augmented_rows.append({"image_id": image_id, "caption": augmented})
        selected_rows.append({"image_id": image_id, "caption": selected})

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "repaired": chair_eval(repaired_rows, args.chair_pkl, output_dir, "repaired"),
        "augmented": chair_eval(augmented_rows, args.chair_pkl, output_dir, "augmented"),
        "selected": chair_eval(selected_rows, args.chair_pkl, output_dir, "selected"),
    }
    selected_count = sum(1 for row in examples if row["selected_source"] == "verified_atomic_additions")
    metrics = {
        "atomic_examples_json": args.atomic_examples_json,
        "audit_json": args.audit_json,
        "num_examples": len(examples),
        "selected_atomic_additions": selected_count,
        "selected_repair_fallback": len(examples) - selected_count,
        "mean_words": {
            "repaired": sum(word_count(row["repaired_caption"]) for row in examples) / len(examples),
            "augmented": sum(word_count(row["augmented_caption"]) for row in examples) / len(examples),
            "selected": sum(word_count(row["selected_caption"]) for row in examples) / len(examples),
        },
        "chair": chair,
        "scope_note": (
            "Verified atomic detail selection. Selection uses only closed-loop target-vs-neighbor "
            "audit results; CHAIR is used only after selection for evaluation."
        ),
    }
    (output_dir / "verified_atomic_detail_selection_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "verified_atomic_detail_selection_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Verified Atomic Detail Selection",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| selected atomic additions | {metrics['selected_atomic_additions']} |",
        f"| selected repair fallback | {metrics['selected_repair_fallback']} |",
        f"| selected mean words | {metrics['mean_words']['selected']:.2f} |",
        f"| selected CHAIRi | {chair['selected']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| repaired CHAIRi | {chair['repaired']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| selected hallucinated mentions | {chair['selected']['total_hallucinated_mentions']} |",
        f"| repaired hallucinated mentions | {chair['repaired']['total_hallucinated_mentions']} |",
    ]
    (output_dir / "verified_atomic_detail_selection_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
