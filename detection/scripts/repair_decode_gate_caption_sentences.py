#!/usr/bin/env python3
"""Sentence-level repair for gated caption smoke outputs.

This is an offline proxy for a decode-time stop/repair policy: if a generated
caption ends with an incomplete sentence fragment, keep only the prefix through
the last complete sentence boundary. It never invents new content.
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
from sinkdetect.open_vocab_claims import open_vocab_candidates

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--examples_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2/gated_generation_examples.json",
    )
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_sentence_repair",
    )
    p.add_argument("--open_vocab_min_token_len", type=int, default=3)
    p.add_argument("--open_vocab_candidate_limit", type=int, default=32)
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repair_caption(caption: str) -> tuple[str, dict[str, Any]]:
    text = caption.strip()
    if not text:
        return text, {"repaired": 0, "removed_tail": "", "reason": "empty"}
    if re.search(r"[.!?][\"')\]]*$", text):
        return text, {"repaired": 0, "removed_tail": "", "reason": "already_complete"}
    matches = list(SENTENCE_END_RE.finditer(text))
    if not matches:
        return text, {"repaired": 0, "removed_tail": "", "reason": "no_sentence_boundary"}
    cut = matches[-1].end()
    repaired = text[:cut].strip()
    removed = text[cut:].strip()
    if not removed:
        return text, {"repaired": 0, "removed_tail": "", "reason": "empty_tail"}
    return repaired, {"repaired": 1, "removed_tail": removed, "reason": "trim_incomplete_tail"}


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
    (output_dir / f"{prefix}_chair_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "overall": overall,
        "total_object_mentions": total_mentions,
        "total_hallucinated_mentions": total_hallucinated,
        "details_file": str(output_dir / f"{prefix}_chair_details.json"),
    }


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:
    args = parse_args()
    examples = read_json(Path(args.examples_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repaired_examples = []
    vanilla_rows = []
    gated_rows = []
    repaired_rows = []
    for example in examples:
        image_id = int(example["image_id"])
        repaired, info = repair_caption(example["gated_caption"])
        vanilla_caption = example.get("vanilla_caption", "")
        gated_caption = example.get("gated_caption", "")
        vanilla_candidates = open_vocab_candidates(
            vanilla_caption,
            min_len=args.open_vocab_min_token_len,
            limit=args.open_vocab_candidate_limit,
        )
        gated_candidates = open_vocab_candidates(
            gated_caption,
            min_len=args.open_vocab_min_token_len,
            limit=args.open_vocab_candidate_limit,
        )
        repaired_candidates = open_vocab_candidates(
            repaired,
            min_len=args.open_vocab_min_token_len,
            limit=args.open_vocab_candidate_limit,
        )
        repaired_examples.append(
            {
                "image_id": image_id,
                "image_path": example.get("image_path", ""),
                "denied_words": [item.get("word", "") for item in example.get("denied_items", [])],
                "vanilla_caption": vanilla_caption,
                "gated_caption": gated_caption,
                "repaired_caption": repaired,
                "repair_info": info,
                "vanilla_words": word_count(vanilla_caption),
                "gated_words": word_count(gated_caption),
                "repaired_words": word_count(repaired),
                "removed_words_by_repair": word_count(gated_caption) - word_count(repaired),
                "vanilla_open_vocab_candidates": vanilla_candidates,
                "gated_open_vocab_candidates": gated_candidates,
                "repaired_open_vocab_candidates": repaired_candidates,
                "repair_removed_open_vocab_candidates": [
                    item for item in gated_candidates if item not in set(repaired_candidates)
                ],
            }
        )
        vanilla_rows.append({"image_id": image_id, "caption": vanilla_caption})
        gated_rows.append({"image_id": image_id, "caption": gated_caption})
        repaired_rows.append({"image_id": image_id, "caption": repaired})

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "repaired": chair_eval(repaired_rows, args.chair_pkl, output_dir, "repaired"),
    }
    num_repaired = sum(int(row["repair_info"]["repaired"]) for row in repaired_examples)
    metrics = {
        "examples_json": args.examples_json,
        "num_examples": len(repaired_examples),
        "num_repaired": num_repaired,
        "mean_gated_words": sum(row["gated_words"] for row in repaired_examples) / len(repaired_examples) if repaired_examples else 0.0,
        "mean_repaired_words": sum(row["repaired_words"] for row in repaired_examples) / len(repaired_examples) if repaired_examples else 0.0,
        "mean_removed_words_by_repair": sum(row["removed_words_by_repair"] for row in repaired_examples) / len(repaired_examples) if repaired_examples else 0.0,
        "chair": chair,
        "scope_note": (
            "Offline sentence-level repair proxy. It trims incomplete trailing sentence fragments "
            "from already generated captions and does not generate replacement text."
        ),
    }
    (output_dir / "sentence_repair_examples.json").write_text(
        json.dumps(repaired_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sentence_repair_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| repaired captions | {metrics['num_repaired']} |",
        f"| mean gated words | {metrics['mean_gated_words']:.2f} |",
        f"| mean repaired words | {metrics['mean_repaired_words']:.2f} |",
        f"| mean removed words | {metrics['mean_removed_words_by_repair']:.2f} |",
        f"| gated CHAIRi | {chair['gated']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| repaired CHAIRi | {chair['repaired']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| gated hallucinated mentions | {chair['gated']['total_hallucinated_mentions']} |",
        f"| repaired hallucinated mentions | {chair['repaired']['total_hallucinated_mentions']} |",
    ]
    (output_dir / "sentence_repair_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
