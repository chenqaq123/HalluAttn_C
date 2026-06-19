#!/usr/bin/env python3
"""Smoke-test the decode-time TDEV object-phrase gate without loading a VLM."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
EDIT_SCRIPT = REPO_ROOT / "detection" / "scripts" / "evaluate_tdev_caption_edit.py"
if str(DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DETECTION_SRC))

from sinkdetect.decode_gate import ObjectPhraseGateLogitsProcessor, build_object_phrase_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mention_matches_csv",
        default="detection/baselines/results/tdev_decode_gate_feasibility/mention_token_matches.csv",
    )
    p.add_argument(
        "--model_path",
        default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    )
    p.add_argument("--chair_source", default="../pas/src/pas/evaluate/chair.py")
    p.add_argument("--output_dir", default="detection/baselines/results/tdev_decode_gate_smoke")
    p.add_argument("--all_mentions", action="store_true", help="Smoke-test all matched mentions, not only TDEV-selected rows.")
    p.add_argument("--max_examples", type=int, default=50)
    return p.parse_args()


def load_edit_module() -> Any:
    spec = importlib.util.spec_from_file_location("tdev_caption_edit", EDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def parse_token_ids(value: str) -> list[int]:
    return [int(item) for item in value.split() if item.strip()]


def is_banned(value: torch.Tensor) -> bool:
    return bool(torch.isneginf(value).item())


def main() -> None:
    args = parse_args()
    edit = load_edit_module()
    rows = read_rows(Path(args.mention_matches_csv))
    selected_only = not args.all_mentions
    if selected_only:
        rows = [row for row in rows if row.get("selected") == "1"]
    rows = [
        row
        for row in rows
        if row.get("near_gen_pos_match") == "1" and row.get("matched_tokens", "").strip()
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    synonyms = edit.load_synonyms(Path(args.chair_source))
    vocab_size = len(tokenizer)

    examples: list[dict[str, object]] = []
    first_blocked = 0
    any_step_blocked = 0
    all_steps_blocked = 0
    total_steps = 0
    blocked_steps = 0

    for row in rows:
        matched = parse_token_ids(row["matched_tokens"])
        phrases = object_phrases(row["word"].strip().lower(), synonyms, edit)
        denied = build_object_phrase_sequences(tokenizer, phrases)
        processor = ObjectPhraseGateLogitsProcessor(
            tokenizer=tokenizer,
            denied_token_sequences=[denied],
            prompt_lengths=0,
            denied_phrase_texts=[[row["word"]] * len(denied)],
            audit_limit=4,
        )

        step_results: list[bool] = []
        for prefix_len in range(len(matched)):
            prefix = matched[:prefix_len]
            next_token = matched[prefix_len]
            input_ids = torch.tensor([prefix], dtype=torch.long)
            scores = torch.zeros((1, vocab_size), dtype=torch.float32)
            scores = processor(input_ids, scores)
            blocked = is_banned(scores[0, next_token])
            step_results.append(blocked)
            total_steps += 1
            blocked_steps += int(blocked)

        first_blocked += int(bool(step_results and step_results[0]))
        any_step_blocked += int(any(step_results))
        all_steps_blocked += int(all(step_results) if step_results else False)
        if len(examples) < args.max_examples:
            examples.append(
                {
                    "object_id": row["object_id"],
                    "image_id": row["image_id"],
                    "word": row["word"],
                    "label": int(row["label"]),
                    "matched_text": row["matched_text"],
                    "matched_tokens": matched,
                    "num_denied_sequences": len(denied),
                    "first_token_blocked": int(bool(step_results and step_results[0])),
                    "any_step_blocked": int(any(step_results)),
                    "all_steps_blocked": int(all(step_results) if step_results else False),
                    "step_blocked": [int(item) for item in step_results],
                    "audit_events": [event.__dict__ for event in processor.events],
                }
            )

    n = len(rows)
    metrics = {
        "mention_matches_csv": args.mention_matches_csv,
        "model_path": args.model_path,
        "selected_only": selected_only,
        "tested_mentions": n,
        "first_token_blocked": first_blocked,
        "first_token_block_rate": first_blocked / n if n else 0.0,
        "any_step_blocked": any_step_blocked,
        "any_step_block_rate": any_step_blocked / n if n else 0.0,
        "all_steps_blocked": all_steps_blocked,
        "all_steps_block_rate": all_steps_blocked / n if n else 0.0,
        "tested_generation_steps": total_steps,
        "blocked_generation_steps": blocked_steps,
        "step_block_rate": blocked_steps / total_steps if total_steps else 0.0,
        "all_steps_passed": bool(n and all_steps_blocked == n and blocked_steps == total_steps),
        "interpretation": (
            "This smoke test does not evaluate caption quality. It verifies that "
            "the prefix-state LogitsProcessor can suppress token sequences that "
            "were selected by the TDEV caption feasibility audit."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "decode_gate_smoke_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "decode_gate_smoke_examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["all_steps_passed"]:
        raise SystemExit("decode gate smoke test did not block every tested selected phrase step")


if __name__ == "__main__":
    main()
