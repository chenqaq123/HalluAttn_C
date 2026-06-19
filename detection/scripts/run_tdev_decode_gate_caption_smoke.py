#!/usr/bin/env python3
"""Run a small LLaVA captioning smoke test with the TDEV object-phrase gate.

This is an oracle-input integration test: denied objects come from existing
TDEV-scored vanilla caption mentions. It verifies end-to-end generation with
``generate(logits_processor=...)``; it is not yet a final deployable method.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
EDIT_SCRIPT = REPO_ROOT / "detection" / "scripts" / "evaluate_tdev_caption_edit.py"
if str(DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DETECTION_SRC))

from sinkdetect.decode_gate import ObjectPhraseGateLogitsProcessor, build_object_phrase_sequences
from sinkdetect.utils import build_caption_prompt, load_model_and_processor


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
    p.add_argument("--coco_path", default="/home/chenguanxu/common_dataset/coco-2014-dataset")
    p.add_argument("--chair_source", default="../pas/src/pas/evaluate/chair.py")
    p.add_argument("--output_dir", default="detection/baselines/results/tdev_decode_gate_caption_smoke")
    p.add_argument("--score", default="hybrid_positive_branch_absence")
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--max_images", type=int, default=4)
    p.add_argument("--image_ids", default="", help="Optional comma-separated COCO image ids.")
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--device", type=int, default=5)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
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


def select_denied_words(rows: list[dict[str, str]], scores: np.ndarray, selected_mask: np.ndarray) -> dict[int, list[dict[str, Any]]]:
    denied: dict[int, list[dict[str, Any]]] = defaultdict(list)
    selected_indices = np.flatnonzero(selected_mask)
    selected_indices = selected_indices[np.argsort(-scores[selected_indices], kind="mergesort")]
    seen: set[tuple[int, str]] = set()
    for idx in selected_indices:
        row = rows[int(idx)]
        image_id = int(row["image_id"])
        word = row["word"].strip().lower()
        key = (image_id, word)
        if key in seen:
            continue
        seen.add(key)
        denied[image_id].append(
            {
                "word": word,
                "score": float(scores[int(idx)]),
                "label": int(row["label"]),
                "object_id": int(row["object_id"]),
            }
        )
    return dict(denied)


def choose_image_ids(args: argparse.Namespace, denied: dict[int, list[dict[str, Any]]]) -> list[int]:
    if args.image_ids.strip():
        return [int(item) for item in args.image_ids.split(",") if item.strip()]
    ranked = sorted(
        denied,
        key=lambda image_id: max(item["score"] for item in denied[image_id]),
        reverse=True,
    )
    return ranked[: args.max_images]


def image_path(coco_path: str, image_id: int) -> Path:
    return Path(coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"


def decode_generated(processor: Any, output_ids: torch.Tensor, prompt_len: int) -> str:
    return processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    edit = load_edit_module()
    rows = read_rows(Path(args.scores_csv))
    scores = edit.build_scores(rows, args)[args.score]
    selected_mask = edit.top_fraction_mask(scores, args.top_frac)
    denied_by_image = select_denied_words(rows, scores, selected_mask)
    image_ids = choose_image_ids(args, denied_by_image)
    synonyms = edit.load_synonyms(Path(args.chair_source))

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(args.model_path, device)
    prompt_text = build_caption_prompt()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for image_id in tqdm(image_ids, desc="TDEV gated captions"):
        path = image_path(args.coco_path, image_id)
        if not path.exists():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device, dtype=torch.float16)
        prompt_len = int(inputs["input_ids"].shape[1])

        denied_items = denied_by_image.get(image_id, [])
        denied_sequences: list[list[int]] = []
        denied_texts: list[str] = []
        for item in denied_items:
            phrases = object_phrases(item["word"], synonyms, edit)
            seqs = build_object_phrase_sequences(processor.tokenizer, phrases)
            denied_sequences.extend(seqs)
            denied_texts.extend([item["word"]] * len(seqs))

        gate = ObjectPhraseGateLogitsProcessor(
            tokenizer=processor.tokenizer,
            denied_token_sequences=[denied_sequences],
            prompt_lengths=prompt_len,
            denied_phrase_texts=[denied_texts],
            audit_limit=80,
        )

        with torch.no_grad():
            vanilla_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            gated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([gate]),
            )

        vanilla_caption = decode_generated(processor, vanilla_ids, prompt_len)
        gated_caption = decode_generated(processor, gated_ids, prompt_len)
        results.append(
            {
                "image_id": image_id,
                "image_path": str(path),
                "denied_items": denied_items,
                "num_denied_sequences": len(denied_sequences),
                "vanilla_caption": vanilla_caption,
                "gated_caption": gated_caption,
                "caption_changed": int(vanilla_caption != gated_caption),
                "gate_events": [event.__dict__ for event in gate.events],
            }
        )

        del inputs, vanilla_ids, gated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {
        "scores_csv": args.scores_csv,
        "score": args.score,
        "top_frac": args.top_frac,
        "model_path": args.model_path,
        "coco_path": args.coco_path,
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "num_images": len(results),
        "changed_captions": sum(row["caption_changed"] for row in results),
        "images_with_gate_events": sum(int(bool(row["gate_events"])) for row in results),
        "total_gate_events_saved": sum(len(row["gate_events"]) for row in results),
        "scope_note": (
            "Oracle smoke test: denied objects are selected from cached TDEV scores "
            "over vanilla captions. This proves generation integration, not final method quality."
        ),
    }
    with (output_dir / "gated_generation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "gated_generation_examples.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
