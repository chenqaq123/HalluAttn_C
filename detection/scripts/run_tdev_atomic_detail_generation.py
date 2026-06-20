#!/usr/bin/env python3
"""Generate atomic missing-detail caption spans from a repaired draft.

This script is deliberately narrower than whole-caption regeneration. It keeps
the claim-local repaired caption as the safe draft and asks the VLM for at most
two short new visual-detail sentences. The output is prepared for the existing
closed-loop verifier by comparing repaired captions against augmented captions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
PAS_SRC = REPO_ROOT.parent / "pas" / "src"
for import_path in (DETECTION_SRC, PAS_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sinkdetect.chair import evaluate_chair, load_chair_evaluator
from sinkdetect.utils import LLAVA_SYSTEM_PROMPT

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
        "--audit_compatible_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_audit_compatible_examples.json",
    )
    parser.add_argument(
        "--model_path",
        default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    )
    parser.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    parser.add_argument("--output_dir", default="detection/baselines/results/tdev_caption_atomic_detail_gen_100")
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--max_new_tokens", type=int, default=72)
    parser.add_argument("--max_additions", type=int, default=2)
    parser.add_argument("--min_sentence_words", type=int, default=5)
    parser.add_argument("--max_sentence_words", type=int, default=16)
    parser.add_argument("--max_overlap", type=float, default=0.65)
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


def build_prompt(repaired_caption: str) -> str:
    draft = repaired_caption.strip() or "No verified draft is available."
    instruction = (
        "You are adding only missing visual details to a verified image caption.\n"
        f"Verified caption:\n{draft}\n\n"
        "Write at most two NEW short sentences. Each sentence must be under 16 words, "
        "must describe concrete visible image content, and must not repeat the verified caption. "
        "Do not mention uncertain, hidden, or absent objects. If no safe new detail is visible, write NONE. "
        "Output only the new sentence(s). ASSISTANT:"
    )
    return f"{LLAVA_SYSTEM_PROMPT} USER: <image>\n{instruction}"


def load_generation_stack(model_path: str, device: torch.device) -> tuple[Any, Any]:
    processor = LlavaProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": str(device)},
    )
    vision_config = getattr(model.config, "vision_config", None)
    if getattr(processor, "patch_size", None) is None and vision_config is not None:
        processor.patch_size = getattr(vision_config, "patch_size", None)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    if not getattr(processor, "num_additional_image_tokens", None):
        processor.num_additional_image_tokens = 1
    model.eval()
    return model, processor


def decode_generated(processor: Any, output_ids: torch.Tensor, prompt_len: int) -> str:
    return processor.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


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
    regen_rows = read_json(Path(args.regen_examples_json))
    if args.max_images > 0:
        regen_rows = regen_rows[: args.max_images]
    audit_by_image = {int(row["image_id"]): row for row in read_json(Path(args.audit_compatible_json))}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, processor = load_generation_stack(args.model_path, device)

    examples = []
    audit_compatible = []
    vanilla_rows = []
    gated_rows = []
    repaired_rows = []
    augmented_rows = []
    for row in tqdm(regen_rows, desc="atomic detail generation"):
        image_id = int(row["image_id"])
        repaired = str(row.get("repaired_caption", "")).strip()
        prompt = build_prompt(repaired)
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
        image.close()
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        raw_additions = decode_generated(processor, generated_ids, prompt_len)
        additions = parse_additions(raw_additions, repaired, args)
        augmented = normalize_caption([repaired] + [item["text"] for item in additions])
        audit_row = audit_by_image[image_id]
        example = {
            "image_id": image_id,
            "image_path": row["image_path"],
            "vanilla_caption": row.get("vanilla_caption", ""),
            "gated_caption": row.get("gated_caption", ""),
            "repaired_caption": repaired,
            "raw_atomic_detail_text": raw_additions,
            "atomic_detail_additions": additions,
            "augmented_caption": augmented,
            "repaired_words": word_count(repaired),
            "augmented_words": word_count(augmented),
            "num_additions": len(additions),
        }
        examples.append(example)
        audit_compatible.append(
            {
                "image_id": image_id,
                "image_path": row["image_path"],
                "denied_items": audit_row.get("denied_items", []),
                "num_denied_sequences": audit_row.get("num_denied_sequences", 0),
                "vanilla_source": "claim_local_repair",
                "vanilla_caption": repaired,
                "gated_caption": augmented,
                "caption_differs_from_reference": int(repaired != augmented),
                "reference_comparison_note": "atomic detail augmented caption vs claim-local repair",
                "gate_events": [],
            }
        )
        vanilla_rows.append({"image_id": image_id, "caption": row.get("vanilla_caption", "")})
        gated_rows.append({"image_id": image_id, "caption": row.get("gated_caption", "")})
        repaired_rows.append({"image_id": image_id, "caption": repaired})
        augmented_rows.append({"image_id": image_id, "caption": augmented})
        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "repaired": chair_eval(repaired_rows, args.chair_pkl, output_dir, "repaired"),
        "augmented": chair_eval(augmented_rows, args.chair_pkl, output_dir, "augmented"),
    }
    metrics = {
        "regen_examples_json": args.regen_examples_json,
        "audit_compatible_json": args.audit_compatible_json,
        "model_path": args.model_path,
        "num_examples": len(examples),
        "max_new_tokens": args.max_new_tokens,
        "max_additions": args.max_additions,
        "min_sentence_words": args.min_sentence_words,
        "max_sentence_words": args.max_sentence_words,
        "max_overlap": args.max_overlap,
        "images_with_additions": sum(int(row["num_additions"] > 0) for row in examples),
        "total_additions": sum(row["num_additions"] for row in examples),
        "mean_words": {
            "repaired": sum(row["repaired_words"] for row in examples) / len(examples) if examples else 0.0,
            "augmented": sum(row["augmented_words"] for row in examples) / len(examples) if examples else 0.0,
        },
        "chair": chair,
        "scope_note": (
            "Atomic detail generation prototype. It asks the VLM for short missing-detail "
            "sentences and prepares repaired-vs-augmented examples for closed-loop claim verification."
        ),
    }
    (output_dir / "atomic_detail_generation_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "atomic_detail_generation_audit_examples.json").write_text(
        json.dumps(audit_compatible, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "atomic_detail_generation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Atomic Detail Generation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| images with additions | {metrics['images_with_additions']} |",
        f"| total additions | {metrics['total_additions']} |",
        f"| augmented mean words | {metrics['mean_words']['augmented']:.2f} |",
        f"| repaired mean words | {metrics['mean_words']['repaired']:.2f} |",
        f"| augmented CHAIRi | {chair['augmented']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| repaired CHAIRi | {chair['repaired']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| augmented hallucinated mentions | {chair['augmented']['total_hallucinated_mentions']} |",
        f"| repaired hallucinated mentions | {chair['repaired']['total_hallucinated_mentions']} |",
    ]
    (output_dir / "atomic_detail_generation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
