#!/usr/bin/env python3
"""Controlled caption regeneration from verified caption drafts.

This prototype tries to recover visible detail after claim acceptance/repair.
It uses the repaired caption as a safe draft, asks LLaVA to rewrite a concise
faithful caption from the image, and applies a phrase gate to block the denied
or unsupported claims found by the closed-loop audit.
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
from transformers import LlavaForConditionalGeneration, LlavaProcessor, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
PAS_SRC = REPO_ROOT.parent / "pas" / "src"
for import_path in (DETECTION_SRC, PAS_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from sinkdetect.chair import evaluate_chair, load_chair_evaluator
from sinkdetect.decode_gate import ObjectPhraseGateLogitsProcessor, build_object_phrase_sequences
from sinkdetect.utils import LLAVA_SYSTEM_PROMPT

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96/gated_generation_examples.json",
    )
    parser.add_argument(
        "--audit_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_audit_ov96/closed_loop_example_audit.json",
    )
    parser.add_argument(
        "--repair_examples_json",
        default="detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair/claim_repair_examples.json",
    )
    parser.add_argument(
        "--model_path",
        default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    )
    parser.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    parser.add_argument("--output_dir", default="detection/baselines/results/tdev_caption_controlled_regen_100_t80")
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--max_images", type=int, default=0, help="0 means all examples.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--soft_penalty", type=float, default=8.0)
    parser.add_argument("--hard_gate", action="store_true")
    parser.add_argument("--prompt_style", choices=("concise", "detail"), default="concise")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(text.split())


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        norm = str(item).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def forbidden_claims(audit_row: dict[str, Any], gated_caption: str) -> list[str]:
    gated_lower = gated_caption.lower()
    claims: list[str] = []
    claims.extend(str(word) for word in audit_row.get("denied_words", []) if phrase_pattern(str(word)).search(gated_lower))
    claims.extend(str(word) for word in audit_row.get("introduced_hallucinated_words", []))
    for claim in audit_row.get("unsupported_open_vocab_claims", []):
        norm = str(claim).strip().lower()
        if norm and norm not in GENERIC_UNSUPPORTED_CLAIMS:
            claims.append(norm)
    claims.extend(str(hit.get("alias", "")) for hit in audit_row.get("introduced_variant_leaks", []))
    claims.extend(str(hit.get("token", "")) for hit in audit_row.get("introduced_root_leaks", []))
    return dedupe(claims)


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


def build_prompt(seed_caption: str, forbidden: list[str], prompt_style: str) -> str:
    forbidden_text = ", ".join(forbidden[:20]) if forbidden else "none"
    seed = seed_caption.strip() or "No safe draft is available."
    if prompt_style == "detail":
        instruction = (
            "Rewrite a faithful image caption from the image and the verified draft below.\n"
            f"Verified draft: {seed}\n"
            f"Forbidden unsupported terms: {forbidden_text}.\n"
            "Preserve the concrete visible details from the verified draft unless the image clearly contradicts them. "
            "You may add only clearly visible details from the image. Do not invent absent objects. "
            "Write a detailed but concise caption in three or four sentences. ASSISTANT:"
        )
    else:
        instruction = (
            "Rewrite a faithful image caption from the image and the verified draft below.\n"
            f"Verified draft: {seed}\n"
            f"Forbidden unsupported terms: {forbidden_text}.\n"
            "Keep visible details when they are clear. Do not invent absent objects. "
            "Write one concise caption, two sentences at most. ASSISTANT:"
        )
    return f"{LLAVA_SYSTEM_PROMPT} USER: <image>\n{instruction}"


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
    examples = read_json(Path(args.examples_json))
    if args.max_images > 0:
        examples = examples[: args.max_images]
    audit_by_image = {int(row["image_id"]): row for row in read_json(Path(args.audit_json))["examples"]}
    repair_by_image = {int(row["image_id"]): row for row in read_json(Path(args.repair_examples_json))}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, processor = load_generation_stack(args.model_path, device)

    regen_examples = []
    vanilla_rows = []
    gated_rows = []
    repaired_rows = []
    regenerated_rows = []
    audit_compatible = []
    for example in tqdm(examples, desc="controlled regeneration"):
        image_id = int(example["image_id"])
        audit_row = audit_by_image[image_id]
        repair_row = repair_by_image[image_id]
        forbidden = forbidden_claims(audit_row, example.get("gated_caption", ""))
        prompt = build_prompt(repair_row.get("repaired_caption", ""), forbidden, args.prompt_style)
        image = Image.open(example["image_path"]).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
        image.close()
        prompt_len = int(inputs["input_ids"].shape[1])
        denied_sequences = []
        denied_texts = []
        for phrase in forbidden:
            seqs = build_object_phrase_sequences(processor.tokenizer, [phrase])
            denied_sequences.extend(seqs)
            denied_texts.extend([phrase] * len(seqs))
        gate = ObjectPhraseGateLogitsProcessor(
            tokenizer=processor.tokenizer,
            denied_token_sequences=[denied_sequences],
            prompt_lengths=prompt_len,
            denied_phrase_texts=[denied_texts],
            penalty=None if args.hard_gate else args.soft_penalty,
            block_first_token_for_multi_token=False,
            audit_limit=80,
        )
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([gate]),
            )
        regenerated_caption = decode_generated(processor, generated_ids, prompt_len)
        row = {
            "image_id": image_id,
            "image_path": example["image_path"],
            "denied_words": audit_row.get("denied_words", []),
            "forbidden_claims": forbidden,
            "vanilla_caption": example.get("vanilla_caption", ""),
            "gated_caption": example.get("gated_caption", ""),
            "repaired_caption": repair_row.get("repaired_caption", ""),
            "regenerated_caption": regenerated_caption,
            "vanilla_words": word_count(example.get("vanilla_caption", "")),
            "gated_words": word_count(example.get("gated_caption", "")),
            "repaired_words": word_count(repair_row.get("repaired_caption", "")),
            "regenerated_words": word_count(regenerated_caption),
            "gate_events": [event.__dict__ for event in gate.events],
        }
        regen_examples.append(row)
        vanilla_rows.append({"image_id": image_id, "caption": row["vanilla_caption"]})
        gated_rows.append({"image_id": image_id, "caption": row["gated_caption"]})
        repaired_rows.append({"image_id": image_id, "caption": row["repaired_caption"]})
        regenerated_rows.append({"image_id": image_id, "caption": regenerated_caption})
        audit_compatible.append(
            {
                "image_id": image_id,
                "image_path": example["image_path"],
                "denied_items": example.get("denied_items", []),
                "num_denied_sequences": len(denied_sequences),
                "vanilla_source": example.get("vanilla_source", ""),
                "vanilla_caption": row["vanilla_caption"],
                "gated_caption": regenerated_caption,
                "caption_differs_from_reference": int(row["vanilla_caption"] != regenerated_caption),
                "reference_comparison_note": "controlled regeneration vs generated vanilla",
                "gate_events": row["gate_events"],
            }
        )
        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    chair = {
        "vanilla": chair_eval(vanilla_rows, args.chair_pkl, output_dir, "vanilla"),
        "gated": chair_eval(gated_rows, args.chair_pkl, output_dir, "gated"),
        "repaired": chair_eval(repaired_rows, args.chair_pkl, output_dir, "repaired"),
        "regenerated": chair_eval(regenerated_rows, args.chair_pkl, output_dir, "regenerated"),
    }
    metrics = {
        "examples_json": args.examples_json,
        "audit_json": args.audit_json,
        "repair_examples_json": args.repair_examples_json,
        "model_path": args.model_path,
        "num_examples": len(regen_examples),
        "max_new_tokens": args.max_new_tokens,
        "prompt_style": args.prompt_style,
        "soft_penalty": None if args.hard_gate else args.soft_penalty,
        "hard_gate": bool(args.hard_gate),
        "mean_words": {
            "vanilla": sum(row["vanilla_words"] for row in regen_examples) / len(regen_examples) if regen_examples else 0.0,
            "gated": sum(row["gated_words"] for row in regen_examples) / len(regen_examples) if regen_examples else 0.0,
            "repaired": sum(row["repaired_words"] for row in regen_examples) / len(regen_examples) if regen_examples else 0.0,
            "regenerated": sum(row["regenerated_words"] for row in regen_examples) / len(regen_examples) if regen_examples else 0.0,
        },
        "images_with_gate_events": sum(int(bool(row["gate_events"])) for row in regen_examples),
        "total_gate_events_saved": sum(len(row["gate_events"]) for row in regen_examples),
        "chair": chair,
        "scope_note": (
            "Controlled regeneration prototype: LLaVA rewrites from the image and a verified repaired "
            "draft while a phrase gate blocks denied or unsupported claims. This is not yet the final "
            "caption mitigation; it tests whether visible detail can be recovered after claim repair."
        ),
    }
    (output_dir / "controlled_regeneration_examples.json").write_text(
        json.dumps(regen_examples, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "controlled_regeneration_audit_compatible_examples.json").write_text(
        json.dumps(audit_compatible, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "controlled_regeneration_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Controlled Caption Regeneration",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| examples | {metrics['num_examples']} |",
        f"| regenerated mean words | {metrics['mean_words']['regenerated']:.2f} |",
        f"| repaired mean words | {metrics['mean_words']['repaired']:.2f} |",
        f"| regenerated CHAIRi | {chair['regenerated']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| repaired CHAIRi | {chair['repaired']['overall'].get('CHAIRi', 0.0):.4f} |",
        f"| regenerated hallucinated mentions | {chair['regenerated']['total_hallucinated_mentions']} |",
        f"| repaired hallucinated mentions | {chair['repaired']['total_hallucinated_mentions']} |",
        f"| images with gate events | {metrics['images_with_gate_events']} |",
    ]
    (output_dir / "controlled_regeneration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
