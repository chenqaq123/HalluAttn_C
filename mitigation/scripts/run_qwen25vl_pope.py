#!/usr/bin/env python3
"""Run vanilla Qwen2.5-VL POPE generation with the local HF processor."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.data import iter_pope_records, load_pope_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate POPE yes/no answers with Qwen2.5-VL")
    p.add_argument("--model_path", default="/home/chenguanxu/common_model/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5")
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_split", default="adversarial")
    p.add_argument("--output_file", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--min_pixels", type=int, default=200704, help="Qwen image min_pixels; 256*28*28 by default")
    p.add_argument("--max_pixels", type=int, default=200704, help="Qwen image max_pixels; 256*28*28 by default")
    p.add_argument("--use_fast_processor", action="store_true")
    return p.parse_args()


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)["question_id"] for line in f if line.strip()}


def build_prompt(processor, question: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question + " Please just answer yes or no."},
        ],
    }]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=args.use_fast_processor,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        local_files_only=True,
        device_map={"": str(device)},
    )
    model.eval()
    model_device = next(model.parameters()).device

    rows = list(iter_pope_records(
        args.pope_dir,
        args.coco_path,
        args.pope_split,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        limit=args.limit,
    ))
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(output_path)

    with output_path.open("a", encoding="utf-8") as out:
        for step, record in enumerate(tqdm(rows, desc=f"qwen25vl:{args.pope_split}:shard{args.shard_idx}"), start=1):
            if record["question_id"] in done:
                continue
            image = load_pope_image(record)
            prompt = build_prompt(processor, record["question"])
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            image.close()
            inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            text = processor.decode(generated, skip_special_tokens=True).strip()
            payload = {
                "question_id": record["question_id"],
                "method": "vanilla",
                "model_family": "qwen2.5-vl",
                "model_path": args.model_path,
                "text": text,
                "question": record["question"],
                "label": record["label"],
                "image": record["image"],
                "qwen25vl": {
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            out.write(json.dumps(payload) + "\n")
            out.flush()
            del inputs, output_ids
            if step % 16 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
