#!/usr/bin/env python3
"""
Stage 1: Generate captions for COCO val2014 images using vanilla LLaVA-1.5-7B.

Output: generation.json with [{image_id, caption, output_ids, prompt_end_idx}]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from sinkdetect.utils import (
    build_caption_prompt,
    find_prompt_end_idx,
    load_coco_val2014,
    load_model_and_processor,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Generate captions with vanilla LLaVA")
    p.add_argument("--model_path", type=str,
                   default="/data/common_model/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/")
    p.add_argument("--coco_path", type=str, default="/data/common_dataset/coco-2014-dataset/")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0,
                   help="This shard's index (0..num_shards-1). Used for data slicing.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of parallel shards.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")

    # Load dataset
    print(f"Loading COCO val2014 from {args.coco_path} ...")
    coco_data = load_coco_val2014(args.coco_path, num_samples=args.num_samples, seed=args.seed)
    print(f"  {len(coco_data)} images sampled (global)")

    # Shard slicing: deterministic stride so the same seed → same global set
    if args.num_shards > 1:
        coco_data = coco_data[args.shard_idx :: args.num_shards]
        print(f"  shard {args.shard_idx}/{args.num_shards}: {len(coco_data)} images")

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model, processor = load_model_and_processor(args.model_path, device)

    # Generate captions
    prompt_text = build_caption_prompt()
    results = []

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for item in tqdm(coco_data, desc="Generating captions"):
        img_path = Path(args.coco_path) / "val2014" / item["file_name"]
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(
            device, dtype=torch.float16
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Extract generated portion
        gen_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        caption = processor.decode(gen_tokens, skip_special_tokens=True).strip()

        # Find prompt end index in the full output
        output_ids_list = output_ids[0].tolist()
        prompt_end_idx = find_prompt_end_idx(output_ids_list)

        results.append({
            "image_id": item["image_id"],
            "caption": caption,
            "output_ids": output_ids_list,
            "prompt_end_idx": prompt_end_idx,
        })

        del inputs, output_ids
        if len(results) % 8 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} captions to {output_path}")


if __name__ == "__main__":
    main()
