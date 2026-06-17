#!/usr/bin/env python3
"""Run one mitigation method on one POPE split or the CHAIR caption manifest."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

import torch
from PIL import Image
from tqdm import tqdm

from sinkdetect.sink_utils import find_vis_bounds
from sinkdetect.utils import build_caption_prompt, load_model_and_processor
from src.data import iter_pope_records, load_chair_manifest, load_pope_image
from src.decoding import generate_vcd_greedy
from src.interventions import install_intervention, set_visual_bounds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate POPE answers or CHAIR captions with mitigation interventions")
    p.add_argument("--task", choices=["pope", "chair"], required=True)
    p.add_argument("--method", choices=["vanilla", "pai", "clearsight", "visattnsink", "vcd"], required=True)
    p.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_dir", default="")
    p.add_argument("--pope_split", default="random")
    p.add_argument("--chair_manifest", default=str(PROJECT_ROOT / "experiments/coco_llava_7b/generation.json"))
    p.add_argument("--output_file", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--start_layer", type=int)
    p.add_argument("--end_layer", type=int)
    p.add_argument("--pai_alpha", type=float, default=0.2)
    p.add_argument("--vaf_enhance", type=float, default=1.15)
    p.add_argument("--vaf_suppress", type=float, default=0.95)
    p.add_argument("--vas_tau", type=float, default=20.0)
    p.add_argument("--vas_rho", type=float, default=0.5)
    p.add_argument("--vas_visual_mass", type=float, default=0.2)
    p.add_argument("--vas_keep", type=float, default=0.6)
    p.add_argument("--vcd_alpha", type=float, default=0.5)
    p.add_argument("--vcd_beta", type=float, default=0.1)
    p.add_argument("--vcd_noise_step", type=int, default=500)
    return p.parse_args()


def _existing_ids(path: Path, key: str) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)[key] for line in f if line.strip()}


def _prepare(model, processor, image: Image.Image, prompt: str, device: torch.device):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)
    vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], image_token_id)
    set_visual_bounds(model, vis_start, vis_end)
    return inputs, vis_start, vis_end


def _generate(model, processor, inputs, max_new_tokens: int) -> str:
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    del output_ids
    return text


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None)
    intervention = install_intervention(
        model,
        args.method,
        device,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        pai_alpha=args.pai_alpha,
        vaf_enhance=args.vaf_enhance,
        vaf_suppress=args.vaf_suppress,
        vas_tau=args.vas_tau,
        vas_rho=args.vas_rho,
        vas_visual_mass=args.vas_visual_mass,
        vas_keep=args.vas_keep,
    )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_key = "question_id" if args.task == "pope" else "image_id"
    existing = _existing_ids(output_path, id_key)

    if args.task == "pope":
        if not args.pope_dir:
            raise ValueError("--pope_dir is required for --task pope")
        records = list(iter_pope_records(
            args.pope_dir,
            args.coco_path,
            args.pope_split,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            limit=args.limit,
        ))
    else:
        records = load_chair_manifest(
            args.chair_manifest,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
            limit=args.limit,
        )

    with output_path.open("a", encoding="utf-8") as out:
        for step, record in enumerate(tqdm(records, desc=f"{args.task}:{args.method}:shard{args.shard_idx}"), start=1):
            if record[id_key] in existing:
                continue
            if args.task == "pope":
                image = load_pope_image(record)
                question = record["question"] + " Please just answer yes or no."
                prompt = f"<image>\nUSER: {question}\nASSISTANT:"
            else:
                image_path = Path(args.coco_path) / "val2014" / f"COCO_val2014_{record['image_id']:012d}.jpg"
                image = Image.open(image_path).convert("RGB")
                prompt = build_caption_prompt()
            inputs, vis_start, vis_end = _prepare(model, processor, image, prompt, device)
            image.close()
            if args.method == "vcd":
                try:
                    stable_id = int(record[id_key])
                except (TypeError, ValueError):
                    stable_id = args.shard_idx * 1_000_000 + step
                torch.manual_seed(args.seed + stable_id)
                text = generate_vcd_greedy(
                    model,
                    processor,
                    inputs,
                    args.max_new_tokens,
                    alpha=args.vcd_alpha,
                    beta=args.vcd_beta,
                    noise_step=args.vcd_noise_step,
                )
            else:
                text = _generate(model, processor, inputs, args.max_new_tokens)
            payload = {
                id_key: record[id_key],
                "method": args.method,
                "text" if args.task == "pope" else "caption": text,
                "visual_span": [vis_start, vis_end],
                "intervention": asdict(intervention),
            }
            if args.method == "vcd":
                payload["vcd"] = {
                    "alpha": args.vcd_alpha,
                    "beta": args.vcd_beta,
                    "noise_step": args.vcd_noise_step,
                    "decode": "greedy",
                }
            if args.task == "pope":
                payload.update({"question": record["question"], "label": record["label"], "image": record["image"]})
            out.write(json.dumps(payload) + "\n")
            out.flush()
            del inputs
            if step % 32 == 0:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
