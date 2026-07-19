#!/usr/bin/env python
"""Extract DHCP (arXiv:2411.18659, ACM MM 2025) attention features for LLaVA-1.5-7B.

DHCP's classifier input is the per-(layer, head) attention distribution over the 576
visual tokens at the object/query token, stacked over all 32 layers -> [32, 32, 576]
(589,824-d), raw (no normalization). Reference: official repo btzyd/DHCP
`qwen_extract_attention.py` (feature = stack over layers of
attentions[l][0][:, token_pos, vis_start:vis_end]).

Saved per shard as fp16 (dhcp_feat: [N, 32, 32, 576]) + object/image/token_pos/gen_pos/label.
Scoring/experiments are done offline (position_dependence_experiment.py / a DHCP MLP).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "detection" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import build_caption_prompt, load_model_and_processor  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--object_cache", default=str(PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl"))
    p.add_argument("--generation_json", default=str(PROJECT_ROOT / "experiments/coco_llava_7b/generation.json"))
    p.add_argument("--coco_path", default="/srv/common_dataset/coco-2014-dataset")
    p.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    records = [json.loads(l) for l in open(args.object_cache, encoding="utf-8")]
    generation = {int(x["image_id"]): x for x in json.load(open(args.generation_json, encoding="utf-8"))}
    by_image: dict[int, list[dict]] = {}
    for r in records:
        by_image.setdefault(int(r["image_id"]), []).append(r)
    image_ids = sorted(by_image)
    shard_ids = image_ids[args.shard_idx :: args.num_shards]

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    prompt_text = build_caption_prompt()
    img_token_id = getattr(processor, "image_token_id", None) or getattr(processor, "image_token_index", 32000)
    n_layers = model.config.text_config.num_hidden_layers  # 32

    out = {k: [] for k in ("object_id", "image_id", "token_pos", "gen_pos", "label")}
    feats = []  # each [n, L, H, N] fp16

    for image_id in tqdm(shard_ids, desc=f"dhcp{args.shard_idx}"):
        entry = generation.get(int(image_id))
        if entry is None:
            continue
        img_path = Path(args.coco_path) / "val2014" / f"COCO_val2014_{int(image_id):012d}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        full_text = prompt_text + " " + str(entry["caption"])
        inputs = processor(images=image, text=full_text, return_tensors="pt").to(device, dtype=torch.float16)
        image.close()
        input_ids = inputs["input_ids"][0]
        try:
            vis_start, vis_end = find_vis_bounds(input_ids, img_token_id)
        except ValueError:
            continue
        seq = input_ids.numel()
        mentions = [r for r in by_image[int(image_id)] if 0 <= int(r["token_pos"]) < seq - 1]
        if not mentions:
            del inputs
            continue
        tpos = torch.tensor([int(r["token_pos"]) for r in mentions], device=device)

        with torch.no_grad():
            o = model.forward(**inputs, output_attentions=True, use_cache=False)
        att = o.attentions  # tuple[L] of [1, H, seq, seq]
        # [L, H, n, N] -> [n, L, H, N]
        stack = torch.stack([att[l][0][:, tpos, vis_start:vis_end] for l in range(n_layers)], dim=0)
        stack = stack.permute(2, 0, 1, 3).contiguous().half().cpu().numpy()  # [n, L, H, N]
        feats.append(stack)
        for r in mentions:
            out["object_id"].append(int(r["object_id"]))
            out["image_id"].append(int(image_id))
            out["token_pos"].append(int(r["token_pos"]))
            out["gen_pos"].append(int(r["gen_pos"]))
            out["label"].append(int(r["label"]))
        del inputs, o, att, stack
        torch.cuda.empty_cache()

    save = {k: np.asarray(v) for k, v in out.items()}
    save["dhcp_feat"] = np.concatenate(feats, axis=0) if feats else np.zeros((0, n_layers, 32, vis_end - vis_start), np.float16)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)  # uncompressed: fp16 attn is dense, compression is slow for little gain
    print(f"[dhcp{args.shard_idx}] wrote {len(save['object_id'])} mentions, feat {save['dhcp_feat'].shape} -> {args.out}")


if __name__ == "__main__":
    main()
