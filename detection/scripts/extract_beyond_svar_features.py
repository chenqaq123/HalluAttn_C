#!/usr/bin/env python
"""Extract per-layer features for the FIXED SVAR / Beyond-ADS / Beyond-CGC scores.

One GPU forward pass per image; for every object mention we store, for all 32
decoder layers:
  - image_mass[l]: head-averaged object-token attention mass onto the 576 image
    patches (query row = token_pos, the position whose forward emits the object
    token) -> used for the fixed SVAR (Jiang et al., sum over layers 5..18).
  - ads[l]:  Attention Dispersion Score (Beyond the Global Scores, arXiv:2604.04863)
             (1 - m_t) * Hhat_t with top-10% foreground, 8-connected blob mass,
             background entropy normalized by log(576).
  - cgc[l]:  Cross-modal Grounding Consistency = mean cosine of the object-token
             hidden state to its top-5%-by-similarity image patches, same layer.

All scoring/AUROC is done offline by evaluate_beyond_svar.py from these features,
so formula tweaks do not require another GPU run.

Faithful specs sourced from the two papers + official repos (see
docs/experiment_results.md 2026-07-08 fix entry).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "detection" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import build_caption_prompt, load_model_and_processor  # noqa: E402

_EPS = 1e-8
_S8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 8-connectivity
_DEBUG = bool(int(os.environ.get("BEYOND_DEBUG", "0")))


def ads_from_row(a: np.ndarray, top_x: float = 0.10, tau_ads: int = 2) -> float:
    """ADS = (1 - m) * Hhat on a single 576-vector head-averaged attention map."""
    P = a.size  # 576
    side = int(round(P ** 0.5))  # 24
    grid = a.reshape(side, side)
    k_fg = max(1, int(round(top_x * P)))
    thresh = np.sort(a)[-k_fg]
    fg = grid >= thresh
    lab, ncomp = label(fg, structure=_S8)
    keep = np.zeros_like(fg)
    m = 0.0
    for c in range(1, ncomp + 1):
        mask = lab == c
        if int(mask.sum()) >= tau_ads:
            keep |= mask
            m += float(grid[mask].sum())
    bmass = grid[~keep]
    Z = float(bmass.sum())
    if Z <= 0:
        h_hat = 0.0
    else:
        e = bmass / Z
        e = e[e > 0]
        h_hat = float(-(e * np.log(e)).sum() / np.log(P))
    return (1.0 - m) * h_hat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--object_cache", default=str(PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl"))
    p.add_argument("--generation_json", default=str(PROJECT_ROOT / "experiments/coco_llava_7b/generation.json"))
    p.add_argument("--coco_path", default="/data/common_dataset/coco-2014-dataset")
    p.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--cgc_k_frac", type=float, default=0.05)
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
    feat = {k: [] for k in ("image_mass", "ads", "cgc")}  # each -> (n, n_layers)

    for image_id in tqdm(shard_ids, desc=f"shard{args.shard_idx}"):
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

        with torch.no_grad():
            outputs = model.forward(**inputs, output_attentions=True, output_hidden_states=True)
        attentions = outputs.attentions            # tuple len 32, each [1,H,seq,seq]
        hidden_states = outputs.hidden_states       # tuple len 33
        seq = input_ids.numel()

        raw = by_image[int(image_id)]
        mentions = [r for r in raw if 0 <= int(r["token_pos"]) < seq - 1]
        if _DEBUG:
            tp = [int(r["token_pos"]) for r in raw]
            print(f"[dbg] img={image_id} seq={seq} vis=({vis_start},{vis_end}) "
                  f"raw={len(raw)} kept={len(mentions)} token_pos={tp[:5]}", file=sys.stderr)
        if not mentions:
            del inputs, outputs
            continue
        tpos = torch.tensor([int(r["token_pos"]) for r in mentions], device=device)
        target = tpos + 1

        # Per-layer head-averaged attention rows at the mention query positions,
        # restricted to the image span -> [n_layers, n_mentions, 576].
        img_rows = torch.empty((n_layers, len(mentions), vis_end - vis_start), dtype=torch.float32)
        cgc = torch.empty((n_layers, len(mentions)), dtype=torch.float32)
        k_cgc = max(1, int(round(args.cgc_k_frac * (vis_end - vis_start))))
        for l in range(n_layers):
            head_avg = attentions[l][0].mean(0)                 # [seq, seq]
            rows = head_avg[tpos, vis_start:vis_end].float()    # [n, 576]
            img_rows[l] = rows.cpu()
            hs = hidden_states[l + 1][0].float()                # output of decoder layer l
            tok = torch.nn.functional.normalize(hs[target], dim=-1)         # [n, d]
            patch = torch.nn.functional.normalize(hs[vis_start:vis_end], dim=-1)  # [576, d]
            sims = tok @ patch.t()                              # [n, 576]
            cgc[l] = sims.topk(k_cgc, dim=-1).values.mean(-1).cpu()

        img_rows_np = img_rows.numpy()                          # [L, n, 576]
        image_mass = img_rows_np.sum(-1).T                      # [n, L]
        ads = np.empty((len(mentions), n_layers), dtype=np.float32)
        for j in range(len(mentions)):
            for l in range(n_layers):
                row = img_rows_np[l, j]
                s = row.sum()
                ads[j, l] = ads_from_row(row / s if s > _EPS else row)

        for i, r in enumerate(mentions):
            out["object_id"].append(int(r["object_id"]))
            out["image_id"].append(int(image_id))
            out["token_pos"].append(int(r["token_pos"]))
            out["gen_pos"].append(int(r["gen_pos"]))
            out["label"].append(int(r["label"]))
        feat["image_mass"].append(image_mass.astype(np.float32))
        feat["ads"].append(ads)
        feat["cgc"].append(cgc.numpy().T.astype(np.float32))    # [n, L]

        del inputs, outputs, attentions, hidden_states
        torch.cuda.empty_cache()

    save = {k: np.asarray(v) for k, v in out.items()}
    for k, v in feat.items():
        save[k] = np.concatenate(v, axis=0) if v else np.zeros((0, n_layers), np.float32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **save)
    print(f"[shard{args.shard_idx}] wrote {len(save['object_id'])} mentions -> {args.out}")


if __name__ == "__main__":
    main()
