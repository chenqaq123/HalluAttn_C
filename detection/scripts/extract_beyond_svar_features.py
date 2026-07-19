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
import math
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
# 4-neighbour discrete Laplacian for LaSCD (arXiv:2605.11559, App. B).
_LAPLACIAN = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
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


def _denoise_renorm(a, mask=None, eps=_EPS):
    """ONLY's per-key outlier-zero (drop keys > mean+std over the key axis) then renorm to sum 1.

    a: [..., K]. If mask (bool, same trailing shape or broadcastable) is given, statistics and the
    distribution are restricted to valid keys (used for causal text keys of variable length).
    """
    if mask is None:
        mean = a.mean(-1, keepdim=True)
        std = a.std(-1, unbiased=True, keepdim=True)          # torch ddof=1, matches ONLY
        a = torch.where(a > mean + std, torch.zeros_like(a), a)
    else:
        m = mask.to(a.dtype)
        a = a * m
        cnt = m.sum(-1, keepdim=True).clamp_min(1.0)
        mean = a.sum(-1, keepdim=True) / cnt
        var = (((a - mean) * m) ** 2).sum(-1, keepdim=True) / (cnt - 1.0).clamp_min(1.0)
        std = var.sqrt()
        a = torch.where((a > mean + std) & mask, torch.zeros_like(a), a) * m
    return a / a.sum(-1, keepdim=True).clamp_min(eps)


def tver_layer_scores(full, vis_start, vis_end, text_mask, eps=1e-6):
    """TVER (ONLY, arXiv:2507.00898): per-head H(text attn)/H(visual attn) at the query row, mean over heads.

    full: [H, n, seq] attention rows at each mention's token_pos. text_mask: [n, seq] bool
    (system + generated-so-far, excluding BOS/image/self-future). Higher TVER = text-prioritizing.
    """
    vis = full[:, :, vis_start:vis_end]                        # [H, n, 576]
    p_vis = _denoise_renorm(vis)
    H_vis = -(p_vis * (p_vis + eps).log()).sum(-1)             # [H, n] nats
    p_text = _denoise_renorm(full, mask=text_mask.unsqueeze(0))
    H_text = -(p_text * (p_text + eps).log()).sum(-1)          # [H, n]
    H_text = torch.nan_to_num(H_text, nan=0.0)
    H_vis = torch.nan_to_num(H_vis, nan=float("inf"))
    return (H_text / H_vis).mean(0)                            # [n] mean over heads


def patch_labels_for_image(seg_model, seg_proc, image, side, device):
    """Mask2Former-Swin-L semantic labels for the CLIP 24x24 patch grid (SAE, arXiv:2603.16558).

    Segments the 336x336 CLIP input, then assigns each 14x14 patch its majority-pixel
    semantic label. Returns (labels[576] in [0,C), C) as a dense LongTensor on `device`.
    """
    res = side * 14  # 24*14 = 336, the CLIP input size
    im = image.resize((res, res))
    inp = seg_proc(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        o = seg_model(**inp)
    sem = seg_proc.post_process_semantic_segmentation(o, target_sizes=[(res, res)])[0]  # [336,336] long
    blocks = sem.reshape(side, 14, side, 14).permute(0, 2, 1, 3).reshape(side * side, 14 * 14)
    maj = torch.mode(blocks, dim=-1).values                      # [576] majority label per patch
    uniq, inv = torch.unique(maj, return_inverse=True)           # densify to [0,C)
    return inv.to(device), int(uniq.numel())


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
    p.add_argument("--seg_model", default="facebook/mask2former-swin-large-coco-panoptic")
    p.add_argument("--no_seg", action="store_true", help="skip Mask2Former; only the seg-free SAE skeleton")
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

    seg_model = seg_proc = None
    if not args.no_seg:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        seg_proc = AutoImageProcessor.from_pretrained(args.seg_model)
        seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(args.seg_model).to(device).eval()
        for prm in seg_model.parameters():
            prm.requires_grad_(False)

    n_layers = model.config.text_config.num_hidden_layers  # 32
    out = {k: [] for k in ("object_id", "image_id", "token_pos", "gen_pos", "label")}
    feat_keys = ["image_mass", "ads", "cgc", "sae_R", "lascd_raw", "lascd_norm", "tver"]
    if seg_model is not None:
        feat_keys.append("sae_seg")          # true SAE (Mask2Former segmentation-region entropy)
    feat = {k: [] for k in feat_keys}         # each -> (n, n_layers)
    lap = _LAPLACIAN.to(device)

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
        # Object-token generating position (= repo's last_pos): used as the query row
        # for attention scores AND as the token hidden state for CGC, matching the
        # official token-grounding-detector code (features/cgc.py reads hs at last_pos).

        # Per-layer, per-mention features at the object-token query row (token_pos).
        nv = vis_end - vis_start                                # 576
        side = int(round(nv ** 0.5))                            # 24
        nm = len(mentions)
        log_nv = math.log(nv)
        img_rows = torch.empty((n_layers, nm, nv), dtype=torch.float32)  # head-avg (SVAR/ADS)
        cgc = torch.empty((n_layers, nm), dtype=torch.float32)
        sae_R = torch.empty((n_layers, nm), dtype=torch.float32)         # SAE-raw reliability R
        lascd_raw = torch.empty((n_layers, nm), dtype=torch.float32)     # LaSCD energy (raw)
        lascd_norm = torch.empty((n_layers, nm), dtype=torch.float32)    # LaSCD energy (mass-normalized)
        tver = torch.empty((n_layers, nm), dtype=torch.float32)          # TVER (ONLY): H(text)/H(vis)
        k_cgc = max(1, int(round(args.cgc_k_frac * nv)))
        # Causal text-key mask per mention: system+generated-so-far, exclude BOS(0)/image/future.
        pos = torch.arange(seq, device=device)
        text_mask = (((pos >= 1) & (pos < vis_start)) |
                     ((pos >= vis_end) & (pos <= tpos.unsqueeze(1))))    # [nm, seq]
        if seg_model is not None:
            image2 = Image.open(img_path).convert("RGB")
            plabel, n_seg = patch_labels_for_image(seg_model, seg_proc, image2, side, device)
            image2.close()
            log_seg = math.log(n_seg) if n_seg > 1 else 1.0
            sae_seg = torch.empty((n_layers, nm), dtype=torch.float32)   # true SAE reliability R
        for l in range(n_layers):
            full = attentions[l][0][:, tpos, :].float()                 # [H, n, seq] per-head full row
            tver[l] = tver_layer_scores(full, vis_start, vis_end, text_mask).cpu()
            Aph = full[:, :, vis_start:vis_end]                          # [H, n, 576] visual slice
            img_rows[l] = Aph.mean(0).cpu()                             # head-avg row for SVAR/ADS
            # SAE (arXiv:2603.16558), segmentation-free: per-head R = M*(1-normEntropy), mean over heads.
            M = Aph.sum(-1)                                              # [H, n] attention mass
            p = Aph / M.unsqueeze(-1).clamp_min(_EPS)
            ent = -(p * p.clamp_min(_EPS).log()).sum(-1)                 # [H, n] natural-log entropy
            sae_R[l] = (M * (1.0 - ent / log_nv)).mean(0).cpu()          # higher R = more reliable
            if seg_model is not None:
                # True SAE: pool p into Mask2Former semantic regions, entropy over regions / log|C|.
                pc = torch.zeros(p.shape[0], nm, n_seg, device=p.device)
                pc.index_add_(2, plabel, p)                              # [H, n, C]
                ent_s = -(pc * pc.clamp_min(_EPS).log()).sum(-1)         # [H, n]
                sae_seg[l] = (M * (1.0 - ent_s / log_seg)).mean(0).cpu()
            # LaSCD (arXiv:2605.11559): per-head Frobenius norm of the 2D Laplacian, mean over heads.
            grid = Aph.reshape(Aph.shape[0] * nm, 1, side, side)
            e_raw = torch.nn.functional.conv2d(grid, lap, padding=1).flatten(1).norm(dim=1)
            lascd_raw[l] = e_raw.reshape(Aph.shape[0], nm).mean(0).cpu()
            gridn = p.reshape(Aph.shape[0] * nm, 1, side, side)          # mass-normalized (sum=1)
            e_norm = torch.nn.functional.conv2d(gridn, lap, padding=1).flatten(1).norm(dim=1)
            lascd_norm[l] = e_norm.reshape(Aph.shape[0], nm).mean(0).cpu()
            # CGC (Beyond): object-token vs top-similarity patch hidden states, same layer.
            hs = hidden_states[l + 1][0].float()
            tok = torch.nn.functional.normalize(hs[tpos], dim=-1)          # generating pos (repo last_pos)
            patch = torch.nn.functional.normalize(hs[vis_start:vis_end], dim=-1)
            sims = tok @ patch.t()
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
        feat["cgc"].append(cgc.numpy().T.astype(np.float32))          # [n, L]
        feat["sae_R"].append(sae_R.numpy().T.astype(np.float32))      # [n, L]
        feat["lascd_raw"].append(lascd_raw.numpy().T.astype(np.float32))
        feat["lascd_norm"].append(lascd_norm.numpy().T.astype(np.float32))
        feat["tver"].append(tver.numpy().T.astype(np.float32))
        if seg_model is not None:
            feat["sae_seg"].append(sae_seg.numpy().T.astype(np.float32))

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
