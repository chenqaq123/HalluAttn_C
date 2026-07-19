#!/usr/bin/env python
"""Generate many clean-map triptychs for case selection.

For each POPE-adversarial `related_present` negative where vanilla answers "no" and
ClearSight answers "yes", render ONE bare figure = [ original | clean attention map |
overlay ], horizontally concatenated, with nothing else (no titles, axes, colorbars,
or boxes). The clean map is the no-RoPE (position-prior-removed) attention with
detected sink tokens deleted — layers 9–14, mean over heads, first-answer-token row.

Candidate pool is rebuilt from source (so it is reproducible without the lost
original selection script), pooled across the requested POPE splits:
  - rows:        mitigation/results/semantic_neighbor_audit/{split}_semantic_neighbor_rows.csv
  - vanilla/CS:  mitigation/results/coco_llava_7b_attention_only/pope/{split}/{vanilla,clearsight}/predictions.jsonl
  - co-occ rank: mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json
Every case is a target-absent `related_present` negative. Cases where the
intervention flips the answer (vanilla=no → ClearSight=yes) are ranked first
(the thesis-relevant case); the rest (vanilla=no, no flip) top up to --limit.
De-duplicated by (image_id, target). Ranked within each tier by co-occurrence
rank asc, split priority (adversarial>popular>random), qid asc. A manifest CSV
records what each file is (including split and whether it flipped).

Sink detection / no-RoPE reuse the pure-torch sink_utils and the helpers in
extract_qualitative_sink_attention.py (transformers 4.48.1 compatible).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))          # mitigation/scripts
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from extract_qualitative_sink_attention import (  # noqa: E402
    KeyNormHook, _no_rope_last_row, _drop_sinks_renorm, resolve_llm_layers,
)
from sinkdetect.sink_utils import auto_detect_sinks, find_vis_bounds  # noqa: E402
from sinkdetect.utils import load_model_and_processor  # noqa: E402

GRID, SCALE = 24, 14
BAND_LO, BAND_HI = 9, 15
CLIP_Q, ALPHA = 99.5, 0.85
POPE_SUFFIX = " Please just answer yes or no."
MAGMA = cm.get_cmap("magma")

AUDIT = PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit"
PRED_BASE = PROJECT_ROOT / "mitigation/results/coco_llava_7b_attention_only/pope"
SPLIT_ORDER = {"adversarial": 0, "popular": 1, "random": 2}


def load_pred(p: Path) -> dict[str, str]:
    d = {}
    for line in p.open():
        r = json.loads(line)
        d[str(r["question_id"])] = r["text"].strip().lower().rstrip(".")
    return d


def cooc_rank(neighbors: list[str], related_present: list[str]) -> int:
    """Best (smallest) co-occurrence rank among the present related objects."""
    ranks = [neighbors.index(o) for o in related_present if o in neighbors]
    return min(ranks) if ranks else 999


def build_pool(splits: list[str], limit: int) -> list[dict]:
    cooc = json.loads((AUDIT / "cooccurrence_neighbors.json").read_text())
    candidates: list[dict] = []
    for split in splits:
        csv_p = AUDIT / f"{split}_semantic_neighbor_rows.csv"
        if not csv_p.exists():
            continue
        V = load_pred(PRED_BASE / split / "vanilla/predictions.jsonl")
        C = load_pred(PRED_BASE / split / "clearsight/predictions.jsonl")
        for r in csv.DictReader(csv_p.open()):
            if r.get("negative_type") != "negative_related_present":
                continue
            qid = str(r["question_id"])
            if V.get(qid) != "no":  # target must be absent AND the model must say "no"
                continue
            flipped = C.get(qid) == "yes"
            rel_present = [o for o in (r.get("related_present") or "").split("|") if o]
            neighbors = cooc.get(r["target"], [])
            if isinstance(neighbors, dict):
                neighbors = list(neighbors)
            candidates.append({
                "split": split,
                "question_id": qid,
                "image_id": str(r["image_id"]),
                "target": r["target"],
                "question": r["question"],
                "related_present": "|".join(rel_present),
                "cooccurrence_rank": cooc_rank(neighbors, rel_present),
                "flip": int(flipped),
            })
    # flip cases first, then strong co-occurrence, then split priority, then qid
    candidates.sort(key=lambda x: (
        -x["flip"], x["cooccurrence_rank"], SPLIT_ORDER.get(x["split"], 9), int(x["question_id"]),
    ))
    pool, seen = [], set()
    for c in candidates:
        key = (c["image_id"], c["target"])  # same image+target => identical map; keep one
        if key in seen:
            continue
        seen.add(key)
        pool.append(c)
        if len(pool) >= limit:
            break
    return pool


def clean_map(model, processor, llm_layers, hooks, band, image_path, question, ratio, image_token_id):
    prompt = f"<image>\nUSER: {question}{POPE_SUFFIX}\nASSISTANT:"
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.float16)
    image.close()
    vs, ve = find_vis_bounds(inputs["input_ids"][0], int(image_token_id))
    with torch.inference_mode():
        out = model(**inputs, output_attentions=True)
    attentions = out.attentions
    last = int(inputs["input_ids"].shape[1] - 1)
    a0 = llm_layers[band[0]].self_attn
    scaling = getattr(a0, "scaling", None) or (getattr(a0, "head_dim", 128) ** -0.5)
    acc = np.zeros(ve - vs, np.float32)
    for li in band:
        kn = hooks[li].key_norms
        all_pos = torch.arange(kn.size(0), device=kn.device)
        vis_pos = all_pos[(all_pos >= vs) & (all_pos < ve)]
        sink_pos = auto_detect_sinks(kn, vis_pos)
        no_rope = _no_rope_last_row(hooks[li], last, float(scaling))
        clean = _drop_sinks_renorm(no_rope, sink_pos)
        acc += clean[vs:ve].cpu().numpy()
    return acc / len(band)


def upsample(m: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(m, SCALE, axis=0), SCALE, axis=1)


def render_triptych(image_336: np.ndarray, cmap_1d: np.ndarray, out_png: Path, sep: int = 8) -> None:
    """image_336: (336,336,3) float [0,1]; cmap_1d: (576,) attention. Writes a bare
    [original | map | overlay] PNG — no titles, axes, colorbars, or boxes."""
    loc = (cmap_1d / cmap_1d.sum()).reshape(GRID, GRID)
    up = upsample(loc)
    vmax = float(np.percentile(loc, CLIP_Q)) or float(loc.max()) or 1.0
    normed = np.clip(up / vmax, 0.0, 1.0)
    map_rgb = MAGMA(normed)[..., :3]
    a = (np.sqrt(normed) * ALPHA)[..., None]
    over_rgb = image_336 * (1.0 - a) + map_rgb * a
    white = np.ones((image_336.shape[0], sep, 3), np.float32)
    comp = np.concatenate([image_336, white, map_rgb, white, over_rgb], axis=1)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "mitigation/results/object_cooccurrence_prior/case_gallery"))
    p.add_argument("--coco_path", default=os.environ.get("COCO_PATH", "/srv/common_dataset/coco-2014-dataset"))
    p.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--splits", nargs="+", default=["adversarial", "popular", "random"])
    p.add_argument("--ratio", type=float, default=0.5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = build_pool(args.splits, args.limit)
    n_flip = sum(c["flip"] for c in pool)
    print(f"pool selected: {len(pool)} cases ({n_flip} flip vanilla=no->CS=yes, "
          f"{len(pool) - n_flip} vanilla=no non-flip) from splits {args.splits}")

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None)
    llm_layers = resolve_llm_layers(model)
    image_token_id = getattr(processor, "image_token_id", None) or getattr(processor, "image_token_index", 32000)
    band = list(range(BAND_LO, BAND_HI))
    hooks = {li: KeyNormHook(llm_layers[li].self_attn) for li in band}
    for li in band:
        llm_layers[li].self_attn.register_forward_pre_hook(hooks[li], with_kwargs=True)

    # 336x336 processed image (Resize 336 bicubic + CenterCrop) to match the visual grid
    from torchvision import transforms
    proc = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
    ])

    manifest = []
    for i, case in enumerate(pool, start=1):
        img_path = Path(args.coco_path) / "val2014" / f"COCO_val2014_{int(case['image_id']):012d}.jpg"
        amap = clean_map(model, processor, llm_layers, hooks, band, img_path, case["question"], args.ratio, image_token_id)
        image = Image.open(img_path).convert("RGB")
        image_336 = np.asarray(proc(image)).astype(np.float32) / 255.0
        image.close()
        name = f"case_{case['split']}_{case['question_id']}_{case['target'].replace(' ', '-')}.png"
        render_triptych(image_336, amap, out_dir / name)
        manifest.append({**case, "file": name})
        print(f"  [{i:3d}/{len(pool)}] {name}  (flip={case['flip']} / related={case['related_present']})")

    with (out_dir / "gallery_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "flip", "question_id", "image_id",
                                          "target", "related_present", "cooccurrence_rank", "question"])
        w.writeheader()
        w.writerows([{k: m[k] for k in w.fieldnames} for m in manifest])
    print(f"\nwrote {len(manifest)} triptychs + gallery_manifest.csv to {out_dir}")


if __name__ == "__main__":
    main()
