#!/usr/bin/env python
"""Regenerate the 14 SELECTED clean-map triptychs (reproducible, cached).

Two phases:
  1. CACHE (needs GPU, run once): compute each case's clean attention map + its
     336x336 processed image, save to a cache npz.
  2. RENDER (no model / no GPU, instant): read the cache and draw one bare
     [ original | clean attention map | overlay ] PNG per case.

On every run: if the cache exists it is used and the model is NOT loaded (so
adjusting colors is instant). Pass --refresh to recompute the attention cache.

Output: `mitigation/results/object_cooccurrence_prior/case_gallery/` — same
filenames, format, and location as before.

>>> To adjust colors / style, edit only the STYLE block below, then just re-run
    (the cached attention is reused; no GPU needed). <<<

Build the cache once:
  HF_HOME=/srv/common_model/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 COCO_PATH=/srv/common_dataset/coco-2014-dataset \
  MODEL_PATH=llava-hf/llava-1.5-7b-hf \
  /home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
    mitigation/scripts/regenerate_selected_cases.py
Then re-style with just:
  /home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
    mitigation/scripts/regenerate_selected_cases.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.cm as cm

# ─────────────────────────────  STYLE — edit these  ─────────────────────────────
# Everything a color/appearance tweak needs lives here. Changing anything here only
# affects RENDER; the cached attention is reused (no GPU needed).
CMAP_NAME = "viridis"     # any matplotlib colormap: "viridis", "inferno", "turbo", "jet", ...
ALPHA = 0.60
ALPHA_GAMMA = 0.45       # opacity curve: alpha = (value/vmax) ** ALPHA_GAMMA * ALPHA
                        #   0.5 = sqrt (current look); 1.0 = linear; >1 = only peaks show
CLIP_Q = 99.5           # color-scale cap percentile (attention is spike-dominated)
SEP_PX = 8              # white gap between the three panels, in pixels
SEP_RGB = (1.0, 1.0, 1.0)   # gap color (0..1); e.g. (0,0,0) for black
BACKGROUND_FOR_MAP = None    # None = plain colormap; or an (r,g,b) to tint the map's zeros
# ────────────────────────────────────────────────────────────────────────────────

# The 14 selected cases: (split, question_id, image_id, target, question).
SELECTED_CASES = [
    ("adversarial", "1090", 142774, "bicycle", "Is there a bicycle in the image?"),
    ("adversarial", "1268", 244455, "traffic light", "Is there a traffic light in the image?"),
    ("adversarial", "1396", 316237, "backpack", "Is there a backpack in the image?"),
    ("adversarial", "1444", 407386, "mouse", "Is there a mouse in the image?"),
    ("adversarial", "1456", 18918, "cup", "Is there a cup in the image?"),
    ("adversarial", "1492", 302405, "handbag", "Is there a handbag in the image?"),
    ("adversarial", "1932", 356424, "spoon", "Is there a spoon in the image?"),
    ("adversarial", "2760", 536073, "bowl", "Is there a bowl in the image?"),
    ("adversarial", "2850", 264155, "bowl", "Is there a bowl in the image?"),
    ("adversarial", "2916", 190788, "spoon", "Is there a spoon in the image?"),
    ("popular", "1332", 287035, "cup", "Is there a cup in the image?"),
    ("popular", "1720", 354976, "cup", "Is there a cup in the image?"),
    ("random", "302", 515904, "bowl", "Is there a bowl in the image?"),
    ("random", "442", 231589, "toothbrush", "Is there a toothbrush in the image?"),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRID, SCALE = 24, 14
BAND_LO, BAND_HI = 9, 15
POPE_SUFFIX = " Please just answer yes or no."


def case_name(split, qid, target) -> str:
    return f"case_{split}_{qid}_{target.replace(' ', '-')}.png"


# ─────────────────────────────  CACHE (needs GPU)  ──────────────────────────────
def build_cache(cache_path: Path, coco_path: str, model_path: str, cache_dir: str | None,
                device_idx: int, ratio: float) -> None:
    """Run the model on all 14 cases and cache (clean map, processed image)."""
    import torch  # lazy: only imported when the cache must be (re)built
    sys.path.insert(0, str(Path(__file__).resolve().parent))       # mitigation/scripts
    sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))
    from extract_qualitative_sink_attention import (
        KeyNormHook, _no_rope_last_row, _drop_sinks_renorm, resolve_llm_layers,
    )
    from sinkdetect.sink_utils import auto_detect_sinks, find_vis_bounds
    from sinkdetect.utils import load_model_and_processor
    from torchvision import transforms

    device = torch.device(f"cuda:{device_idx}")
    model, processor = load_model_and_processor(model_path, device, cache_dir=cache_dir or None)
    llm_layers = resolve_llm_layers(model)
    image_token_id = getattr(processor, "image_token_id", None) or getattr(processor, "image_token_index", 32000)
    band = list(range(BAND_LO, BAND_HI))
    hooks = {li: KeyNormHook(llm_layers[li].self_attn) for li in band}
    for li in band:
        llm_layers[li].self_attn.register_forward_pre_hook(hooks[li], with_kwargs=True)
    proc = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
    ])

    names, maps, images = [], [], []
    for i, (split, qid, image_id, target, question) in enumerate(SELECTED_CASES, start=1):
        img_path = Path(coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        prompt = f"<image>\nUSER: {question}{POPE_SUFFIX}\nASSISTANT:"
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
        vs, ve = find_vis_bounds(inputs["input_ids"][0], int(image_token_id))
        with torch.inference_mode():
            model(**inputs, output_attentions=True)
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
        maps.append((acc / len(band)).astype(np.float32))
        images.append(np.asarray(proc(image)).astype(np.uint8))
        names.append(case_name(split, qid, target))
        image.close()
        print(f"  [cache {i:2d}/{len(SELECTED_CASES)}] {names[-1]}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        names=np.array(names),
        maps=np.stack(maps),          # (14, 576) clean attention
        images=np.stack(images),      # (14, 336, 336, 3) uint8
    )
    print(f"cached attention for {len(names)} cases -> {cache_path}")


# ─────────────────────────────  RENDER (no GPU)  ────────────────────────────────
def upsample(m: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(m, SCALE, axis=0), SCALE, axis=1)


def _to_u8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def compute_panels(image_336: np.ndarray, cmap_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three panels as float[0,1] RGB arrays: (original, clean map, overlay).
    Appearance is controlled by the STYLE block."""
    cmap = cm.get_cmap(CMAP_NAME)
    loc = (cmap_1d / cmap_1d.sum()).reshape(GRID, GRID)
    up = upsample(loc)
    vmin = 0.0025
    CLIP_Q = 98.5
    vmax = float(np.percentile(loc, CLIP_Q)) or float(loc.max()) or 1.0

    normed = np.clip((up - vmin) / (vmax - vmin), 0.0, 1.0)

    map_rgb = cmap(normed)[..., :3].astype(np.float32)
    if BACKGROUND_FOR_MAP is not None:
        bg = np.array(BACKGROUND_FOR_MAP, np.float32)
        w = normed[..., None]
        map_rgb = bg * (1.0 - w) + map_rgb * w

    a = (np.power(normed, ALPHA_GAMMA) * ALPHA)[..., None]
    over_rgb = image_336 * (1.0 - a) + cmap(normed)[..., :3].astype(np.float32) * a
    return image_336, map_rgb, over_rgb


def _hconcat(panels: list[np.ndarray]) -> np.ndarray:
    """Concatenate panels left-to-right with a whitespace gap (SEP_PX/SEP_RGB) between them."""
    h = panels[0].shape[0]
    sep = np.ones((h, SEP_PX, 3), np.float32) * np.array(SEP_RGB, np.float32)
    pieces = []
    for i, panel in enumerate(panels):
        if i:
            pieces.append(sep)
        pieces.append(panel)
    return np.concatenate(pieces, axis=1)


def render_triptych(image_336: np.ndarray, cmap_1d: np.ndarray, out_png: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Save the bare [ original | clean map | overlay ] combined PNG and return the panels."""
    original, map_rgb, over_rgb = compute_panels(image_336, cmap_1d)
    Image.fromarray(_to_u8(_hconcat([original, map_rgb, over_rgb]))).save(out_png)
    return original, map_rgb, over_rgb


def export_panels(original: np.ndarray, map_rgb: np.ndarray, over_rgb: np.ndarray,
                  stem: str, export_dir: Path) -> None:
    """Export the three requested results, per case:
      1. triptych [ original | map | overlay ]  -> PNG
      2. [ original | overlay ] side by side     -> PNG
      3. overlay alone                            -> PDF
    All horizontal joins use the STYLE whitespace gap (SEP_PX / SEP_RGB)."""
    export_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_u8(_hconcat([original, map_rgb, over_rgb]))).save(export_dir / f"{stem}_triptych.png")
    Image.fromarray(_to_u8(_hconcat([original, over_rgb]))).save(export_dir / f"{stem}_original-overlay.png")
    Image.fromarray(_to_u8(over_rgb)).convert("RGB").save(export_dir / f"{stem}_overlay.pdf")


def render_from_cache(cache_path: Path, out_dir: Path, export_dir: Path) -> None:
    data = np.load(cache_path, allow_pickle=True)
    by_name = {n: (data["maps"][i], data["images"][i]) for i, n in enumerate(data["names"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (split, qid, image_id, target, question) in enumerate(SELECTED_CASES, start=1):
        name = case_name(split, qid, target)
        amap, image_u8 = by_name[name]
        original, map_rgb, over_rgb = render_triptych(
            image_u8.astype(np.float32) / 255.0, amap.astype(np.float32), out_dir / name
        )
        # per request, three results per case: triptych PNG, original|overlay PNG, overlay PDF
        export_panels(original, map_rgb, over_rgb, stem=name[:-4], export_dir=export_dir)
        manifest.append({"file": name, "split": split, "question_id": qid,
                         "image_id": image_id, "target": target, "question": question})
        print(f"  [render {i:2d}/{len(SELECTED_CASES)}] {name}")
    with (out_dir / "gallery_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "question_id", "image_id", "target", "question"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\nrendered {len(manifest)} triptychs -> {out_dir}")
    print(f"exported per case: triptych PNG + original|overlay PNG + overlay PDF -> {export_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "mitigation/results/object_cooccurrence_prior/case_gallery"))
    p.add_argument("--export_dir", default=None,
                   help="per-case exports go here (default: <out_dir>/export): triptych PNG, "
                        "original|overlay side-by-side PNG, and overlay-only PDF")
    p.add_argument("--cache", default=None, help="attention cache npz (default: <out_dir>/selected_cases_cache.npz)")
    p.add_argument("--refresh", action="store_true", help="recompute the attention cache (needs GPU)")
    p.add_argument("--coco_path", default=os.environ.get("COCO_PATH", "/srv/common_dataset/coco-2014-dataset"))
    p.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.5)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    cache_path = Path(args.cache) if args.cache else out_dir / "selected_cases_cache.npz"
    export_dir = Path(args.export_dir) if args.export_dir else out_dir / "export"

    if args.refresh or not cache_path.exists():
        why = "refresh requested" if args.refresh else f"no cache at {cache_path}"
        print(f"building attention cache ({why}) — loading model...")
        build_cache(cache_path, args.coco_path, args.model_path, args.cache_dir, args.device, args.ratio)
    else:
        print(f"using cached attention: {cache_path} (no model loaded)")
    render_from_cache(cache_path, out_dir, export_dir)


if __name__ == "__main__":
    main()
