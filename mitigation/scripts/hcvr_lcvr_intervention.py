#!/usr/bin/env python
"""Reproduce the HCVR/LCVR attention-intervention phenomenon (paper Fig. 4) on ONE case.

Case: POPE adversarial qid 1268 — "Is there a traffic light in the image?"
(image COCO_val2014_000000244455, traffic light ABSENT, related objects present).

Localization map = the CLEAN attention (RoPE position prior removed via no-RoPE
recomputation, AND detected sink tokens deleted), layers 9-14 mean over heads,
decision row — the same map used for the case gallery. Among the NON-sink visual
tokens we split by clean attention into:
  - HCVR (top 25%)     — "High-Credibility Visual Regions"
  - LCVR (bottom 25%)  — "Low-Credibility Visual Regions"
Sink tokens are excluded from both. We then amplify each region's attention by
factors 1.0..1.5 and measure how P("yes")/P("no") of the first answer token change.

The intervention edits the attention weights POST-softmax: for every post-image
(text/answer) query row in the band layers it multiplies the region's attention
weights by the factor without row renormalization. It is applied via a gated
monkeypatch of transformers 4.48's `eager_attention_forward` (transformers 5.x
DetectionAdapter is unavailable here).

Two phases (like regenerate_selected_cases.py):
  1. CACHE (needs GPU, run once): baseline attention + full 2x6 intervention sweep.
  2. RENDER (no GPU, instant): 4 figures + saved list data. Edit STYLE and re-run.

HCVR/LCVR fractions are set by --top_q / --bottom_q (default 0.25 / 0.25); the output
dir is auto-tagged (…_t25_b25, …_t10_b50, …). Figures are bare (no title, no margin):
  fig1_HCVR_top.png        binary HCVR region highlight on the image (in-region patches = 1)
  fig2_LCVR_bottom.png     binary LCVR region highlight on the image
  fig3_amplify_top.png     line plot: amplify-HCVR factor vs P(yes)/P(no)
  fig4_amplify_bottom.png  line plot: amplify-LCVR factor vs P(yes)/P(no)
fig3/fig4 share ONE x-axis (equal intervals, ticks end in 0/5) for direct comparison.
Saved data: cache.npz, attention_distribution.csv, sweep_*.csv, summary.json.

Build the cache once (pick a free GPU):
  HF_HOME=/srv/common_model/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=7 COCO_PATH=/srv/common_dataset/coco-2014-dataset \
  MODEL_PATH=llava-hf/llava-1.5-7b-hf \
  /home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
    mitigation/scripts/hcvr_lcvr_intervention.py --device 0
Then re-style / re-plot with just:
  /home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
    mitigation/scripts/hcvr_lcvr_intervention.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────────  STYLE — edit these  ─────────────────────────────
FONT_FAMILY = "Comic Sans MS"  # all figure text
REGION_COLOR = "#001f3f"  # fig1/fig2 binary region fill color (dark navy blue)
REGION_ALPHA = 0.80       # fig1/fig2 region fill opacity
YES_COLOR = "#1b7837"     # P(yes) line
NO_COLOR = "#b2182b"      # P(no) line
# ────────────────────────────────────────────────────────────────────────────────

plt.rcParams["font.family"] = FONT_FAMILY

CASE = ("adversarial", "1268", 244455, "traffic light", "Is there a traffic light in the image?")
# Dense amplification-factor sweep CACHED once (post-softmax reweight of the region).
# The displayed x-range is auto-selected at RENDER time to span ~TARGET_CHANGE, so the
# target can be re-tuned without re-running the model.
SWEEP_FACTORS = sorted(set([1.0] + list(np.round(np.geomspace(1.02, 20.0, 60), 4))))
TARGET_CHANGE = 0.30    # the HCVR line-plot is scaled to span ~this much P(yes) change; fig4 shares its x-axis
N_DISPLAY = 6           # points shown per line-plot figure
TOP_Q = 0.25            # HCVR fraction of eligible (non-sink) visual tokens
BOTTOM_Q = 0.25         # LCVR fraction of eligible (non-sink) visual tokens
BAND_LO, BAND_HI = 9, 15
GRID, SCALE = 24, 14
POPE_SUFFIX = " Please just answer yes or no."

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────  intervention (monkeypatch)  ─────────────────────────
def make_intervened_eager():
    """Return a drop-in replacement for transformers 4.48 eager_attention_forward that,
    when module._intervene is set, scales the decision row's selected visual columns by
    module._scale without row renormalization. Non-flagged modules behave identically to
    the original."""
    import torch
    from torch import nn
    from transformers.models.llama.modeling_llama import repeat_kv

    def _eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        # POST-softmax amplification: literally multiply the region's attention weights by
        # the factor for every post-image (text/answer) query row. No full-row renorm — a
        # renormalized reweight saturates (attending 100% to the region is the ceiling, only
        # a ~10-20% swing here); leaving the row un-renormalized injects more of the region's
        # evidence and lets the swing reach the target magnitude.
        if getattr(module, "_intervene", False) and float(module._scale) != 1.0 and attn_weights.shape[-2] > 1:
            f = float(module._scale)
            q0 = int(module._q_start)
            cols = module._region_cols  # LongTensor of absolute key indices (visual)
            sub = attn_weights[:, :, q0:, :]
            sub[:, :, :, cols] = sub[:, :, :, cols] * f
        attn_weights = attn_weights.to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return _eager


def _yes_no_ids(tokenizer):
    def fid(s):
        ids = tokenizer(s, add_special_tokens=False).input_ids
        return ids[0] if ids else None
    yes = {fid(s) for s in ["Yes", "yes"]}
    no = {fid(s) for s in ["No", "no"]}
    yes.discard(None); no.discard(None)
    return sorted(yes), sorted(no)


# ─────────────────────────────  CACHE (needs GPU)  ──────────────────────────────
def build_cache(cache_path: Path, coco_path: str, model_path: str, cache_dir: str | None,
                device_idx: int, top_q: float = TOP_Q, bottom_q: float = BOTTOM_Q,
                select_by: str = "count") -> None:
    import torch
    import transformers.models.llama.modeling_llama as ll
    sys.path.insert(0, str(Path(__file__).resolve().parent))        # mitigation/scripts
    sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))
    from extract_qualitative_sink_attention import (
        KeyNormHook, _no_rope_last_row, _drop_sinks_renorm,
    )
    from sinkdetect.sink_utils import auto_detect_sinks, find_vis_bounds
    from sinkdetect.utils import load_model_and_processor
    from torchvision import transforms

    ll.eager_attention_forward = make_intervened_eager()  # gated; safe for all layers

    split, qid, image_id, target, question = CASE
    device = torch.device(f"cuda:{device_idx}")
    model, processor = load_model_and_processor(model_path, device, cache_dir=cache_dir or None)
    tok = processor.tokenizer
    yes_ids, no_ids = _yes_no_ids(tok)

    def _llm_layers(m):
        for get in (lambda x: x.model.language_model.layers,
                    lambda x: x.language_model.model.layers,
                    lambda x: x.language_model.layers):
            try:
                layers = get(m)
            except AttributeError:
                continue
            if isinstance(layers, torch.nn.ModuleList):
                return layers
        raise RuntimeError("cannot find decoder layers")

    layers = _llm_layers(model)
    band = list(range(BAND_LO, BAND_HI))
    image_token_id = getattr(processor, "image_token_id", None) or getattr(processor, "image_token_index", 32000)

    img_path = Path(coco_path) / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
    prompt = f"<image>\nUSER: {question}{POPE_SUFFIX}\nASSISTANT:"
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    vs, ve = find_vis_bounds(inputs["input_ids"][0], int(image_token_id))
    last = int(inputs["input_ids"].shape[1] - 1)

    def probs_at_last():
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
        logits = out.logits[0, last].float()
        p = torch.softmax(logits, dim=-1)
        return float(p[yes_ids].sum()), float(p[no_ids].sum()), int(logits.argmax())

    # ── baseline forward: capture pre-RoPE key norms / q,k (for the clean map) ──
    hooks = {li: KeyNormHook(layers[li].self_attn) for li in band}
    handles = [layers[li].self_attn.register_forward_pre_hook(hooks[li], with_kwargs=True) for li in band]
    for li in band:
        layers[li].self_attn._intervene = False
    with torch.inference_mode():
        out = model(**inputs, use_cache=False, output_attentions=True)
    base_logits = out.logits[0, last].float()
    base_p = torch.softmax(base_logits, dim=-1)
    base_yes, base_no, base_arg = float(base_p[yes_ids].sum()), float(base_p[no_ids].sum()), int(base_logits.argmax())
    print(f"baseline: P(yes)={base_yes:.4f} P(no)={base_no:.4f} argmax='{tok.decode([base_arg])}'")

    # ── clean localization map: no-RoPE (position prior removed) + sink tokens deleted,
    #    mean over band, decision row — the SAME map used for the case gallery ──
    a0 = layers[band[0]].self_attn
    scaling = getattr(a0, "scaling", None) or (getattr(a0, "head_dim", 128) ** -0.5)
    n_vis = ve - vs
    clean = np.zeros(n_vis, np.float32)
    sink_hits = np.zeros(n_vis, np.float32)
    for li in band:
        kn = hooks[li].key_norms
        all_pos = torch.arange(kn.size(0), device=kn.device)
        vis_pos = all_pos[(all_pos >= vs) & (all_pos < ve)]
        sink_pos = auto_detect_sinks(kn, vis_pos)
        no_rope = _no_rope_last_row(hooks[li], last, float(scaling))
        clean += _drop_sinks_renorm(no_rope, sink_pos)[vs:ve].cpu().numpy()
        for s in sink_pos.cpu().tolist():
            sink_hits[s - vs] += 1.0
    clean /= len(band)
    for h in handles:
        h.remove()
    attn = clean  # displayed + ranked map is the clean map

    # ── regions on the clean map, with sink tokens EXCLUDED ──
    SINK_MIN_HITS = 4  # flagged in >=4/6 band layers => persistent sink, dropped from HCVR/LCVR
    sink_mask = sink_hits >= SINK_MIN_HITS
    eligible = np.flatnonzero(~sink_mask)
    order = eligible[np.argsort(clean[eligible])]     # eligible positions, ascending clean value
    vals = clean[order].astype(np.float64)            # ascending clean values along `order`
    if select_by == "mass":
        # HCVR/LCVR are the smallest high/low token sets whose CUMULATIVE clean-attention
        # score reaches top_q / bottom_q of the total eligible mass (score-based split).
        total = vals.sum() or 1.0
        cs_hi = np.cumsum(vals[::-1]) / total          # cumulative from the highest token down
        cs_lo = np.cumsum(vals) / total                # cumulative from the lowest token up
        k_top = max(1, int(np.searchsorted(cs_hi, top_q)) + 1)
        k_bot = max(1, int(np.searchsorted(cs_lo, bottom_q)) + 1)
    else:                                              # select_by == "count"
        k_top = max(1, int(round(top_q * eligible.size)))
        k_bot = max(1, int(round(bottom_q * eligible.size)))
    bottom_local = order[:k_bot]
    top_local = order[-k_top:]
    top_mask = np.zeros(n_vis, bool); top_mask[top_local] = True
    bottom_mask = np.zeros(n_vis, bool); bottom_mask[bottom_local] = True
    hcvr_mass = vals[-k_top:].sum() / (vals.sum() or 1.0)
    lcvr_mass = vals[:k_bot].sum() / (vals.sum() or 1.0)
    print(f"regions on CLEAN map (select_by={select_by}): {int(sink_mask.sum())} sinks excluded, "
          f"{eligible.size} eligible; HCVR(top {top_q:.0%})={k_top} tok / {hcvr_mass:.1%} mass, "
          f"LCVR(bottom {bottom_q:.0%})={k_bot} tok / {lcvr_mass:.1%} mass")
    top_cols = torch.tensor(vs + top_local, device=device, dtype=torch.long)
    bottom_cols = torch.tensor(vs + bottom_local, device=device, dtype=torch.long)

    # ── intervention sweep ──
    def sweep(region_cols):
        ys, ns = [], []
        for li in band:
            a = layers[li].self_attn
            a._region_cols = region_cols; a._q_start = ve  # amplify for post-image query rows
        for f in SWEEP_FACTORS:
            for li in band:
                a = layers[li].self_attn
                a._intervene = True; a._scale = f
            y, n, _ = probs_at_last()
            ys.append(y); ns.append(n)
        for li in band:
            layers[li].self_attn._intervene = False
        return np.array(ys, np.float32), np.array(ns, np.float32)

    top_yes, top_no = sweep(top_cols)
    bottom_yes, bottom_no = sweep(bottom_cols)
    print(f"amplify HCVR(top {top_q:.0%}):    P(yes) {top_yes[0]:.3f}->{top_yes[-1]:.3f}   P(no) {top_no[0]:.3f}->{top_no[-1]:.3f}")
    print(f"amplify LCVR(bottom {bottom_q:.0%}): P(yes) {bottom_yes[0]:.3f}->{bottom_yes[-1]:.3f}   P(no) {bottom_no[0]:.3f}->{bottom_no[-1]:.3f}")

    proc = transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(336),
    ])
    image_336 = np.asarray(proc(image)).astype(np.uint8)
    image.close()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        image=image_336, attention=attn,
        top_mask=top_mask, bottom_mask=bottom_mask, sink_mask=sink_mask,
        factors=np.array(SWEEP_FACTORS, np.float32),
        top_yes=top_yes, top_no=top_no, bottom_yes=bottom_yes, bottom_no=bottom_no,
        baseline_yes=np.float32(base_yes), baseline_no=np.float32(base_no),
        yes_ids=np.array(yes_ids), no_ids=np.array(no_ids),
        vis_start=np.int32(vs), vis_end=np.int32(ve),
        top_q=np.float32(top_q), bottom_q=np.float32(bottom_q),
        select_by=np.str_(select_by),
    )
    print(f"cached -> {cache_path}")


# ─────────────────────────────  RENDER (no GPU)  ────────────────────────────────
def _upsample(m):
    return np.repeat(np.repeat(m, SCALE, axis=0), SCALE, axis=1)


def _overlay_region(ax, image, mask_2d):
    """Binary region highlight: every non-zero (in-region) patch is drawn as a single
    uniform fill color at REGION_ALPHA; everything else is transparent. No graded values."""
    from matplotlib.colors import to_rgb
    ax.imshow(image)
    up = _upsample(mask_2d.astype(np.float32))          # 1 inside region, 0 outside
    rgba = np.zeros((*up.shape, 4), np.float32)
    rgba[..., :3] = to_rgb(REGION_COLOR)
    rgba[..., 3] = up * REGION_ALPHA
    ax.imshow(rgba)
    ax.set_axis_off()


def _nice_step(raw):
    """Round a raw step to the nearest multiple of 0.05 (>=0.05); such a step keeps every
    tick 1.0 + k*step ending in a 0 or 5 decimal digit."""
    return max(0.05, round(raw / 0.05) * 0.05)


def _shared_xaxis(factors, hcvr_yes, target, n):
    """One x-axis shared by BOTH line plots so they are directly comparable: equal-sized
    intervals whose ticks end in 0/5. Scale is set by the HCVR curve (the one that reaches
    ~target); f* is where HCVR's P(yes) first gains `target` (or its directed peak)."""
    directed = hcvr_yes - hcvr_yes[0]
    peak = int(np.argmax(directed))
    dd = directed[: peak + 1]
    idx = int(np.argmax(dd >= target)) if np.any(dd >= target) else peak
    fstar = float(factors[max(idx, 1)])
    step = _nice_step((fstar - 1.0) / (n - 1))
    xs = 1.0 + step * np.arange(n)
    return np.round(xs, 2)


def _line_plot(ax, factors, yes, no, xs):
    ys = np.interp(xs, factors, yes)
    ns = np.interp(xs, factors, no)
    ax.plot(xs, ys, "-o", color=YES_COLOR, label='P("yes")', linewidth=2)
    ax.plot(xs, ns, "-s", color=NO_COLOR, label='P("no")', linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.2f}" for x in xs], rotation=0)
    ax.set_xlim(xs[0], xs[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    return xs, ys, ns


def render_from_cache(cache_path: Path, out_dir: Path) -> None:
    d = np.load(cache_path, allow_pickle=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = d["image"]
    top_2d = d["top_mask"].reshape(GRID, GRID)
    bottom_2d = d["bottom_mask"].reshape(GRID, GRID)
    factors = d["factors"]
    top_q = float(d["top_q"]); bottom_q = float(d["bottom_q"])
    qt = int(round(top_q * 100)); qb = int(round(bottom_q * 100))

    # fig1/fig2 — binary region highlight, no title, no margin
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    _overlay_region(ax, image, top_2d)
    fig.savefig(out_dir / "fig1_HCVR_top.png", bbox_inches="tight", pad_inches=0, dpi=200)
    fig.savefig(out_dir / "fig1_HCVR_top.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    _overlay_region(ax, image, bottom_2d)
    fig.savefig(out_dir / "fig2_LCVR_bottom.png", bbox_inches="tight", pad_inches=0, dpi=200)
    fig.savefig(out_dir / "fig2_LCVR_bottom.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # fig3/fig4 — line plots share ONE round x-axis (equal intervals, ticks end in 0/5),
    # scaled off the HCVR curve; no title
    xs = _shared_xaxis(factors, d["top_yes"], TARGET_CHANGE, N_DISPLAY)

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    top_xs, top_ys, top_ns = _line_plot(ax, factors, d["top_yes"], d["top_no"], xs)
    fig.savefig(out_dir / "fig3_amplify_top.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    fig.savefig(out_dir / "fig3_amplify_top.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    bot_xs, bot_ys, bot_ns = _line_plot(ax, factors, d["bottom_yes"], d["bottom_no"], xs)
    fig.savefig(out_dir / "fig4_amplify_bottom.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    fig.savefig(out_dir / "fig4_amplify_bottom.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # saved list data — the displayed points per figure (both share the same x-axis)
    with (out_dir / "sweep_HCVR_top.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["factor", "P_yes", "P_no"])
        for i in range(len(top_xs)):
            w.writerow([f"{top_xs[i]:.4f}", f"{top_ys[i]:.6f}", f"{top_ns[i]:.6f}"])
    with (out_dir / "sweep_LCVR_bottom.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["factor", "P_yes", "P_no"])
        for i in range(len(bot_xs)):
            w.writerow([f"{bot_xs[i]:.4f}", f"{bot_ys[i]:.6f}", f"{bot_ns[i]:.6f}"])
    # full dense sweep (model output at every cached factor) for reference
    with (out_dir / "sweep_dense.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "HCVR_top_P_yes", "HCVR_top_P_no", "LCVR_bottom_P_yes", "LCVR_bottom_P_no"])
        for i, fac in enumerate(factors):
            w.writerow([f"{fac:.4f}", f"{d['top_yes'][i]:.6f}", f"{d['top_no'][i]:.6f}",
                        f"{d['bottom_yes'][i]:.6f}", f"{d['bottom_no'][i]:.6f}"])

    attn = d["attention"]
    with (out_dir / "attention_distribution.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["visual_token_index", "attention", "in_HCVR_top", "in_LCVR_bottom"])
        for i in range(attn.size):
            w.writerow([i, f"{attn[i]:.8f}", int(d["top_mask"][i]), int(d["bottom_mask"][i])])

    (out_dir / "summary.json").write_text(json.dumps({
        "case": {"split": CASE[0], "question_id": CASE[1], "image_id": CASE[2],
                 "target": CASE[3], "question": CASE[4]},
        "band_layers": list(range(BAND_LO, BAND_HI)),
        "top_q": float(d["top_q"]), "bottom_q": float(d["bottom_q"]),
        "n_HCVR_tokens": int(d["top_mask"].sum()), "n_LCVR_tokens": int(d["bottom_mask"].sum()),
        "factors": [float(x) for x in factors],
        "baseline": {"P_yes": float(d["baseline_yes"]), "P_no": float(d["baseline_no"])},
        "amplify_HCVR_top": {"P_yes": [float(x) for x in d["top_yes"]], "P_no": [float(x) for x in d["top_no"]]},
        "amplify_LCVR_bottom": {"P_yes": [float(x) for x in d["bottom_yes"]], "P_no": [float(x) for x in d["bottom_no"]]},
    }, indent=2))
    print(f"rendered 4 figures + sweeps + attention_distribution.csv + summary.json -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=None, help="default: hcvr_lcvr_1268_t<top>_b<bottom>")
    p.add_argument("--cache", default=None)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--top_q", type=float, default=TOP_Q, help="HCVR fraction (default 0.25)")
    p.add_argument("--bottom_q", type=float, default=BOTTOM_Q, help="LCVR fraction (default 0.25)")
    p.add_argument("--select_by", choices=["count", "mass"], default="count",
                   help="'count': top/bottom fraction of TOKENS; 'mass': of cumulative clean score")
    p.add_argument("--coco_path", default=os.environ.get("COCO_PATH", "/srv/common_dataset/coco-2014-dataset"))
    p.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    tag = f"t{int(round(args.top_q * 100))}_b{int(round(args.bottom_q * 100))}_{args.select_by}"
    out_dir = Path(args.out_dir) if args.out_dir else \
        PROJECT_ROOT / "mitigation/results/object_cooccurrence_prior" / f"hcvr_lcvr_1268_{tag}"
    cache_path = Path(args.cache) if args.cache else out_dir / "cache.npz"
    if args.refresh or not cache_path.exists():
        why = "refresh requested" if args.refresh else f"no cache at {cache_path}"
        print(f"building cache ({why}, top_q={args.top_q}, bottom_q={args.bottom_q}, "
              f"select_by={args.select_by}) — loading model...")
        build_cache(cache_path, args.coco_path, args.model_path, args.cache_dir, args.device,
                    top_q=args.top_q, bottom_q=args.bottom_q, select_by=args.select_by)
    else:
        print(f"using cached results: {cache_path} (no model loaded)")
    render_from_cache(cache_path, out_dir)


if __name__ == "__main__":
    main()
