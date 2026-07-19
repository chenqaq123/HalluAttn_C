#!/usr/bin/env python
"""Re-run the two H3 qualitative cases to add sink-token detection/deletion and
position-bias (no-RoPE) attention maps.

The H3 qualitative npz files hold only vanilla / ClearSight visual attention; they
carry no key norms and no full seq×seq attention, so sink detection and purification
cannot be computed offline. This script reproduces the *vanilla* first-answer-token
decision for each saved case (same image, same POPE prompt) and captures, at the
last prefill position:

  original           — mean-over-heads visual attention row (reproduces vanilla_attention)
  sink_only          — after sink-token deletion (key-norm sinks zeroed on text rows)
  topmass_only       — after top-mass visual masking only
  purified           — sink deletion + top-mass masking (full purification)
  no_rope            — attention recomputed with pre-RoPE q/k (position-bias removed)
plus per-visual-token key norms (the sink diagnostic — sinks have low key norm) and
the per-layer detected sink positions.

This mirrors the reference AttnAdapter
(MM2026/Hallu_Sink_Correlation/utils/AttnAdapter.py) — pre-RoPE key L2 norm for sink
detection, cumulative top-mass masking over the visual tokens — and the SinkDetect
DetectionAdapter, but without importing either (both target transformers 5.x /
``TransformersKwargs``, unavailable in this env). Attention is read from
``output_attentions``; pre-RoPE key norms and no-RoPE attention come from a
forward pre-hook; sink detection and purification reuse the pure-torch
``detection.src.sinkdetect.sink_utils`` unchanged.

Faithfulness gate: the reproduced ``original`` map is asserted close to the saved
``vanilla_attention`` before anything is written — this also validates that the
local transformers version reproduces the run that made the npz.

Outputs, per case, next to the existing artifacts:
  qualitative_case_{i}_sink_attention.npz
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DET_SRC = PROJECT_ROOT / "detection" / "src"
if str(DET_SRC) not in sys.path:
    sys.path.insert(0, str(DET_SRC))

# sink_utils and utils import only torch / the Llava classes — no transformers 5.x API.
from sinkdetect.sink_utils import auto_detect_sinks, find_vis_bounds, purify_attention  # noqa: E402
from sinkdetect.utils import load_model_and_processor  # noqa: E402

BAND_LO, BAND_HI = 9, 15  # layers 9..14 inclusive — the H3 attention band
POPE_SUFFIX = " Please just answer yes or no."


def resolve_llm_layers(model):
    """Locate the decoder ModuleList across transformers layouts.

    5.x: model.model.language_model.layers
    4.48 LLaVA: model.language_model.model.layers  (language_model is a *ForCausalLM)
    fallback: model.language_model.layers
    """
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
    ]
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if isinstance(layers, torch.nn.ModuleList):
            return layers
    raise RuntimeError(f"Cannot locate decoder layers in {type(model).__name__}")


class KeyNormHook:
    """Forward pre-hook on a layer's self_attn: capture pre-RoPE per-token key L2 norm
    (mean over batch, heads) and the pre-RoPE q/k for the last query position, so
    no-RoPE attention for that row can be recomputed."""

    def __init__(self, attn_module):
        self.attn = attn_module
        self.key_norms = None       # (seq,)
        self.q_prerope = None       # (heads, seq, head_dim)
        self.k_prerope = None       # (kv_heads, seq, head_dim)

    def __call__(self, module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None:
            return
        cfg = module.config
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        hd = getattr(module, "head_dim", cfg.hidden_size // n_heads)
        bsz, seq, _ = hs.shape
        # Project in the module's native dtype (half), then upcast for stable norms.
        q = module.q_proj(hs).view(bsz, seq, n_heads, hd).transpose(1, 2).float()  # (b, H, S, d)
        k = module.k_proj(hs).view(bsz, seq, n_kv, hd).transpose(1, 2).float()     # (b, Hkv, S, d)
        self.key_norms = k.norm(p=2, dim=-1).mean(dim=(0, 1)).detach()             # (S,)
        self.q_prerope = q[0].detach()
        self.k_prerope = k[0].detach()


def _no_rope_last_row(hook: KeyNormHook, last: int, scaling: float) -> torch.Tensor:
    """Mean-over-heads no-RoPE attention of the last query row over ALL keys (S,).
    The last prefill row attends causally to every position, so no mask is needed."""
    q = hook.q_prerope[:, last, :]                       # (H, d)
    k = hook.k_prerope                                   # (Hkv, S, d)
    n_rep = q.shape[0] // k.shape[0]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=0)            # (H, S, d)
    logits = torch.einsum("hd,hsd->hs", q, k) * scaling  # (H, S)
    return torch.softmax(logits, dim=-1).mean(dim=0)     # (S,)


def _drop_sinks_renorm(row: torch.Tensor, sink_pos: torch.Tensor) -> torch.Tensor:
    """Zero the sink columns of a full attention row (over all keys) and renormalize
    to sum 1 — the row-level analogue of purify_attention's sink-removal step."""
    clean = row.clone()
    if sink_pos.numel() > 0:
        clean[sink_pos] = 0.0
    return clean / clean.sum().clamp_min(1e-10)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results_dir",
        default=str(PROJECT_ROOT / "mitigation/results/object_cooccurrence_prior"),
    )
    p.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.5, help="top-mass keep ratio")
    p.add_argument("--band_lo", type=int, default=BAND_LO)
    p.add_argument("--band_hi", type=int, default=BAND_HI)
    p.add_argument("--repro_atol", type=float, default=5e-3)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    meta = json.loads((results_dir / "qualitative_cases.json").read_text())
    cases = meta["cases"]
    band = list(range(args.band_lo, args.band_hi))

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(
        args.model_path, device, cache_dir=args.cache_dir or None
    )
    llm_layers = resolve_llm_layers(model)
    image_token_id = getattr(processor, "image_token_id", None) or getattr(
        processor, "image_token_index", 32000
    )

    hooks = {li: KeyNormHook(llm_layers[li].self_attn) for li in band}
    handles = [
        llm_layers[li].self_attn.register_forward_pre_hook(hooks[li], with_kwargs=True)
        for li in band
    ]

    # "clean" is the headline map: position prior removed (no-RoPE) AND sink tokens
    # deleted. "no_rope"/"sink_only"/"purified" are the single-factor references.
    variants = ["original", "sink_only", "topmass_only", "purified", "no_rope", "clean"]

    try:
        for r, case in enumerate(cases):
            i = r + 1
            question = case["question"] + POPE_SUFFIX
            prompt = f"<image>\nUSER: {question}\nASSISTANT:"
            image = Image.open(case["image_path"]).convert("RGB")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                device, dtype=torch.float16
            )
            image.close()
            vs, ve = find_vis_bounds(inputs["input_ids"][0], int(image_token_id))
            n_vis = ve - vs

            with torch.inference_mode():
                out = model(**inputs, output_attentions=True)
            attentions = out.attentions  # tuple over layers: (b, H, S, S)
            last = int(inputs["input_ids"].shape[1] - 1)
            scaling = getattr(llm_layers[band[0]].self_attn, "scaling", None)
            if scaling is None:
                hd = getattr(
                    llm_layers[band[0]].self_attn, "head_dim",
                    model.config.text_config.hidden_size // model.config.text_config.num_attention_heads,
                )
                scaling = hd ** -0.5

            collected = {name: np.zeros((len(band), n_vis), np.float32) for name in variants}
            key_norms = np.zeros((len(band), n_vis), np.float32)
            sink_hits = np.zeros(n_vis, np.float32)
            sink_positions_per_layer: dict[int, list[int]] = {}

            for j, li in enumerate(band):
                mean_attn = attentions[li][0].float().mean(dim=0)  # (S, S) mean over heads
                kn = hooks[li].key_norms                            # (S,)
                key_norms[j] = kn[vs:ve].cpu().numpy()

                all_pos = torch.arange(kn.size(0), device=kn.device)
                vis_pos = all_pos[(all_pos >= vs) & (all_pos < ve)]
                sink_pos = auto_detect_sinks(kn, vis_pos)
                sinks_local = [int(s - vs) for s in sink_pos.cpu().tolist()]
                sink_positions_per_layer[li] = sinks_local
                for s in sinks_local:
                    sink_hits[s] += 1.0

                pkw = dict(sink_pos=sink_pos, vis_start=vs, vis_end=ve, ratio=args.ratio)
                sink_only = purify_attention(mean_attn, **pkw, remove_sinks=True, apply_top_mass=False)
                topmass = purify_attention(mean_attn, **pkw, remove_sinks=False, apply_top_mass=True)
                purified = purify_attention(mean_attn, **pkw, remove_sinks=True, apply_top_mass=True)
                no_rope_row = _no_rope_last_row(hooks[li], last, float(scaling))  # (S,)
                clean_row = _drop_sinks_renorm(no_rope_row, sink_pos)             # no-RoPE + sink-free

                collected["original"][j] = mean_attn[last, vs:ve].cpu().numpy()
                collected["sink_only"][j] = sink_only[last, vs:ve].cpu().numpy()
                collected["topmass_only"][j] = topmass[last, vs:ve].cpu().numpy()
                collected["purified"][j] = purified[last, vs:ve].cpu().numpy()
                collected["no_rope"][j] = no_rope_row[vs:ve].cpu().numpy()
                collected["clean"][j] = clean_row[vs:ve].cpu().numpy()

            means = {name: collected[name].mean(axis=0) for name in variants}

            saved = np.load(results_dir / f"qualitative_case_{i}_attention.npz")["vanilla_attention"].astype(np.float32)
            max_abs = float(np.abs(means["original"] - saved).max())
            status = "OK" if max_abs <= args.repro_atol else "MISMATCH"
            print(
                f"[case {i}] reproduced original vs saved vanilla_attention: "
                f"max|Δ|={max_abs:.2e} ({status}); sink hits/token max="
                f"{int(sink_hits.max())}/{len(band)}; "
                f"key-norm(mean-band) range=[{key_norms.mean(0).min():.2f},{key_norms.mean(0).max():.2f}]"
            )
            if status == "MISMATCH":
                raise SystemExit(
                    f"case {i}: reproduced original attention does not match saved vanilla "
                    f"(max|Δ|={max_abs:.2e} > {args.repro_atol}); refusing to write."
                )

            out_path = results_dir / f"qualitative_case_{i}_sink_attention.npz"
            np.savez_compressed(
                out_path,
                band_layers=np.array(band, np.int32),
                vis_start=np.int32(vs),
                vis_end=np.int32(ve),
                ratio=np.float32(args.ratio),
                key_norms_by_layer=key_norms,
                key_norms_mean=key_norms.mean(axis=0).astype(np.float32),
                sink_hits=sink_hits,
                sink_positions_json=np.bytes_(json.dumps(sink_positions_per_layer).encode()),
                **{f"{name}_mean": means[name] for name in variants},
                **{f"{name}_by_layer": collected[name] for name in variants},
            )
            print(f"  wrote {out_path}")
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
