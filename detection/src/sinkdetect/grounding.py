"""
Counterfactual Visual Grounding (CVG) and related distribution-shape scores.

The classical PAS family summarises text→visual attention with a single scalar
(e.g. ``A[q, V].sum()``). That ignores the *shape* of the attention over the
576 visual tokens. Real visual grounding should:

  (a) concentrate mass on a small set of content-relevant visual tokens,
  (b) be different from the "what a generic prompt would look at" baseline,
  (c) agree across layers.

We compute three families of scores per object mention:

  cvg_*   — negative KL/JSD between the object-query attention over visual
            tokens and a content-independent "null" estimated from the
            instruction tokens of the same image (a free, in-matrix null).
  conc_*  — direct concentration metrics on the sink-removed visual
            distribution (entropy, top-k mass, max).
  clc_*   — cross-layer consistency: generalised JSD of the sink-removed
            visual distributions across all instrumented layers.

All scores are sign-aligned so that larger value ⇒ more likely hallucination.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch

_EPS = 1e-10


# ─── Distribution primitives ─────────────────────────────────────────────────

def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    """Row-normalize a non-negative vector to a probability distribution."""
    s = x.sum(dim=-1, keepdim=True).clamp(min=_EPS)
    return x / s


def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a probability vector (nats)."""
    return -(p * (p.clamp(min=_EPS)).log()).sum(dim=-1)


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) in nats. p, q must be valid distributions of the same shape."""
    p = p.clamp(min=_EPS)
    q = q.clamp(min=_EPS)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence (nats)."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _generalised_jsd(P: torch.Tensor) -> torch.Tensor:
    """Generalised JSD across rows of P (n, k). Equals H(mean) − mean(H(rows))."""
    m = P.mean(dim=0)
    return _entropy(m) - _entropy(P).mean()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sinks_within_visual(
    sink_positions: Sequence[int],
    vis_start: int,
    vis_end: int,
) -> list[int]:
    """Return sink positions expressed as offsets within the [vis_start, vis_end)
    range. Drops any sink that lies outside the visual span (shouldn't happen)."""
    return [p - vis_start for p in sink_positions if vis_start <= p < vis_end]


def _strip_sinks(vec: torch.Tensor, sink_rel: list[int]) -> torch.Tensor:
    """Zero out the given indices in a 1-D non-negative vector (does NOT renormalize)."""
    if not sink_rel:
        return vec
    out = vec.clone()
    out[sink_rel] = 0.0
    return out


# ─── Public: per-mention, per-layer score computation ────────────────────────

def compute_grounding_scores_for_mention(
    token_pos: int,
    attn_layers: Sequence[torch.Tensor],
    sink_stats_layers: Sequence[dict],
    instruction_mask: torch.Tensor,
    vis_start: int,
    vis_end: int,
    score_prefix: str = "",
    local_null_positions: Sequence[int] | None = None,
    strip_sink_tokens: bool = True,
) -> dict[str, float]:
    """Compute CVG / concentration / CLC scores for a single object mention.

    Args:
        token_pos:           Absolute query position (= object's preceding token).
        attn_layers:         List[T (seq, seq)] one per layer.
        sink_stats_layers:   List[dict] with 'sink_positions' (absolute indices).
        instruction_mask:    1-D bool tensor over the full sequence; True at
                             instruction tokens (system + user prompt + ASSISTANT:).
        vis_start, vis_end:  Visual-token span (dynamic per sample).
        score_prefix:        Optional prefix for every emitted key, e.g.
                             "purified_" for scores computed on purified attention.
        local_null_positions: Optional absolute positions of nearby generated
                             non-object tokens used as a local language-context null.
        strip_sink_tokens:   If False, leave detected sink visual positions in
                             the score distributions. Used for the no-sink-removal
                             ablation.

    Returns:
        Dict[score_name -> float]. Sign convention: higher ⇒ hallucination.
    """
    n_layers = len(attn_layers)
    n_v = vis_end - vis_start
    out: dict[str, float] = {}

    # Per-layer sink-removed visual distributions for the object query.
    # Stored to feed the cross-layer CLC computation.
    obj_clean_layers: list[torch.Tensor] = []

    # ── Per-layer scores ─────────────────────────────────────────────────────
    local_null_idx = None
    if local_null_positions:
        local_null_idx = torch.tensor(
            list(local_null_positions),
            dtype=torch.long,
            device=attn_layers[0].device,
        )

    for l, A in enumerate(attn_layers):
        sink_rel = _sinks_within_visual(
            sink_stats_layers[l].get("sink_positions", []),
            vis_start, vis_end,
        )
        if not strip_sink_tokens:
            sink_rel = []

        # (1) Object-query attention restricted to visual span, sink-stripped, renormalized.
        a_obj = A[token_pos, vis_start:vis_end].float()
        a_obj_clean = _safe_normalize(_strip_sinks(a_obj, sink_rel))
        obj_clean_layers.append(a_obj_clean)

        # (2) Instruction-token null: average of attention rows at instruction
        # positions, sink-stripped, renormalized. This is the "generic prompt
        # gaze over this specific image" — free of content from the object word.
        instr_rows = A[instruction_mask, vis_start:vis_end].float()
        if instr_rows.shape[0] == 0:
            a_null = torch.full((n_v,), 1.0 / n_v, device=A.device)
        else:
            a_null = instr_rows.mean(dim=0)
        a_null_clean = _safe_normalize(_strip_sinks(a_null, sink_rel))

        # (3) Local generated-token null: nearby non-object tokens in the
        # caption. This is a deliberately less-clean ablation than the fixed
        # instruction null because it carries local syntax and caption context.
        if local_null_idx is not None and local_null_idx.numel() > 0:
            local_rows = A[local_null_idx, vis_start:vis_end].float()
            a_local_null = local_rows.mean(dim=0)
            a_local_null_clean = _safe_normalize(_strip_sinks(a_local_null, sink_rel))
        else:
            a_local_null_clean = None

        # (4) Uniform-over-non-sink null (purely architectural baseline).
        u = torch.ones(n_v, device=A.device)
        if sink_rel:
            u[sink_rel] = 0.0
        a_unif = _safe_normalize(u)

        # ── CVG scores: closeness to null ────────────────────────────────────
        # The instruction-token null is a language/prompt-prior visual gaze.
        # If the object query stays close to that prior, it has not developed
        # object-specific grounding. We therefore NEGATE the divergences so
        # high score = close to null = more likely hallucination.
        kl_instr = _kl(a_obj_clean, a_null_clean).item()
        kl_unif = _kl(a_obj_clean, a_unif).item()
        jsd_instr = _jsd(a_obj_clean, a_null_clean).item()

        out[f"{score_prefix}cvg_kl_instr_layer_{l}"] = -kl_instr
        out[f"{score_prefix}cvg_kl_uniform_layer_{l}"] = -kl_unif
        out[f"{score_prefix}cvg_jsd_instr_layer_{l}"] = -jsd_instr
        if a_local_null_clean is not None:
            out[f"{score_prefix}cvg_kl_local_nonobj_layer_{l}"] = -_kl(
                a_obj_clean, a_local_null_clean
            ).item()
            out[f"{score_prefix}cvg_jsd_local_nonobj_layer_{l}"] = -_jsd(
                a_obj_clean, a_local_null_clean
            ).item()

        # ── Concentration scores ─────────────────────────────────────────────
        # Higher entropy = more diffuse = hallu, so NO negation.
        ent = _entropy(a_obj_clean).item()
        out[f"{score_prefix}conc_entropy_layer_{l}"] = ent

        # Top-k mass: higher mass on top-k = peaked = grounded ⇒ NEGATE.
        sorted_vals, _ = a_obj_clean.sort(descending=True)
        out[f"{score_prefix}conc_top1_mass_layer_{l}"] = -sorted_vals[0].item()
        out[f"{score_prefix}conc_top5_mass_layer_{l}"] = -sorted_vals[:5].sum().item()
        out[f"{score_prefix}conc_top10_mass_layer_{l}"] = -sorted_vals[:10].sum().item()

        # Max-to-mean ratio: a unitless peakedness signal.
        out[f"{score_prefix}conc_max_over_mean_layer_{l}"] = -(
            a_obj_clean.max().item() / (a_obj_clean.mean().item() + _EPS)
        )

    # ── Global (cross-layer-averaged matrix) versions of CVG ─────────────────
    # We average the per-layer obj_clean and null distributions then re-score.
    # This is the "what does the model on average look at after deconfounding"
    # signal. (Different from averaging per-layer scores.)
    obj_avg = torch.stack(obj_clean_layers).mean(dim=0)
    obj_avg = _safe_normalize(obj_avg)

    # Build a layer-averaged null distribution.
    null_layers = []
    local_null_layers = []
    for l, A in enumerate(attn_layers):
        sink_rel = _sinks_within_visual(
            sink_stats_layers[l].get("sink_positions", []),
            vis_start, vis_end,
        )
        if not strip_sink_tokens:
            sink_rel = []
        instr_rows = A[instruction_mask, vis_start:vis_end].float()
        a_null = (instr_rows.mean(dim=0) if instr_rows.shape[0] > 0
                  else torch.full((n_v,), 1.0 / n_v, device=A.device))
        a_null_clean = _safe_normalize(_strip_sinks(a_null, sink_rel))
        null_layers.append(a_null_clean)

        if local_null_idx is not None and local_null_idx.numel() > 0:
            local_rows = A[local_null_idx, vis_start:vis_end].float()
            a_local_null = local_rows.mean(dim=0)
            local_null_layers.append(_safe_normalize(_strip_sinks(a_local_null, sink_rel)))
    null_avg = _safe_normalize(torch.stack(null_layers).mean(dim=0))

    out[f"{score_prefix}global_cvg_kl_instr"] = -_kl(obj_avg, null_avg).item()
    out[f"{score_prefix}global_cvg_jsd_instr"] = -_jsd(obj_avg, null_avg).item()
    if local_null_layers:
        local_null_avg = _safe_normalize(torch.stack(local_null_layers).mean(dim=0))
        out[f"{score_prefix}global_cvg_kl_local_nonobj"] = -_kl(
            obj_avg, local_null_avg
        ).item()
        out[f"{score_prefix}global_cvg_jsd_local_nonobj"] = -_jsd(
            obj_avg, local_null_avg
        ).item()

    out[f"{score_prefix}global_conc_entropy"] = _entropy(obj_avg).item()
    sorted_vals, _ = obj_avg.sort(descending=True)
    out[f"{score_prefix}global_conc_top5_mass"] = -sorted_vals[:5].sum().item()
    out[f"{score_prefix}global_conc_top10_mass"] = -sorted_vals[:10].sum().item()
    out[f"{score_prefix}global_conc_max_over_mean"] = -(
        obj_avg.max().item() / (obj_avg.mean().item() + _EPS)
    )

    # ── Cross-Layer Consistency (CLC) ─────────────────────────────────────────
    # Stack per-layer obj_clean → (L, |V|). Generalised JSD = H(mean) − mean(H).
    # Low value = layers agree = grounded. Higher = hallu (NO negation).
    P = torch.stack(obj_clean_layers)  # (L, |V|)
    out[f"{score_prefix}clc_gen_jsd"] = _generalised_jsd(P).item()
    out[f"{score_prefix}clc_mean_pairwise_jsd"] = _mean_pairwise_jsd(P).item()

    # CLC restricted to "middle-late" layers (14..28 for LLaVA-7B which has 32),
    # where grounding tends to be cleanest in the literature. Robust to off-by-one.
    lo, hi = max(0, n_layers // 2 - 2), min(n_layers, n_layers - 4)
    if hi - lo >= 2:
        out[f"{score_prefix}clc_gen_jsd_midlate"] = _generalised_jsd(P[lo:hi]).item()

    # ── Argmax consistency (a top-1 agreement signal) ────────────────────────
    # Fraction of layer-pairs that share the same argmax visual position.
    argmaxes = P.argmax(dim=-1)  # (L,)
    if argmaxes.numel() > 1:
        agree = (argmaxes.unsqueeze(0) == argmaxes.unsqueeze(1)).float()
        agree = agree.triu(diagonal=1).sum() / (argmaxes.numel() * (argmaxes.numel() - 1) / 2)
        # high agreement = grounded ⇒ NEGATE
        out[f"{score_prefix}clc_argmax_agree"] = -agree.item()

    return out


def _mean_pairwise_jsd(P: torch.Tensor) -> torch.Tensor:
    """Vectorised mean pairwise JSD over rows of P (n, k)."""
    n = P.shape[0]
    if n < 2:
        return torch.zeros(())
    # Compute pairwise JSD without explicit loop.
    Pi = P.unsqueeze(1)  # (n, 1, k)
    Pj = P.unsqueeze(0)  # (1, n, k)
    M = 0.5 * (Pi + Pj)
    # KL(Pi || M) + KL(Pj || M) then 0.5 average
    log_M = M.clamp(min=_EPS).log()
    kl_iM = (Pi.clamp(min=_EPS) * (Pi.clamp(min=_EPS).log() - log_M)).sum(dim=-1)
    kl_jM = (Pj.clamp(min=_EPS) * (Pj.clamp(min=_EPS).log() - log_M)).sum(dim=-1)
    jsd_mat = 0.5 * (kl_iM + kl_jM)  # (n, n)
    # Mean of upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=P.device), diagonal=1)
    return jsd_mat[mask].mean()


def compute_per_head_scores(
    token_pos: int,
    per_head_attn: dict[int, torch.Tensor],
    sink_stats_layers: dict[int, dict],
    instruction_mask: torch.Tensor,
    vis_start: int,
    vis_end: int,
) -> dict[str, float]:
    """Compute per-head analysis scores for key layers.

    Instead of averaging over heads, we compute per-head visual attention
    mass and then derive inter-head statistics (variance, agreement).

    Args:
        token_pos:        Absolute query position.
        per_head_attn:    Mapping layer_idx → (n_heads, seq, seq) tensor.
        sink_stats_layers: Mapping layer_idx → dict with 'sink_positions'.
        instruction_mask: 1-D bool tensor over the full sequence.
        vis_start, vis_end: Visual-token span.

    Returns:
        Dict[score_name -> float]. Sign convention: higher ⇒ hallucination.
    """
    out: dict[str, float] = {}

    for layer_idx, attn_heads in per_head_attn.items():
        n_heads = attn_heads.shape[0]
        sink_rel = _sinks_within_visual(
            sink_stats_layers.get(layer_idx, {}).get("sink_positions", []),
            vis_start, vis_end,
        )
        n_v = vis_end - vis_start

        # Per-head visual attention mass (sink-stripped, renormalized)
        head_vis_mass = []
        head_conc = []
        head_argmax = []
        for h in range(n_heads):
            a_h = attn_heads[h, token_pos, vis_start:vis_end].float()
            a_h_clean = _strip_sinks(a_h, sink_rel)
            head_vis_mass.append(a_h_clean.sum().item())
            a_h_norm = _safe_normalize(a_h_clean)
            head_conc.append(_entropy(a_h_norm).item())
            head_argmax.append(a_h_clean.argmax().item())

        head_vis_mass_t = torch.tensor(head_vis_mass, device=attn_heads.device)
        head_conc_t = torch.tensor(head_conc, device=attn_heads.device)

        out[f"ph_vis_mass_std_layer_{layer_idx}"] = head_vis_mass_t.std().item()
        out[f"ph_vis_mass_mean_layer_{layer_idx}"] = -head_vis_mass_t.mean().item()

        mode_count = Counter(head_argmax).most_common(1)[0][1]
        out[f"ph_argmax_agreement_layer_{layer_idx}"] = -(mode_count / n_heads)

        # ── Per-head concentration spread ─────────────────────────────────
        # High std of per-head entropy = heads disagree on peakedness.
        out[f"ph_entropy_std_layer_{layer_idx}"] = head_conc_t.std().item()

        # ── Top-1 head visual mass (the head with most visual attention) ──
        # If even the best head has low visual mass, hallucination is likely.
        out[f"ph_vis_mass_max_layer_{layer_idx}"] = -head_vis_mass_t.max().item()

        # ── Bottom-1 head visual mass ─────────────────────────────────────
        # The worst head: high value (after negation) = even the worst head is grounded.
        out[f"ph_vis_mass_min_layer_{layer_idx}"] = -head_vis_mass_t.min().item()

    return out


def compute_cross_image_scores(
    token_pos: int,
    orig_attn_layers: Sequence[torch.Tensor],
    sink_stats_layers: Sequence[dict],
    instruction_mask: torch.Tensor,
    vis_start: int,
    vis_end: int,
    cross_image_nulls: Sequence[Sequence[torch.Tensor]],
) -> dict[str, float]:
    """Compute CVG scores using cross-image null distributions.

    The instruction null from other images provides an image-independent
    "what does the model typically look at" baseline. If the current object
    query's visual attention is far from cross-image nulls, it means the
    query has image-specific grounding (not hallucination).

    Args:
        token_pos:        Absolute query position.
        orig_attn_layers: List[T (seq, seq)] one per layer.
        sink_stats_layers: List[dict] with 'sink_positions'.
        instruction_mask: 1-D bool tensor over the full sequence.
        vis_start, vis_end: Visual-token span.
        cross_image_nulls: List of per-layer null distributions from other
                           images. Each element is a list of (n_v,) tensors,
                           one per layer. Must have same n_v as current image.

    Returns:
        Dict[score_name -> float]. Sign convention: higher ⇒ hallucination.
    """
    n_layers = len(orig_attn_layers)
    n_v = vis_end - vis_start
    out: dict[str, float] = {}

    if not cross_image_nulls:
        return out

    for l, A in enumerate(orig_attn_layers):
        sink_rel = _sinks_within_visual(
            sink_stats_layers[l].get("sink_positions", []),
            vis_start, vis_end,
        )

        # Current image's object query distribution
        a_obj = A[token_pos, vis_start:vis_end].float()
        a_obj_clean = _safe_normalize(_strip_sinks(a_obj, sink_rel))

        # Collect cross-image nulls for this layer
        cross_nulls_this_layer = []
        for null_layers in cross_image_nulls:
            if l < len(null_layers):
                null_l = null_layers[l]
                if null_l.shape[0] == n_v:
                    cross_nulls_this_layer.append(null_l)

        if not cross_nulls_this_layer:
            continue

        # Average cross-image null
        cross_null_avg = _safe_normalize(
            torch.stack(cross_nulls_this_layer).mean(dim=0)
        )

        # KL/JSD from cross-image null: LARGE distance ⇒ object query is
        # image-specific (grounded) ⇒ NOT hallucination ⇒ NEGATE.
        kl_cross = _kl(a_obj_clean, cross_null_avg).item()
        jsd_cross = _jsd(a_obj_clean, cross_null_avg).item()

        out[f"cvg_kl_crossimg_layer_{l}"] = -kl_cross
        out[f"cvg_jsd_crossimg_layer_{l}"] = -jsd_cross

    # Global cross-image score
    obj_clean_layers = []
    for l, A in enumerate(orig_attn_layers):
        sink_rel = _sinks_within_visual(
            sink_stats_layers[l].get("sink_positions", []),
            vis_start, vis_end,
        )
        a_obj = A[token_pos, vis_start:vis_end].float()
        a_obj_clean = _safe_normalize(_strip_sinks(a_obj, sink_rel))
        obj_clean_layers.append(a_obj_clean)

    obj_avg = _safe_normalize(torch.stack(obj_clean_layers).mean(dim=0))

    # Collect cross-image nulls and average across layers + images
    cross_nulls_global = []
    for null_layers in cross_image_nulls:
        null_stack = []
        for l in range(min(n_layers, len(null_layers))):
            if null_layers[l].shape[0] == n_v:
                null_stack.append(null_layers[l])
        if null_stack:
            cross_nulls_global.append(_safe_normalize(torch.stack(null_stack).mean(dim=0)))

    if cross_nulls_global:
        cross_global_avg = _safe_normalize(torch.stack(cross_nulls_global).mean(dim=0))
        out["global_cvg_kl_crossimg"] = -_kl(obj_avg, cross_global_avg).item()
        out["global_cvg_jsd_crossimg"] = -_jsd(obj_avg, cross_global_avg).item()

    return out
