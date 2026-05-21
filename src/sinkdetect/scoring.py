"""
Hallucination detection score computation and AUROC evaluation.

Computes per-object shape scores from original and purified attention maps.
PAS-style attention-mass baselines are intentionally not emitted by the
current pipeline.
"""

from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .grounding import compute_grounding_scores_for_mention, compute_per_head_scores, compute_cross_image_scores


def compute_all_scores(
    mentions: list[dict],
    orig_attn_layers: list[torch.Tensor],
    sink_only_attn_layers: list[torch.Tensor],
    topmass_only_attn_layers: list[torch.Tensor],
    purified_attn_layers: list[torch.Tensor],
    sink_stats_layers: list[dict],
    token_masks: dict[str, torch.Tensor],
    prompt_end_idx: int,
    vis_start: int,
    vis_end: int,
    no_rope_attn_layers: Optional[list[torch.Tensor]] = None,
    no_rope_sink_only_attn_layers: Optional[list[torch.Tensor]] = None,
    no_rope_topmass_only_attn_layers: Optional[list[torch.Tensor]] = None,
    no_rope_purified_attn_layers: Optional[list[torch.Tensor]] = None,
    per_head_attn: Optional[dict[int, torch.Tensor]] = None,
    cross_image_nulls: Optional[list[list[torch.Tensor]]] = None,
) -> tuple[dict[str, list[Optional[float]]], list[int]]:
    """Compute all hallucination detection scores for a single sample.

    Args:
        mentions: List of first-mention dicts with {word, pos, hallucinated}.
                  pos is relative to the start of the generated caption.
        orig_attn_layers: List of (seq_len, seq_len) original attention
                          matrices, one per layer.
        sink_only_attn_layers: Same shape, sink-removal-only matrices.
        topmass_only_attn_layers: Same shape, top-mass-only matrices.
        purified_attn_layers: Same shape, purified attention matrices.
        no_rope_*_attn_layers: Optional pre-RoPE-Q/K attention branches and
            their sink/top-mass purification variants.
        sink_stats_layers: List of dicts with sink_count, sink_positions.
        token_masks: Dict with instruction_mask and output_mask bool tensors.
        prompt_end_idx: Absolute index of first generated token.
        per_head_attn: Optional mapping layer_idx → (n_heads, seq, seq) tensor,
                       for key layers only (PER_HEAD_LAYERS).
        cross_image_nulls: Optional list of per-layer null distributions from
                           other images. Each element is a list of (n_v,) tensors.
                           Used for cross-image CVG scores.

    Returns:
        Dict mapping score names to lists of values (one per mention).
        None values indicate the score could not be computed.
    """
    n_layers = len(orig_attn_layers)
    instruction_mask = token_masks["instruction_mask"]
    output_mask = token_masks["output_mask"]

    # Initialize score containers
    scores: dict[str, list[Optional[float]]] = {}
    labels: list[int] = []
    object_positions = set()
    for m in mentions:
        obj_query_pos = m["pos"] + prompt_end_idx
        object_positions.update({obj_query_pos, obj_query_pos + 1, obj_query_pos + 2})

    for mention in mentions:
        token_pos = mention["pos"] + prompt_end_idx  # absolute position
        local_null_positions = _nearest_nonobject_positions(
            token_pos=token_pos,
            output_mask=output_mask,
            object_positions=object_positions,
        )

        # ── Counterfactual Visual Grounding family ──────────────────────────
        # Adds: cvg_kl_instr_*, cvg_kl_uniform_*, cvg_jsd_instr_*,
        #       conc_entropy_*, conc_top{1,5,10}_mass_*, conc_max_over_mean_*,
        #       clc_gen_jsd, clc_mean_pairwise_jsd, clc_gen_jsd_midlate,
        #       clc_argmax_agree, plus their global_* layer-averaged versions.
        grounding_scores = compute_grounding_scores_for_mention(
            token_pos=token_pos,
            attn_layers=orig_attn_layers,
            sink_stats_layers=sink_stats_layers,
            instruction_mask=instruction_mask,
            vis_start=vis_start,
            vis_end=vis_end,
            local_null_positions=local_null_positions,
        )
        for k, v in grounding_scores.items():
            _append_score(scores, k, v)

        # Sink-removal-only ablation.
        sink_only_grounding_scores = compute_grounding_scores_for_mention(
            token_pos=token_pos,
            attn_layers=sink_only_attn_layers,
            sink_stats_layers=sink_stats_layers,
            instruction_mask=instruction_mask,
            vis_start=vis_start,
            vis_end=vis_end,
            score_prefix="sink_only_",
            local_null_positions=local_null_positions,
            strip_sink_tokens=True,
        )
        for k, v in sink_only_grounding_scores.items():
            _append_score(scores, k, v)

        # Top-mass-only ablation before sink removal.
        topmass_only_grounding_scores = compute_grounding_scores_for_mention(
            token_pos=token_pos,
            attn_layers=topmass_only_attn_layers,
            sink_stats_layers=sink_stats_layers,
            instruction_mask=instruction_mask,
            vis_start=vis_start,
            vis_end=vis_end,
            score_prefix="topmass_only_",
            local_null_positions=local_null_positions,
            strip_sink_tokens=False,
        )
        for k, v in topmass_only_grounding_scores.items():
            _append_score(scores, k, v)

        # Same shape-based grounding scores computed on purified attention.
        # This is sink-removal + top-mass purification; it is not a strict
        # no-RoPE attention branch.
        purified_grounding_scores = compute_grounding_scores_for_mention(
            token_pos=token_pos,
            attn_layers=purified_attn_layers,
            sink_stats_layers=sink_stats_layers,
            instruction_mask=instruction_mask,
            vis_start=vis_start,
            vis_end=vis_end,
            score_prefix="purified_",
            local_null_positions=local_null_positions,
        )
        for k, v in purified_grounding_scores.items():
            _append_score(scores, k, v)

        if no_rope_attn_layers is not None:
            no_rope_grounding_scores = compute_grounding_scores_for_mention(
                token_pos=token_pos,
                attn_layers=no_rope_attn_layers,
                sink_stats_layers=sink_stats_layers,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
                score_prefix="no_rope_",
                local_null_positions=local_null_positions,
            )
            for k, v in no_rope_grounding_scores.items():
                _append_score(scores, k, v)

        if no_rope_sink_only_attn_layers is not None:
            no_rope_sink_only_scores = compute_grounding_scores_for_mention(
                token_pos=token_pos,
                attn_layers=no_rope_sink_only_attn_layers,
                sink_stats_layers=sink_stats_layers,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
                score_prefix="no_rope_sink_only_",
                local_null_positions=local_null_positions,
                strip_sink_tokens=True,
            )
            for k, v in no_rope_sink_only_scores.items():
                _append_score(scores, k, v)

        if no_rope_topmass_only_attn_layers is not None:
            no_rope_topmass_only_scores = compute_grounding_scores_for_mention(
                token_pos=token_pos,
                attn_layers=no_rope_topmass_only_attn_layers,
                sink_stats_layers=sink_stats_layers,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
                score_prefix="no_rope_topmass_only_",
                local_null_positions=local_null_positions,
                strip_sink_tokens=False,
            )
            for k, v in no_rope_topmass_only_scores.items():
                _append_score(scores, k, v)

        if no_rope_purified_attn_layers is not None:
            no_rope_purified_scores = compute_grounding_scores_for_mention(
                token_pos=token_pos,
                attn_layers=no_rope_purified_attn_layers,
                sink_stats_layers=sink_stats_layers,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
                score_prefix="no_rope_purified_",
                local_null_positions=local_null_positions,
            )
            for k, v in no_rope_purified_scores.items():
                _append_score(scores, k, v)

        # ── Per-head scores (if available) ────────────────────────────────
        if per_head_attn:
            # Build sink_stats dict keyed by layer_idx for per-head function
            ph_sink_stats = {
                l: sink_stats_layers[l] for l in per_head_attn.keys()
                if l < len(sink_stats_layers)
            }
            ph_scores = compute_per_head_scores(
                token_pos=token_pos,
                per_head_attn=per_head_attn,
                sink_stats_layers=ph_sink_stats,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
            )
            for k, v in ph_scores.items():
                _append_score(scores, k, v)

        # ── Cross-image CVG scores (if available) ─────────────────────
        if cross_image_nulls:
            ci_scores = compute_cross_image_scores(
                token_pos=token_pos,
                orig_attn_layers=orig_attn_layers,
                sink_stats_layers=sink_stats_layers,
                instruction_mask=instruction_mask,
                vis_start=vis_start,
                vis_end=vis_end,
                cross_image_nulls=cross_image_nulls,
            )
            for k, v in ci_scores.items():
                _append_score(scores, k, v)

        # Label
        labels.append(1 if mention["hallucinated"] else 0)

    return scores, labels


def _append_score(scores: dict, key: str, value):
    """Append a value to a score list, creating the list if needed."""
    if key not in scores:
        scores[key] = []
    scores[key].append(value)


def _nearest_nonobject_positions(
    token_pos: int,
    output_mask: torch.Tensor,
    object_positions: set[int],
    max_tokens: int = 8,
    window: int = 16,
) -> list[int]:
    """Pick nearby generated non-object query positions for a local null.

    The instruction null is the clean primary null. This local null is an
    ablation: it asks whether the object query looks like the surrounding
    caption language rather than like an object-specific visual query.
    """
    candidates = torch.nonzero(output_mask, as_tuple=False).flatten().tolist()
    candidates = [
        int(pos)
        for pos in candidates
        if pos != token_pos
        and pos not in object_positions
        and abs(int(pos) - token_pos) <= window
    ]
    candidates.sort(key=lambda pos: (abs(pos - token_pos), pos))
    return candidates[:max_tokens]


def compute_auroc(
    scores: dict[str, list],
    labels: list[int],
) -> dict[str, float]:
    """Compute AUROC for each score type.

    Returns:
        Dict mapping score names to AUROC values, sorted descending.
        Only includes scores with valid AUROC (≥2 classes present).
    """
    roc_auc = {}
    labels_arr = np.array(labels)

    for key, values in scores.items():
        values_arr = np.array(values, dtype=float)
        if values_arr.shape[0] != labels_arr.shape[0]:
            n = min(values_arr.shape[0], labels_arr.shape[0])
            values_arr = values_arr[:n]
            labels_for_key = labels_arr[:n]
        else:
            labels_for_key = labels_arr
        # Filter out None/NaN
        valid_mask = np.isfinite(values_arr)
        if not valid_mask.any():
            continue
        valid_labels = labels_for_key[valid_mask]
        valid_values = values_arr[valid_mask]
        if len(set(valid_labels)) < 2:
            continue
        try:
            auc = roc_auc_score(valid_labels, valid_values)
            roc_auc[key] = round(float(auc), 4)
        except ValueError:
            continue

    # Sort by AUROC descending
    roc_auc = dict(sorted(roc_auc.items(), key=lambda x: x[1], reverse=True))
    return roc_auc
