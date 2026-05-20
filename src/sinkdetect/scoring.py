"""
Hallucination detection score computation and AUROC evaluation.

Computes per-object detection scores from both original and purified
attention maps, plus sink statistics and attention shift signals.
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
    per_head_attn: Optional[dict[int, torch.Tensor]] = None,
    cross_image_nulls: Optional[list[list[torch.Tensor]]] = None,
) -> tuple[dict[str, list[Optional[float]]], list[int]]:
    """Compute all hallucination detection scores for a single sample.

    Args:
        mentions: List of first-mention dicts with {word, pos, hallucinated}.
                  pos is relative to the start of the generated caption.
        orig_attn_layers: List of (seq_len, seq_len) original attention
                          matrices, one per layer, on CPU.
        sink_only_attn_layers: Same shape, sink-removal-only matrices.
        topmass_only_attn_layers: Same shape, top-mass-only matrices.
        purified_attn_layers: Same shape, purified attention matrices.
        sink_stats_layers: List of dicts with sink_count, sink_positions.
        token_masks: Dict with bos_mask, image_mask, instruction_mask,
                     output_mask — CPU bool tensors.
        prompt_end_idx: Absolute index of first generated token.
        per_head_attn: Optional mapping layer_idx → (n_heads, seq, seq) CPU
                       float, for key layers only (PER_HEAD_LAYERS).
        cross_image_nulls: Optional list of per-layer null distributions from
                           other images. Each element is a list of (n_v,) tensors.
                           Used for cross-image CVG scores.

    Returns:
        Dict mapping score names to lists of values (one per mention).
        None values indicate the score could not be computed.
    """
    n_layers = len(orig_attn_layers)
    bos_mask = token_masks["bos_mask"]
    image_mask = token_masks["image_mask"]
    instruction_mask = token_masks["instruction_mask"]
    output_mask = token_masks["output_mask"]

    # Pre-compute global (mean across layers) attention matrices
    orig_attn_stack = torch.stack(orig_attn_layers)  # (n_layers, seq_len, seq_len)
    sink_only_stack = torch.stack(sink_only_attn_layers)
    topmass_only_stack = torch.stack(topmass_only_attn_layers)
    purified_attn_stack = torch.stack(purified_attn_layers)

    global_orig = orig_attn_stack.mean(dim=0)  # (seq_len, seq_len)
    global_sink_only = sink_only_stack.mean(dim=0)
    global_topmass_only = topmass_only_stack.mean(dim=0)
    global_purified = purified_attn_stack.mean(dim=0)

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

        # ── Per-layer scores ──
        for layer_idx in range(n_layers):
            orig = orig_attn_layers[layer_idx]
            sink_only = sink_only_attn_layers[layer_idx]
            topmass_only = topmass_only_attn_layers[layer_idx]
            purified = purified_attn_layers[layer_idx]

            # Original attention scores (same as PAS)
            _append_score(
                scores, f"orig_prelim_attn_layer_{layer_idx}",
                orig[token_pos, prompt_end_idx:].sum().item(),
            )
            _append_score(
                scores, f"orig_image_attn_layer_{layer_idx}",
                -orig[token_pos, image_mask].sum().item(),  # negated
            )
            _append_score(
                scores, f"orig_bos_attn_layer_{layer_idx}",
                -orig[token_pos, bos_mask].sum().item(),  # negated
            )

            # Sink-removal-only attention scores
            _append_score(
                scores, f"sink_only_prelim_attn_layer_{layer_idx}",
                sink_only[token_pos, prompt_end_idx:].sum().item(),
            )
            _append_score(
                scores, f"sink_only_image_attn_layer_{layer_idx}",
                -sink_only[token_pos, image_mask].sum().item(),
            )
            _append_score(
                scores, f"sink_only_bos_attn_layer_{layer_idx}",
                -sink_only[token_pos, bos_mask].sum().item(),
            )

            # Top-mass-only attention scores
            _append_score(
                scores, f"topmass_only_prelim_attn_layer_{layer_idx}",
                topmass_only[token_pos, prompt_end_idx:].sum().item(),
            )
            _append_score(
                scores, f"topmass_only_image_attn_layer_{layer_idx}",
                -topmass_only[token_pos, image_mask].sum().item(),
            )
            _append_score(
                scores, f"topmass_only_bos_attn_layer_{layer_idx}",
                -topmass_only[token_pos, bos_mask].sum().item(),
            )

            # Purified attention scores
            _append_score(
                scores, f"purified_prelim_attn_layer_{layer_idx}",
                purified[token_pos, prompt_end_idx:].sum().item(),
            )
            _append_score(
                scores, f"purified_image_attn_layer_{layer_idx}",
                -purified[token_pos, image_mask].sum().item(),  # negated
            )
            _append_score(
                scores, f"purified_bos_attn_layer_{layer_idx}",
                -purified[token_pos, bos_mask].sum().item(),  # negated
            )

            # Sink attention mass (negated: more sink attn ⇒ less visual grounding ⇒ hallucination)
            sink_pos = sink_stats_layers[layer_idx].get("sink_positions", [])
            if sink_pos:
                sink_pos_t = torch.tensor(sink_pos, dtype=torch.long, device=orig.device)
                _append_score(
                    scores, f"sink_attn_mass_layer_{layer_idx}",
                    -orig[token_pos, sink_pos_t].sum().item(),
                )
            else:
                _append_score(scores, f"sink_attn_mass_layer_{layer_idx}", 0.0)

            # Sink count
            _append_score(
                scores, f"sink_count_layer_{layer_idx}",
                float(sink_stats_layers[layer_idx]["sink_count"]),
            )

            # Attention shift scores
            orig_img_raw = orig[token_pos, image_mask].sum().item()
            purified_img_raw = purified[token_pos, image_mask].sum().item()
            orig_prelim_raw = orig[token_pos, prompt_end_idx:].sum().item()
            purified_prelim_raw = purified[token_pos, prompt_end_idx:].sum().item()

            _append_score(
                scores, f"attn_shift_visual_layer_{layer_idx}",
                -(purified_img_raw - orig_img_raw),  # negated: positive shift = non-hallu
            )
            _append_score(
                scores, f"attn_shift_prelim_layer_{layer_idx}",
                purified_prelim_raw - orig_prelim_raw,
            )

        # ── Global scores (mean across layers) ──
        _append_score(
            scores, "global_orig_prelim_attn",
            global_orig[token_pos, prompt_end_idx:].sum().item(),
        )
        _append_score(
            scores, "global_orig_image_attn",
            -global_orig[token_pos, image_mask].sum().item(),
        )
        _append_score(
            scores, "global_orig_bos_attn",
            -global_orig[token_pos, bos_mask].sum().item(),
        )

        _append_score(
            scores, "global_purified_prelim_attn",
            global_purified[token_pos, prompt_end_idx:].sum().item(),
        )
        _append_score(
            scores, "global_sink_only_prelim_attn",
            global_sink_only[token_pos, prompt_end_idx:].sum().item(),
        )
        _append_score(
            scores, "global_sink_only_image_attn",
            -global_sink_only[token_pos, image_mask].sum().item(),
        )
        _append_score(
            scores, "global_sink_only_bos_attn",
            -global_sink_only[token_pos, bos_mask].sum().item(),
        )
        _append_score(
            scores, "global_topmass_only_prelim_attn",
            global_topmass_only[token_pos, prompt_end_idx:].sum().item(),
        )
        _append_score(
            scores, "global_topmass_only_image_attn",
            -global_topmass_only[token_pos, image_mask].sum().item(),
        )
        _append_score(
            scores, "global_topmass_only_bos_attn",
            -global_topmass_only[token_pos, bos_mask].sum().item(),
        )
        _append_score(
            scores, "global_purified_image_attn",
            -global_purified[token_pos, image_mask].sum().item(),
        )
        _append_score(
            scores, "global_purified_bos_attn",
            -global_purified[token_pos, bos_mask].sum().item(),
        )

        # Global sink mass (negated: more sink attn ⇒ less visual grounding ⇒ hallucination)
        total_sink_mass = 0.0
        for layer_idx in range(n_layers):
            sink_pos = sink_stats_layers[layer_idx].get("sink_positions", [])
            if sink_pos:
                sink_pos_t = torch.tensor(
                    sink_pos,
                    dtype=torch.long,
                    device=orig_attn_layers[layer_idx].device,
                )
                total_sink_mass += orig_attn_layers[layer_idx][token_pos, sink_pos_t].sum().item()
        _append_score(scores, "global_sink_attn_mass", -(total_sink_mass / n_layers))

        # Global sink count
        total_sink_count = sum(s["sink_count"] for s in sink_stats_layers)
        _append_score(scores, "global_sink_count", float(total_sink_count))

        # Global attention shift
        g_orig_img = global_orig[token_pos, image_mask].sum().item()
        g_pur_img = global_purified[token_pos, image_mask].sum().item()
        g_orig_prelim = global_orig[token_pos, prompt_end_idx:].sum().item()
        g_pur_prelim = global_purified[token_pos, prompt_end_idx:].sum().item()

        _append_score(scores, "global_attn_shift_visual", -(g_pur_img - g_orig_img))
        _append_score(scores, "global_attn_shift_prelim", g_pur_prelim - g_orig_prelim)

        # ── NEW: Counterfactual Visual Grounding family ─────────────────────
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
        # These are the main de-biased variants: sink/RoPE-like visual bias is
        # removed before comparing the object query to null distributions.
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
