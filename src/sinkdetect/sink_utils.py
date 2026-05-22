"""
Sink detection and attention purification utilities.

Ported from LogitsAnalysis/sink_dynamic_method/sink_dynamic_adapter.py,
adapted for single-branch hallucination detection.

The visual-token span is NOT fixed — it depends on the prompt template
(e.g. the LLaVA-1.5 system prompt sits before <image>). Use
``find_vis_bounds`` at runtime and set the resulting bounds on each
DetectionAdapter before the forward pass.
"""

import torch

# Default fallback bounds (BOS + 576 image tokens), only used if no system
# prompt is in front of <image>. Real bounds are set dynamically in detect.py.
VIS_START = 1
VIS_END = 577
TEXT_START = 577


def find_vis_bounds(input_ids: torch.Tensor, image_token_id: int) -> tuple[int, int]:
    """Return [start, end) of the contiguous run of ``image_token_id`` in input_ids.

    Args:
        input_ids: 1-D tensor of token IDs.
        image_token_id: The token id used for image placeholder slots (e.g. 32000
                        for LLaVA-1.5).
    """
    mask = input_ids == image_token_id
    if not mask.any():
        raise ValueError(
            f"No image tokens (id={image_token_id}) found in input_ids."
        )
    positions = mask.nonzero(as_tuple=True)[0]
    return int(positions[0].item()), int(positions[-1].item()) + 1


def auto_detect_sinks(
    key_norms: torch.Tensor,
    vis_pos: torch.Tensor,
    min_sinks: int = 3,
    max_sinks: int = 30,
) -> torch.Tensor:
    """Return sink token positions detected via statistical threshold on key L2 norm.

    Sink tokens have abnormally low key norms.  We threshold at mean - 1.5*std
    (statistical outliers on the low end), then clamp to [min_sinks, max_sinks].
    """
    vis_norms = key_norms[vis_pos]
    mean_norm = vis_norms.mean()
    std_norm = vis_norms.std()

    threshold = mean_norm - 1.5 * std_norm
    sink_mask = vis_norms < threshold
    sink_count = int(sink_mask.sum().item())

    if sink_count < min_sinks:
        sink_pos = vis_pos[torch.argsort(vis_norms)[:min_sinks]]
    elif sink_count > max_sinks:
        sink_pos = vis_pos[torch.argsort(vis_norms)[:max_sinks]]
    else:
        sink_pos = vis_pos[sink_mask]
    return sink_pos


def top_mass_mask_visual(
    attn_weights: torch.Tensor,
    ratio: float = 0.5,
    vis_start: int = VIS_START,
    vis_end: int = VIS_END,
) -> torch.Tensor:
    """Compute a top-mass mask over visual tokens.

    Keeps the smallest set of visual-token positions whose cumulative
    attention mass >= ``ratio`` of the total visual attention mass.
    Non-visual positions are all unmasked (True).
    """
    mask = torch.ones_like(attn_weights, dtype=torch.bool)
    vis_attn = attn_weights[..., vis_start:vis_end]
    sorted_vals, sorted_idx = torch.sort(vis_attn, dim=-1, descending=True)
    cumsum = sorted_vals.cumsum(dim=-1)
    total = sorted_vals.sum(dim=-1, keepdim=True)
    keep_sorted = cumsum <= ratio * total
    keep_sorted[..., 0] = True  # always keep at least the top-1 token
    vis_mask = torch.zeros_like(vis_attn, dtype=torch.bool)
    vis_mask.scatter_(-1, sorted_idx, keep_sorted)
    mask[..., vis_start:vis_end] = vis_mask
    return mask


def purify_attention(
    attn: torch.Tensor,
    sink_pos: torch.Tensor,
    vis_start: int = VIS_START,
    vis_end: int = VIS_END,
    ratio: float = 0.5,
    remove_sinks: bool = True,
    apply_top_mass: bool = True,
) -> torch.Tensor:
    """Purify an attention matrix by suppressing sinks and applying top-mass masking.

    Args:
        attn: Attention weights of shape (seq_len, seq_len), already softmax'd
              and mean-over-heads reduced.
        sink_pos: 1-D tensor of sink token positions within the visual span.
        vis_start, vis_end: Visual token range.
        ratio: Fraction of visual attention mass to retain.
        remove_sinks: If False, keep sink columns and only apply top-mass
              visual masking. This is the "RoPE/top-mass only" ablation.
        apply_top_mass: If False, skip visual top-mass masking. This isolates
              the effect of sink removal.

    Returns:
        Purified attention matrix of the same shape, with rows renormalized.
    """
    purified = attn.clone()

    # Step 1: Optionally zero out sink attention for all non-visual query
    # positions. Pre-image text rows have zero visual attention due to
    # causal masking, so this effectively targets post-image text queries
    # (instruction tokens after the image + generated tokens).
    text_start = vis_end
    if remove_sinks and sink_pos.numel() > 0 and purified.size(0) > text_start:
        purified[text_start:, sink_pos] = 0.0

    # Step 2: Optionally apply top-mass visual masking
    if apply_top_mass:
        top_mask = top_mass_mask_visual(purified, ratio=ratio, vis_start=vis_start, vis_end=vis_end)
        purified[~top_mask] = 0.0

    # Step 3: Renormalize each row to sum to 1
    row_sums = purified.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    purified = purified / row_sums

    return purified


def compute_sink_attn_mass(
    attn: torch.Tensor,
    sink_pos: torch.Tensor,
    prompt_end_idx: int,
) -> torch.Tensor:
    """Compute how much original attention each text position devotes to sinks.

    Args:
        attn: Original attention matrix (seq_len, seq_len), mean-over-heads.
        sink_pos: 1-D tensor of sink token positions.
        prompt_end_idx: Index of first generated token.

    Returns:
        1-D tensor of shape (seq_len - prompt_end_idx,) with sink attention
        mass for each generated token position.
    """
    if sink_pos.numel() == 0:
        return torch.zeros(attn.size(0) - prompt_end_idx, device=attn.device)
    return attn[prompt_end_idx:, sink_pos].sum(dim=-1)
