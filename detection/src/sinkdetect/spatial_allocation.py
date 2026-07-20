"""Compact spatial statistics for per-head visual attention maps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

FEATURE_NAMES = (
    "visual_mass",
    "top10_mass",
    "top10_neighbor_coherence",
)

_EPS = 1e-12


def remove_attention_keys(
    attention_rows: torch.Tensor, key_positions: torch.Tensor
) -> torch.Tensor:
    """Remove selected key columns and renormalize complete attention rows."""
    if attention_rows.ndim != 3:
        raise ValueError(
            f"Expected attention rows shaped (heads, rows, sequence), got {attention_rows.shape}"
        )
    cleaned = attention_rows.float().clone()
    if key_positions.numel():
        cleaned[:, :, key_positions.long()] = 0.0
    row_sum = cleaned.sum(dim=-1, keepdim=True)
    if bool((row_sum <= _EPS).any()):
        raise ValueError("Removing keys emptied at least one complete attention row")
    return cleaned / row_sum


def spatial_allocation_features(
    attention_rows: torch.Tensor,
    vis_start: int,
    vis_end: int,
    top_fraction: float = 0.10,
) -> torch.Tensor:
    """Return mass, top-k concentration, and 8-neighbor coherence per row.

    ``attention_rows`` is normalized over the complete causal prefix. Spatial
    shape statistics are computed after normalizing its visual slice. The
    returned tensor has shape ``(heads, rows, len(FEATURE_NAMES))``.
    """
    if attention_rows.ndim != 3:
        raise ValueError(
            f"Expected attention rows shaped (heads, rows, sequence), got {attention_rows.shape}"
        )
    if not 0 <= vis_start < vis_end <= attention_rows.shape[-1]:
        raise ValueError(
            f"Invalid visual span [{vis_start}, {vis_end}) for {attention_rows.shape[-1]} keys"
        )
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    visual = attention_rows[:, :, vis_start:vis_end].float()
    visual_mass = visual.sum(dim=-1)
    patch_count = visual.shape[-1]
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(f"Visual patch count {patch_count} is not a square grid")

    uniform = torch.full_like(visual, 1.0 / patch_count)
    probabilities = torch.where(
        visual_mass.unsqueeze(-1) > _EPS,
        visual / visual_mass.unsqueeze(-1).clamp_min(_EPS),
        uniform,
    )
    top_count = max(1, int(round(top_fraction * patch_count)))
    top_values, top_indices = probabilities.topk(top_count, dim=-1)
    top_mass = top_values.sum(dim=-1)

    support = torch.zeros_like(probabilities, dtype=torch.bool)
    support.scatter_(-1, top_indices, True)
    flat_support = support.reshape(-1, 1, side, side).float()
    kernel = torch.ones((1, 1, 3, 3), device=visual.device, dtype=torch.float32)
    kernel[:, :, 1, 1] = 0.0
    neighbor_count = F.conv2d(flat_support, kernel, padding=1)
    degree = F.conv2d(torch.ones_like(flat_support), kernel, padding=1)
    neighbor_fraction = (neighbor_count / degree.clamp_min(1.0)).reshape_as(
        probabilities
    )
    coherence = (
        probabilities * support.to(probabilities.dtype) * neighbor_fraction
    ).sum(dim=-1) / top_mass.clamp_min(_EPS)

    return torch.stack((visual_mass, top_mass, coherence), dim=-1)
