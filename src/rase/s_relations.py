"""Model-agnostic construction and validation of head-relation matrices."""

from __future__ import annotations

import numpy as np
import torch


def canonical_edge_heads(heads: int) -> np.ndarray:
    """Return canonical strict-upper head-pair coordinates."""

    if heads < 2:
        raise ValueError("heads must be at least two")
    return np.stack(np.triu_indices(heads, k=1), axis=-1).astype(np.int32)


def attention_to_s_edges(attention: torch.Tensor) -> torch.Tensor:
    """Convert visual attention to strict-upper Bhattacharyya similarities.

    ``attention`` has shape ``(..., heads, visual_tokens)``. Each head is
    normalized over the supplied visual tokens before computing

        S_ij = sum_v sqrt(p_i(v) * p_j(v)).

    The result has shape ``(..., heads * (heads - 1) / 2)`` in canonical
    strict-upper order. No model, dataset, layer, head, or image-size setting
    is assumed.
    """

    if attention.ndim < 2:
        raise ValueError("attention must end in (heads, visual_tokens)")
    if attention.shape[-2] < 2 or attention.shape[-1] < 1:
        raise ValueError("attention must contain at least two heads and one token")
    values = attention.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("attention contains non-finite values")
    if bool((values < 0).any()):
        raise ValueError("attention must be non-negative")
    mass = values.sum(dim=-1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("every head must assign positive visual mass")
    probabilities = values / mass
    roots = probabilities.sqrt()
    similarity = roots @ roots.transpose(-1, -2)
    upper = torch.triu_indices(
        attention.shape[-2], attention.shape[-2], offset=1, device=attention.device
    )
    return similarity[..., upper[0], upper[1]].clamp_(0.0, 1.0)


def attention_to_selected_s_edges(
    attention: torch.Tensor,
    edge_heads: torch.Tensor,
) -> torch.Tensor:
    """Compute only explicitly selected Bhattacharyya head-pair similarities."""

    if attention.ndim < 2:
        raise ValueError("attention must end in (heads, visual_tokens)")
    if edge_heads.ndim != 2 or edge_heads.shape[1] != 2:
        raise ValueError("edge_heads must have shape (selected_edges, 2)")
    if edge_heads.numel() and (
        int(edge_heads.min()) < 0 or int(edge_heads.max()) >= attention.shape[-2]
    ):
        raise ValueError("edge_heads contains an out-of-range head index")
    values = attention.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("attention contains non-finite values")
    if bool((values < 0).any()):
        raise ValueError("attention must be non-negative")
    mass = values.sum(dim=-1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("every head must assign positive visual mass")
    roots = (values / mass).sqrt()
    left = roots.index_select(-2, edge_heads[:, 0])
    right = roots.index_select(-2, edge_heads[:, 1])
    return (left * right).sum(dim=-1).clamp_(0.0, 1.0)
