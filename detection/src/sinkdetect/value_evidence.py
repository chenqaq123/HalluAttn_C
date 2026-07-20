"""Source-decomposed visual value evidence utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PER_HEAD_FEATURE_NAMES = (
    "visual_mass",
    "value_norm",
    "target_dot",
    "target_cosine",
    "target_gain",
)

JOINT_FEATURE_NAMES = (
    "mean_visual_mass",
    "value_norm",
    "target_dot",
    "target_cosine",
    "target_gain",
)

CONFIDENCE_FEATURE_NAMES = (
    "selected_logprob",
    "entropy",
    "max_logprob",
    "selected_margin",
    "normalized_rank",
)

_EPS = 1e-12


def repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    """Repeat key/value heads to the query-head count."""
    if states.ndim != 3:
        raise ValueError(f"Expected (heads, sequence, dim), got {states.shape}")
    if groups < 1:
        raise ValueError(f"groups must be positive, got {groups}")
    if groups == 1:
        return states
    return states.repeat_interleave(groups, dim=0)


def causal_attention_rows(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Compute pre-RoPE attention rows over each query's complete causal prefix.

    Args:
        query: Query states with shape ``(query_heads, rows, head_dim)``.
        key: Key states with shape ``(kv_heads, sequence, head_dim)``.
        query_positions: Absolute sequence position for each requested row.
        scaling: Attention scaling factor.
    """
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError(f"Expected rank-3 Q/K, got {query.shape} and {key.shape}")
    if query.shape[1] != query_positions.numel():
        raise ValueError("query rows and query_positions do not align")
    if query.shape[0] % key.shape[0] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    expanded_key = repeat_kv(key, query.shape[0] // key.shape[0])
    logits = torch.einsum("hrd,hsd->hrs", query.float(), expanded_key.float())
    logits.mul_(float(scaling))
    key_positions = torch.arange(key.shape[1], device=query_positions.device)
    causal = key_positions.unsqueeze(0) <= query_positions.reshape(-1, 1)
    logits.masked_fill_(~causal.unsqueeze(0), torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1)


def project_source_contributions(
    attention_rows: torch.Tensor,
    value_states: torch.Tensor,
    output_weight: torch.Tensor,
    source_start: int,
    source_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one source span's value contribution through each output-head slice.

    Returns:
        ``(contributions, source_mass)`` where contributions has shape
        ``(rows, heads, hidden_size)`` and source_mass has shape ``(rows, heads)``.
    """
    if attention_rows.ndim != 3 or value_states.ndim != 3:
        raise ValueError("attention_rows and value_states must be rank 3")
    heads, rows, sequence = attention_rows.shape
    if not 0 <= source_start < source_end <= sequence:
        raise ValueError(
            f"Invalid source span [{source_start}, {source_end}) for sequence {sequence}"
        )
    if heads % value_states.shape[0] != 0:
        raise ValueError("attention heads must be divisible by value heads")
    values = repeat_kv(value_states, heads // value_states.shape[0]).float()
    weights = attention_rows[:, :, source_start:source_end].float()
    source_values = values[:, source_start:source_end]
    head_outputs = torch.einsum("hrv,hvd->hrd", weights, source_values)
    head_dim = head_outputs.shape[-1]
    if output_weight.shape[1] != heads * head_dim:
        raise ValueError(
            f"Output projection {output_weight.shape} does not match {heads}x{head_dim} heads"
        )
    head_weight = output_weight.float().reshape(output_weight.shape[0], heads, head_dim)
    contributions = torch.einsum("hrd,ohd->rho", head_outputs, head_weight)
    return contributions, weights.sum(dim=-1).transpose(0, 1)


def reconstruct_attention_output(
    attention_rows: torch.Tensor,
    value_states: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the bias-free attention output for selected query rows."""
    contributions, _ = project_source_contributions(
        attention_rows,
        value_states,
        output_weight,
        source_start=0,
        source_end=attention_rows.shape[-1],
    )
    return contributions.sum(dim=1)


def _target_logits(
    hidden: torch.Tensor,
    target_directions: torch.Tensor,
    final_norm: torch.nn.Module,
) -> torch.Tensor:
    normalized = final_norm(hidden)
    return (normalized.float() * target_directions.float()).sum(dim=-1)


def evidence_features(
    residual_rows: torch.Tensor,
    contributions: torch.Tensor,
    source_mass: torch.Tensor,
    target_ids: torch.Tensor,
    final_norm: torch.nn.Module,
    unembedding_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-head and joint target-aligned visual evidence features."""
    if contributions.ndim != 3:
        raise ValueError("contributions must have shape (rows, heads, hidden_size)")
    rows, heads, hidden_size = contributions.shape
    if residual_rows.shape != (rows, hidden_size):
        raise ValueError("residual_rows do not align with contributions")
    if source_mass.shape != (rows, heads):
        raise ValueError("source_mass does not align with contributions")
    if target_ids.shape != (rows,):
        raise ValueError("target_ids do not align with contributions")

    directions = unembedding_weight[target_ids].float()
    norms = contributions.float().norm(dim=-1)
    dots = torch.einsum("rho,ro->rh", contributions.float(), directions)
    direction_norms = directions.norm(dim=-1, keepdim=True)
    cosines = dots / (norms * direction_norms).clamp_min(_EPS)

    baseline_logits = _target_logits(residual_rows, directions, final_norm)
    head_hidden = residual_rows[:, None, :] + contributions
    head_directions = directions[:, None, :].expand(rows, heads, hidden_size)
    head_logits = _target_logits(head_hidden, head_directions, final_norm)
    gains = head_logits - baseline_logits[:, None]
    per_head = torch.stack([source_mass, norms, dots, cosines, gains], dim=-1)

    joint_contribution = contributions.sum(dim=1)
    joint_norm = joint_contribution.float().norm(dim=-1)
    joint_dot = (joint_contribution.float() * directions).sum(dim=-1)
    joint_cosine = joint_dot / (
        joint_norm * directions.norm(dim=-1)
    ).clamp_min(_EPS)
    joint_gain = _target_logits(
        residual_rows + joint_contribution, directions, final_norm
    ) - baseline_logits
    joint = torch.stack(
        [source_mass.mean(dim=1), joint_norm, joint_dot, joint_cosine, joint_gain],
        dim=-1,
    )
    return per_head, joint


def confidence_features(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed confidence statistics for tokens selected after query rows."""
    if logits.ndim != 2 or input_ids.ndim != 1 or query_positions.ndim != 1:
        raise ValueError("Expected logits (sequence,vocab), IDs (sequence), positions (rows)")
    selected_positions = query_positions + 1
    if int(selected_positions.max()) >= input_ids.numel():
        raise ValueError("A query row has no following selected token")
    selected_ids = input_ids[selected_positions]
    row_logits = logits[query_positions].float()
    log_probs = F.log_softmax(row_logits, dim=-1)
    probabilities = log_probs.exp()
    selected = log_probs.gather(1, selected_ids[:, None]).squeeze(1)
    entropy = -(probabilities * log_probs).sum(dim=-1)
    max_logprob = log_probs.max(dim=-1).values

    selected_logits = row_logits.gather(1, selected_ids[:, None]).squeeze(1)
    alternatives = row_logits.clone()
    alternatives.scatter_(1, selected_ids[:, None], -torch.inf)
    margin = selected_logits - alternatives.max(dim=-1).values
    rank = (row_logits > selected_logits[:, None]).sum(dim=-1).float()
    rank /= max(row_logits.shape[-1] - 1, 1)
    features = torch.stack([selected, entropy, max_logprob, margin, rank], dim=-1)
    return features, selected_ids
