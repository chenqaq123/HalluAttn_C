"""Shared, pre-result generation termination audits."""

from __future__ import annotations

from typing import Any

import torch


PREMATURE_EOS_CONTENT_TOKENS = 16


def _eos_ids(tokenizer: Any) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(value) for value in eos}


def continuation_audit(
    full_ids: torch.Tensor,
    prompt_length: int,
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> dict[str, int | bool]:
    """Describe stopping without calling every ordinary EOS premature.

    ``generated_tokens`` preserves the raw decoder length, including a final
    EOS when present. ``content_tokens`` removes that one final EOS.
    """

    continuation = full_ids[0, prompt_length:].tolist()
    terminated_by_eos = bool(
        continuation and int(continuation[-1]) in _eos_ids(tokenizer)
    )
    content_tokens = len(continuation) - int(terminated_by_eos)
    hit_max_new_tokens = len(continuation) >= max_new_tokens
    return {
        "generated_tokens": len(continuation),
        "content_tokens": content_tokens,
        "terminated_by_eos": terminated_by_eos,
        "hit_max_new_tokens": hit_max_new_tokens,
        "premature_eos": (
            terminated_by_eos
            and content_tokens < PREMATURE_EOS_CONTENT_TOKENS
        ),
    }
