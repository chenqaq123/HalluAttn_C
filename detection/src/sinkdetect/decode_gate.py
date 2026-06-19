"""Decode-time object-phrase gates for target-discriminative captioning."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
from transformers import LogitsProcessor


def _dedupe_seqs(seqs: Iterable[list[int]]) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for seq in seqs:
        if not seq:
            continue
        key = tuple(seq)
        if key in seen:
            continue
        seen.add(key)
        out.append(seq)
    return out


def _dedupe_text(items: Iterable[str]) -> list[str]:
    return list(OrderedDict.fromkeys(item.strip() for item in items if item and item.strip()))


def trim_leading_whitespace_tokens(tokenizer: Any, seq: list[int]) -> list[int]:
    """Drop tokenizer whitespace placeholders before the first content token."""
    for idx, token_id in enumerate(seq):
        if tokenizer.decode([token_id]).strip():
            return seq[idx:]
    return seq


def build_object_phrase_sequences(
    tokenizer: Any,
    phrases: Iterable[str],
    *,
    include_capitalized: bool = True,
    trim_leading_whitespace: bool = True,
) -> list[list[int]]:
    """Tokenize object phrases for prefix-state generation blocking.

    LLaVA-1.5 uses a LLaMA tokenizer where a leading-space form often starts
    with a standalone whitespace token. A decoder should not ban that whitespace
    globally, so by default we trim leading whitespace-only tokens and gate on
    the first content token and its phrase continuation.
    """
    forms: list[str] = []
    for phrase in _dedupe_text(phrases):
        variants = [phrase, f" {phrase}"]
        if include_capitalized:
            cap = phrase[:1].upper() + phrase[1:]
            variants.extend([cap, f" {cap}"])
        forms.extend(variants)

    seqs: list[list[int]] = []
    for form in _dedupe_text(forms):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if trim_leading_whitespace:
            ids = trim_leading_whitespace_tokens(tokenizer, ids)
        if ids:
            seqs.append(ids)
    return _dedupe_seqs(seqs)


@dataclass
class GateEvent:
    """Small audit record for a decode-gate suppression step."""

    batch_index: int
    generated_step: int
    prefix_len: int
    banned_token_id: int
    banned_token_text: str
    phrase_text: str = ""


@dataclass
class ObjectPhraseGateLogitsProcessor(LogitsProcessor):
    """Suppress object-phrase continuations with a prefix-state gate.

    The processor receives one denied phrase-sequence list per batch item. At
    each generation step it checks whether the already-generated suffix is a
    prefix of a denied phrase and sets the next token for that phrase to
    ``-inf``. With an empty suffix, this blocks the first content token of the
    denied phrase. With a partial suffix, it blocks phrase completion.

    This is intentionally narrow: it does not score vision evidence itself. The
    caller is responsible for deciding which object phrases are unsupported for
    each image.
    """

    tokenizer: Any
    denied_token_sequences: list[list[list[int]]]
    prompt_lengths: list[int] | int
    denied_phrase_texts: list[list[str]] | None = None
    penalty: float | None = None
    min_prefix_len_to_block: int = 0
    block_first_token_for_multi_token: bool = True
    audit_limit: int = 200
    events: list[GateEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.prompt_lengths, int):
            self.prompt_lengths = [self.prompt_lengths] * len(self.denied_token_sequences)
        if len(self.prompt_lengths) != len(self.denied_token_sequences):
            raise ValueError("prompt_lengths must be an int or match the batch size")
        if self.denied_phrase_texts is None:
            self.denied_phrase_texts = [[""] * len(seqs) for seqs in self.denied_token_sequences]
        if len(self.denied_phrase_texts) != len(self.denied_token_sequences):
            raise ValueError("denied_phrase_texts must match the batch size")
        if self.min_prefix_len_to_block < 0:
            raise ValueError("min_prefix_len_to_block must be non-negative")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        if batch_size != len(self.denied_token_sequences):
            raise ValueError(
                f"batch size {batch_size} does not match gate config {len(self.denied_token_sequences)}"
            )

        for batch_idx in range(batch_size):
            start = int(self.prompt_lengths[batch_idx])
            generated = input_ids[batch_idx, start:].tolist()
            for seq_idx, seq in enumerate(self.denied_token_sequences[batch_idx]):
                prefix_len = self._matched_prefix_len(generated, seq)
                if (
                    prefix_len is None
                    or prefix_len >= len(seq)
                    or prefix_len < self.min_prefix_len_to_block
                    or (
                        prefix_len == 0
                        and len(seq) > 1
                        and not self.block_first_token_for_multi_token
                    )
                ):
                    continue
                banned_token = int(seq[prefix_len])
                if self.penalty is None:
                    scores[batch_idx, banned_token] = -float("inf")
                else:
                    scores[batch_idx, banned_token] -= float(self.penalty)
                if len(self.events) < self.audit_limit:
                    text = self.tokenizer.decode([banned_token])
                    phrase_text = self.denied_phrase_texts[batch_idx][seq_idx]
                    self.events.append(
                        GateEvent(
                            batch_index=batch_idx,
                            generated_step=len(generated),
                            prefix_len=prefix_len,
                            banned_token_id=banned_token,
                            banned_token_text=text,
                            phrase_text=phrase_text,
                        )
                    )
        return scores

    @staticmethod
    def _matched_prefix_len(generated: list[int], seq: list[int]) -> int | None:
        max_prefix = min(len(generated), len(seq) - 1)
        for prefix_len in range(max_prefix, -1, -1):
            if prefix_len == 0:
                return 0
            if generated[-prefix_len:] == seq[:prefix_len]:
                return prefix_len
        return None
