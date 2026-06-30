"""Shared target-discriminative evidence utilities.

This module is intentionally small and task-agnostic. Dataset-specific scripts
build contrast sets, load images, and choose output modes; the functions here
provide the common claim templates and VLM-internal evidence readouts used by
the no-external-detector TDEV family.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


PLURAL_OBJECT_NAMES = {"scissors", "skis"}


@dataclass(frozen=True)
class ClaimEvidence:
    yes_logit: float
    no_logit: float
    yes_margin: float
    yes_prob: float


@dataclass(frozen=True)
class VisualClaim:
    """A candidate visual claim to be verified against an image."""

    claim_id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class ContrastSet:
    """One target claim and the alternatives it must beat."""

    target: VisualClaim
    alternatives: tuple[VisualClaim, ...]
    format: str

    def all_claims(self) -> tuple[VisualClaim, ...]:
        return (self.target, *self.alternatives)


def read_neighbor_map(path: str | Path, top_k: int) -> dict[str, list[str]]:
    """Read the semantic-neighbor table used by existence-style adapters."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {obj: [str(item["object"]) for item in items[:top_k]] for obj, items in raw.items()}


def normalize_object_name(obj: str) -> str:
    return str(obj).strip().lower().replace("_", " ")


def indefinite_article(noun: str) -> str:
    return "an" if noun[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def existence_question(obj: str) -> str:
    """Template a COCO object name as a yes/no existence question."""
    obj = normalize_object_name(obj)
    if obj in PLURAL_OBJECT_NAMES:
        return f"Are there any {obj} in the image?"
    return f"Is there {indefinite_article(obj)} {obj} in the image?"


def existence_claim(obj: str) -> str:
    """Template an object name as a declarative visual claim."""
    obj = normalize_object_name(obj)
    if obj in PLURAL_OBJECT_NAMES:
        return f"There are {obj} in the image."
    return f"There is {indefinite_article(obj)} {obj} in the image."


def build_existence_contrast(target_obj: str, neighbor_objs: Sequence[str]) -> ContrastSet:
    target_obj = normalize_object_name(target_obj)
    alternatives = tuple(
        VisualClaim(claim_id=normalize_object_name(obj), text=existence_claim(obj), source="semantic_neighbor")
        for obj in neighbor_objs
    )
    return ContrastSet(
        target=VisualClaim(claim_id=target_obj, text=existence_claim(target_obj), source="target"),
        alternatives=alternatives,
        format="existence",
    )


def build_true_false_contrast(claim: str, negated_claim: str) -> ContrastSet:
    return ContrastSet(
        target=VisualClaim(claim_id="true", text=str(claim).strip(), source="target"),
        alternatives=(VisualClaim(claim_id="false", text=str(negated_claim).strip(), source="negation"),),
        format="true_false",
    )


def build_mcq_contrast(options: Sequence[str], target_index: int) -> ContrastSet:
    if target_index < 0 or target_index >= len(options):
        raise IndexError(f"target_index={target_index} out of range for {len(options)} options")
    claims = [
        VisualClaim(claim_id=str(idx), text=str(option).strip(), source="mcq_option")
        for idx, option in enumerate(options)
    ]
    alternatives = tuple(claim for idx, claim in enumerate(claims) if idx != target_index)
    return ContrastSet(target=claims[target_index], alternatives=alternatives, format="mcq")


def yes_no_prompt(question: str) -> str:
    return f"<image>\nUSER: {question} Please just answer yes or no.\nASSISTANT:"


def claim_verification_prompt(claim: str) -> str:
    return f"<image>\nUSER: {claim} Is this true? Please just answer yes or no.\nASSISTANT:"


def first_token_candidates(tokenizer, word: str) -> list[int]:
    """Return single-token variants for a yes/no answer word."""
    ids: list[int] = []
    for text in (word, word.capitalize(), word.upper()):
        toks = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(toks[0])
    if not ids:
        raise ValueError(f"Could not tokenize candidate word {word!r} as single-token variants")
    return ids


def logsumexp_ids(logits: torch.Tensor, ids: Sequence[int]) -> float:
    vals = logits[torch.tensor(list(ids), dtype=torch.long, device=logits.device)].float()
    return float(torch.logsumexp(vals, dim=0).item())


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def answer_evidence_from_logits(logits: torch.Tensor, yes_ids: Sequence[int], no_ids: Sequence[int]) -> ClaimEvidence:
    yes_logit = logsumexp_ids(logits, yes_ids)
    no_logit = logsumexp_ids(logits, no_ids)
    yes_margin = yes_logit - no_logit
    return ClaimEvidence(
        yes_logit=yes_logit,
        no_logit=no_logit,
        yes_margin=yes_margin,
        yes_prob=sigmoid(yes_margin),
    )


def discriminative_margin(target_score: float, alternative_scores: Sequence[float]) -> float:
    if not alternative_scores:
        return float(target_score)
    return float(target_score) - max(float(score) for score in alternative_scores)


def score_yes_no_question(
    model,
    processor,
    image,
    question: str,
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
    device: torch.device,
) -> ClaimEvidence:
    prompt = yes_no_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    with torch.inference_mode():
        outputs = model.forward(**inputs, output_hidden_states=False, output_attentions=False)
    evidence = answer_evidence_from_logits(outputs.logits[0, -1].float(), yes_ids, no_ids)
    del inputs, outputs
    return evidence
