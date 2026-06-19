"""Open-vocabulary object-claim candidate utilities.

These helpers are intentionally lightweight and model-free. They support the
caption-gate audits by separating two steps: discovering object-like phrases
outside the CHAIR/COCO vocabulary and mapping those phrases back to canonical
verification targets.
"""

from __future__ import annotations

import re
from typing import Any

OPEN_VOCAB_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "closer",
    "could",
    "each",
    "from",
    "front",
    "has",
    "have",
    "having",
    "in",
    "inside",
    "into",
    "is",
    "its",
    "likely",
    "located",
    "near",
    "of",
    "on",
    "one",
    "or",
    "other",
    "overall",
    "possibly",
    "prominently",
    "seen",
    "side",
    "suggests",
    "taking",
    "that",
    "the",
    "there",
    "these",
    "this",
    "to",
    "two",
    "up",
    "visible",
    "with",
}

OPEN_VOCAB_NONCLAIMS = {
    "addition",
    "atmosphere",
    "building",
    "caption",
    "center",
    "end",
    "features",
    "historical",
    "image",
    "large",
    "left",
    "nostalgic",
    "objects",
    "old-fashioned",
    "parked",
    "placed",
    "portion",
    "right",
    "scene",
    "setting",
    "significant",
    "surface",
}


def content_tokens(caption: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9-]*", caption.lower())


def open_vocab_candidates(caption: str, min_len: int = 3, limit: int = 32) -> list[str]:
    """Discover lightweight object-like candidates outside fixed vocabularies."""
    tokens = content_tokens(caption)
    candidates: list[str] = []
    seen: set[str] = set()

    def keep(token: str) -> bool:
        return (
            len(token) >= min_len
            and token not in OPEN_VOCAB_STOPWORDS
            and token not in OPEN_VOCAB_NONCLAIMS
            and not token.isdigit()
        )

    def add(phrase: str) -> None:
        if phrase and phrase not in seen:
            candidates.append(phrase)
            seen.add(phrase)

    for idx, token in enumerate(tokens):
        if not keep(token):
            continue
        add(token)
        if idx > 0 and keep(tokens[idx - 1]):
            add(f"{tokens[idx - 1]} {token}")
        if idx > 1 and keep(tokens[idx - 2]) and keep(tokens[idx - 1]):
            add(f"{tokens[idx - 2]} {tokens[idx - 1]} {token}")
        if len(candidates) >= limit:
            break
    return candidates[:limit]


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


def token_stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def char_ngrams(text: str, n: int = 3) -> set[str]:
    clean = "".join(lexical_tokens(text))
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[idx : idx + n] for idx in range(len(clean) - n + 1)}


def lexical_match_score(candidate: str, target: str) -> float:
    """Score whether an open-vocabulary candidate names a canonical target."""
    cand_tokens = lexical_tokens(candidate)
    target_tokens = lexical_tokens(target)
    if not cand_tokens or not target_tokens:
        return 0.0
    cand_stems = [token_stem(token) for token in cand_tokens]
    target_stems = [token_stem(token) for token in target_tokens]
    cand_joined = "".join(cand_stems)
    target_joined = "".join(target_stems)

    scores: list[float] = []
    for t_stem in target_stems:
        token_scores = []
        for c_stem in cand_stems:
            if c_stem == t_stem:
                token_scores.append(1.0)
            elif len(t_stem) >= 4 and c_stem.startswith(t_stem):
                token_scores.append(0.95)
            elif len(c_stem) >= 4 and t_stem.startswith(c_stem):
                token_scores.append(0.90)
            else:
                common = 0
                for left, right in zip(c_stem, t_stem):
                    if left != right:
                        break
                    common += 1
                if common == len(t_stem) and len(t_stem) <= 3 and c_stem != t_stem:
                    token_scores.append(0.5)
                else:
                    token_scores.append(common / max(len(t_stem), 1))
        scores.append(max(token_scores) if token_scores else 0.0)

    token_score = sum(scores) / len(scores)
    if len(target_joined) >= 4 and target_joined in cand_joined:
        token_score = max(token_score, 0.95)

    cand_grams = char_ngrams(candidate)
    target_grams = char_ngrams(target)
    if cand_grams and target_grams and target_joined in cand_joined:
        ngram_score = len(cand_grams & target_grams) / len(target_grams)
        if len(target_joined) <= 3 and cand_joined != target_joined:
            ngram_score = min(ngram_score, 0.5)
    else:
        ngram_score = 0.0
    return max(token_score, ngram_score)


def denied_item(denied_items: list[dict[str, Any]], word: str) -> dict[str, Any]:
    for item in denied_items:
        if str(item.get("word", "")).strip().lower() == word:
            return item
    return {}


def auto_map_candidate(
    candidate: str,
    denied_items: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Map a candidate phrase to the closest denied canonical target."""
    best_word = ""
    best_score = 0.0
    for item in denied_items:
        word = str(item.get("word", "")).strip().lower()
        score = lexical_match_score(candidate, word)
        if score > best_score or (score == best_score and len(word) > len(best_word)):
            best_word = word
            best_score = score
    mapped = denied_item(denied_items, best_word) if best_score >= threshold else {}
    return {
        "candidate": candidate,
        "mapped_word": best_word if mapped else "",
        "mapping_score": best_score,
        "mapped_target_score": mapped.get("target_score"),
        "mapped_best_neighbor": mapped.get("best_neighbor"),
        "mapped_best_neighbor_score": mapped.get("best_neighbor_score"),
        "mapped_tdev_margin": mapped.get("tdev_margin"),
        "mapped_two_stage_present": mapped.get("two_stage_present"),
    }
