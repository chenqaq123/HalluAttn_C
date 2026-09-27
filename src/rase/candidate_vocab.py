"""Benchmark-specific object-phrase candidate matching for rollback."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateMention:
    word: str
    surface: str
    rel_anchor: int
    vocabulary_origin: str
    matched_term: str


@dataclass(frozen=True)
class _VocabularyEntry:
    identity: str
    surface: str
    origin: str


class ObjectCandidateVocabulary:
    """Match object phrases from a configured vocabulary."""

    SCHEMA = "sinkdetect-object-candidate-vocabulary-v1"

    def __init__(self, payload: dict[str, Any], nlp: Any) -> None:
        if payload.get("schema") != self.SCHEMA:
            raise ValueError("unsupported object candidate vocabulary schema")
        benchmarks = payload.get("benchmarks")
        if not isinstance(benchmarks, dict) or not benchmarks:
            raise ValueError("candidate vocabulary has no benchmark entries")
        self.metadata = payload.get("metadata", {})
        self.compiled: dict[
            str, tuple[dict[tuple[str, ...], _VocabularyEntry], int]
        ] = {}
        for benchmark, specification in benchmarks.items():
            entries = specification.get("entries")
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"candidate vocabulary for {benchmark!r} is empty")
            phrases: dict[tuple[str, ...], _VocabularyEntry] = {}
            max_length = 0
            for raw in entries:
                entry = _VocabularyEntry(
                    identity=str(raw["identity"]),
                    surface=str(raw["surface"]),
                    origin=str(raw["origin"]),
                )
                key = self._phrase_key(nlp, entry.surface)
                if not key:
                    continue
                incumbent = phrases.get(key)
                if (
                    incumbent is None
                    or self._origin_rank(entry.origin)
                    < self._origin_rank(incumbent.origin)
                ):
                    phrases[key] = entry
                max_length = max(max_length, len(key))
            if not phrases:
                raise ValueError(
                    f"candidate vocabulary for {benchmark!r} has no parseable phrases"
                )
            self.compiled[str(benchmark)] = (phrases, max_length)

    @staticmethod
    def _origin_rank(origin: str) -> int:
        return {
            "chair": 0,
            "chair_domain": 1,
            "chair_overlap": 1,
            "amber_extension": 2,
        }.get(origin, 3)

    @staticmethod
    def _lemma(token: Any) -> str:
        value = str(
            getattr(token, "lemma_", "") or getattr(token, "text", "")
        ).lower().strip()
        if value == "-pron-":
            value = str(getattr(token, "text", "")).lower().strip()
        return value

    @classmethod
    def _phrase_key(cls, nlp: Any, text: str) -> tuple[str, ...]:
        normalized = text.lower().replace("-", " ").replace("_", " ")
        return tuple(
            cls._lemma(token)
            for token in nlp(normalized)
            if not bool(getattr(token, "is_space", False))
            and not bool(getattr(token, "is_punct", False))
            and cls._lemma(token)
        )

    @classmethod
    def from_file(
        cls, path: str | Path, nlp: Any
    ) -> "ObjectCandidateVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload, nlp)

    def benchmark_summary(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for benchmark, (phrases, _max_length) in self.compiled.items():
            counts: dict[str, int] = {"terms": len(phrases)}
            for entry in phrases.values():
                counts[entry.origin] = counts.get(entry.origin, 0) + 1
            result[benchmark] = counts
        return result

    def mentions(
        self,
        nlp: Any,
        tokenizer: Any,
        generated_ids: list[int],
        *,
        benchmark: str,
    ) -> list[CandidateMention]:
        """Return unique first object-identity mentions, longest phrase first."""

        if benchmark not in self.compiled:
            raise ValueError(f"no candidate vocabulary for benchmark {benchmark!r}")
        phrases, max_length = self.compiled[benchmark]
        text = tokenizer.decode(generated_ids)
        starts = [
            len(tokenizer.decode(generated_ids[:index]))
            for index in range(len(generated_ids))
        ]
        tokens = list(nlp(text))
        mentions: list[CandidateMention] = []
        seen: set[str] = set()
        start = 0
        while start < len(tokens):
            matched: tuple[_VocabularyEntry, int] | None = None
            for length in range(min(max_length, len(tokens) - start), 0, -1):
                span = tokens[start : start + length]
                if any(
                    bool(getattr(token, "is_space", False))
                    or bool(getattr(token, "is_punct", False))
                    for token in span
                ):
                    continue
                key = tuple(self._lemma(token) for token in span)
                entry = phrases.get(key)
                if entry is None:
                    continue
                if str(getattr(span[-1], "pos_", "")) not in {"NOUN", "PROPN"}:
                    continue
                matched = (entry, length)
                break
            if matched is None:
                start += 1
                continue
            entry, length = matched
            first = tokens[start]
            last = tokens[start + length - 1]
            token_index = bisect_right(starts, int(first.idx)) - 1
            rel_anchor = token_index - 1
            if rel_anchor >= 0 and entry.identity not in seen:
                seen.add(entry.identity)
                surface_end = int(last.idx) + len(str(last.text))
                mentions.append(
                    CandidateMention(
                        word=entry.identity,
                        surface=text[int(first.idx) : surface_end],
                        rel_anchor=rel_anchor,
                        vocabulary_origin=entry.origin,
                        matched_term=entry.surface,
                    )
                )
            start += length
        return mentions
