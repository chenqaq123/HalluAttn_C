import re

import pytest

from rase.candidate_vocab import ObjectCandidateVocabulary


class _Token:
    def __init__(self, text, lemma, pos, idx, *, punct=False):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.idx = idx
        self.is_space = False
        self.is_punct = punct


class _NLP:
    POS = {
        "a": "DET",
        "and": "CCONJ",
        "with": "ADP",
        "near": "ADP",
    }

    def __call__(self, text):
        result = []
        for match in re.finditer(r"\w+|[^\w\s]", text):
            raw = match.group(0)
            lower = raw.lower()
            punct = not lower.isalnum()
            lemma = {"cars": "car"}.get(lower, lower)
            result.append(
                _Token(
                    raw,
                    lemma,
                    "PUNCT" if punct else self.POS.get(lower, "NOUN"),
                    match.start(),
                    punct=punct,
                )
            )
        return result


class _Tokenizer:
    PIECES = {
        1: "A",
        2: " scene",
        3: " with",
        4: " air",
        5: " and",
        6: " traffic",
        7: " light",
        8: " near",
        9: " cars",
        10: ".",
    }

    @classmethod
    def decode(cls, ids, **_kwargs):
        return "".join(cls.PIECES[value] for value in ids)


def _vocabulary():
    return ObjectCandidateVocabulary(
        {
            "schema": "rase-object-candidate-vocabulary-v1",
            "benchmarks": {
                "amber-generative": {
                    "entries": [
                        {
                            "identity": "traffic light",
                            "surface": "traffic light",
                            "origin": "amber_extension",
                        },
                        {
                            "identity": "light",
                            "surface": "light",
                            "origin": "amber_extension",
                        },
                        {
                            "identity": "car",
                            "surface": "car",
                            "origin": "chair_overlap",
                        },
                    ]
                }
            },
        },
        _NLP(),
    )


def test_amber_vocabulary_matches_longest_object_phrase_and_plural():
    mentions = _vocabulary().mentions(
        _NLP(),
        _Tokenizer(),
        list(range(1, 11)),
        benchmark="amber-generative",
    )
    assert [
        (
            item.word,
            item.surface,
            item.rel_anchor,
            item.vocabulary_origin,
        )
        for item in mentions
    ] == [
        ("traffic light", "traffic light", 4, "amber_extension"),
        ("car", "cars", 7, "chair_overlap"),
    ]


def test_non_vocabulary_nouns_are_not_detector_candidates():
    words = [
        item.word
        for item in _vocabulary().mentions(
            _NLP(),
            _Tokenizer(),
            list(range(1, 11)),
            benchmark="amber-generative",
        )
    ]
    assert "air" not in words
    assert "scene" not in words


def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError, match="no candidate vocabulary"):
        _vocabulary().mentions(
            _NLP(), _Tokenizer(), [1, 2], benchmark="unknown"
        )
