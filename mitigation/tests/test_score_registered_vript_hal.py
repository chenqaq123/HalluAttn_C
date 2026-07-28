import json
from pathlib import Path

import numpy as np

from mitigation.scripts.score_registered_vript_hal import (
    _paired_bootstrap,
    parse_rows,
    score,
)


class FakeMetric:
    def gt_words(self, contents):
        return contents

    def precision_recall(self, gt_words, prediction):
        precision, recall = map(float, prediction.split(","))
        return precision, recall, ["object"], [precision > 0]

    def word_matches_gt(self, word, gt_words):
        return word in gt_words


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_paired_bootstrap_is_deterministic_and_paired():
    vanilla = np.array([0.1, 0.2, 0.3])
    candidate = vanilla + 0.1
    first = _paired_bootstrap(candidate, vanilla, repeats=1000, seed=17)
    second = _paired_bootstrap(candidate, vanilla, repeats=1000, seed=17)
    assert first == second
    assert np.isclose(first["delta"], 0.1)
    assert first["ci95_low"] > 0


def test_score_validates_and_reports_registered_rows(tmp_path):
    annotations = tmp_path / "annotations.jsonl"
    vanilla = tmp_path / "vanilla.jsonl"
    ours = tmp_path / "ours.jsonl"
    _write_jsonl(
        annotations,
        [
            {"clip_id": "a", "content": ["grounded"]},
            {"clip_id": "b", "content": ["grounded"]},
        ],
    )
    common = {
        "frame_indices": [0, 1],
        "generated_tokens": 2,
        "words": 2,
        "wall_seconds": 1.0,
    }
    _write_jsonl(
        vanilla,
        [
            {"clip_id": "a", "caption": "0.4,0.4", **common},
            {"clip_id": "b", "caption": "0.5,0.5", **common},
        ],
    )
    _write_jsonl(
        ours,
        [
            {
                "clip_id": "a",
                "caption": "0.6,0.5",
                "actions": [{"word": "hallucinated"}],
                **common,
            },
            {
                "clip_id": "b",
                "caption": "0.7,0.6",
                "actions": [{"word": "grounded"}],
                **common,
            },
        ],
    )
    payload = score(
        annotations,
        {"vanilla": str(vanilla), "ours": str(ours)},
        expected_clips=2,
        bootstrap=1000,
        seed=17,
        metric=FakeMetric(),
    )
    assert payload["frame_identity_pass"]
    assert payload["rows"]["ours"]["removal_precision"] == 0.5
    assert payload["paired_vs_vanilla"]["ours"]["precision"]["ci95_low"] > 0
    assert payload["h2_gate"]["model_supports_h2"]


def test_parse_rows_rejects_missing_vanilla():
    try:
        parse_rows(["ours=result.jsonl"])
    except ValueError as error:
        assert "vanilla" in str(error)
    else:
        raise AssertionError("missing vanilla row was accepted")
