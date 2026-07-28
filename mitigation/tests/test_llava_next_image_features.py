from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mitigation.scripts.extract_llava_next_image_features import (
    FEATURE_KEYS,
    load_resume_checkpoint,
)


def _payload(image_ids: list[int]) -> dict[str, np.ndarray]:
    size = len(image_ids)
    return {
        "object_id": np.arange(size, dtype=np.int64),
        "image_id": np.asarray(image_ids, dtype=np.int64),
        "token_pos": np.arange(size, dtype=np.int32),
        "gen_pos": np.arange(size, dtype=np.int32),
        "label": np.zeros(size, dtype=np.int32),
        "word": np.asarray(["object"] * size, dtype="U64"),
        "no_rope": np.zeros((size, 2, 2, 2, 1), dtype=np.float32),
        "first_logp": np.zeros(size, dtype=np.float64),
    }


class LlavaNextImageFeatureCheckpointTests(unittest.TestCase):
    def test_matching_checkpoint_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "features.jsonl"
            npz = root / "features.npz"
            rows = [
                {"image_id": 10, "mentions": [{"word": "cat"}]},
                {"image_id": 20, "mentions": []},
            ]
            jsonl.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            np.savez_compressed(npz, **_payload([10]))
            values, done = load_resume_checkpoint(jsonl, npz)
            self.assertEqual(done, {10, 20})
            self.assertEqual(set(values), set(FEATURE_KEYS))
            self.assertEqual(len(values["label"]), 1)

    def test_missing_npz_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "features.jsonl"
            jsonl.write_text('{"image_id": 10, "mentions": []}\n')
            with self.assertRaisesRegex(RuntimeError, "both exist"):
                load_resume_checkpoint(
                    jsonl, Path(directory) / "features.npz"
                )

    def test_duplicate_jsonl_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "features.jsonl"
            npz = root / "features.npz"
            row = {"image_id": 10, "mentions": [{"word": "cat"}]}
            jsonl.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            np.savez_compressed(npz, **_payload([10]))
            with self.assertRaisesRegex(RuntimeError, "duplicate JSONL"):
                load_resume_checkpoint(jsonl, npz)

    def test_mention_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "features.jsonl"
            npz = root / "features.npz"
            jsonl.write_text(
                json.dumps(
                    {
                        "image_id": 10,
                        "mentions": [{"word": "cat"}, {"word": "dog"}],
                    }
                )
                + "\n"
            )
            np.savez_compressed(npz, **_payload([10]))
            with self.assertRaisesRegex(RuntimeError, "mention counts"):
                load_resume_checkpoint(jsonl, npz)


if __name__ == "__main__":
    unittest.main()
