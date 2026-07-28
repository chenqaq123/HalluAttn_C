from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mitigation.scripts.audit_llava_next_image_features import audit_shard


def _row(image_id: int, word: str | None = "cat") -> dict:
    mentions = []
    if word is not None:
        mentions.append(
            {
                "word": word,
                "rel_anchor": 3,
                "hallucinated": False,
            }
        )
    return {
        "image_id": image_id,
        "caption": f"a {word}" if word else "empty",
        "mentions": mentions,
    }


def _payload(image_ids: list[int]) -> dict[str, np.ndarray]:
    size = len(image_ids)
    return {
        "object_id": np.asarray(
            [image_id * 128 for image_id in image_ids],
            dtype=np.int64,
        ),
        "image_id": np.asarray(image_ids, dtype=np.int64),
        "token_pos": np.full(size, 10, dtype=np.int32),
        "gen_pos": np.full(size, 3, dtype=np.int32),
        "label": np.zeros(size, dtype=np.int32),
        "word": np.asarray(["cat"] * size, dtype="U64"),
        "no_rope": np.zeros((size, 2, 2, 2, 1), dtype=np.float32),
        "first_logp": np.zeros(size, dtype=np.float64),
        "feature_names": np.asarray(["visual_mass"], dtype="U32"),
    }


def _write_shard(
    root: Path,
    rows: list[dict],
    feature_image_ids: list[int],
) -> tuple[Path, Path]:
    jsonl = root / "features.jsonl"
    npz = root / "features.npz"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    np.savez_compressed(npz, **_payload(feature_image_ids))
    return jsonl, npz


class LlavaNextImageFeatureAuditTests(unittest.TestCase):
    def test_valid_shard_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl, npz = _write_shard(
                Path(directory),
                [_row(10), _row(20, None)],
                [10],
            )
            result = audit_shard(jsonl, npz, [10, 20], set())
            self.assertTrue(result["valid"])
            self.assertEqual(result["mentions"], 1)

    def test_identical_duplicates_are_archived_and_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = _row(10)
            jsonl, npz = _write_shard(
                root,
                [duplicate, duplicate, _row(20, None)],
                [10],
            )
            result = audit_shard(
                jsonl,
                npz,
                [10, 20],
                set(),
                normalize_duplicates=True,
            )
            self.assertEqual(result["duplicate_extra_rows_removed"], 1)
            self.assertTrue(Path(result["archive"]).is_file())
            normalized = [
                json.loads(line)
                for line in jsonl.read_text().splitlines()
            ]
            self.assertEqual(
                [row["image_id"] for row in normalized], [10, 20]
            )

    def test_divergent_duplicates_fail_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl, npz = _write_shard(
                root,
                [_row(10), _row(10, "dog")],
                [10],
            )
            before = jsonl.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "not identical"):
                audit_shard(
                    jsonl,
                    npz,
                    [10],
                    set(),
                    normalize_duplicates=True,
                )
            self.assertEqual(jsonl.read_bytes(), before)

    def test_incomplete_duplicate_shard_is_not_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = _row(10)
            jsonl, npz = _write_shard(
                root,
                [duplicate, duplicate],
                [10],
            )
            before = jsonl.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "complete"):
                audit_shard(
                    jsonl,
                    npz,
                    [10, 20],
                    set(),
                    normalize_duplicates=True,
                )
            self.assertEqual(jsonl.read_bytes(), before)

    def test_mention_payload_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl, npz = _write_shard(root, [_row(10)], [10])
            payload = _payload([10])
            payload["label"][0] = 1
            np.savez_compressed(npz, **payload)
            with self.assertRaisesRegex(RuntimeError, "mention mismatch"):
                audit_shard(jsonl, npz, [10], set())

    def test_excluded_image_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl, npz = _write_shard(
                Path(directory), [_row(10)], [10]
            )
            with self.assertRaisesRegex(RuntimeError, "Excluded"):
                audit_shard(jsonl, npz, [10], {10})


if __name__ == "__main__":
    unittest.main()
