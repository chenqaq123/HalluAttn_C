from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mitigation.scripts.merge_registered_generation_shards import merge


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class MergeRegisteredGenerationShardsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.jsonl"
        _write_jsonl(
            self.manifest,
            [
                {"sample_id": 0},
                {"sample_id": 1},
                {"sample_id": 2},
                {"sample_id": 3},
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _shard(
        self, index: int, rows: list[dict], *, method: str = "diml"
    ) -> Path:
        path = self.root / f"part{index}.jsonl"
        _write_jsonl(path, rows)
        path.with_suffix(".jsonl.meta.json").write_text(
            json.dumps(
                {
                    "model": {"kind": "qwen25"},
                    "method": method,
                    "manifest": str(self.manifest.resolve()),
                    "shard_idx": index,
                    "num_shards": 2,
                    "registered_samples_in_shard": len(rows),
                    "completed_samples": len(rows),
                }
            )
        )
        return path

    def test_merges_in_manifest_order(self) -> None:
        first = self._shard(
            0,
            [{"sample_id": 0, "caption": "a"}, {"sample_id": 2, "caption": "c"}],
        )
        second = self._shard(
            1,
            [{"sample_id": 1, "caption": "b"}, {"sample_id": 3, "caption": "d"}],
        )
        output = self.root / "merged.jsonl"
        metadata = merge(
            manifest_path=self.manifest,
            shard_paths=[second, first],
            output_path=output,
            id_key="sample_id",
        )
        self.assertEqual(
            [row["sample_id"] for row in _read_jsonl(output)],
            [0, 1, 2, 3],
        )
        self.assertEqual(metadata["completed_samples"], 4)
        self.assertTrue(metadata["merge"]["preserved_manifest_order"])

    def test_rejects_cross_method_mix(self) -> None:
        first = self._shard(
            0,
            [{"sample_id": 0}, {"sample_id": 2}],
        )
        second = self._shard(
            1,
            [{"sample_id": 1}, {"sample_id": 3}],
            method="allpath",
        )
        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            merge(
                manifest_path=self.manifest,
                shard_paths=[first, second],
                output_path=self.root / "merged.jsonl",
                id_key="sample_id",
            )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
