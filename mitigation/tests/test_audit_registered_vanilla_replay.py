from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mitigation.scripts.audit_registered_vanilla_replay import _saved


class RegisteredVanillaReplayTests(unittest.TestCase):
    def test_saved_rejects_duplicate_ids_across_shards(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                (root / f"shard{index}.jsonl").write_text(
                    json.dumps(
                        {
                            "sample_id": 7,
                            "method": "vanilla",
                            "caption": "same",
                        }
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(ValueError, "Duplicate saved ID"):
                _saved([str(root / "shard*.jsonl")], "sample_id")

    def test_saved_rejects_nonvanilla_method(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "row.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": 7,
                        "method": "diml",
                        "caption": "caption",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "expected vanilla"):
                _saved([str(path)], "sample_id")

    def test_saved_accepts_one_complete_vanilla_row(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "row.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "clip_id": "clip-1",
                        "method": "vanilla",
                        "caption": "caption",
                    }
                )
                + "\n"
            )
            self.assertEqual(
                _saved([str(path)], "clip_id")["clip-1"]["caption"],
                "caption",
            )


if __name__ == "__main__":
    unittest.main()
