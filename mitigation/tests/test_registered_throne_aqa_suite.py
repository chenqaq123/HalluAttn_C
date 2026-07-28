from __future__ import annotations

import unittest
from types import SimpleNamespace

from mitigation.scripts.run_registered_throne_aqa_suite import validate_jobs


class RegisteredThroneAQASuiteTests(unittest.TestCase):
    def test_accepts_distinct_outputs(self):
        validate_jobs(
            SimpleNamespace(
                job=[
                    ["vanilla*.jsonl", "vanilla.json"],
                    ["pai*.jsonl", "pai.json"],
                ],
                shard_idx=2,
                num_shards=8,
            )
        )

    def test_rejects_duplicate_outputs(self):
        with self.assertRaisesRegex(ValueError, "distinct output"):
            validate_jobs(
                SimpleNamespace(
                    job=[
                        ["vanilla*.jsonl", "same.json"],
                        ["pai*.jsonl", "same.json"],
                    ],
                    shard_idx=2,
                    num_shards=8,
                )
            )

    def test_rejects_invalid_shard(self):
        with self.assertRaisesRegex(ValueError, "Invalid shard"):
            validate_jobs(
                SimpleNamespace(
                    job=[["vanilla*.jsonl", "vanilla.json"]],
                    shard_idx=8,
                    num_shards=8,
                )
            )


if __name__ == "__main__":
    unittest.main()
