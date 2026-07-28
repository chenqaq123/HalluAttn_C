from __future__ import annotations

import unittest

from mitigation.scripts.run_registered_throne_shard_queue import (
    MODELS,
    formula_command,
    ours_command,
)


class ThroneShardQueueTests(unittest.TestCase):
    def test_formula_command_freezes_all_registered_formula_rows(self):
        command = formula_command("qwen3", MODELS["qwen3"], 3, 8)
        joined = " ".join(command)
        for method in ("vanilla", "pai", "clearsight", "diml", "allpath"):
            self.assertIn(f"--job {method} ", joined)
            self.assertIn(f"{method}_shard3.jsonl", joined)
        self.assertIn("--shard_idx 3 --num_shards 8", joined)
        self.assertIn("--max_new_tokens 512 --seed 927", joined)
        self.assertIn("--pai_gate", command)

    def test_ours_command_is_image_rollback_with_frozen_budget(self):
        command = ours_command("llava_next", MODELS["llava_next"], 7, 8)
        joined = " ".join(command)
        self.assertIn("--media image", joined)
        self.assertIn("--shard_idx 7 --num_shards 8", joined)
        self.assertIn("--max_actions 5", joined)
        self.assertIn("ours_shard7.jsonl", joined)


if __name__ == "__main__":
    unittest.main()
