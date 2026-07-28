from __future__ import annotations

import unittest
from unittest.mock import patch

from mitigation.scripts.run_open_caption_formula_suite import (
    parse_args as parse_formula_args,
)
from mitigation.scripts.run_registered_throne_shard_queue import (
    EVALUATORS,
    METHODS,
    MODELS,
    aqa_command,
    formula_command,
    ours_command,
    score_command,
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
        with patch("sys.argv", command[1:]):
            parsed = parse_formula_args()
        self.assertEqual(parsed.seed, 927)
        self.assertEqual(parsed.shard_idx, 3)
        self.assertEqual(len(parsed.job), 5)

    def test_llava_gate_paths_match_promoted_audits(self):
        config = MODELS["llava_next"]
        self.assertEqual(
            config["replay"],
            (
                "experiments/registered_caption/llava_next/"
                "vanilla_replay_20.json"
            ),
        )
        self.assertEqual(
            config["rollback"],
            (
                "experiments/registered_caption/llava_next/"
                "rollback_identity_5.json"
            ),
        )

    def test_ours_command_is_image_rollback_with_frozen_budget(self):
        command = ours_command("llava_next", MODELS["llava_next"], 7, 8)
        joined = " ".join(command)
        self.assertIn("--media image", joined)
        self.assertIn("--shard_idx 7 --num_shards 8", joined)
        self.assertIn("--max_actions 5", joined)
        self.assertIn("ours_shard7.jsonl", joined)

    def test_aqa_suite_contains_every_registered_caption_row(self):
        command = aqa_command("xl", EVALUATORS["xl"], 4, 8)
        joined = " ".join(command)
        self.assertIn("run_registered_throne_aqa_suite.py", joined)
        for config in MODELS.values():
            for method in METHODS:
                self.assertIn(
                    f"{config['directory']}/throne/{method}_shard*.jsonl",
                    joined,
                )
                self.assertIn(
                    f"aqa/xl/{method}_shard4.json",
                    joined,
                )
        self.assertIn("--batch_size 64", joined)

    def test_score_command_requires_all_six_rows_and_official_vote(self):
        command = score_command(MODELS["qwen25"])
        joined = " ".join(command)
        for method in METHODS:
            self.assertIn(f"--caption_row {method}=", joined)
            self.assertIn(f"--aqa_row {method}=", joined)
        self.assertIn("--voting_threshold 9", joined)
        self.assertIn("--bootstrap 10000", joined)


if __name__ == "__main__":
    unittest.main()
