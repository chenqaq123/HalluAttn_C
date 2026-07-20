from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.scripts.evaluate_position_balanced_oof import (  # noqa: E402
    position_balanced_indices,
)


class PositionBalancedOofTest(unittest.TestCase):
    def test_balances_classes_in_each_valid_position_bin(self) -> None:
        labels = np.asarray([0] * 4 + [1] * 2 + [0] * 3 + [1] * 5 + [0] * 3)
        positions = np.asarray([0] * 6 + [10] * 8 + [20] * 3)

        selected, bins = position_balanced_indices(
            labels,
            positions,
            bin_width=10,
            min_bin_count=4,
            rng=np.random.default_rng(17),
        )

        self.assertEqual([item["bin_start"] for item in bins], [0, 10])
        self.assertEqual(selected.size, 10)
        for bin_start in (0, 10):
            in_bin = positions[selected] // 10 == bin_start // 10
            self.assertEqual(int(labels[selected[in_bin]].sum()), int(in_bin.sum() // 2))
        self.assertFalse(np.any(positions[selected] == 20))


if __name__ == "__main__":
    unittest.main()
