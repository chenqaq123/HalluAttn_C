from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.scripts.analyze_allocation_probe import (  # noqa: E402
    HEADS,
    pair_feature_indices,
)


class AllocationProbeAuditTest(unittest.TestCase):
    def test_pair_feature_indices_keep_both_rows(self) -> None:
        pairs = np.asarray([0, HEADS + 3, 31 * HEADS + 31])

        indices = pair_feature_indices(pairs)

        self.assertEqual(
            indices.tolist(),
            [0, 32, 67, 99, 2015, 2047],
        )


if __name__ == "__main__":
    unittest.main()
