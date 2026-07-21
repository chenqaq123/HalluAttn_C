from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.scripts.compare_position_balanced_detectors import (  # noqa: E402
    _assert_aligned,
    _fit_adscgc_logistic_oof,
)


class PositionBalancedDetectorComparisonTest(unittest.TestCase):
    def test_adscgc_oof_is_grouped_and_finite(self) -> None:
        image_ids = np.repeat(np.arange(10), 2)
        labels = np.tile(np.asarray([0, 1]), 10)
        signal = labels[:, None].astype(np.float64)
        raw = {
            "ads": np.concatenate((signal, 1.0 - signal), axis=1),
            "cgc": np.concatenate((2.0 * signal, -signal), axis=1),
            "label": labels,
            "image_id": image_ids,
        }

        scores = _fit_adscgc_logistic_oof(raw, folds=5)

        self.assertEqual(scores.shape, labels.shape)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores[labels == 1].min()), float(scores[labels == 0].max()))

    def test_alignment_rejects_mismatched_object_ids(self) -> None:
        reference = {
            "object_id": np.asarray([0, 1]),
            "label": np.asarray([0, 1]),
            "gen_pos": np.asarray([2, 3]),
        }
        with self.assertRaisesRegex(ValueError, "object_id"):
            _assert_aligned(
                "test",
                np.asarray([1, 0]),
                reference["label"],
                reference["gen_pos"],
                reference,
            )


if __name__ == "__main__":
    unittest.main()
