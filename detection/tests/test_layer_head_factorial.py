from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.scripts.evaluate_layer_head_factorial import (  # noqa: E402
    Cell,
    average_precision,
    best_balanced_threshold,
    capacity_transform,
    enumerate_cells,
    make_representation,
)
from detection.scripts.select_single_layer_nested import inner_image_split  # noqa: E402
from detection.scripts.analyze_layer_head_factorial import holm_adjust  # noqa: E402


class LayerHeadFactorialTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        self.allocation = rng.normal(size=(12, 32, 2, 32)).astype(np.float32)

    def test_registered_grid_has_117_unique_cells(self) -> None:
        cells = enumerate_cells()
        self.assertEqual(len(cells), 117)
        self.assertEqual(len({cell.name for cell in cells}), 117)

    def test_head_mean_and_deviation(self) -> None:
        mean_cell = Cell("mean", (3, 4), "head_mean", "test")
        deviation_cell = Cell("deviation", (3, 4), "head_deviation", "test")
        per_head_cell = Cell("per_head", (3, 4), "per_head", "test")

        mean = make_representation(self.allocation, mean_cell)
        deviation = make_representation(self.allocation, deviation_cell)
        per_head = make_representation(self.allocation, per_head_cell)

        self.assertEqual(mean.shape, (12, 4))
        self.assertEqual(deviation.shape, (12, 128))
        self.assertEqual(per_head.shape, (12, 128))
        shaped = deviation.reshape(12, 2, 2, 32)
        np.testing.assert_allclose(shaped.mean(axis=-1), 0.0, atol=1e-6)

    def test_capacity_transform_zero_pads_without_test_fit(self) -> None:
        raw = np.arange(60, dtype=np.float32).reshape(12, 5)
        train = np.arange(8)
        transformed, metadata = capacity_transform(
            raw,
            train,
            output_dim=8,
            device=torch.device("cpu"),
            seed=17,
            pca_iterations=2,
        )
        altered = raw.copy()
        altered[8:] += 10000
        altered_transform, _ = capacity_transform(
            altered,
            train,
            output_dim=8,
            device=torch.device("cpu"),
            seed=17,
            pca_iterations=2,
        )
        np.testing.assert_allclose(transformed[:8], altered_transform[:8])
        np.testing.assert_allclose(transformed[:, 5:], 0.0)
        self.assertEqual(metadata["method"], "identity_zero_pad")

    def test_pca_is_deterministic_and_train_only(self) -> None:
        raw = self.allocation[:, :2].reshape(12, -1)
        train = np.arange(8)
        left, metadata = capacity_transform(
            raw,
            train,
            output_dim=8,
            device=torch.device("cpu"),
            seed=927,
            pca_iterations=2,
        )
        altered = raw.copy()
        altered[8:] += 10000
        right, _ = capacity_transform(
            altered,
            train,
            output_dim=8,
            device=torch.device("cpu"),
            seed=927,
            pca_iterations=2,
        )
        np.testing.assert_allclose(left[:8], right[:8], atol=1e-5)
        self.assertEqual(metadata["method"], "train_fold_pca")

    def test_train_threshold_and_average_precision(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = best_balanced_threshold(labels, scores)
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(average_precision(labels, scores), 1.0)

    def test_inner_image_split_is_group_disjoint_and_deterministic(self) -> None:
        images = np.arange(100, dtype=np.int64)
        left_train, left_val = inner_image_split(images, 0.2, 8100)
        right_train, right_val = inner_image_split(images, 0.2, 8100)
        self.assertEqual(left_train.size, 80)
        self.assertEqual(left_val.size, 20)
        self.assertEqual(np.intersect1d(left_train, left_val).size, 0)
        np.testing.assert_array_equal(left_train, right_train)
        np.testing.assert_array_equal(left_val, right_val)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.20)


if __name__ == "__main__":
    unittest.main()
