from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.spatial_allocation import (  # noqa: E402
    remove_attention_keys,
    spatial_allocation_features,
)


class SpatialAllocationTest(unittest.TestCase):
    def test_mass_and_concentration_use_different_normalizations(self) -> None:
        row = torch.zeros(1, 1, 8)
        row[:, :, :4] = torch.tensor([0.20, 0.10, 0.05, 0.05])
        row[:, :, 4:] = 0.15
        features = spatial_allocation_features(row, 0, 4, top_fraction=0.25)
        self.assertAlmostEqual(float(features[0, 0, 0]), 0.40, places=6)
        self.assertAlmostEqual(float(features[0, 0, 1]), 0.50, places=6)

    def test_contiguous_top_support_has_higher_neighbor_coherence(self) -> None:
        patch_count = 16
        contiguous = torch.full((patch_count,), 1e-4)
        contiguous[[0, 1, 4, 5]] = 1.0
        scattered = torch.full((patch_count,), 1e-4)
        scattered[[0, 3, 12, 15]] = 1.0
        rows = torch.stack((contiguous, scattered)).reshape(1, 2, patch_count)
        rows /= rows.sum(dim=-1, keepdim=True)
        features = spatial_allocation_features(rows, 0, patch_count, top_fraction=0.25)
        self.assertAlmostEqual(
            float(features[0, 0, 1]), float(features[0, 1, 1]), places=6
        )
        self.assertGreater(float(features[0, 0, 2]), float(features[0, 1, 2]))

    def test_full_support_has_unit_neighbor_coherence(self) -> None:
        row = torch.full((2, 3, 16), 1.0 / 16)
        features = spatial_allocation_features(row, 0, 16, top_fraction=1.0)
        torch.testing.assert_close(features[..., 2], torch.ones(2, 3))

    def test_sink_removal_renormalizes_complete_row(self) -> None:
        row = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        cleaned = remove_attention_keys(row, torch.tensor([1, 3]))
        torch.testing.assert_close(cleaned.sum(dim=-1), torch.ones(1, 1))
        torch.testing.assert_close(cleaned[0, 0], torch.tensor([0.25, 0.0, 0.75, 0.0]))
        torch.testing.assert_close(row, torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]))


if __name__ == "__main__":
    unittest.main()
