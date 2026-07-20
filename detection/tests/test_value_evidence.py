from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

DETECTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DETECTION_ROOT / "src"))

from sinkdetect.value_evidence import (  # noqa: E402
    causal_attention_rows,
    confidence_features,
    evidence_features,
    project_source_contributions,
    reconstruct_attention_output,
)
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _conditional_weights,
)


class ValueEvidenceTest(unittest.TestCase):
    def test_uniform_bin_weights_equalize_each_bin_and_class(self) -> None:
        labels = np.asarray(
            [0] * 24 + [1] * 6 + [0] * 10 + [1] * 30 + [0] * 10 + [1] * 10
        )
        positions = np.asarray([0] * 30 + [10] * 40 + [20] * 20)

        size_weights = _conditional_weights(labels, positions, 10, mode="size")
        uniform_weights = _conditional_weights(
            labels, positions, 10, mode="uniform_bin"
        )

        size_totals = [
            float(size_weights[positions == value].sum()) for value in (0, 10)
        ]
        uniform_totals = [
            float(uniform_weights[positions == value].sum()) for value in (0, 10)
        ]
        self.assertAlmostEqual(size_totals[1] / size_totals[0], 4.0 / 3.0, places=5)
        self.assertAlmostEqual(uniform_totals[0], uniform_totals[1], places=5)
        self.assertTrue(np.all(uniform_weights[positions == 20] == 0))
        for position in (0, 10):
            class_totals = [
                float(
                    uniform_weights[
                        (positions == position) & (labels == label)
                    ].sum()
                )
                for label in (0, 1)
            ]
            self.assertAlmostEqual(class_totals[0], class_totals[1], places=5)

    def test_causal_attention_rows_masks_future_keys(self) -> None:
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        rows = causal_attention_rows(query, key, torch.tensor([0, 1]), scaling=1.0)
        self.assertTrue(torch.allclose(rows.sum(dim=-1), torch.ones(1, 2)))
        self.assertEqual(float(rows[0, 0, 1:].sum()), 0.0)
        self.assertEqual(float(rows[0, 1, 2]), 0.0)

    def test_head_projection_reconstructs_direct_output(self) -> None:
        torch.manual_seed(3)
        attention = torch.softmax(torch.randn(2, 3, 5), dim=-1)
        values = torch.randn(2, 5, 4)
        output_weight = torch.randn(7, 8)
        contributions, mass = project_source_contributions(
            attention, values, output_weight, 1, 4
        )
        self.assertEqual(contributions.shape, (3, 2, 7))
        self.assertEqual(mass.shape, (3, 2))

        full = torch.einsum("hrs,hsd->hrd", attention, values)
        expected = torch.nn.functional.linear(
            full.transpose(0, 1).reshape(3, 8), output_weight
        )
        reconstructed = reconstruct_attention_output(attention, values, output_weight)
        self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-5))

    def test_target_aligned_contribution_has_positive_gain(self) -> None:
        residual = torch.zeros(1, 3)
        contribution = torch.tensor([[[2.0, 0.0, 0.0]]])
        mass = torch.ones(1, 1)
        unembedding = torch.eye(3)
        per_head, joint = evidence_features(
            residual,
            contribution,
            mass,
            torch.tensor([0]),
            torch.nn.Identity(),
            unembedding,
        )
        self.assertGreater(float(per_head[0, 0, 4]), 0.0)
        self.assertAlmostEqual(float(per_head[0, 0, 2]), 2.0)
        self.assertAlmostEqual(float(joint[0, 4]), 2.0)

    def test_confidence_features_follow_query_rows(self) -> None:
        logits = torch.tensor(
            [[0.0, 2.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 4.0]]
        )
        ids = torch.tensor([0, 1, 0])
        features, selected = confidence_features(logits, ids, torch.tensor([0, 1]))
        self.assertEqual(features.shape, (2, 5))
        self.assertTrue(torch.equal(selected, torch.tensor([1, 0])))
        self.assertTrue(torch.isfinite(features).all())
        self.assertAlmostEqual(float(features[0, 4]), 0.0)


if __name__ == "__main__":
    unittest.main()
