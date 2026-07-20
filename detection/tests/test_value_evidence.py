from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


class ValueEvidenceTest(unittest.TestCase):
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
