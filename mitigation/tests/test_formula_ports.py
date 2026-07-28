from __future__ import annotations

import unittest

import torch
from torch import nn

from mitigation.src.formula_ports import FormulaAttentionPort, FormulaPortConfig


class FakeAttention(nn.Module):
    def __init__(self, layer_idx: int = 2):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = 1


class FakeModeling:
    @staticmethod
    def eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        logits = query @ key.transpose(2, 3) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask
        weights = torch.softmax(logits.float(), dim=-1).to(query.dtype)
        return (weights @ value).transpose(1, 2).contiguous(), weights


class FormulaPortTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.module = FakeAttention()
        self.query = torch.randn(1, 2, 1, 4)
        self.key = torch.randn(1, 2, 5, 4)
        self.value = torch.randn(1, 2, 5, 4)

    def run_port(self, config):
        with FormulaAttentionPort(FakeModeling, FakeAttention, config) as port:
            output, weights = FakeModeling.eager_attention_forward(
                self.module,
                self.query,
                self.key,
                self.value,
                None,
                0.5,
            )
        return output, weights, port.audit

    def test_pai_identity_matches_vanilla(self):
        baseline, weights, _ = self.run_port(
            FormulaPortConfig("vanilla", 0, 4, [1, 2])
        )
        output, modified_weights, audit = self.run_port(
            FormulaPortConfig("pai", 0, 4, [1, 2], pai_alpha=0.0)
        )
        torch.testing.assert_close(output, baseline)
        torch.testing.assert_close(modified_weights, weights)
        self.assertEqual(audit.modified_calls, 0)

    def test_diml_shares_head_mean_signal(self):
        _, baseline_weights, _ = self.run_port(
            FormulaPortConfig("vanilla", 0, 4, [1, 2])
        )
        _, modified_weights, audit = self.run_port(
            FormulaPortConfig("diml", 0, 4, [1, 2], diml_alpha=0.5)
        )
        self.assertFalse(torch.equal(modified_weights, baseline_weights))
        self.assertEqual(audit.layers_modified, {2})
        self.assertGreater(audit.max_abs_tensor_delta, 0)

    def test_allpath_scales_selected_head(self):
        baseline, _, _ = self.run_port(
            FormulaPortConfig("vanilla", 0, 4, [1, 2])
        )
        output, _, audit = self.run_port(
            FormulaPortConfig(
                "allpath",
                0,
                4,
                [1, 2],
                allpath_good_heads={2: [0]},
                allpath_in_scale=2.0,
            )
        )
        self.assertFalse(torch.equal(output, baseline))
        self.assertEqual(audit.layers_modified, {2})

    def test_suspended_branch_is_unmodified(self):
        config = FormulaPortConfig("pai", 0, 4, [1, 2], pai_alpha=0.5)
        with FormulaAttentionPort(FakeModeling, FakeAttention, config) as port:
            with port.suspended():
                suspended, _ = FakeModeling.eager_attention_forward(
                    self.module,
                    self.query,
                    self.key,
                    self.value,
                    None,
                    0.5,
                )
        baseline, _, _ = self.run_port(
            FormulaPortConfig("vanilla", 0, 4, [1, 2])
        )
        torch.testing.assert_close(suspended, baseline)


if __name__ == "__main__":
    unittest.main()
