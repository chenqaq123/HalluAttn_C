from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from mitigation.src.formula_ports import (
    FormulaAttentionPort,
    FormulaPortConfig,
    generate_vcd_greedy_multimodal,
)


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


class FakeGenerationModel:
    def __init__(self):
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.generation_config = SimpleNamespace(eos_token_id=4)
        self.model = SimpleNamespace(rope_deltas=None)
        self.prepare_calls = 0
        self.logits_to_keep_calls = []

    def _supports_logits_to_keep(self):
        return True

    def _get_initial_cache_position(
        self, seq_length, device, model_kwargs
    ):
        model_kwargs["cache_position"] = torch.arange(
            seq_length, device=device
        )
        return model_kwargs

    def prepare_inputs_for_generation(
        self, sequence, cache_position, **kwargs
    ):
        self.prepare_calls += 1
        return {
            "input_ids": (
                sequence
                if int(cache_position[0]) == 0
                else sequence[:, -1:]
            ),
            "cache_position": cache_position,
            "logits_to_keep": kwargs["logits_to_keep"],
        }

    def __call__(
        self,
        input_ids,
        cache_position,
        logits_to_keep,
        return_dict=True,
    ):
        del input_ids, return_dict
        self.logits_to_keep_calls.append(logits_to_keep)
        token = min(2 + int(cache_position[-1]), 4)
        logits = torch.full((1, 1, 6), -10.0)
        logits[0, 0, token] = 10.0
        self.model.rope_deltas = torch.tensor([[int(cache_position[-1])]])
        return SimpleNamespace(logits=logits, past_key_values=None)

    def _update_model_kwargs_for_generation(
        self, output, model_kwargs, is_encoder_decoder
    ):
        del output, is_encoder_decoder
        model_kwargs["cache_position"] = (
            model_kwargs["cache_position"][-1:] + 1
        )
        return model_kwargs


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

    def test_vcd_uses_generation_preparation_and_appends_eos(self):
        model = FakeGenerationModel()
        tokenizer = SimpleNamespace(eos_token_id=4)
        full_ids, audit = generate_vcd_greedy_multimodal(
            model,
            tokenizer,
            {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
                "pixel_values": torch.ones(1, 1),
            },
            max_new_tokens=5,
            alpha=0.0,
            beta=0.1,
            noise_step=500,
            seed=17,
        )
        self.assertEqual(full_ids.tolist(), [[1, 2, 3, 4]])
        self.assertTrue(audit["stopped_by_eos"])
        self.assertEqual(audit["generated_tokens"], 3)
        self.assertEqual(model.prepare_calls, 6)
        self.assertEqual(model.logits_to_keep_calls, [1] * 6)


if __name__ == "__main__":
    unittest.main()
