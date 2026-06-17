"""Attention-only mitigation ports for HuggingFace LLaVA-1.5.

The ports mirror the intervention component of each official implementation.
PAI intentionally excludes its CFG/logit-refinement branch in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg


@dataclass
class InterventionConfig:
    method: str
    start_layer: int
    end_layer: int
    pai_alpha: float = 0.2
    vaf_enhance: float = 1.15
    vaf_suppress: float = 0.95
    vas_tau: float = 20.0
    vas_rho: float = 0.5
    vas_visual_mass: float = 0.2
    vas_keep: float = 0.6


class AttentionIntervention(LlamaAttention):
    """Eager attention implementation with an optional inference intervention."""

    def __init__(self, config: LlamaConfig, layer_idx: int, intervention: InterventionConfig):
        super().__init__(config, layer_idx=layer_idx)
        self.intervention = intervention
        self.vis_start = 1
        self.vis_end = 577
        self.sink_positions: Optional[torch.Tensor] = None
        self.audit_enabled = False
        self.audit_records: list[dict[str, float | int | None]] = []

    def reset_audit(self) -> None:
        self.audit_records.clear()

    def capture_sink_candidates(self, hidden_states: torch.Tensor) -> None:
        """Official VisAttnSink DimProspector, evaluated once on prompt tokens."""
        if (self.intervention.method != "visattnsink" and not self.audit_enabled) or hidden_states.shape[1] <= self.vis_end:
            return
        dims = torch.tensor([2533, 1415], device=hidden_states.device, dtype=torch.long)
        rms = hidden_states.float()
        rms = rms * torch.rsqrt(rms.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        score = rms.abs().index_select(-1, dims).amax(dim=-1)[0]
        self.sink_positions = torch.nonzero(score > self.intervention.vas_tau, as_tuple=False).flatten()

    def _mass_summary(self, weights: torch.Tensor) -> dict[str, float | None]:
        selected = weights[:, :, -1:, :]
        kv_len = selected.shape[-1]
        vis_start = min(self.vis_start, kv_len)
        vis_end = min(self.vis_end, kv_len)

        visual = selected[..., vis_start:vis_end].sum(dim=-1).mean()
        prefix = selected[..., :vis_start].sum(dim=-1).mean()
        nonvisual = 1.0 - visual
        sink_mass = None
        if self.sink_positions is not None and self.sink_positions.numel() > 0:
            sinks = self.sink_positions[self.sink_positions < kv_len]
            if sinks.numel() > 0:
                sink_mass = selected.index_select(-1, sinks).sum(dim=-1).mean()

        return {
            "visual_mass": float(visual.detach().cpu()),
            "prefix_mass": float(prefix.detach().cpu()),
            "nonvisual_mass": float(nonvisual.detach().cpu()),
            "sink_mass": None if sink_mass is None else float(sink_mass.detach().cpu()),
        }

    def _record_audit(self, pre_weights: torch.Tensor, post_weights: torch.Tensor, q_len: int, kv_len: int) -> None:
        if not self.audit_enabled:
            return
        pre = self._mass_summary(pre_weights)
        post = self._mass_summary(post_weights)
        record: dict[str, float | int | None] = {
            "layer": int(self.layer_idx),
            "q_len": int(q_len),
            "kv_len": int(kv_len),
            "num_sinks": int(self.sink_positions.numel()) if self.sink_positions is not None else 0,
        }
        for key, value in pre.items():
            record[f"pre_{key}"] = value
        for key, value in post.items():
            record[f"post_{key}"] = value
            pre_value = pre[key]
            record[f"delta_{key}"] = None if value is None or pre_value is None else float(value - pre_value)
        self.audit_records.append(record)

    def _text_query_slice(self, q_len: int, kv_len: int) -> slice:
        query_start = kv_len - q_len
        local_start = max(self.vis_end - query_start, 0)
        return slice(local_start, q_len)

    def _apply_logit_intervention(self, logits: torch.Tensor) -> torch.Tensor:
        method = self.intervention.method
        q_len, kv_len = logits.shape[-2:]
        if self.vis_end > kv_len:
            return logits
        if method == "pai":
            visual = logits[:, :, -1:, self.vis_start:self.vis_end]
            logits[:, :, -1:, self.vis_start:self.vis_end] = (
                visual + visual.abs() * self.intervention.pai_alpha
            )
        elif method == "clearsight":
            q_slice = self._text_query_slice(q_len, kv_len)
            logits[:, :, q_slice, self.vis_start:self.vis_end] *= self.intervention.vaf_enhance
            logits[:, :, q_slice, :self.vis_start] *= self.intervention.vaf_suppress
        return logits

    def _apply_visattnsink(self, weights: torch.Tensor) -> torch.Tensor:
        if self.sink_positions is None or self.sink_positions.numel() == 0:
            return weights
        q_len, kv_len = weights.shape[-2:]
        q_slice = self._text_query_slice(q_len, kv_len)
        selected = weights[:, :, q_slice, :]
        if selected.numel() == 0:
            return weights
        sinks = self.sink_positions[self.sink_positions < kv_len]
        if sinks.numel() == 0:
            return weights
        vis_sinks = sinks[(sinks >= self.vis_start) & (sinks < self.vis_end)]
        if vis_sinks.numel() == 0:
            return weights
        visual = selected[..., self.vis_start:self.vis_end]
        visual_total = visual.sum(dim=-1).clamp_min(1e-8)
        sink_visual_mass = selected.index_select(-1, vis_sinks).sum(dim=-1)
        eligible = (sink_visual_mass / visual_total <= self.intervention.vas_rho) & (
            visual_total >= self.intervention.vas_visual_mass
        )
        if not eligible.any():
            return weights
        original_sink_mass = selected.index_select(-1, sinks).sum(dim=-1)
        redistributed = selected.clone()
        redistributed[..., sinks] *= self.intervention.vas_keep
        non_sink_visual = visual.clone()
        non_sink_visual[..., vis_sinks - self.vis_start] = 0
        denom = non_sink_visual.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        addition = (
            original_sink_mass * (1.0 - self.intervention.vas_keep)
        ).unsqueeze(-1) * non_sink_visual / denom
        redistributed[..., self.vis_start:self.vis_end] += addition
        selected = torch.where(eligible.unsqueeze(-1), redistributed, selected)
        weights[:, :, q_slice, :] = selected
        return weights

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if past_key_values is not None:
            key, value = past_key_values.update(
                key,
                value,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position},
            )
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        logits = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        q_len, kv_len = logits.shape[-2:]
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, : key.shape[-2]]
        pre_weights = None
        if self.audit_enabled:
            pre_weights = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        if self.intervention.method in {"pai", "clearsight"}:
            logits = self._apply_logit_intervention(logits)
        weights = nn.functional.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        if pre_weights is None:
            pre_weights = weights
        if self.intervention.method == "visattnsink":
            weights = self._apply_visattnsink(weights)
        self._record_audit(pre_weights, weights, q_len=q_len, kv_len=kv_len)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        output = self.o_proj(output.reshape(*input_shape, -1))
        return output, weights


def get_llm_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    raise RuntimeError(f"Cannot locate language model layers in {type(model).__name__}")


def _default_layers(method: str) -> tuple[int, int]:
    if method == "pai":
        return 2, 32
    if method == "clearsight":
        return 9, 15
    if method == "visattnsink":
        return 2, 32
    return 0, 0


def install_intervention(model, method: str, device, **kwargs) -> InterventionConfig:
    if method in {"vanilla", "vcd"}:
        return InterventionConfig(method=method, start_layer=0, end_layer=0)
    start, end = _default_layers(method)
    intervention = InterventionConfig(
        method=method,
        start_layer=int(kwargs.get("start_layer", start) if kwargs.get("start_layer") is not None else start),
        end_layer=int(kwargs.get("end_layer", end) if kwargs.get("end_layer") is not None else end),
        pai_alpha=float(kwargs.get("pai_alpha", 0.2)),
        vaf_enhance=float(kwargs.get("vaf_enhance", 1.15)),
        vaf_suppress=float(kwargs.get("vaf_suppress", 0.95)),
        vas_tau=float(kwargs.get("vas_tau", 20.0)),
        vas_rho=float(kwargs.get("vas_rho", 0.5)),
        vas_visual_mass=float(kwargs.get("vas_visual_mass", 0.2)),
        vas_keep=float(kwargs.get("vas_keep", 0.6)),
    )
    layers = get_llm_layers(model)
    for layer_idx in range(intervention.start_layer, min(intervention.end_layer, len(layers))):
        original = layers[layer_idx].self_attn
        adapter = AttentionIntervention(original.config, layer_idx, intervention)
        adapter.load_state_dict(original.state_dict())
        adapter.to(device=device, dtype=next(original.parameters()).dtype)
        layers[layer_idx].self_attn = adapter
        if method == "visattnsink":
            def _capture(_module, args, idx=layer_idx):
                current = get_llm_layers(model)[idx].self_attn
                current.capture_sink_candidates(args[0])

            layers[layer_idx].register_forward_pre_hook(_capture)
    return intervention


def set_visual_bounds(model, vis_start: int, vis_end: int) -> None:
    for layer in get_llm_layers(model):
        if isinstance(layer.self_attn, AttentionIntervention):
            layer.self_attn.vis_start = vis_start
            layer.self_attn.vis_end = vis_end
