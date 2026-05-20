"""
DetectionAdapter: single-branch attention adapter for hallucination detection.

Computes both original and purified attention maps during a standard forward pass.
The model output is unchanged — only the purified attention is stored as an
attribute for post-hoc extraction.

Based on the three-branch AttnAdapter from LogitsAnalysis, simplified for
single-branch detection.
"""

from typing import Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    eager_attention_forward,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.processing_utils import Unpack

from .sink_utils import (
    VIS_START as DEFAULT_VIS_START,
    VIS_END as DEFAULT_VIS_END,
    auto_detect_sinks,
    purify_attention,
)

# Layers for which per-head attention is stored (key layers identified by
# analysis: early layers 0-1 carry strongest signal, layers 10/22/31 cover
# mid and late). Per-head storage is expensive (~32× memory vs mean-over-heads),
# so we restrict it to these layers.
PER_HEAD_LAYERS: list[int] = [0, 1, 10, 22, 31]


class DetectionAdapter(LlamaAttention):
    """Single-branch attention adapter that computes both original and purified
    attention maps for hallucination detection.

    Per-iteration knobs (set externally before each forward):
        vis_start, vis_end — bounds of the visual token span.
        text_start         — first text query position (defaults to vis_end).

    After a forward pass, the following attributes are populated:
        last_original_attn  — (seq_len, seq_len) mean-over-heads original attn, on device
        last_sink_only_attn — sink removal without top-mass masking, on device
        last_topmass_only_attn — top-mass masking without sink removal, on device
        last_purified_attn — sink removal plus top-mass masking, on device
        last_sink_stats     — dict with layer_idx, sink_count, sink_positions, ratio
    """

    def __init__(self, config: LlamaConfig, layer_idx: int, purify_ratio: float = 0.5):
        super().__init__(config, layer_idx=layer_idx)
        self.purify_ratio = purify_ratio
        self.vis_start: int = DEFAULT_VIS_START
        self.vis_end: int = DEFAULT_VIS_END
        self.text_start: int = DEFAULT_VIS_END
        self.last_original_attn: Optional[torch.Tensor] = None
        self.last_sink_only_attn: Optional[torch.Tensor] = None
        self.last_topmass_only_attn: Optional[torch.Tensor] = None
        self.last_purified_attn: Optional[torch.Tensor] = None
        self.last_per_head_attn: Optional[torch.Tensor] = None  # (n_heads, seq, seq) CPU, only for PER_HEAD_LAYERS
        self.last_sink_stats: dict = {}

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ── Capture pre-RoPE key norms for sink detection ──
        # Per-head L2 norm, averaged over (batch, heads)
        key_norms = key_states.float().norm(p=2, dim=-1).mean(dim=(0, 1))  # (seq_len,)
        all_pos = torch.arange(key_norms.size(0), device=key_norms.device)
        vis_pos = all_pos[(all_pos >= self.vis_start) & (all_pos < self.vis_end)]
        sink_pos = auto_detect_sinks(key_norms, vis_pos) if vis_pos.numel() > 0 else vis_pos

        # ── Apply RoPE ──
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ── Handle KV cache ──
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # ── Compute original attention via standard path ──
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # ── Store original attention (mean over heads, on CPU) ──
        if attn_weights is not None:
            # attn_weights: (batch, n_heads, seq_q, seq_k). Keep the reduced
            # matrices on device; moving every layer to CPU dominates runtime.
            orig_mean = attn_weights.mean(dim=1).squeeze(0).detach()
            self.last_original_attn = orig_mean

            # ── Compute and store purified attention variants ──
            self._sink_pos = sink_pos.detach()
            self._orig_attn_for_purify = orig_mean

            # Sink removal only: isolates the effect of removing sinks.
            sink_only = purify_attention(
                orig_mean,
                sink_pos.detach(),
                prompt_end_idx=self.text_start,
                vis_start=self.vis_start,
                vis_end=self.vis_end,
                ratio=self.purify_ratio,
                remove_sinks=True,
                apply_top_mass=False,
            )
            self.last_sink_only_attn = sink_only

            # Top-mass visual masking only: isolates the effect of top-mass.
            topmass_only = purify_attention(
                orig_mean,
                sink_pos.detach(),
                prompt_end_idx=self.text_start,
                vis_start=self.vis_start,
                vis_end=self.vis_end,
                ratio=self.purify_ratio,
                remove_sinks=False,
                apply_top_mass=True,
            )
            self.last_topmass_only_attn = topmass_only

            # Full purification: sink removal plus top-mass visual masking.
            purified = purify_attention(
                orig_mean,
                sink_pos.detach(),
                prompt_end_idx=self.text_start,
                vis_start=self.vis_start,
                vis_end=self.vis_end,
                ratio=self.purify_ratio,
                remove_sinks=True,
                apply_top_mass=True,
            )
            self.last_purified_attn = purified

            # ── Store per-head attention for key layers ──
            if self.layer_idx in PER_HEAD_LAYERS:
                # (n_heads, seq_q, seq_k), squeeze batch dim, keep on device
                self.last_per_head_attn = attn_weights.squeeze(0).detach()
            else:
                self.last_per_head_attn = None
        else:
            self.last_original_attn = None
            self.last_sink_only_attn = None
            self.last_topmass_only_attn = None
            self.last_purified_attn = None
            self.last_per_head_attn = None
            self._sink_pos = None

        # ── Store sink stats ──
        self.last_sink_stats = {
            "layer_idx": int(self.layer_idx),
            "sink_count": int(sink_pos.numel()),
            "sink_positions": sink_pos.detach().cpu().tolist() if sink_pos.numel() > 0 else [],
            "ratio": float(self.purify_ratio),
        }

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def get_llm_layers(model) -> torch.nn.ModuleList:
    """Get the LLM decoder layers from a LlavaForConditionalGeneration model.

    Handles transformers 5.0.0 path change:
      old: model.language_model.layers
      new: model.model.language_model.layers
    """
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        return model.language_model.layers
    raise RuntimeError(
        f"Cannot find LLM layers in {type(model).__name__}. "
        f"Available attributes: {dir(model)}"
    )


def inject_detection_adapter(
    model,
    start_layer: int,
    end_layer: int,
    device: torch.device,
    ratio: float = 0.5,
):
    """Replace LlamaAttention layers with DetectionAdapter instances.

    The adapter loads the existing attention weights, so the model output
    remains identical to the unmodified model.
    """
    layers = get_llm_layers(model)
    for idx in range(start_layer, min(end_layer, len(layers))):
        layer = layers[idx]
        adapter = DetectionAdapter(
            layer.self_attn.config,
            layer_idx=idx,
            purify_ratio=ratio,
        )
        adapter.load_state_dict(layer.self_attn.state_dict())
        adapter = adapter.to(device=device, dtype=next(model.parameters()).dtype)
        layer.self_attn = adapter
