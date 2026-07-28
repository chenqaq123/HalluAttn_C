"""Formula-faithful attention ports used by the open-caption matrix.

The released PAI, ClearSight, DIML, and AllPath implementations patch old
LLaMA/Qwen model forks.  This module applies the disclosed formulas at the
shared eager-attention boundary in current Hugging Face model classes.  It is
deliberately architecture-light: callers provide the visual-token positions
and the text-decoder module to patch.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn


@dataclass
class FormulaPortConfig:
    """Frozen parameters for one formula-faithful intervention."""

    method: str
    start_layer: int
    end_layer: int
    visual_positions: Sequence[int]
    pai_alpha: float = 0.5
    clearsight_enhance: float = 1.15
    clearsight_suppress: float = 0.95
    diml_alpha: float = 0.5
    allpath_good_heads: Mapping[int, Sequence[int]] = field(default_factory=dict)
    allpath_hallu_heads: Mapping[int, Sequence[int]] = field(default_factory=dict)
    allpath_in_scale: float = 2.0
    allpath_de_scale: float = 0.0

    def validate(self) -> None:
        if self.method not in {"vanilla", "pai", "clearsight", "diml", "allpath"}:
            raise ValueError(f"Unknown formula port: {self.method}")
        if self.start_layer < 0 or self.end_layer < self.start_layer:
            raise ValueError("Invalid layer interval")
        if any(int(position) < 0 for position in self.visual_positions):
            raise ValueError("Visual positions must be non-negative")


@dataclass
class FormulaPortAudit:
    """Runtime evidence that the intended tensor and token span were touched."""

    method: str
    requested_start_layer: int
    requested_end_layer: int
    visual_position_count: int
    calls: int = 0
    modified_calls: int = 0
    layers_seen: set[int] = field(default_factory=set)
    layers_modified: set[int] = field(default_factory=set)
    min_visual_position: int | None = None
    max_visual_position: int | None = None
    max_abs_tensor_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layers_seen"] = sorted(self.layers_seen)
        result["layers_modified"] = sorted(self.layers_modified)
        return result


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, kv_heads, groups, length, head_dim)
        .reshape(batch, kv_heads * groups, length, head_dim)
    )


class FormulaAttentionPort:
    """Context manager replacing one module's eager attention function."""

    def __init__(
        self,
        modeling_module: Any,
        text_attention_classes: type[nn.Module] | tuple[type[nn.Module], ...],
        config: FormulaPortConfig,
    ):
        config.validate()
        self.modeling_module = modeling_module
        self.text_attention_classes = text_attention_classes
        self.config = config
        self.original: Callable[..., Any] | None = None
        positions = sorted({int(position) for position in config.visual_positions})
        self.visual_positions = torch.tensor(positions, dtype=torch.long)
        self.active = True
        self.audit = FormulaPortAudit(
            method=config.method,
            requested_start_layer=config.start_layer,
            requested_end_layer=config.end_layer,
            visual_position_count=len(positions),
            min_visual_position=positions[0] if positions else None,
            max_visual_position=positions[-1] if positions else None,
        )

    def __enter__(self) -> "FormulaAttentionPort":
        if self.original is not None:
            raise RuntimeError("FormulaAttentionPort cannot be entered twice")
        self.original = self.modeling_module.eager_attention_forward
        self.modeling_module.eager_attention_forward = self._forward
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.original is not None:
            self.modeling_module.eager_attention_forward = self.original
            self.original = None

    @contextmanager
    def suspended(self) -> Iterator[None]:
        previous = self.active
        self.active = False
        try:
            yield
        finally:
            self.active = previous

    def _selected(self, module: nn.Module) -> tuple[bool, int]:
        if not isinstance(module, self.text_attention_classes):
            return False, -1
        layer = int(getattr(module, "layer_idx", -1))
        self.audit.calls += 1
        self.audit.layers_seen.add(layer)
        return (
            self.active
            and self.config.method != "vanilla"
            and self.config.start_layer <= layer < self.config.end_layer,
            layer,
        )

    def _valid_visual_positions(self, kv_len: int, device: torch.device) -> torch.Tensor:
        positions = self.visual_positions
        if positions.numel() == 0:
            return positions.to(device)
        positions = positions[positions < kv_len]
        return positions.to(device)

    def _modify_logits(
        self,
        logits: torch.Tensor,
        layer: int,
        visual: torch.Tensor,
        *,
        after_mask: bool,
    ) -> tuple[torch.Tensor, bool]:
        method = self.config.method
        q_len, kv_len = logits.shape[-2:]
        changed = False

        if method == "clearsight" and not after_mask:
            # Official VAF modifies only text queries after the image block in
            # prefill, and the sole query row during cached decoding.
            query_start = kv_len - q_len
            last_visual = int(visual.max().item())
            query_rows = torch.arange(q_len, device=logits.device)
            query_rows = query_rows[query_start + query_rows > last_visual]
            if query_rows.numel():
                before = logits[:, :, query_rows, :].detach()
                logits[:, :, query_rows[:, None], visual] *= (
                    self.config.clearsight_enhance
                )
                first_visual = int(visual.min().item())
                if first_visual > 0:
                    logits[:, :, query_rows, :first_visual] *= (
                        self.config.clearsight_suppress
                    )
                delta = (logits[:, :, query_rows, :] - before).abs().max()
                self.audit.max_abs_tensor_delta = max(
                    self.audit.max_abs_tensor_delta, float(delta.cpu())
                )
                changed = bool(delta > 0)

        if after_mask and method in {"pai", "diml"}:
            before = logits[:, :, -1:, visual].detach()
            selected = logits[:, :, -1:, visual]
            if method == "pai":
                selected = selected + self.config.pai_alpha * selected.abs()
            else:
                selected = selected + self.config.diml_alpha * selected.abs().mean(
                    dim=1, keepdim=True
                )
            logits[:, :, -1:, visual] = selected
            delta = (selected - before).abs().max()
            self.audit.max_abs_tensor_delta = max(
                self.audit.max_abs_tensor_delta, float(delta.cpu())
            )
            changed = bool(delta > 0)

        if changed:
            self.audit.modified_calls += 1
            self.audit.layers_modified.add(layer)
        return logits, changed

    def _apply_allpath(
        self, head_output: torch.Tensor, layer: int
    ) -> tuple[torch.Tensor, bool]:
        # head_output is (batch, heads, query, head_dim).  Scaling before the
        # output projection is algebraically identical to scaling each
        # per-head projected contribution in the official implementation.
        head_count = head_output.shape[1]
        scale = torch.ones(
            head_count, dtype=head_output.dtype, device=head_output.device
        )
        good = [
            int(head)
            for head in self.config.allpath_good_heads.get(layer, ())
            if 0 <= int(head) < head_count
        ]
        hallu = [
            int(head)
            for head in self.config.allpath_hallu_heads.get(layer, ())
            if 0 <= int(head) < head_count and int(head) not in good
        ]
        if good:
            scale[good] = self.config.allpath_in_scale
        if hallu:
            scale[hallu] = self.config.allpath_de_scale
        changed = bool(torch.any(scale != 1))
        if changed:
            before = head_output.detach()
            head_output = head_output * scale[None, :, None, None]
            delta = (head_output - before).abs().max()
            self.audit.max_abs_tensor_delta = max(
                self.audit.max_abs_tensor_delta, float(delta.cpu())
            )
            self.audit.modified_calls += 1
            self.audit.layers_modified.add(layer)
        return head_output, changed

    def _forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        selected, layer = self._selected(module)
        if not selected:
            if self.original is None:
                raise RuntimeError("Formula port is not installed")
            return self.original(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )

        key_states = _repeat_kv(key, int(module.num_key_value_groups))
        value_states = _repeat_kv(value, int(module.num_key_value_groups))
        logits = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        visual = self._valid_visual_positions(key_states.shape[-2], logits.device)
        if visual.numel() == 0:
            if self.original is None:
                raise RuntimeError("Formula port is not installed")
            return self.original(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )

        logits, _ = self._modify_logits(
            logits, layer, visual, after_mask=False
        )
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, : key_states.shape[-2]]
        logits, _ = self._modify_logits(
            logits, layer, visual, after_mask=True
        )

        weights = nn.functional.softmax(
            logits, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        weights = nn.functional.dropout(
            weights, p=dropout, training=module.training
        )
        head_output = torch.matmul(weights, value_states)
        if self.config.method == "allpath":
            head_output, _ = self._apply_allpath(head_output, layer)
        output = head_output.transpose(1, 2).contiguous()
        return output, weights


def _eos_ids(model, tokenizer) -> set[int]:
    values: list[Any] = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ]
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    return result


def _initialize_generation_branch(
    model, inputs: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Initialize the same cache/position state used by GenerationMixin."""

    input_ids = inputs["input_ids"]
    model_kwargs = {
        key: value for key, value in inputs.items() if key != "input_ids"
    }
    model_kwargs["use_cache"] = True
    # GenerationMixin.generate() requests only the final-token logits when the
    # model supports that optimization.  Besides saving memory, this is
    # semantically required for models such as LLaVA-NeXT-Video whose
    # prepare_inputs_for_generation() default is logits_to_keep=None: passing
    # None indexes the hidden states with a new axis and returns rank-4 logits.
    if (
        model._supports_logits_to_keep()
        and "logits_to_keep" not in model_kwargs
    ):
        model_kwargs["logits_to_keep"] = 1
    # Qwen-VL requires cache_position so prepare_inputs_for_generation can
    # construct multimodal RoPE indices and propagate rope_deltas. Directly
    # calling forward happened to match LLaVA but silently diverged on Qwen.
    model_kwargs = model._get_initial_cache_position(
        input_ids.shape[-1], input_ids.device, model_kwargs
    )
    return input_ids, model_kwargs


def _generation_branch_step(
    model,
    sequence: torch.Tensor,
    model_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    rope_owner = getattr(model, "model", None)
    branch_rope_deltas = model_kwargs.pop(
        "_sinkdetect_rope_deltas", None
    )
    if (
        branch_rope_deltas is not None
        and rope_owner is not None
        and hasattr(rope_owner, "rope_deltas")
    ):
        rope_owner.rope_deltas = branch_rope_deltas
    model_inputs = model.prepare_inputs_for_generation(
        sequence, **model_kwargs
    )
    output = model(**model_inputs, return_dict=True)
    model_kwargs = model._update_model_kwargs_for_generation(
        output,
        model_kwargs,
        is_encoder_decoder=model.config.is_encoder_decoder,
    )
    if rope_owner is not None and hasattr(rope_owner, "rope_deltas"):
        model_kwargs["_sinkdetect_rope_deltas"] = rope_owner.rope_deltas
    return output.logits[:, -1, :], model_kwargs


def generate_pai_greedy(
    model,
    tokenizer,
    conditional_inputs: Mapping[str, Any],
    unconditional_inputs: Mapping[str, Any],
    port: FormulaAttentionPort,
    *,
    max_new_tokens: int = 512,
    guidance_scale: float = 2.0,
    plausibility_cutoff: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Deterministic PAI CFG, matching the official log-softmax formula."""

    conditional_ids, conditional_kwargs = _initialize_generation_branch(
        model, conditional_inputs
    )
    unconditional_ids, unconditional_kwargs = _initialize_generation_branch(
        model, unconditional_inputs
    )
    current_conditional = conditional_ids
    current_unconditional = unconditional_ids
    generated: list[torch.Tensor] = []
    eos = _eos_ids(model, tokenizer)
    stopped_by_eos = False

    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            conditional_logits, conditional_kwargs = _generation_branch_step(
                model,
                current_conditional,
                conditional_kwargs,
            )
            with port.suspended():
                unconditional_logits, unconditional_kwargs = (
                    _generation_branch_step(
                        model,
                        current_unconditional,
                        unconditional_kwargs,
                    )
                )

            scores = torch.log_softmax(conditional_logits.float(), dim=-1)
            unconditional_scores = torch.log_softmax(
                unconditional_logits.float(), dim=-1
            )
            guided = (
                float(guidance_scale) * (scores - unconditional_scores)
                + unconditional_scores
            )
            if plausibility_cutoff > 0:
                cutoff = (
                    math.log(float(plausibility_cutoff))
                    + scores.max(dim=-1, keepdim=True).values
                )
                guided = guided.masked_fill(scores < cutoff, -torch.inf)
            next_token = torch.argmax(guided, dim=-1)
            if int(next_token[0]) in eos:
                generated.append(next_token)
                stopped_by_eos = True
                break
            generated.append(next_token)
            current_conditional = torch.cat(
                [current_conditional, next_token[:, None]], dim=-1
            )
            current_unconditional = torch.cat(
                [current_unconditional, next_token[:, None]], dim=-1
            )

    if generated:
        continuation = torch.stack(generated, dim=1)
    else:
        continuation = conditional_ids.new_empty((conditional_ids.shape[0], 0))
    full_ids = torch.cat([conditional_ids, continuation], dim=-1)
    return full_ids, {
        "generated_tokens": int(continuation.shape[-1]),
        "stopped_by_eos": stopped_by_eos,
        "guidance_scale": float(guidance_scale),
        "plausibility_cutoff": float(plausibility_cutoff),
    }


def _diffusion_noise(
    tensor: torch.Tensor,
    *,
    noise_step: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    step = max(0, min(int(noise_step), 999))
    betas = torch.linspace(
        -6, 6, 1000, device=tensor.device, dtype=torch.float32
    )
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alpha_products = torch.cumprod(1.0 - betas, dim=0)
    alpha = torch.sqrt(alpha_products[step]).to(tensor.dtype)
    sigma = torch.sqrt(1.0 - alpha_products[step]).to(tensor.dtype)
    noise = torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )
    return alpha * tensor + sigma * noise


def _offload_dynamic_cache(cache) -> bool:
    """Temporarily move a Transformers dynamic KV cache to CPU."""

    layers = getattr(cache, "layers", None)
    if not layers:
        return False
    initialized = False
    for layer in layers:
        if getattr(layer, "is_initialized", False):
            layer.offload()
            initialized = True
    if initialized and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return initialized


def _prefetch_dynamic_cache(cache) -> None:
    """Move a temporarily offloaded dynamic KV cache back to its device."""

    layers = getattr(cache, "layers", None)
    if not layers:
        return
    for layer in layers:
        if getattr(layer, "is_initialized", False):
            layer.prefetch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_vcd_greedy_multimodal(
    model,
    tokenizer,
    inputs: Mapping[str, Any],
    *,
    max_new_tokens: int = 512,
    alpha: float = 0.5,
    beta: float = 0.1,
    noise_step: int = 500,
    seed: int = 927,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Video/image VCD with every processed visual tensor independently noised."""

    visual_keys = [
        key
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
        and key in {"pixel_values", "pixel_values_videos"}
    ]
    if not visual_keys:
        raise ValueError("VCD requires pixel_values or pixel_values_videos")
    generator_device = inputs[visual_keys[0]].device
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    distorted = dict(inputs)
    for key in visual_keys:
        distorted[key] = _diffusion_noise(
            inputs[key], noise_step=noise_step, generator=generator
        )

    input_ids, clean_kwargs = _initialize_generation_branch(model, inputs)
    distorted_ids, distorted_kwargs = _initialize_generation_branch(
        model, distorted
    )
    current = input_ids
    current_distorted = distorted_ids
    generated: list[torch.Tensor] = []
    eos = _eos_ids(model, tokenizer)
    stopped_by_eos = False
    branch_calls = 0
    clean_initial_cache_offloads = 0

    with torch.inference_mode():
        for _ in range(int(max_new_tokens)):
            logits, clean_kwargs = _generation_branch_step(
                model,
                current,
                clean_kwargs,
            )
            # On long video prompts the second full-sequence branch can exceed
            # a 24 GB card solely because the first branch's KV cache remains
            # resident.  Move that cache to CPU only for the distorted
            # branch's initial prompt pass, then restore it unchanged.  Both
            # caches stay on GPU for all single-token decoding steps.
            restore_clean_cache = (
                branch_calls == 0
                and _offload_dynamic_cache(
                    clean_kwargs.get("past_key_values")
                )
            )
            clean_initial_cache_offloads += int(restore_clean_cache)
            distorted_logits, distorted_kwargs = _generation_branch_step(
                model,
                current_distorted,
                distorted_kwargs,
            )
            if restore_clean_cache:
                _prefetch_dynamic_cache(
                    clean_kwargs.get("past_key_values")
                )
            branch_calls += 1
            adjusted = (1.0 + float(alpha)) * logits - float(alpha) * (
                distorted_logits
            )
            if beta > 0:
                cutoff = (
                    math.log(float(beta))
                    + logits.max(dim=-1, keepdim=True).values
                )
                adjusted = adjusted.masked_fill(logits < cutoff, -torch.inf)
            next_token = torch.argmax(adjusted, dim=-1)
            if int(next_token[0]) in eos:
                generated.append(next_token)
                stopped_by_eos = True
                break
            generated.append(next_token)
            current = torch.cat([current, next_token[:, None]], dim=-1)
            current_distorted = torch.cat(
                [current_distorted, next_token[:, None]], dim=-1
            )

    if generated:
        continuation = torch.stack(generated, dim=1)
    else:
        continuation = input_ids.new_empty((input_ids.shape[0], 0))
    return torch.cat([input_ids, continuation], dim=-1), {
        "generated_tokens": int(continuation.shape[-1]),
        "stopped_by_eos": stopped_by_eos,
        "alpha": float(alpha),
        "beta": float(beta),
        "noise_step": int(noise_step),
        "perturbed_branch_calls": branch_calls,
        "visual_tensor_keys": visual_keys,
        "clean_initial_cache_cpu_offloads": clean_initial_cache_offloads,
    }
