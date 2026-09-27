"""Small, explicit adapters for supported image-to-text model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


SUPPORTED_MODEL_TYPES = {"llava", "llava_next", "qwen2_5_vl", "qwen3_vl"}


def model_type(value: Any) -> str:
    config = getattr(value, "config", value)
    name = str(getattr(config, "model_type", ""))
    if name not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"unsupported model_type {name!r}; expected one of "
            f"{sorted(SUPPORTED_MODEL_TYPES)}"
        )
    return name


def decoder_layers(model: Any):
    """Return the text decoder layers for every supported family."""

    return model.model.language_model.layers


def text_config(model: Any):
    return model.config.text_config


def set_attention_implementation(model: Any, name: str) -> None:
    """Select the text decoder attention backend."""

    model.config._attn_implementation = name
    model.config.text_config._attn_implementation = name
    model.model.language_model.config._attn_implementation = name


def effective_prompt(processor: Any, family: str, instruction: str) -> str:
    """Build the exact model-facing prompt from a human-readable instruction."""

    if family in {"llava", "llava_next"}:
        return instruction
    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def processor_inputs(processor: Any, family: str, image: Any, prompt: str):
    """Create a batch of one while preserving each processor's public API."""

    if family in {"llava", "llava_next"}:
        return processor(images=image, text=prompt, return_tensors="pt")
    return processor(images=[image], text=[prompt], return_tensors="pt")


def image_token_id(model: Any) -> int:
    family = model_type(model)
    attribute = "image_token_index" if family in {"llava", "llava_next"} else "image_token_id"
    return int(getattr(model.config, attribute))


@dataclass(frozen=True)
class VisualLayout:
    positions: torch.Tensor
    grid: tuple[int, int] | None
    expanded_count: int
    scope: str


def visual_layout(
    model: Any,
    inputs: dict,
    input_ids: torch.Tensor,
    *,
    scope: str,
    llava_base_tokens: int,
) -> VisualLayout:
    """Resolve visual key positions and their native post-merge spatial grid."""

    family = model_type(model)
    positions = torch.where(input_ids[0] == image_token_id(model))[0]
    if positions.numel() == 0:
        raise ValueError("processor output contains no expanded image tokens")
    expanded = int(positions.numel())
    if family in {"llava", "llava_next"}:
        if expanded < llava_base_tokens:
            raise ValueError(
                f"LLaVA-NeXT emitted {expanded} image tokens, fewer than "
                f"the {llava_base_tokens} base-grid tokens"
            )
        if scope == "base":
            return VisualLayout(
                positions=positions[:llava_base_tokens],
                grid=(24, 24),
                expanded_count=expanded,
                scope=scope,
            )
        return VisualLayout(
            positions=positions,
            grid=None,
            expanded_count=expanded,
            scope=scope,
        )
    if scope != "all":
        raise ValueError(f"{family} supports only --visual-token-scope all")
    grid_thw = inputs.get("image_grid_thw")
    if not isinstance(grid_thw, torch.Tensor) or grid_thw.shape != (1, 3):
        raise ValueError("Qwen image replay requires one image_grid_thw row")
    temporal, raw_height, raw_width = [int(value) for value in grid_thw[0].tolist()]
    merge = int(model.config.vision_config.spatial_merge_size)
    if temporal != 1 or raw_height % merge or raw_width % merge:
        raise ValueError(f"invalid Qwen image grid: {grid_thw.tolist()}")
    grid = (raw_height // merge, raw_width // merge)
    if grid[0] * grid[1] != expanded:
        raise ValueError(
            f"Qwen visual grid {grid} has {grid[0] * grid[1]} cells but "
            f"input_ids contain {expanded} image tokens"
        )
    return VisualLayout(
        positions=positions,
        grid=grid,
        expanded_count=expanded,
        scope=scope,
    )


def repeat_kv_for_family(family: str):
    if family == "llava":
        from transformers.models.llama.modeling_llama import repeat_kv
    elif family == "llava_next":
        from transformers.models.mistral.modeling_mistral import repeat_kv
    elif family == "qwen2_5_vl":
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
    elif family == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
    else:  # pragma: no cover - guarded by model_type
        raise ValueError(f"unsupported model family: {family}")
    return repeat_kv
