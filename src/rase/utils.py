"""Explicit local model and processor loading for supported VLM families."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from .model_adapter import model_type


def load_model_and_processor(
    model_path: str,
    device: torch.device,
    *,
    attn_implementation: str,
    dtype: torch.dtype,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    use_fast_processor: bool = False,
) -> tuple[Any, Any]:
    """Load an explicitly named local LLaVA-NeXT or Qwen-VL checkpoint."""

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    family = model_type(config)
    processor_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "use_fast": use_fast_processor,
    }
    if family not in {"llava", "llava_next"}:
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    if family in {"llava", "llava_next"}:
        vision_config = getattr(model.config, "vision_config", None)
        if getattr(processor, "patch_size", None) is None and vision_config is not None:
            processor.patch_size = getattr(vision_config, "patch_size", None)
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            strategy = getattr(model.config, "vision_feature_select_strategy", None)
            if strategy is None:
                raise ValueError(
                    "checkpoint does not declare vision_feature_select_strategy"
                )
            processor.vision_feature_select_strategy = strategy
    model.eval()
    return model, processor


def torch_dtype(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {name}") from error
