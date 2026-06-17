"""Decoding-time mitigation utilities."""

from __future__ import annotations

import math
from typing import Any

import torch


def add_diffusion_noise(pixel_values: torch.Tensor, noise_step: int) -> torch.Tensor:
    """Apply the diffusion-style image corruption used by VCD.

    This follows the official VCD utility: a 1000-step sigmoid beta schedule and
    direct sampling from q(x_t | x_0). The input is the processed LLaVA image
    tensor, matching the point where the official LLaVA integration applies the
    corruption.
    """
    step = max(0, min(int(noise_step), 999))
    device = pixel_values.device
    betas = torch.linspace(-6, 6, 1000, device=device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alpha = torch.sqrt(alphas_prod[step]).to(dtype=pixel_values.dtype)
    sigma = torch.sqrt(1.0 - alphas_prod[step]).to(dtype=pixel_values.dtype)
    return alpha * pixel_values + sigma * torch.randn_like(pixel_values)


def _eos_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for value in [getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "sep_token_id", None)]:
        if value is not None:
            ids.add(int(value))
    return ids


def _forward_next_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    past_key_values: Any | None = None,
):
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    outputs = model(**kwargs)
    logits = outputs.logits[:, -1, :]
    return logits, outputs.past_key_values


def generate_vcd_greedy(
    model,
    processor,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    alpha: float = 0.5,
    beta: float = 0.1,
    noise_step: int = 500,
) -> str:
    """Generate with a deterministic VCD variant for controlled POPE/CHAIR runs.

    Official VCD contrasts logits from the original image and a distorted image:

        (1 + alpha) * logits(image) - alpha * logits(noisy_image)

    and applies an adaptive plausibility constraint based on the original-image
    logits. The official code samples from this distribution; this project uses
    greedy decoding for the other mitigation baselines, so this port takes the
    argmax after the same contrastive adjustment.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pixel_values = inputs["pixel_values"]
    pixel_values_cd = add_diffusion_noise(pixel_values, noise_step=noise_step)

    eos_ids = _eos_ids(processor.tokenizer)
    generated: list[int] = []
    beta = float(beta)
    alpha = float(alpha)
    past = None
    past_cd = None
    current_input_ids = input_ids

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, past = _forward_next_logits(
                model,
                current_input_ids,
                attention_mask,
                pixel_values=pixel_values if past is None else None,
                past_key_values=past,
            )
            logits_cd, past_cd = _forward_next_logits(
                model,
                current_input_ids,
                attention_mask,
                pixel_values=pixel_values_cd if past_cd is None else None,
                past_key_values=past_cd,
            )
            cd_logits = (1.0 + alpha) * logits - alpha * logits_cd
            if beta > 0:
                cutoff = math.log(beta) + logits.max(dim=-1, keepdim=True).values
                masked = cd_logits.masked_fill(logits < cutoff, -torch.inf)
                if torch.isfinite(masked).any(dim=-1).all():
                    cd_logits = masked
            next_token = torch.argmax(cd_logits, dim=-1)
            token_id = int(next_token[0].detach().cpu())
            if token_id in eos_ids:
                break
            generated.append(token_id)
            current_input_ids = next_token[:, None]
            attention_mask = torch.cat([attention_mask, torch.ones_like(current_input_ids)], dim=-1)

    if not generated:
        return ""
    return processor.decode(generated, skip_special_tokens=True).strip()
