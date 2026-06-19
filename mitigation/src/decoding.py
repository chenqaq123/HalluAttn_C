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
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = inputs_embeds
    elif input_ids is not None:
        kwargs["input_ids"] = input_ids
    else:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
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


def _symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    probs_a = torch.softmax(logits_a.float(), dim=-1).clamp_min(1e-10)
    probs_b = torch.softmax(logits_b.float(), dim=-1).clamp_min(1e-10)
    kl_ab = torch.sum(probs_a * (probs_a.log() - probs_b.log()), dim=-1)
    kl_ba = torch.sum(probs_b * (probs_b.log() - probs_a.log()), dim=-1)
    return 0.5 * (kl_ab + kl_ba)


def generate_nolan_greedy(
    model,
    processor,
    inputs: dict[str, torch.Tensor],
    text_only_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    alpha_scale: float = 0.8,
) -> tuple[str, dict[str, Any]]:
    """Generate with a deterministic NoLan-compatible decoding rule.

    The official NoLan code monkey-patches ``GenerationMixin.sample`` and samples
    from logits adjusted by a text-only branch. This guarded port keeps this
    repository's deterministic POPE protocol by taking the greedy token after the
    same adaptive contrast:

        alpha = alpha_scale * (tanh(1 / symmetric_kl) + 1)
        logits = (1 + alpha) * logits_mm - alpha * logits_text

    It should be reported as a compatible port unless the official NoLan stack is
    used directly.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pixel_values = inputs["pixel_values"]

    text_input_ids = text_only_inputs["input_ids"]
    text_attention_mask = text_only_inputs.get("attention_mask")
    if text_attention_mask is None:
        text_attention_mask = torch.ones_like(text_input_ids)

    eos_ids = _eos_ids(processor.tokenizer)
    generated: list[int] = []
    alpha_values: list[float] = []
    kl_values: list[float] = []
    past = None
    past_text = None
    current_input_ids = input_ids
    current_text_input_ids = text_input_ids

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, past = _forward_next_logits(
                model,
                current_input_ids,
                attention_mask,
                pixel_values=pixel_values if past is None else None,
                past_key_values=past,
            )
            logits_text, past_text = _forward_next_logits(
                model,
                current_text_input_ids,
                text_attention_mask,
                pixel_values=None,
                past_key_values=past_text,
            )
            kl = _symmetric_kl(logits, logits_text).clamp_min(1e-6)
            alpha = (torch.tanh(1.0 / kl) + 1.0) * float(alpha_scale)
            adjusted = (1.0 + alpha[:, None]) * logits - alpha[:, None] * logits_text
            next_token = torch.argmax(adjusted, dim=-1)
            token_id = int(next_token[0].detach().cpu())
            alpha_values.append(float(alpha[0].detach().cpu()))
            kl_values.append(float(kl[0].detach().cpu()))
            if token_id in eos_ids:
                break
            generated.append(token_id)
            current_input_ids = next_token[:, None]
            current_text_input_ids = next_token[:, None]
            attention_mask = torch.cat([attention_mask, torch.ones_like(current_input_ids)], dim=-1)
            text_attention_mask = torch.cat(
                [text_attention_mask, torch.ones_like(current_text_input_ids)],
                dim=-1,
            )

    text = processor.decode(generated, skip_special_tokens=True).strip() if generated else ""
    info = {
        "decode": "greedy_compatible_port",
        "alpha_scale": float(alpha_scale),
        "mean_alpha": sum(alpha_values) / len(alpha_values) if alpha_values else 0.0,
        "mean_symmetric_kl": sum(kl_values) / len(kl_values) if kl_values else 0.0,
        "num_steps": len(alpha_values),
    }
    return text, info


class _VisionHookStore:
    def __init__(self):
        self.query: torch.Tensor | None = None
        self.key: torch.Tensor | None = None

    def hook_query(self, _module, _args, output):
        self.query = output

    def hook_key(self, _module, _args, output):
        self.key = output


def _vision_feature_layer(model) -> int:
    layer = getattr(model.config, "vision_feature_layer", -2)
    if isinstance(layer, list):
        if len(layer) != 1:
            raise ValueError("DAMRO controlled port expects one vision feature layer")
        return int(layer[0])
    return int(layer)


def _select_damro_outlier_features(
    model,
    pixel_values: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, list[int]]:
    """Return projected CLIP outlier-token features selected by CLS attention.

    The official DAMRO implementation hooks the vision transformer's query/key
    projections at the selected CLIP layer, ranks spatial tokens by CLS-to-patch
    attention, projects only the top-k patch features, and uses them as the
    negative visual branch for contrastive decoding.
    """
    llava = model.model
    feature_layer = _vision_feature_layer(model)
    strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    if strategy != "default":
        raise ValueError(f"DAMRO controlled port expects default vision feature selection, got {strategy!r}")

    vision_layers = llava.vision_tower.vision_model.encoder.layers
    layer = vision_layers[feature_layer]
    store = _VisionHookStore()
    handle_q = layer.self_attn.q_proj.register_forward_hook(store.hook_query)
    handle_k = layer.self_attn.k_proj.register_forward_hook(store.hook_key)
    try:
        image_outputs = llava.vision_tower(pixel_values, output_hidden_states=True)
    finally:
        handle_q.remove()
        handle_k.remove()
    if store.query is None or store.key is None:
        raise RuntimeError("Failed to capture vision query/key projections for DAMRO")

    selected = image_outputs.hidden_states[feature_layer][:, 1:]
    query = store.query
    key = store.key
    attn = torch.matmul(query, key.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
    attn = torch.softmax(attn.float(), dim=-1)
    cls_attn = attn[:, 0, 1:]
    k = max(1, min(int(topk), selected.shape[1]))
    indices = torch.topk(cls_attn, k=k, dim=1).indices
    gather_index = indices.unsqueeze(-1).expand(-1, -1, selected.shape[-1])
    outlier_features = torch.gather(selected, dim=1, index=gather_index)
    outlier_features = llava.multi_modal_projector(outlier_features)
    return outlier_features.to(dtype=pixel_values.dtype), indices[0].detach().cpu().tolist()


def _build_embeds_with_image_features(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_token_id = getattr(model.config, "image_token_index", 32000)
    token_embeds = model.get_input_embeddings()(input_ids).to(image_features.dtype)
    rows = []
    masks = []
    for batch_idx in range(input_ids.shape[0]):
        image_positions = torch.nonzero(input_ids[batch_idx] == image_token_id, as_tuple=False).flatten()
        if image_positions.numel() == 0:
            raise ValueError("DAMRO requires image placeholder tokens in the prompt")
        start = int(image_positions.min())
        end = int(image_positions.max()) + 1
        row = torch.cat(
            [
                token_embeds[batch_idx, :start],
                image_features[batch_idx],
                token_embeds[batch_idx, end:],
            ],
            dim=0,
        )
        mask = torch.cat(
            [
                attention_mask[batch_idx, :start],
                torch.ones(image_features.shape[1], device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask[batch_idx, end:],
            ],
            dim=0,
        )
        rows.append(row)
        masks.append(mask)

    max_len = max(row.shape[0] for row in rows)
    padded_rows = []
    padded_masks = []
    for row, mask in zip(rows, masks):
        pad = max_len - row.shape[0]
        if pad:
            row = torch.cat([row, torch.zeros(pad, row.shape[-1], device=row.device, dtype=row.dtype)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad, device=mask.device, dtype=mask.dtype)], dim=0)
        padded_rows.append(row)
        padded_masks.append(mask)
    return torch.stack(padded_rows, dim=0), torch.stack(padded_masks, dim=0)


def generate_damro_greedy(
    model,
    processor,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    alpha: float = 2.0,
    beta: float = 0.1,
    topk: int = 10,
) -> tuple[str, list[int]]:
    """Generate with a deterministic DAMRO-style contrastive decoder.

    This is a controlled HuggingFace port: the negative branch uses only the
    top-k CLIP spatial tokens selected by CLS attention, while decoding remains
    greedy to match the rest of this repository's POPE/CHAIR runs.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pixel_values = inputs["pixel_values"]

    outlier_features, outlier_indices = _select_damro_outlier_features(model, pixel_values, topk=topk)
    negative_embeds, negative_mask = _build_embeds_with_image_features(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_features=outlier_features,
    )

    eos_ids = _eos_ids(processor.tokenizer)
    generated: list[int] = []
    beta = float(beta)
    alpha = float(alpha)
    past = None
    past_negative = None
    current_input_ids = input_ids
    current_negative_embeds: torch.Tensor | None = negative_embeds

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, past = _forward_next_logits(
                model,
                current_input_ids,
                attention_mask,
                pixel_values=pixel_values if past is None else None,
                past_key_values=past,
            )
            if current_negative_embeds is not None:
                logits_negative, past_negative = _forward_next_logits(
                    model,
                    None,
                    negative_mask,
                    past_key_values=past_negative,
                    inputs_embeds=current_negative_embeds,
                )
                current_negative_embeds = None
            else:
                logits_negative, past_negative = _forward_next_logits(
                    model,
                    current_input_ids,
                    negative_mask,
                    past_key_values=past_negative,
                )
            cd_logits = (1.0 + alpha) * logits - alpha * logits_negative
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
            negative_mask = torch.cat([negative_mask, torch.ones_like(current_input_ids)], dim=-1)

    if not generated:
        return "", outlier_indices
    return processor.decode(generated, skip_special_tokens=True).strip(), outlier_indices


def generate_opera_beam(
    model,
    processor,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    image_start: int,
    image_end: int,
    num_beams: int = 5,
    scale_factor: float = 50.0,
    threshold: int = 15,
    num_attn_candidates: int = 5,
    penalty_weights: float = 1.0,
) -> str:
    """Generate through the official OPERA beam-search hook when available.

    OPERA's released implementation modifies ``transformers.generate`` and adds
    an ``opera_decoding`` beam-search path. This wrapper deliberately does not
    emulate OPERA with a greedy approximation: if the installed transformers
    package does not contain the official hook, it fails before producing any
    baseline output.
    """
    key_position = {
        "image_start": int(image_start),
        "image_end": int(image_end),
        "response_start": int(inputs["input_ids"].shape[1]),
    }
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=int(num_beams),
                output_attentions=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                opera_decoding=True,
                key_position=key_position,
                scale_factor=float(scale_factor),
                threshold=int(threshold),
                num_attn_candidates=int(num_attn_candidates),
                penalty_weights=float(penalty_weights),
            )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        opera_keys = (
            "opera_decoding",
            "key_position",
            "scale_factor",
            "num_attn_candidates",
            "penalty_weights",
        )
        if any(key in message for key in opera_keys):
            raise RuntimeError(
                "OPERA requires the official modified transformers generate() "
                "implementation from shikiw/OPERA. The current environment "
                "rejected OPERA-specific generation arguments, so no OPERA "
                "baseline was produced."
            ) from exc
        raise
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    del output_ids
    return text
