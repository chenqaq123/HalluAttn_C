"""Modern-architecture head identification for the AllPath formula port.

The released AllPath head lists cover LLaVA-1.5 and legacy Qwen-VL.  The
registered open-caption protocol therefore re-identifies heads on the
official, disjoint ResampledCHAIR images.  This module implements the two
disclosed scores without modifying the target model:

* format/T2T: target-token log-probability change from adding one projected
  attention-head output to the layer residual;
* image/I2T: attention mass on the referenced object's instance mask for
  segmented mentions, or on the whole visual block when no instance exists.

Generation is followed by one teacher-forced replay at the object-generating
rows.  Full hidden states and full attention tensors are never retained.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch


@dataclass
class AllPathReplay:
    """Compact tensors captured for the selected generating rows."""

    residual: torch.Tensor  # (layers, mentions, hidden), CPU
    projected_heads: torch.Tensor  # (layers, mentions, heads, hidden), CPU
    visual_attention: torch.Tensor  # (layers, heads, mentions, visual), CPU
    layers: int
    heads: int
    visual_tokens: int


def first_chair_mentions(
    tokenizer,
    evaluator,
    eval_dict: dict[str, Any],
    generated_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return CHAIR first mentions with their generating-row anchors."""

    from pas.evaluate.chair import caption_to_words

    token_ids = [int(value) for value in generated_ids]
    text = tokenizer.decode(token_ids)
    starts = [
        len(tokenizer.decode(token_ids[:index]))
        for index in range(len(token_ids))
    ]
    grounded = set(eval_dict["mscoco_gt_words"])
    _, _, _, double_words, original_words = caption_to_words(
        evaluator, eval_dict["caption"]
    )

    surfaces: list[str] = []
    labels: list[bool] = []
    word_index = 0
    for canonical in eval_dict["mscoco_generated_words"]:
        while (
            word_index < len(double_words)
            and evaluator.inverse_synonym_dict.get(double_words[word_index])
            != canonical
        ):
            word_index += 1
        if word_index >= len(double_words):
            return []
        surfaces.append(original_words[word_index])
        labels.append(canonical not in grounded)
        word_index += 1

    mentions: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    for surface, hallucinated, canonical in zip(
        surfaces,
        labels,
        eval_dict["mscoco_generated_words"],
    ):
        match = re.compile(
            r"\b" + re.escape(surface) + r"\b", re.IGNORECASE
        ).search(text, pos=offset)
        if match is None:
            return []
        token_index = bisect_right(starts, match.start()) - 1
        offset = match.end()
        if canonical in seen:
            continue
        seen.add(canonical)
        if token_index - 1 < 0:
            continue
        mentions.append(
            {
                "word": canonical,
                "surface": surface,
                "rel_anchor": token_index - 1,
                "hallucinated": bool(hallucinated),
            }
        )
    return mentions


def overlap_grid(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Map a binary instance mask to row-major visual cells by any overlap."""

    if height <= 0 or width <= 0:
        raise ValueError("Grid dimensions must be positive")
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {source.shape}")
    row_edges = np.rint(
        np.linspace(0, source.shape[0], height + 1)
    ).astype(np.int64)
    col_edges = np.rint(
        np.linspace(0, source.shape[1], width + 1)
    ).astype(np.int64)
    result = np.zeros((height, width), dtype=bool)
    for row, (start_row, end_row) in enumerate(
        zip(row_edges[:-1], row_edges[1:])
    ):
        for column, (start_col, end_col) in enumerate(
            zip(col_edges[:-1], col_edges[1:])
        ):
            result[row, column] = bool(
                source[start_row:end_row, start_col:end_col].any()
            )
    return result


def _llava_unpadded_shape(
    original_height: int,
    original_width: int,
    current_height: int,
    current_width: int,
) -> tuple[int, int]:
    """Mirror transformers.models.llava_next.unpad_image shape arithmetic."""

    original_ratio = original_width / original_height
    current_ratio = current_width / current_height
    if original_ratio > current_ratio:
        new_height = int(
            round(original_height * (current_width / original_width), 7)
        )
        padding = (current_height - new_height) // 2
        return current_height - 2 * padding, current_width
    new_width = int(
        round(original_width * (current_height / original_height), 7)
    )
    padding = (current_width - new_width) // 2
    return current_height, current_width - 2 * padding


def visual_instance_mask(
    kind: str,
    model,
    inputs: dict[str, Any],
    instance_mask: np.ndarray,
    expected_tokens: int,
) -> np.ndarray:
    """Build a mask in the exact visual-token packing order."""

    if kind in {"qwen25", "qwen3"}:
        grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
        merge = int(model.config.vision_config.spatial_merge_size)
        height = int(grid[1]) // merge
        width = int(grid[2]) // merge
        result = overlap_grid(instance_mask, height, width).reshape(-1)
    elif kind == "llava_next":
        from transformers.models.llava_next.modeling_llava_next import (
            get_anyres_image_grid_shape,
        )

        config = model.config
        base = (
            int(config.vision_config.image_size)
            // int(config.vision_config.patch_size)
        )
        original_height, original_width = (
            int(value)
            for value in inputs["image_sizes"][0].detach().cpu().tolist()
        )
        patch_height, patch_width = get_anyres_image_grid_shape(
            [original_height, original_width],
            config.image_grid_pinpoints,
            int(config.vision_config.image_size),
        )
        current_height = int(patch_height) * base
        current_width = int(patch_width) * base
        unpadded_height, unpadded_width = _llava_unpadded_shape(
            original_height,
            original_width,
            current_height,
            current_width,
        )
        base_mask = overlap_grid(instance_mask, base, base).reshape(-1)
        detail = overlap_grid(
            instance_mask, unpadded_height, unpadded_width
        )
        # LLaVA-NeXT inserts one learned newline vector after each detail row.
        detail_with_newline = np.concatenate(
            [
                np.concatenate([row, np.zeros(1, dtype=bool)])
                for row in detail
            ]
        )
        result = np.concatenate([base_mask, detail_with_newline])
    else:
        raise ValueError(f"Unsupported AllPath identification kind: {kind}")

    result = np.asarray(result, dtype=np.float32)
    if result.size != int(expected_tokens):
        raise RuntimeError(
            f"{kind} visual-mask length {result.size} does not match "
            f"the processor's {expected_tokens} visual tokens"
        )
    return result


def _decoder_layers(model) -> Sequence[torch.nn.Module]:
    language_model = model.model.language_model
    layers = getattr(language_model, "layers", None)
    if layers is None and hasattr(language_model, "model"):
        layers = getattr(language_model.model, "layers", None)
    if layers is None:
        raise AttributeError("Could not locate text decoder layers")
    return layers


@contextmanager
def capture_allpath_replay(
    captioner,
    full_ids: torch.Tensor,
    inputs: dict[str, Any],
    rows: Sequence[int],
    visual_positions: torch.Tensor,
) -> Iterator[dict[str, Any]]:
    """Capture official-format residual/head states and image attention."""

    selected_rows = torch.tensor(
        [int(value) for value in rows], dtype=torch.long
    )
    visual = visual_positions.detach().to(dtype=torch.long)
    layers = _decoder_layers(captioner.model)
    residual_current: dict[int, torch.Tensor] = {}
    residual: dict[int, torch.Tensor] = {}
    projected: dict[int, torch.Tensor] = {}
    attention: dict[int, torch.Tensor] = {}
    hooks = []

    def make_pre_hook(layer_index: int):
        def pre_hook(module, args, kwargs):
            hidden = (
                args[0]
                if args
                else kwargs.get("hidden_states")
            )
            if hidden is None:
                raise RuntimeError(
                    f"Missing residual input at decoder layer {layer_index}"
                )
            residual_current[layer_index] = hidden

        return pre_hook

    for index, layer in enumerate(layers):
        hooks.append(
            layer.register_forward_pre_hook(
                make_pre_hook(index), with_kwargs=True
            )
        )

    modeling_module = captioner.decoder.modeling_module
    original = modeling_module.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
        output, weights = original(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )
        if not isinstance(module, captioner.decoder.attention_classes):
            return output, weights
        layer_index = int(module.layer_idx)
        hidden = residual_current.pop(layer_index)
        row_index = selected_rows.to(hidden.device)
        if int(row_index.max()) >= hidden.shape[1]:
            raise IndexError(
                f"Replay row outside sequence at layer {layer_index}"
            )
        residual[layer_index] = (
            hidden[0, row_index].detach().to("cpu", dtype=torch.bfloat16)
        )

        # Eager attention returns (batch, query, heads, head_dim).  Slice the
        # output projection by head exactly as the released AllPath patch.
        selected_output = output[0, row_index].permute(1, 0, 2)
        head_count = selected_output.shape[0]
        head_dim = selected_output.shape[-1]
        projection = module.o_proj.weight.T.reshape(
            head_count, head_dim, module.o_proj.out_features
        )
        contributions = torch.matmul(
            selected_output, projection
        ).permute(1, 0, 2)
        projected[layer_index] = contributions.detach().to(
            "cpu", dtype=torch.bfloat16
        )

        visual_index = visual.to(weights.device)
        if int(visual_index.max()) >= weights.shape[-1]:
            raise IndexError(
                f"Visual token outside attention key span at layer {layer_index}"
            )
        attention[layer_index] = (
            weights[0, :, row_index][:, :, visual_index]
            .detach()
            .to("cpu", dtype=torch.float32)
        )
        return output, weights

    modeling_module.eager_attention_forward = wrapped
    captured: dict[str, Any] = {}
    try:
        yield captured
        vision = {
            key: value
            for key, value in inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }
        with torch.inference_mode():
            result = captioner.model(
                input_ids=full_ids.to(captioner.device),
                attention_mask=torch.ones_like(
                    full_ids, device=captioner.device
                ),
                use_cache=False,
                return_dict=True,
                **vision,
            )
        del result
        expected = set(range(len(layers)))
        for name, values in (
            ("residual", residual),
            ("projected heads", projected),
            ("visual attention", attention),
        ):
            missing = expected - values.keys()
            if missing:
                raise RuntimeError(
                    f"Missing {name} at layers {sorted(missing)}"
                )
        captured["replay"] = AllPathReplay(
            residual=torch.stack(
                [residual[index] for index in range(len(layers))]
            ),
            projected_heads=torch.stack(
                [projected[index] for index in range(len(layers))]
            ),
            visual_attention=torch.stack(
                [attention[index] for index in range(len(layers))]
            ),
            layers=len(layers),
            heads=int(captioner.decoder.heads),
            visual_tokens=int(visual.numel()),
        )
    finally:
        modeling_module.eager_attention_forward = original
        for hook in hooks:
            hook.remove()


def _target_log_probabilities(
    lm_head: torch.nn.Module,
    representations: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Exact target log probabilities with bounded vocabulary-logit memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    device = lm_head.weight.device
    dtype = lm_head.weight.dtype
    results = []
    with torch.inference_mode():
        for start in range(0, representations.shape[0], chunk_size):
            stop = min(start + chunk_size, representations.shape[0])
            hidden = representations[start:stop].to(device, dtype=dtype)
            target = targets[start:stop].to(device, dtype=torch.long)
            logits = lm_head(hidden).float()
            values = torch.log_softmax(logits, dim=-1).gather(
                1, target[:, None]
            )
            results.append(values[:, 0].cpu())
            del hidden, target, logits, values
    return torch.cat(results)


def allpath_scores(
    replay: AllPathReplay,
    lm_head: torch.nn.Module,
    target_ids: Sequence[int],
    instance_masks: Sequence[np.ndarray | None],
    *,
    logit_chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Compute per-mention format and image scores."""

    targets = torch.tensor([int(value) for value in target_ids])
    mention_count = int(targets.numel())
    if replay.residual.shape[1] != mention_count:
        raise ValueError("Target count does not match captured replay rows")
    if len(instance_masks) != mention_count:
        raise ValueError("Instance-mask count does not match targets")

    baseline = replay.residual
    layer_count, _, hidden_size = baseline.shape
    head_count = replay.projected_heads.shape[2]
    head_states = baseline[:, :, None, :] + replay.projected_heads
    baseline_targets = targets[None, :].expand(
        layer_count, mention_count
    ).reshape(-1)
    head_targets = targets[None, :, None].expand(
        layer_count, mention_count, head_count
    ).reshape(-1)
    baseline_logp = _target_log_probabilities(
        lm_head,
        baseline.reshape(-1, hidden_size),
        baseline_targets,
        chunk_size=logit_chunk_size,
    ).reshape(layer_count, mention_count)
    head_logp = _target_log_probabilities(
        lm_head,
        head_states.reshape(-1, hidden_size),
        head_targets,
        chunk_size=logit_chunk_size,
    ).reshape(layer_count, mention_count, head_count)
    format_scores = (
        head_logp - baseline_logp[:, :, None]
    ).permute(1, 0, 2)

    normalized_masks = []
    image_safe = []
    for mask in instance_masks:
        if mask is None:
            normalized_masks.append(
                np.ones(replay.visual_tokens, dtype=np.float32)
            )
            image_safe.append(False)
            continue
        value = np.asarray(mask, dtype=np.float32)
        if value.shape != (replay.visual_tokens,):
            raise ValueError(
                f"Expected visual mask {(replay.visual_tokens,)}, got "
                f"{value.shape}"
            )
        normalized_masks.append(value)
        image_safe.append(True)
    mask_tensor = torch.from_numpy(np.stack(normalized_masks))
    image_scores = (
        replay.visual_attention.permute(2, 0, 1, 3)
        * mask_tensor[:, None, None, :]
    ).sum(dim=-1)
    return (
        format_scores.numpy().astype(np.float32),
        image_scores.numpy().astype(np.float32),
        image_safe,
    )


def aggregate_head_scores(
    safe_sum: np.ndarray,
    safe_count: int,
    hallu_sum: np.ndarray,
    hallu_count: int,
) -> np.ndarray:
    """Official AllPath safe-mean minus hallucinated-mean aggregation."""

    if safe_count <= 0 or hallu_count <= 0:
        raise ValueError(
            "Both grounded and hallucinated identification examples are required"
        )
    return (
        np.asarray(safe_sum, dtype=np.float64) / int(safe_count)
        - np.asarray(hallu_sum, dtype=np.float64) / int(hallu_count)
    )


def proportional_head_counts(total_heads: int) -> dict[str, int]:
    """Map the official 1,024-head CHAIR counts once per architecture."""

    return {
        "good_format": int(math.floor(40 * total_heads / 1024 + 0.5)),
        "hallu_format": int(math.floor(40 * total_heads / 1024 + 0.5)),
        "good_image": int(math.floor(50 * total_heads / 1024 + 0.5)),
        "hallu_image": 0,
    }


def ranked_heads(scores: np.ndarray) -> list[dict[str, Any]]:
    """Return ascending official AllPath ranking."""

    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError(f"Expected layer × head scores, got {values.shape}")
    result = [
        {
            "layer": int(layer),
            "head": int(head),
            "score": float(values[layer, head]),
        }
        for layer in range(values.shape[0])
        for head in range(values.shape[1])
    ]
    return sorted(result, key=lambda item: item["score"])


def selected_head_payload(
    format_scores: np.ndarray,
    image_scores: np.ndarray,
) -> dict[str, Any]:
    """Select official CHAIR paths and normalize them for the formula port."""

    format_rank = ranked_heads(format_scores)
    image_rank = ranked_heads(image_scores)
    counts = proportional_head_counts(len(format_rank))
    hallu_format = format_rank[: counts["hallu_format"]]
    good_format = (
        format_rank[-counts["good_format"] :]
        if counts["good_format"]
        else []
    )
    good_image = (
        image_rank[-counts["good_image"] :]
        if counts["good_image"]
        else []
    )
    good_source = good_format + good_image
    hallu_source = hallu_format

    def grouped(items: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for item in items:
            key = str(int(item["layer"]))
            result.setdefault(key, []).append(int(item["head"]))
        return {
            layer: sorted(set(heads))
            for layer, heads in result.items()
        }

    return {
        "counts": counts,
        "selections": {
            "hallu_format": hallu_format,
            "good_format": good_format,
            "good_image": good_image,
            "hallu_image": [],
        },
        "good_heads": grouped(good_source),
        "hallu_heads": grouped(hallu_source),
        "format_ranked": format_rank,
        "image_ranked": image_rank,
    }
