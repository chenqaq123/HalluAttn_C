"""Disclosed-core SmartSight scoring for the registered video matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class TemporalTrace:
    frame_attention: np.ndarray
    visual_attention: np.ndarray
    generated_text_attention: np.ndarray


class TemporalAttentionCapture:
    """Capture last-query temporal attention without returning full maps."""

    def __init__(
        self,
        modeling_module: Any,
        attention_classes,
        frame_groups: Sequence[tuple[torch.Tensor, float]],
        prompt_length: int,
    ):
        self.modeling_module = modeling_module
        self.attention_classes = attention_classes
        self.frame_groups = [
            (positions.detach().cpu().long(), float(divisor))
            for positions, divisor in frame_groups
        ]
        self.prompt_length = int(prompt_length)
        self.original = None
        self._current_frames: list[torch.Tensor] = []
        self._current_visual: list[float] = []
        self._current_text: list[float] = []
        self._last_layer = -1
        self._steps_frames: list[np.ndarray] = []
        self._steps_visual: list[float] = []
        self._steps_text: list[float] = []

    def __enter__(self):
        self.original = self.modeling_module.eager_attention_forward
        self.modeling_module.eager_attention_forward = self._forward
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._finish_step()
        if self.original is not None:
            self.modeling_module.eager_attention_forward = self.original
            self.original = None

    def _finish_step(self) -> None:
        if not self._current_frames:
            return
        frames = torch.stack(self._current_frames).mean(dim=0)
        self._steps_frames.append(frames.float().cpu().numpy())
        self._steps_visual.append(float(np.mean(self._current_visual)))
        self._steps_text.append(float(np.mean(self._current_text)))
        self._current_frames.clear()
        self._current_visual.clear()
        self._current_text.clear()
        self._last_layer = -1

    def _forward(
        self,
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        if self.original is None:
            raise RuntimeError("Temporal capture is not installed")
        output, weights = self.original(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )
        if not isinstance(module, self.attention_classes):
            return output, weights
        layer = int(getattr(module, "layer_idx", -1))
        if self._current_frames and layer <= self._last_layer:
            self._finish_step()
        self._last_layer = layer

        last = weights[0, :, -1, :].float().mean(dim=0)
        frame_values = []
        for positions, divisor in self.frame_groups:
            valid = positions[positions < last.shape[-1]].to(last.device)
            value_for_frame = (
                last.index_select(0, valid).sum() / divisor
                if valid.numel()
                else last.new_zeros(())
            )
            frame_values.append(value_for_frame)
        frames = torch.stack(frame_values)
        visual = float(frames.sum().detach().cpu())
        text_start = min(self.prompt_length, last.shape[-1])
        text = float(last[text_start:].sum().detach().cpu())
        self._current_frames.append(frames.detach().cpu())
        self._current_visual.append(visual)
        self._current_text.append(text)
        return output, weights

    def trace(self) -> TemporalTrace:
        self._finish_step()
        if not self._steps_frames:
            raise ValueError("No text-decoder attention was captured")
        return TemporalTrace(
            frame_attention=np.stack(self._steps_frames),
            visual_attention=np.asarray(
                self._steps_visual, dtype=np.float64
            ),
            generated_text_attention=np.asarray(
                self._steps_text, dtype=np.float64
            ),
        )


def frame_groups(
    kind: str,
    inputs: dict[str, Any],
    visual_positions: torch.Tensor,
    *,
    frames: int = 32,
) -> list[tuple[torch.Tensor, float]]:
    """Map LLM visual-token positions back to the 32 registered frames."""

    positions = visual_positions.detach().cpu().long()
    if kind in {"qwen25", "qwen3"}:
        temporal = int(inputs["video_grid_thw"][0, 0].detach().cpu())
        if positions.numel() % temporal or frames % temporal:
            raise ValueError(
                "Qwen visual tokens cannot be mapped evenly to frames"
            )
        per_temporal = positions.numel() // temporal
        repeats = frames // temporal
        groups = []
        for temporal_index in range(temporal):
            group = positions[
                temporal_index
                * per_temporal : (temporal_index + 1)
                * per_temporal
            ]
            groups.extend((group, float(repeats)) for _ in range(repeats))
        return groups
    if positions.numel() % frames:
        raise ValueError(
            f"{kind}: {positions.numel()} visual tokens not divisible by {frames}"
        )
    per_frame = positions.numel() // frames
    return [
        (
            positions[
                frame * per_frame : (frame + 1) * per_frame
            ],
            1.0,
        )
        for frame in range(frames)
    ]


def video_token_features(
    captioner,
    inputs: dict[str, Any],
    *,
    frames: int = 32,
) -> torch.Tensor:
    """Return post-projector visual tokens as (frames, patches, hidden)."""

    with torch.inference_mode():
        if captioner.kind in {"qwen25", "qwen3"}:
            features = captioner.model.get_video_features(
                inputs["pixel_values_videos"],
                inputs["video_grid_thw"],
            )
            if captioner.kind == "qwen3":
                features = features[0]
            features = features[0]
            temporal = int(
                inputs["video_grid_thw"][0, 0].detach().cpu()
            )
            if features.shape[0] % temporal or frames % temporal:
                raise ValueError("Invalid Qwen video feature grid")
            features = features.reshape(
                temporal, features.shape[0] // temporal, features.shape[-1]
            )
            return features.repeat_interleave(
                frames // temporal, dim=0
            )
        features = captioner.model.get_video_features(
            inputs["pixel_values_videos"]
        )
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 4 and features.shape[0] == 1:
            features = features[0]
        if features.ndim != 3 or features.shape[0] != frames:
            raise ValueError(
                f"Unexpected LLaVA video feature shape: {features.shape}"
            )
        return features


def _spatial_coordinates(patches: int, device: torch.device) -> torch.Tensor:
    height = int(np.floor(np.sqrt(patches)))
    while height > 1 and patches % height:
        height -= 1
    width = patches // height
    rows = torch.linspace(0, 1, height, device=device)
    cols = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1)


def motion_segments(
    features: torch.Tensor,
) -> tuple[list[list[int]], list[float], float]:
    """Median-threshold adjacent-frame Hungarian segmentation."""

    features = torch.nn.functional.normalize(features.float(), dim=-1)
    frame_count, patches = features.shape[:2]
    coordinates = _spatial_coordinates(patches, features.device)
    spatial_cost = torch.cdist(coordinates, coordinates).square()
    distances = []
    for frame in range(1, frame_count):
        semantic_cost = (
            1.0
            - (features[frame - 1] @ features[frame].transpose(0, 1))
        ).clamp_min(0)
        cost = (semantic_cost + spatial_cost).detach().cpu().numpy()
        row, column = linear_sum_assignment(cost)
        distances.append(float(cost[row, column].sum()))
    threshold = float(np.median(distances))
    boundaries = [
        index + 1
        for index, distance in enumerate(distances)
        if distance > threshold
    ]
    starts = [0, *boundaries]
    ends = [*boundaries, frame_count]
    segments = [
        list(range(start, end))
        for start, end in zip(starts, ends)
        if end > start
    ]
    return segments, distances, threshold


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values / total
    positive = probabilities > 0
    return float(
        -(probabilities[positive] * np.log(probabilities[positive])).sum()
    )


def score_trace(
    trace: TemporalTrace,
    segments: Sequence[Sequence[int]],
    *,
    alpha: float = 1.2,
    window: int = 10,
) -> dict[str, Any]:
    """Compute VAV, frame/segment entropy, and the TAC selection score."""

    denominator = np.maximum(
        trace.generated_text_attention, np.finfo(np.float64).tiny
    )
    ratios = trace.visual_attention / denominator
    ratios[trace.generated_text_attention <= 0] = np.inf
    vav_index = len(ratios) - 1
    for end in range(window - 1, len(ratios)):
        if np.all(ratios[end - window + 1 : end + 1] < alpha):
            vav_index = end
            break
    averaged = trace.frame_attention[: vav_index + 1].mean(axis=0)
    frame_entropy = _entropy(averaged)
    segment_attention = np.asarray(
        [averaged[list(segment)].sum() for segment in segments],
        dtype=np.float64,
    )
    segment_entropy = _entropy(segment_attention)
    return {
        "tac": frame_entropy + segment_entropy,
        "frame_entropy": frame_entropy,
        "segment_entropy": segment_entropy,
        "vav_index": int(vav_index),
        "attention_steps": int(len(ratios)),
        "vav_ratio": float(ratios[vav_index]),
        "visual_text_ratios": ratios.tolist(),
        "frame_attention": averaged.tolist(),
        "segment_attention": segment_attention.tolist(),
    }
