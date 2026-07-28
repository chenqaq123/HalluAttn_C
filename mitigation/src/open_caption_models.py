"""Uniform image/video caption adapters for the registered benchmark matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

IMAGE_PIXELS_QWEN25 = 392 * 392
IMAGE_PIXELS_QWEN3 = 200704
VIDEO_FRAMES = 32
VIDEO_SIZE = 392
VRIPT_SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of "
    "events, the detail and movement of objects, and the action and pose of "
    "persons."
)


@dataclass(frozen=True)
class DecoderSpec:
    modeling_module: Any
    attention_classes: type[torch.nn.Module] | tuple[type[torch.nn.Module], ...]
    layers: int
    heads: int


def _move(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _snapshot_fingerprint(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    important = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "chat_template.json",
        "model.safetensors.index.json",
    ]
    digest = hashlib.sha256()
    files = []
    for name in important:
        file_path = root / name
        if not file_path.is_file():
            continue
        content = file_path.read_bytes()
        digest.update(name.encode())
        digest.update(content)
        files.append(name)
    return {
        "path": str(root.resolve()),
        "metadata_files": files,
        "metadata_sha256": digest.hexdigest(),
    }


class OpenCaptionModel:
    """Load one frozen model and normalize preparation/generation APIs."""

    def __init__(
        self,
        kind: str,
        model_path: str,
        device: str | torch.device,
        *,
        max_new_tokens: int = 512,
    ):
        self.kind = str(kind)
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._load()

    def _load(self) -> None:
        from transformers import (
            AutoProcessor,
            LlavaNextForConditionalGeneration,
            LlavaNextVideoForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )

        processor_kwargs: dict[str, Any] = {"local_files_only": True}
        if self.kind == "qwen25":
            processor_kwargs.update(
                min_pixels=IMAGE_PIXELS_QWEN25,
                max_pixels=IMAGE_PIXELS_QWEN25,
            )
            model_class = Qwen2_5_VLForConditionalGeneration
        elif self.kind == "qwen3":
            processor_kwargs.update(
                min_pixels=IMAGE_PIXELS_QWEN3,
                max_pixels=IMAGE_PIXELS_QWEN3,
            )
            model_class = Qwen3VLForConditionalGeneration
        elif self.kind == "llava_next":
            model_class = LlavaNextForConditionalGeneration
        elif self.kind == "llava_next_video":
            model_class = LlavaNextVideoForConditionalGeneration
        else:
            raise ValueError(f"Unknown model kind: {self.kind}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, **processor_kwargs
        )
        self.tokenizer = self.processor.tokenizer
        self.model = model_class.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
            device_map={"": str(self.device)},
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.decoder = self._decoder_spec()

    def _decoder_spec(self) -> DecoderSpec:
        if self.kind == "qwen25":
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

            config = getattr(self.model.config, "text_config", self.model.config)
            return DecoderSpec(
                modeling_module=modeling_qwen2_5_vl,
                attention_classes=modeling_qwen2_5_vl.Qwen2_5_VLAttention,
                layers=int(config.num_hidden_layers),
                heads=int(config.num_attention_heads),
            )
        if self.kind == "qwen3":
            from transformers.models.qwen3_vl import modeling_qwen3_vl

            config = self.model.config.text_config
            return DecoderSpec(
                modeling_module=modeling_qwen3_vl,
                attention_classes=modeling_qwen3_vl.Qwen3VLTextAttention,
                layers=int(config.num_hidden_layers),
                heads=int(config.num_attention_heads),
            )

        from transformers.models.llama import modeling_llama

        config = self.model.config.text_config
        return DecoderSpec(
            modeling_module=modeling_llama,
            attention_classes=modeling_llama.LlamaAttention,
            layers=int(config.num_hidden_layers),
            heads=int(config.num_attention_heads),
        )

    @staticmethod
    def _messages(
        prompt: str,
        media_type: str | None,
        *,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        content: list[dict[str, Any]] = []
        if media_type:
            content.append({"type": media_type})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _chat_text(
        self,
        prompt: str,
        media_type: str | None,
        *,
        system: str | None = None,
    ) -> str:
        messages = self._messages(prompt, media_type, system=system)
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def prepare_image(self, image, prompt: str) -> dict[str, Any]:
        text = self._chat_text(prompt, "image")
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        )
        return _move(dict(inputs), self.device)

    def prepare_text_only(
        self, prompt: str, *, system: str | None = None
    ) -> dict[str, Any]:
        text = self._chat_text(prompt, None, system=system)
        inputs = self.processor(text=[text], return_tensors="pt")
        return _move(dict(inputs), self.device)

    @staticmethod
    def _uniform_video(video_path: str) -> tuple[np.ndarray, list[int]]:
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0))
        if len(reader) == 0:
            raise ValueError(f"Video has no decodable frames: {video_path}")
        indices = np.rint(
            np.linspace(0, len(reader) - 1, VIDEO_FRAMES)
        ).astype(np.int64)
        frames = reader.get_batch(indices).asnumpy()
        return frames, indices.tolist()

    def prepare_video(
        self,
        video_path: str,
        prompt: str = "Describe this video in detail.",
        *,
        system: str | None = VRIPT_SYSTEM,
    ) -> tuple[dict[str, Any], list[int]]:
        if self.kind in {"qwen25", "qwen3"}:
            return self._prepare_qwen_video(
                video_path, prompt, system=system
            )
        if self.kind != "llava_next_video":
            raise ValueError(f"{self.kind} is not registered for video input")
        frames, indices = self._uniform_video(video_path)
        text = self._chat_text(prompt, "video", system=system)
        inputs = self.processor(
            text=[text],
            videos=[frames],
            return_tensors="pt",
        )
        return _move(dict(inputs), self.device), indices

    def _prepare_qwen_video(
        self, video_path: str, prompt: str, *, system: str | None
    ) -> tuple[dict[str, Any], list[int]]:
        from qwen_vl_utils import process_vision_info, vision_process

        def _no_torchvision_fallback(element):
            raise RuntimeError(
                f"decord failed, fallback disabled: {element.get('video')}"
            )

        vision_process.VIDEO_READER_BACKENDS["torchvision"] = (
            _no_torchvision_fallback
        )
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "nframes": VIDEO_FRAMES,
                        "resized_height": VIDEO_SIZE,
                        "resized_width": VIDEO_SIZE,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        _, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        # qwen_vl_utils does not expose indices.  Freeze the same uniform
        # decord indices separately for the cross-method frame audit.
        _, indices = self._uniform_video(video_path)
        return _move(dict(inputs), self.device), indices

    def visual_positions(
        self, inputs: dict[str, Any], *, media_type: str
    ) -> torch.Tensor:
        if media_type not in {"image", "video"}:
            raise ValueError(media_type)
        config = self.model.config
        candidates = (
            ["image_token_id", "image_token_index"]
            if media_type == "image"
            else ["video_token_id", "video_token_index"]
        )
        token_id = None
        for name in candidates:
            if hasattr(config, name):
                token_id = int(getattr(config, name))
                break
        if token_id is None:
            raise ValueError(
                f"No {media_type} token ID found in {type(config).__name__}"
            )
        positions = torch.where(inputs["input_ids"][0] == token_id)[0]
        if positions.numel() == 0:
            raise ValueError(
                f"Processor emitted no {media_type} placeholder tokens "
                f"(token ID {token_id})"
            )
        return positions

    def generate(
        self,
        inputs: dict[str, Any],
        *,
        do_sample: bool = False,
        temperature: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, Any] = {
            **inputs,
            "do_sample": bool(do_sample),
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
        }
        if do_sample:
            kwargs["temperature"] = float(
                temperature if temperature is not None else 0.7
            )
        if generator is not None:
            kwargs["generator"] = generator
        with torch.inference_mode():
            return self.model.generate(**kwargs)

    def decode_continuation(
        self, full_ids: torch.Tensor, prompt_length: int
    ) -> str:
        return self.tokenizer.decode(
            full_ids[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def metadata(self) -> dict[str, Any]:
        vision_shapes = {}
        return {
            "kind": self.kind,
            "snapshot": _snapshot_fingerprint(self.model_path),
            "dtype": str(next(self.model.parameters()).dtype),
            "attention_backend": "eager",
            "decoder_layers": self.decoder.layers,
            "decoder_heads": self.decoder.heads,
            "max_new_tokens": self.max_new_tokens,
            "processor_class": type(self.processor).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "generation_config": json.loads(
                self.model.generation_config.to_json_string()
            ),
            "vision_shapes": vision_shapes,
        }


def realized_vision_shapes(inputs: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
        and key
        not in {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }
    }
