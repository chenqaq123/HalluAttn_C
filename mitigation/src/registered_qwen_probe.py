"""Registered prompt/video extensions for the existing Qwen probe classes."""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
import torch

from mitigation.src.qwen25vl_probe import (
    Qwen25Captioner,
    VRIPT_SYSTEM,
)
from mitigation.src.qwen3vl_probe import Qwen3Captioner

FRAME_COUNT = 32
FRAME_SIZE = 392


class _RegisteredQwenMixin:
    def _image_prompt(self, instruction: str) -> str:
        return self.processor.apply_chat_template(
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

    def prompt_inputs(self, image, instruction: str):
        inputs = self.processor(
            text=[self._image_prompt(instruction)],
            images=[image],
            return_tensors="pt",
        )
        return {
            key: (
                value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

    @staticmethod
    def _vision_kwargs(inputs: dict) -> dict:
        return {
            key: value
            for key, value in inputs.items()
            if key not in ("input_ids", "attention_mask")
        }

    def generate_ids(
        self,
        inputs: dict,
        prefix_ids: torch.Tensor | None = None,
        banned: set[int] | frozenset[int] = frozenset(),
    ) -> torch.Tensor:
        input_ids = (
            prefix_ids if prefix_ids is not None else inputs["input_ids"]
        )
        kwargs = dict(self._vision_kwargs(inputs))
        if banned:
            kwargs["bad_words_ids"] = [
                [int(token)] for token in sorted(banned)
            ]
        with torch.inference_mode():
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.0,
                temperature=None,
                top_k=None,
                top_p=None,
                **kwargs,
            )

    def rollback_once(
        self,
        inputs: dict,
        prefix_ids: torch.Tensor,
        banned_first_ids: set[int],
        *,
        prompt_length: int,
    ) -> torch.Tensor:
        """Ban alternatives only at the rollback slot, then resume normally."""
        vision = dict(self._vision_kwargs(inputs))
        with torch.inference_mode():
            output = self.model(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                use_cache=False,
                return_dict=True,
                **vision,
            )
            logits = output.logits[:, -1, :].float()
            if banned_first_ids:
                banned = torch.tensor(
                    sorted(banned_first_ids),
                    dtype=torch.long,
                    device=logits.device,
                )
                logits.index_fill_(1, banned, -torch.inf)
            next_token = torch.argmax(logits, dim=-1)[:, None]
            resumed_prefix = torch.cat([prefix_ids, next_token], dim=-1)
            eos_values = self.model.generation_config.eos_token_id
            eos_ids = {
                int(value)
                for value in (
                    eos_values
                    if isinstance(eos_values, (list, tuple))
                    else [eos_values]
                )
                if value is not None
            }
            if int(next_token[0, 0]) in eos_ids:
                return resumed_prefix
            used = int(resumed_prefix.shape[-1] - prompt_length)
            remaining = max(0, int(self.max_new_tokens) - used)
            if remaining == 0:
                return resumed_prefix
            return self.model.generate(
                input_ids=resumed_prefix,
                attention_mask=torch.ones_like(resumed_prefix),
                do_sample=False,
                max_new_tokens=remaining,
                repetition_penalty=1.0,
                temperature=None,
                top_k=None,
                top_p=None,
                **vision,
            )

    def noun_mentions(self, nlp, generated_ids: list[int]) -> list[dict]:
        text = self.tokenizer.decode(generated_ids)
        starts = [
            len(self.tokenizer.decode(generated_ids[:index]))
            for index in range(len(generated_ids))
        ]
        mentions = []
        seen = set()
        for token in nlp(text):
            if token.pos_ not in ("NOUN", "PROPN"):
                continue
            lemma = token.lemma_.lower().strip()
            if not lemma or lemma in seen:
                continue
            token_index = bisect_right(starts, token.idx) - 1
            if token_index - 1 < 0:
                continue
            seen.add(lemma)
            mentions.append(
                {
                    "word": lemma,
                    "surface": token.text,
                    "rel_anchor": token_index - 1,
                }
            )
        return mentions


class RegisteredQwen25Captioner(
    _RegisteredQwenMixin, Qwen25Captioner
):
    pass


class RegisteredQwen3Captioner(_RegisteredQwenMixin, Qwen3Captioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_token_id = int(self.model.config.video_token_id)

    def video_prompt_inputs(
        self,
        video_path: str,
        instruction: str = "Describe this video in detail.",
        system: str | None = VRIPT_SYSTEM,
    ) -> dict:
        from qwen_vl_utils import process_vision_info, vision_process

        def _no_torchvision_fallback(element):
            raise RuntimeError(
                f"decord failed, fallback disabled: {element.get('video')}"
            )

        vision_process.VIDEO_READER_BACKENDS["torchvision"] = (
            _no_torchvision_fallback
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "nframes": FRAME_COUNT,
                        "resized_height": FRAME_SIZE,
                        "resized_width": FRAME_SIZE,
                    },
                    {"type": "text", "text": instruction},
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
        return {
            key: (
                value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in inputs.items()
        }

    def replay_features(
        self, inputs: dict, full_ids: torch.Tensor, anchors: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        input_ids = full_ids.to(self.device)
        visual_positions = torch.where(
            input_ids[0] == self.video_token_id
        )[0]
        if visual_positions.numel() == 0:
            visual_positions = torch.where(
                input_ids[0] == self.image_token_id
            )[0]
        rows = torch.tensor(
            [position for anchor in anchors for position in (anchor, anchor + 1)],
            dtype=torch.long,
        )
        with (
            self._capture_mass(rows, visual_positions) as captured,
            torch.inference_mode(),
        ):
            result = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
                **self._vision_kwargs(inputs),
            )
        missing = set(range(self.layer_count)) - captured.keys()
        if missing:
            raise RuntimeError(f"Missing layers: {sorted(missing)}")
        stacked = torch.stack(
            [captured[layer] for layer in range(self.layer_count)], dim=0
        )
        count = len(anchors)
        features = (
            stacked.reshape(self.layer_count, self.head_count, count, 2)
            .permute(2, 0, 3, 1)
            .numpy()
            .astype(np.float32)
        )
        logits = result.logits[0].float()
        first_logp = np.zeros(count, dtype=np.float64)
        for index, anchor in enumerate(anchors):
            target = input_ids[0, anchor + 1]
            first_logp[index] = float(
                torch.log_softmax(logits[anchor], dim=-1)[target].item()
            )
        del result
        return features, first_logp
