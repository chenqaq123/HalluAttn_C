"""Registered LLaVA-NeXT image/video allocation probe and rollback adapter."""

from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
from transformers.models.llama import modeling_llama

from mitigation.src.open_caption_models import (
    OpenCaptionModel,
    VRIPT_SYSTEM,
)


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch, kv_heads, length, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, kv_heads, groups, length, head_dim)
        .reshape(batch, kv_heads * groups, length, head_dim)
    )


class RegisteredLlavaCaptioner:
    """Expose the interface shared by registered Qwen rollback runners."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        *,
        kind: str,
        max_new_tokens: int = 512,
    ):
        if kind not in {"llava_next", "llava_next_video"}:
            raise ValueError(kind)
        self.kind = kind
        self.device = device
        self.base = OpenCaptionModel(
            kind,
            model_path,
            device,
            max_new_tokens=max_new_tokens,
        )
        self.model = self.base.model
        self.processor = self.base.processor
        self.tokenizer = self.base.tokenizer
        self.decoder = self.base.decoder
        self.max_new_tokens = int(max_new_tokens)
        self.layer_count = int(self.decoder.layers)
        self.head_count = int(self.decoder.heads)

    @staticmethod
    def _vision_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in inputs.items()
            if key not in {"input_ids", "attention_mask"}
        }

    def prompt_inputs(self, image, instruction: str) -> dict[str, Any]:
        if self.kind != "llava_next":
            raise ValueError(f"{self.kind} is not an image-caption adapter")
        return self.base.prepare_image(image, instruction)

    def video_prompt_inputs(
        self,
        video_path: str,
        instruction: str = "Describe this video in detail.",
        system: str | None = VRIPT_SYSTEM,
    ) -> dict[str, Any]:
        if self.kind != "llava_next_video":
            raise ValueError(f"{self.kind} is not a video-caption adapter")
        inputs, _ = self.base.prepare_video(
            video_path, instruction, system=system
        )
        return inputs

    def generate_ids(
        self,
        inputs: dict[str, Any],
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

    def caption_text(
        self, full_ids: torch.Tensor, prompt_length: int
    ) -> str:
        return self.tokenizer.decode(
            full_ids[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def noun_mentions(self, nlp, generated_ids: list[int]) -> list[dict]:
        text = self.tokenizer.decode(generated_ids)
        starts = [
            len(self.tokenizer.decode(generated_ids[:index]))
            for index in range(len(generated_ids))
        ]
        mentions = []
        seen = set()
        for token in nlp(text):
            if token.pos_ not in {"NOUN", "PROPN"}:
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

    def _visual_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        media = "video" if self.kind == "llava_next_video" else "image"
        return self.base.visual_positions(
            {"input_ids": input_ids}, media_type=media
        )

    @contextmanager
    def _capture_mass(
        self, query_rows: torch.Tensor, visual_positions: torch.Tensor
    ) -> Iterator[dict[int, torch.Tensor]]:
        """Capture pre-RoPE visual mass for all LLaMA text layers."""

        original_attention = modeling_llama.eager_attention_forward
        original_rope = modeling_llama.apply_rotary_pos_emb
        captured: dict[int, torch.Tensor] = {}
        pre_rope_qk: list[Any] = [None]

        def wrapped_rope(query, key, cos, sin, *args, **kwargs):
            pre_rope_qk[0] = (query, key)
            return original_rope(
                query, key, cos, sin, *args, **kwargs
            )

        def wrapped(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        ):
            output, weights = original_attention(
                module,
                query,
                key,
                value,
                attention_mask,
                *args,
                **kwargs,
            )
            if not isinstance(module, modeling_llama.LlamaAttention):
                return output, weights
            if pre_rope_qk[0] is None:
                raise RuntimeError("Missing pre-RoPE LLaMA Q/K")
            pre_query, pre_key = pre_rope_qk[0]
            key_states = _repeat_kv(
                pre_key, int(module.num_key_value_groups)
            )
            rows = query_rows.to(pre_query.device)
            visual = visual_positions.to(pre_query.device)
            logits = torch.matmul(
                pre_query[:, :, rows, :].float(),
                key_states.transpose(2, 3).float(),
            ) * float(module.scaling)
            if attention_mask is not None:
                logits += attention_mask[
                    :, :, rows, : key_states.shape[-2]
                ].float()
            probabilities = torch.softmax(logits, dim=-1)
            captured[int(module.layer_idx)] = (
                probabilities[0][:, :, visual]
                .sum(dim=-1)
                .detach()
                .cpu()
            )
            pre_rope_qk[0] = None
            return output, weights

        modeling_llama.eager_attention_forward = wrapped
        modeling_llama.apply_rotary_pos_emb = wrapped_rope
        try:
            yield captured
        finally:
            modeling_llama.eager_attention_forward = original_attention
            modeling_llama.apply_rotary_pos_emb = original_rope

    def replay_features(
        self,
        inputs: dict[str, Any],
        full_ids: torch.Tensor,
        anchors: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        input_ids = full_ids.to(self.device)
        visual_positions = self._visual_positions(input_ids)
        rows = torch.tensor(
            [
                position
                for anchor in anchors
                for position in (anchor, anchor + 1)
            ],
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
            raise RuntimeError(
                f"Missing LLaVA layers: {sorted(missing)}"
            )
        stacked = torch.stack(
            [
                captured[layer]
                for layer in range(self.layer_count)
            ],
            dim=0,
        )
        count = len(anchors)
        features = (
            stacked.reshape(
                self.layer_count, self.head_count, count, 2
            )
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

    def rollback_once(
        self,
        inputs: dict[str, Any],
        prefix_ids: torch.Tensor,
        banned_first_ids: set[int],
        *,
        prompt_length: int,
    ) -> torch.Tensor:
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
            resumed = torch.cat([prefix_ids, next_token], dim=-1)
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
                return resumed
            used = int(resumed.shape[-1] - prompt_length)
            remaining = max(0, self.max_new_tokens - used)
            if remaining == 0:
                return resumed
            # The first full-sequence forward can materialize a very large
            # logits tensor for video prompts.  Do not keep it alive while
            # generate() performs its own prefilling pass.
            del output, logits
            return self.model.generate(
                input_ids=resumed,
                attention_mask=torch.ones_like(resumed),
                do_sample=False,
                max_new_tokens=remaining,
                repetition_penalty=1.0,
                temperature=None,
                top_k=None,
                top_p=None,
                **vision,
            )
