"""Qwen3-VL captioning, mention anchoring, and No-RoPE mass extraction.

Protocol: docs/iteration_protocols/qwen3vl_transfer.md. Reuses the XEV
pre-RoPE capture mechanism; anchors follow the LLaVA convention
(anchor = generating row before the mention's first subtoken; post row =
anchor + 1).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl

CAPTION_INSTRUCTION = "Describe this image in detail."


class Qwen3Captioner:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        pixels: int = 200704,
        max_new_tokens: int = 256,
    ):
        self.device = device
        torch.cuda.set_device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            min_pixels=pixels,
            max_pixels=pixels,
        )
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
            device_map={"": str(device)},
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.image_token_id = int(self.model.config.image_token_id)
        self.layer_count = int(self.model.config.text_config.num_hidden_layers)
        self.head_count = int(self.model.config.text_config.num_attention_heads)
        self.max_new_tokens = max_new_tokens
        self.prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": CAPTION_INSTRUCTION},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ---------- generation ----------
    def prompt_inputs(self, image):
        inputs = self.processor(
            text=[self.prompt], images=[image], return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate_ids(
        self,
        inputs: dict,
        prefix_ids: torch.Tensor | None = None,
        banned: set[int] | frozenset[int] = frozenset(),
    ) -> torch.Tensor:
        """Greedy continuation; returns the full sequence ids (1, S)."""
        input_ids = (
            prefix_ids if prefix_ids is not None else inputs["input_ids"]
        )
        kwargs = {}
        if banned:
            kwargs["bad_words_ids"] = [[int(t)] for t in sorted(banned)]
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                **kwargs,
            )
        return output

    def caption_text(self, full_ids: torch.Tensor, prompt_len: int) -> str:
        return self.tokenizer.decode(
            full_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

    # ---------- mention anchoring (Qwen byte-BPE) ----------
    def first_mentions(
        self, evaluator, eval_dict: dict, generated_ids: list[int]
    ) -> list[dict]:
        """CHAIR first mentions with generating-row anchors, Qwen tokens.

        Returns [{word, rel_anchor, hallucinated}] where rel_anchor is the
        index (in the generated sequence) of the token BEFORE the mention's
        first subtoken; rel_anchor == -1 (mention at position 0) is skipped.
        """
        from pas.evaluate.chair import caption_to_words

        # Byte-level BPE tokens (âĢĻ etc.) break \b matching; build the true
        # decoded text and per-token character offsets via cumulative decode.
        text = self.tokenizer.decode(generated_ids)
        starts = [
            len(self.tokenizer.decode(generated_ids[:index]))
            for index in range(len(generated_ids))
        ]

        non_hallu = set(eval_dict["mscoco_gt_words"])
        _, _, _, double_words, original_words = caption_to_words(
            evaluator, eval_dict["caption"]
        )
        words_to_find, labels = [], []
        word_index = 0
        for word in eval_dict["mscoco_generated_words"]:
            while (
                word_index < len(double_words)
                and evaluator.inverse_synonym_dict.get(double_words[word_index])
                != word
            ):
                word_index += 1
            if word_index >= len(double_words):
                return []
            words_to_find.append(original_words[word_index])
            labels.append(word not in non_hallu)
            word_index += 1

        mentions = []
        seen = set()
        offset = 0
        for word, label, canonical in zip(
            words_to_find, labels, eval_dict["mscoco_generated_words"]
        ):
            pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            match = pattern.search(text, pos=offset)
            if not match:
                return []
            token_index = bisect_right(starts, match.start()) - 1
            offset = match.end()
            if canonical in seen:
                continue
            seen.add(canonical)
            if token_index - 1 >= 0:
                mentions.append(
                    {
                        "word": canonical,
                        "rel_anchor": token_index - 1,
                        "hallucinated": bool(label),
                    }
                )
        return mentions

    # ---------- No-RoPE mass extraction ----------
    @contextmanager
    def _capture_mass(
        self, query_rows: torch.Tensor, visual_positions: torch.Tensor
    ):
        """Capture per-layer, per-head no-RoPE visual mass at query rows."""
        original_attention = modeling_qwen3_vl.eager_attention_forward
        original_rope = modeling_qwen3_vl.apply_rotary_pos_emb
        captured: dict[int, torch.Tensor] = {}
        pre_rope_qk: list = [None]

        def wrapped_rope(query, key, cos, sin, *args, **kwargs):
            pre_rope_qk[0] = (query, key)
            return original_rope(query, key, cos, sin, *args, **kwargs)

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            output, weights = original_attention(
                module, query, key, value, attention_mask, *args, **kwargs
            )
            if isinstance(module, modeling_qwen3_vl.Qwen3VLTextAttention):
                if pre_rope_qk[0] is None:
                    raise RuntimeError("Missing pre-RoPE Q/K")
                pre_query, pre_key = pre_rope_qk[0]
                key_states = modeling_qwen3_vl.repeat_kv(
                    pre_key, module.num_key_value_groups
                )
                rows = query_rows.to(pre_query.device)
                logits = torch.matmul(
                    pre_query[:, :, rows, :].float(),
                    key_states.transpose(2, 3).float(),
                ) * float(module.scaling)
                if attention_mask is not None:
                    logits += attention_mask[:, :, rows, : key_states.shape[-2]].float()
                probs = torch.softmax(logits, dim=-1)
                mass = probs[0][:, :, visual_positions.to(probs.device)].sum(-1)
                captured[int(module.layer_idx)] = mass.cpu()  # (heads, rows)
                pre_rope_qk[0] = None
            return output, weights

        modeling_qwen3_vl.eager_attention_forward = wrapped
        modeling_qwen3_vl.apply_rotary_pos_emb = wrapped_rope
        try:
            yield captured
        finally:
            modeling_qwen3_vl.eager_attention_forward = original_attention
            modeling_qwen3_vl.apply_rotary_pos_emb = original_rope

    def replay_features(
        self, inputs: dict, full_ids: torch.Tensor, anchors: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-forced replay; returns (features, first_logp).

        features: (n_mentions, layers, 2, heads) — rows are (pre=anchor,
        post=anchor+1) No-RoPE visual mass. first_logp: log-probability of
        the token at anchor+1 from the anchor row.
        """
        input_ids = full_ids.to(self.device)
        visual_positions = torch.where(input_ids[0] == self.image_token_id)[0]
        rows = torch.tensor(
            [p for a in anchors for p in (a, a + 1)], dtype=torch.long
        )
        with (
            self._capture_mass(rows, visual_positions) as captured,
            torch.inference_mode(),
        ):
            result = self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                use_cache=False,
                return_dict=True,
            )
        missing = set(range(self.layer_count)) - captured.keys()
        if missing:
            raise RuntimeError(f"Missing layers: {sorted(missing)}")
        stacked = torch.stack(
            [captured[i] for i in range(self.layer_count)], dim=0
        )  # (L, H, 2*n)
        n = len(anchors)
        features = (
            stacked.reshape(self.layer_count, self.head_count, n, 2)
            .permute(2, 0, 3, 1)
            .numpy()
            .astype(np.float32)
        )  # (n, L, 2, H)
        logits = result.logits[0].float()
        first_logp = np.zeros(n, dtype=np.float64)
        for index, anchor in enumerate(anchors):
            target = input_ids[0, anchor + 1]
            first_logp[index] = float(
                torch.log_softmax(logits[anchor], dim=-1)[target].item()
            )
        del result
        return features, first_logp
