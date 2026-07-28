"""Qwen2.5-VL captioning, mention anchoring, and No-RoPE mass extraction.

Protocol: docs/iteration_protocols/qwen25vl_video_transfer.md. Mirrors
qwen3vl_probe.py; differences are the M-RoPE hook
(`apply_multimodal_rotary_pos_emb`), the text-attention class
(`Qwen2_5_VLAttention`), and dual image/video input modes at the SmartSight
budget (392x392 per frame, 32 uniform frames).
"""

from __future__ import annotations

import re
from bisect import bisect_right
from contextlib import contextmanager

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

CAPTION_INSTRUCTION = "Describe this image in detail."
VIDEO_INSTRUCTION = "Describe this video in detail."
VRIPT_SYSTEM = (
    "Carefully watch the video and pay attention to the cause and sequence of "
    "events, the detail and movement of objects, and the action and pose of "
    "persons."
)
FRAME_COUNT = 32
FRAME_SIZE = 392
IMAGE_PIXELS = FRAME_SIZE * FRAME_SIZE


class Qwen25Captioner:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        max_new_tokens: int = 256,
    ):
        self.device = device
        torch.cuda.set_device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            min_pixels=IMAGE_PIXELS,
            max_pixels=IMAGE_PIXELS,
        )
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
        self.video_token_id = int(self.model.config.video_token_id)
        self.layer_count = int(self.model.config.num_hidden_layers)
        self.head_count = int(self.model.config.num_attention_heads)
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

    # ---------- inputs ----------
    def prompt_inputs(self, image):
        inputs = self.processor(
            text=[self.prompt], images=[image], return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def video_prompt_inputs(
        self,
        video_path: str,
        instruction: str = VIDEO_INSTRUCTION,
        system: str | None = VRIPT_SYSTEM,
    ) -> dict:
        from qwen_vl_utils import process_vision_info, vision_process

        # decord failures must surface as errors, not silently grind through
        # torchvision's whole-file decode (OOM on hour-long 60fps videos).
        def _no_torchvision_fallback(ele):
            raise RuntimeError(f"decord failed, fallback disabled: {ele.get('video')}")

        vision_process.VIDEO_READER_BACKENDS["torchvision"] = _no_torchvision_fallback

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
        try:
            _, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
        except ValueError as error:
            # Videos shorter than FRAME_COUNT frames: take all frames (even).
            match = re.search(r"interval \[(\d+), (\d+)\]", str(error))
            if not match:
                raise
            messages[-1]["content"][0]["nframes"] = max(
                2, int(match.group(2)) // 2 * 2
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
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

    @staticmethod
    def _vision_kwargs(inputs: dict) -> dict:
        return {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask")
        }

    # ---------- generation ----------
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
        kwargs = dict(self._vision_kwargs(inputs))
        if banned:
            kwargs["bad_words_ids"] = [[int(t)] for t in sorted(banned)]
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
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

    def noun_mentions(self, nlp, generated_ids: list[int]) -> list[dict]:
        """Generic NOUN/PROPN first-mention anchors (spacy), Qwen tokens.

        Returns [{word (lemma), surface, rel_anchor}] deduped by lemma, in
        order of first appearance. rel_anchor is the index of the token
        BEFORE the mention's first subtoken (the generating row); mentions
        at position 0 are skipped. Anchoring reuses the byte-BPE-safe
        cumulative-decode offset map (â/Ġ markers break \\b matching).
        """
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

    # ---------- No-RoPE mass extraction ----------
    @contextmanager
    def _capture_mass(
        self, query_rows: torch.Tensor, visual_positions: torch.Tensor
    ):
        """Capture per-layer, per-head no-RoPE visual mass at query rows."""
        original_attention = modeling_qwen2_5_vl.eager_attention_forward
        original_rope = modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb
        captured: dict[int, torch.Tensor] = {}
        pre_rope_qk: list = [None]

        def wrapped_rope(query, key, cos, sin, *args, **kwargs):
            pre_rope_qk[0] = (query, key)
            return original_rope(query, key, cos, sin, *args, **kwargs)

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            output, weights = original_attention(
                module, query, key, value, attention_mask, *args, **kwargs
            )
            if isinstance(module, modeling_qwen2_5_vl.Qwen2_5_VLAttention):
                if pre_rope_qk[0] is None:
                    raise RuntimeError("Missing pre-RoPE Q/K")
                pre_query, pre_key = pre_rope_qk[0]
                key_states = modeling_qwen2_5_vl.repeat_kv(
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

        modeling_qwen2_5_vl.eager_attention_forward = wrapped
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = wrapped_rope
        try:
            yield captured
        finally:
            modeling_qwen2_5_vl.eager_attention_forward = original_attention
            modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = original_rope

    def replay_features(
        self, inputs: dict, full_ids: torch.Tensor, anchors: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-forced replay; returns (features, first_logp).

        features: (n_mentions, layers, 2, heads) — rows are (pre=anchor,
        post=anchor+1) No-RoPE visual mass over image tokens in image mode
        or video tokens in video mode. first_logp: log-probability of the
        token at anchor+1 from the anchor row.
        """
        input_ids = full_ids.to(self.device)
        visual_positions = torch.where(input_ids[0] == self.video_token_id)[0]
        if visual_positions.numel() == 0:
            visual_positions = torch.where(
                input_ids[0] == self.image_token_id
            )[0]
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
                use_cache=False,
                return_dict=True,
                **self._vision_kwargs(inputs),
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
