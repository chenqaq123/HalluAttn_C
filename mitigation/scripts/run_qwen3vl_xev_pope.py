#!/usr/bin/env python3
"""Run a Qwen3-VL pilot of region-grounded alternative-evidence verification."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mitigation.src.data import iter_pope_records, load_pope_image  # noqa: E402
from mitigation.src.xev import (  # noqa: E402
    combine_global_target_support,
    counterbalanced_pairwise_score,
    evidence_margin,
    extract_target,
    identity_switch_certificate,
    inventory_lexically_supports_target,
    inventory_supports_target,
    lexical_equivalent,
    make_counterfactual_views,
    pairwise_identity_certificate,
    normalize_phrase,
    pairwise_question,
    parse_candidate_list,
    parse_relation_labels,
    propose_diverse_regions,
    propose_regions,
    random_bbox_like,
    relative_verification_score,
    robust_local_identity_proposal,
    robust_local_target_support,
    select_pairwise_candidates,
)


DEFAULT_MODEL = "/home/chenguanxu/models/Qwen3-VL-8B-Instruct"
DEFAULT_POPE = "/srv/common_dataset/POPE"
DEFAULT_COCO = "/srv/common_dataset/coco-2014-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--pope_dir", default=DEFAULT_POPE)
    parser.add_argument("--coco_root", default=DEFAULT_COCO)
    parser.add_argument("--pope_split", default="adversarial")
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--knowledge_cache",
        default=str(PROJECT_ROOT / "experiments/qwen3vl_xev/coco_knowledge.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--question_ids", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--attention_layers", default="18:30")
    parser.add_argument(
        "--attention_map",
        choices=("no_rope", "raw"),
        default="no_rope",
        help="Attention map used only for region localization.",
    )
    parser.add_argument("--proposal_mass", type=float, default=0.30)
    parser.add_argument("--max_regions", type=int, default=1)
    parser.add_argument(
        "--proposal_strategy",
        choices=("connected", "diverse"),
        default="connected",
    )
    parser.add_argument("--closed_k", type=int, default=8)
    parser.add_argument("--max_open_candidates", type=int, default=6)
    parser.add_argument("--pairwise_top_k", type=int, default=0)
    parser.add_argument("--global_support_guard", action="store_true")
    parser.add_argument(
        "--local_candidate_agreement",
        choices=("both", "closed", "any"),
        default="both",
    )
    parser.add_argument(
        "--local_pairwise_gate",
        choices=("preference", "directional"),
        default="preference",
    )
    parser.add_argument(
        "--local_relation_gate",
        choices=("competing", "non_support"),
        default="competing",
    )
    parser.add_argument(
        "--global_support_matcher",
        choices=("relations", "aliases"),
        default="relations",
    )
    parser.add_argument(
        "--global_support_sources",
        choices=("inventories", "complement", "either", "regions"),
        default="inventories",
    )
    parser.add_argument("--global_inventory_limit", type=int, default=12)
    parser.add_argument("--global_inventory_repeats", type=int, default=1)
    parser.add_argument("--global_candidate_max_new_tokens", type=int, default=64)
    parser.add_argument("--margin_batch_size", type=int, default=4)
    parser.add_argument("--candidate_max_new_tokens", type=int, default=32)
    parser.add_argument("--relation_max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--modes",
        default="target,random,closed,oracle,open",
        help="Comma-separated score modes. Open mode enables candidate generation.",
    )
    return parser.parse_args()


def parse_layer_range(spec: str, layer_count: int) -> set[int]:
    if ":" in spec:
        start_text, stop_text = spec.split(":", 1)
        start = int(start_text) if start_text else 0
        stop = int(stop_text) if stop_text else layer_count
        layers = set(range(start, stop))
    else:
        layers = {int(value) for value in spec.split(",") if value.strip()}
    invalid = sorted(index for index in layers if not 0 <= index < layer_count)
    if not layers or invalid:
        raise ValueError(
            f"Invalid attention layers {spec!r} for {layer_count} layers: {invalid}"
        )
    return layers


def image_id_from_record(record: dict) -> int:
    match = re.search(r"(\d+)$", str(record.get("image", "")))
    if not match:
        raise ValueError(f"Cannot extract COCO image id from {record.get('image')!r}")
    return int(match.group(1))


def _pair_key(left: str, right: str) -> str:
    return "|||".join(sorted((left, right)))


def build_coco_knowledge(coco_root: Path, cache_path: Path) -> dict:
    """Build COCO val inventories and train-set NPMI neighbors once."""

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    train = json.loads(
        (coco_root / "annotations/instances_train2014.json").read_text(encoding="utf-8")
    )
    val = json.loads(
        (coco_root / "annotations/instances_val2014.json").read_text(encoding="utf-8")
    )
    category_names = {
        int(row["id"]): normalize_phrase(row["name"]) for row in train["categories"]
    }

    train_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in train["annotations"]:
        train_by_image[int(annotation["image_id"])].add(
            category_names[int(annotation["category_id"])]
        )

    category_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for names in train_by_image.values():
        category_counts.update(names)
        for left, right in itertools.combinations(sorted(names), 2):
            pair_counts[_pair_key(left, right)] += 1

    image_count = len(train_by_image)
    neighbors: dict[str, list[dict]] = defaultdict(list)
    for pair, count in pair_counts.items():
        left, right = pair.split("|||", 1)
        p_xy = count / image_count
        p_left = category_counts[left] / image_count
        p_right = category_counts[right] / image_count
        npmi = math.log(p_xy / (p_left * p_right)) / -math.log(p_xy)
        neighbors[left].append({"name": right, "npmi": npmi, "count": count})
        neighbors[right].append({"name": left, "npmi": npmi, "count": count})
    for rows in neighbors.values():
        rows.sort(key=lambda row: (row["npmi"], row["count"]), reverse=True)

    val_category_names = {
        int(row["id"]): normalize_phrase(row["name"]) for row in val["categories"]
    }
    val_by_image: dict[int, set[str]] = defaultdict(set)
    for annotation in val["annotations"]:
        val_by_image[int(annotation["image_id"])].add(
            val_category_names[int(annotation["category_id"])]
        )

    payload = {
        "coco_categories": sorted(category_names.values()),
        "train_image_count": image_count,
        "neighbors": dict(neighbors),
        "val_objects": {
            str(image_id): sorted(names) for image_id, names in val_by_image.items()
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(cache_path)
    return payload


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {str(json.loads(line)["question_id"]) for line in handle if line.strip()}


class Qwen3VLEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(args.device)
        torch.cuda.set_device(self.device)
        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="eager",
            device_map={"": str(self.device)},
        )
        self.model.eval()
        self.yes_token_id = self._single_token_id("Yes")
        self.no_token_id = self._single_token_id("No")
        self.image_token_id = int(self.model.config.image_token_id)
        self.merge_size = int(self.processor.image_processor.merge_size)
        layer_count = int(self.model.config.text_config.num_hidden_layers)
        self.attention_layers = parse_layer_range(args.attention_layers, layer_count)
        self.attention_map = args.attention_map

    def _single_token_id(self, text: str) -> int:
        ids = self.processor.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Expected {text!r} to be one token, got {ids}")
        return int(ids[0])

    def _chat_prompt(self, text: str, has_image: bool) -> str:
        content = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def question_prompt(self, target: str, question: str | None = None) -> str:
        question_text = (
            question.strip()
            if question is not None
            else f"Is there a {target} in the image?"
        )
        return self._chat_prompt(
            f"{question_text} Please answer Yes or No only.", has_image=True
        )

    def move_inputs(self, inputs: dict) -> dict:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def margins(
        self,
        images: list,
        hypotheses: list[str],
        batch_size: int,
        questions: list[str | None] | None = None,
    ) -> list[float]:
        questions = questions or [None] * len(hypotheses)
        if len(questions) != len(hypotheses):
            raise ValueError("questions and hypotheses must have equal length")
        prompts = [
            self.question_prompt(hypothesis, question)
            for hypothesis, question in zip(hypotheses, questions)
        ]
        return self._margins_for_prompts(images, prompts, batch_size)

    def _margins_for_prompts(
        self, images: list, prompts: list[str], batch_size: int
    ) -> list[float]:
        if len(images) != len(prompts):
            raise ValueError("images and prompts must have equal length")
        results: list[float] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_images = images[start : start + batch_size]
            inputs = self.processor(
                text=batch_prompts,
                images=batch_images,
                padding=True,
                return_tensors="pt",
            )
            inputs = self.move_inputs(inputs)
            with torch.inference_mode():
                logits = self.model(
                    **inputs, use_cache=False, logits_to_keep=1, return_dict=True
                ).logits[:, -1, :]
            margins = logits[:, self.yes_token_id] - logits[:, self.no_token_id]
            results.extend(float(value) for value in margins.float().cpu())
            del inputs, logits, margins
        return results

    def pairwise_comparisons(
        self,
        image,
        target: str,
        candidates: list[dict],
        batch_size: int,
    ) -> list[dict]:
        prompts = []
        for item in candidates:
            candidate = item["candidate"]
            prompts.extend(
                [
                    self._chat_prompt(
                        pairwise_question(target, candidate), has_image=True
                    ),
                    self._chat_prompt(
                        pairwise_question(candidate, target), has_image=True
                    ),
                ]
            )
        margins = self._margins_for_prompts(
            [image] * len(prompts), prompts, batch_size
        )
        comparisons = []
        for item, target_first, alternative_first in zip(
            candidates, margins[::2], margins[1::2]
        ):
            preference, common_bias = counterbalanced_pairwise_score(
                target_first, alternative_first
            )
            comparisons.append(
                {
                    **item,
                    "target_first_margin": target_first,
                    "alternative_first_margin": alternative_first,
                    "target_preference": preference,
                    "common_yes_bias": common_bias,
                }
            )
        return comparisons

    @contextmanager
    def _capture_visual_rows(self, visual_positions: torch.Tensor):
        original_attention = modeling_qwen3_vl.eager_attention_forward
        original_rope = modeling_qwen3_vl.apply_rotary_pos_emb
        rows: dict[int, torch.Tensor] = {}
        wanted_layers = self.attention_layers
        pre_rope_qk: tuple[torch.Tensor, torch.Tensor] | None = None

        def wrapped_rope(query, key, cos, sin, *args, **kwargs):
            nonlocal pre_rope_qk
            pre_rope_qk = (query, key)
            return original_rope(query, key, cos, sin, *args, **kwargs)

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            nonlocal pre_rope_qk
            output, weights = original_attention(
                module, query, key, value, attention_mask, *args, **kwargs
            )
            if isinstance(module, modeling_qwen3_vl.Qwen3VLTextAttention):
                layer_index = int(module.layer_idx)
                if layer_index in wanted_layers:
                    indices = visual_positions.to(weights.device)
                    if self.attention_map == "raw":
                        visual_row = weights[0, :, -1, indices].float().mean(0)
                    else:
                        if pre_rope_qk is None:
                            raise RuntimeError(
                                f"Missing pre-RoPE Q/K for attention layer {layer_index}"
                            )
                        pre_query, pre_key = pre_rope_qk
                        key_states = modeling_qwen3_vl.repeat_kv(
                            pre_key, module.num_key_value_groups
                        )
                        no_rope_logits = torch.matmul(
                            pre_query[:, :, -1:, :].float(),
                            key_states.transpose(2, 3).float(),
                        ) * float(module.scaling)
                        if attention_mask is not None:
                            no_rope_logits += attention_mask[
                                :, :, -1:, : key_states.shape[-2]
                            ].float()
                        no_rope_weights = torch.softmax(no_rope_logits, dim=-1)
                        visual_row = no_rope_weights[0, :, -1, indices].mean(0)
                    rows[layer_index] = visual_row.cpu()
                pre_rope_qk = None
            return output, weights

        modeling_qwen3_vl.eager_attention_forward = wrapped
        if self.attention_map == "no_rope":
            modeling_qwen3_vl.apply_rotary_pos_emb = wrapped_rope
        try:
            yield rows
        finally:
            modeling_qwen3_vl.eager_attention_forward = original_attention
            modeling_qwen3_vl.apply_rotary_pos_emb = original_rope

    def attention_and_margin(
        self, image, target: str, question: str
    ) -> tuple[np.ndarray, float, dict]:
        prompt = self.question_prompt(target, question)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        visual_positions = torch.where(input_ids == self.image_token_id)[0]
        grid_thw = inputs["image_grid_thw"][0].tolist()
        expected_tokens = int(np.prod(grid_thw)) // self.merge_size**2
        if len(visual_positions) != expected_tokens:
            raise RuntimeError(
                f"Visual token mismatch: ids={len(visual_positions)} grid={grid_thw} expected={expected_tokens}"
            )
        inputs = self.move_inputs(inputs)
        with (
            self._capture_visual_rows(visual_positions) as rows,
            torch.inference_mode(),
        ):
            logits = self.model(
                **inputs, use_cache=False, logits_to_keep=1, return_dict=True
            ).logits[0, -1]
        missing = sorted(self.attention_layers - rows.keys())
        if missing:
            raise RuntimeError(f"Failed to capture attention layers: {missing}")
        attention = torch.stack([rows[index] for index in sorted(rows)]).mean(0).numpy()
        attention = np.maximum(attention, 0.0)
        attention /= max(float(attention.sum()), np.finfo(np.float32).eps)
        grid_height = int(grid_thw[1]) // self.merge_size
        grid_width = int(grid_thw[2]) // self.merge_size
        attention = attention.reshape(grid_height, grid_width)
        margin = float(
            (logits[self.yes_token_id] - logits[self.no_token_id]).float().cpu()
        )
        metadata = {
            "map_type": self.attention_map,
            "grid_thw": [int(value) for value in grid_thw],
            "attention_grid": [grid_height, grid_width],
            "layers": sorted(self.attention_layers),
            "visual_tokens": int(len(visual_positions)),
        }
        del inputs, logits
        return attention, margin, metadata

    def generate(self, prompt_text: str, image=None, max_new_tokens: int = 32) -> str:
        has_image = image is not None
        prompt = self._chat_prompt(prompt_text, has_image=has_image)
        kwargs = {"text": [prompt], "return_tensors": "pt"}
        if has_image:
            kwargs["images"] = [image]
        inputs = self.move_inputs(self.processor(**kwargs))
        input_length = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        generated = output_ids[0, input_length:]
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        del inputs, output_ids
        return text


class RelationFilter:
    def __init__(self, engine: Qwen3VLEngine, max_new_tokens: int) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.cache: dict[tuple[str, str], str] = {}

    def classify(
        self, target: str, candidates: list[str]
    ) -> tuple[dict[str, str], dict]:
        target = normalize_phrase(target)
        normalized = list(
            dict.fromkeys(normalize_phrase(value) for value in candidates if value)
        )
        pending = []
        for candidate in normalized:
            key = (target, candidate)
            if lexical_equivalent(target, candidate):
                self.cache[key] = "S"
            elif candidate.startswith(target) and candidate != target:
                # Compounds such as "snowboard boot" and agents such as
                # "snowboarder" contain the target string but are not subtypes.
                self.cache[key] = "N"
            elif key not in self.cache:
                pending.append(candidate)

        raw_outputs = []
        parsed_ok = True
        if pending:
            for chunk_start in range(0, len(pending), 8):
                chunk = pending[chunk_start : chunk_start + 8]
                listing = "\n".join(
                    f"{index}. {value}" for index, value in enumerate(chunk, 1)
                )
                prompt = (
                    "Classify how each candidate relates to the target object phrase.\n\n"
                    f"Target: {target}\nCandidates:\n{listing}\n\n"
                    "Use exactly one label per candidate:\n"
                    "S = the same identity, synonym, or a more specific subtype that "
                    "supports the target.\n"
                    "C = a different whole-object identity that could occupy the same "
                    "local region instead of the target.\n"
                    "N = a broader class, part, accessory, agent, material, marking, "
                    "trace, associated object, scene word, attribute, or unrelated "
                    "object.\n\n"
                    "Do not label a phrase S merely because it contains the target word; "
                    "its head noun must denote the target or a subtype. For example, for "
                    'target "snowboard": "snowboarder" and "snowboard boot" are N, '
                    'while "skis" is C. Objects labeled C may coexist elsewhere in the '
                    "same image; the question is whether they are competing local "
                    'identities. Return only indexed labels such as "1: C".'
                )
                raw = self.engine.generate(prompt, max_new_tokens=self.max_new_tokens)
                raw_outputs.append(raw)
                labels = parse_relation_labels(raw, len(chunk))
                if labels is None:
                    parsed_ok = False
                    labels = ["N"] * len(chunk)
                for candidate, label in zip(chunk, labels):
                    self.cache[(target, candidate)] = label

        relations = {
            candidate: self.cache[(target, candidate)] for candidate in normalized
        }
        return relations, {
            "raw": raw_outputs,
            "parsed_ok": parsed_ok,
            "classified": len(pending),
        }


def open_candidates(
    engine: Qwen3VLEngine, keep_image, limit: int, max_new_tokens: int
) -> tuple[list[str], str]:
    prompt = (
        "The image keeps one region clear and blurs the rest. Without knowing any "
        f"question target, list up to {limit} concrete object identities visibly "
        "supported by the clear region.\n\n"
        "Use singular noun phrases at the most specific recognizable level. Do "
        "not list attributes, parts, scene types, broader classes, or explanations. "
        "Return a comma-separated list only."
    )
    raw = engine.generate(prompt, image=keep_image, max_new_tokens=max_new_tokens)
    return parse_candidate_list(raw, limit=limit), raw


def global_candidates(
    engine: Qwen3VLEngine,
    image,
    limit: int,
    max_new_tokens: int,
    variant: int = 0,
) -> tuple[list[str], str]:
    if variant % 2 == 0:
        prompt = (
            f"List up to {limit} concrete object identities visibly supported anywhere "
            "in this image. Include small but recognizable objects. Use singular noun "
            "phrases at the most specific recognizable level. Do not list attributes, "
            "parts, scene types, broader classes, or explanations. Return a "
            "comma-separated list only."
        )
    else:
        prompt = (
            "Inspect the entire image systematically from left to right and foreground "
            f"to background. List up to {limit} concrete object identities that you can "
            "clearly distinguish. Do not guess a related category when its visual "
            "identity is uncertain. Use specific singular noun phrases, with no parts, "
            "attributes, scene words, or explanations. Return a comma-separated list only."
        )
    raw = engine.generate(prompt, image=image, max_new_tokens=max_new_tokens)
    return parse_candidate_list(raw, limit=limit), raw


def source_score(
    target: str,
    candidates: list[str],
    relations: dict[str, str],
    evidences: dict[str, float],
    allow_support: bool,
) -> dict:
    support = (
        [evidences[name] for name in candidates if relations.get(name) == "S"]
        if allow_support
        else []
    )
    alternatives = [
        evidences[name] for name in candidates if relations.get(name) == "C"
    ]
    score, support_term, penalty = relative_verification_score(
        evidences[target], support, alternatives
    )
    return {
        "score": score,
        "support": support_term,
        "alternative_penalty": penalty,
        "support_count": len(support),
        "alternative_count": len(alternatives),
    }


def process_record(
    record: dict,
    engine: Qwen3VLEngine,
    relation_filter: RelationFilter,
    knowledge: dict,
    args: argparse.Namespace,
    modes: set[str],
) -> dict:
    started = time.perf_counter()
    target = extract_target(record["question"])
    image_id = image_id_from_record(record)
    image = load_pope_image(record)
    attention, baseline_margin, attention_metadata = engine.attention_and_margin(
        image, target, record["question"]
    )
    proposal_fn = (
        propose_diverse_regions
        if args.proposal_strategy == "diverse"
        else propose_regions
    )
    proposals = proposal_fn(
        attention,
        image.size,
        mass_fraction=args.proposal_mass,
        max_regions=args.max_regions,
    )

    closed_candidates = [
        row["name"]
        for row in knowledge["neighbors"].get(target, [])[: args.closed_k]
        if not lexical_equivalent(row["name"], target)
    ]
    oracle_candidates = [
        name
        for name in knowledge["val_objects"].get(str(image_id), [])
        if not lexical_equivalent(name, target)
    ]
    region_rows = []
    for region_index, proposal in enumerate(proposals):
        keep, drop = make_counterfactual_views(image, proposal.bbox)
        generated_candidates: list[str] = []
        candidate_raw = ""
        if "open" in modes:
            generated_candidates, candidate_raw = open_candidates(
                engine,
                keep,
                limit=args.max_open_candidates,
                max_new_tokens=args.candidate_max_new_tokens,
            )

        source_candidates = {
            "closed": closed_candidates if "closed" in modes else [],
            "oracle": oracle_candidates if "oracle" in modes else [],
            "open": generated_candidates if "open" in modes else [],
        }
        all_candidates = list(
            dict.fromkeys(
                candidate
                for candidates in source_candidates.values()
                for candidate in candidates
                if not lexical_equivalent(candidate, target)
            )
        )
        relations, relation_metadata = relation_filter.classify(target, all_candidates)

        hypotheses = [target, *all_candidates]
        hypothesis_questions = [record["question"], *([None] * len(all_candidates))]
        keep_margins = engine.margins(
            [keep] * len(hypotheses),
            hypotheses,
            batch_size=args.margin_batch_size,
            questions=hypothesis_questions,
        )
        drop_margins = engine.margins(
            [drop] * len(hypotheses),
            hypotheses,
            batch_size=args.margin_batch_size,
            questions=hypothesis_questions,
        )
        keep_margin_map = {
            hypothesis: margin
            for hypothesis, margin in zip(hypotheses, keep_margins)
        }
        evidences = {
            hypothesis: evidence_margin(keep_margin, drop_margin)
            for hypothesis, keep_margin, drop_margin in zip(
                hypotheses, keep_margins, drop_margins
            )
        }

        pairwise_candidates = (
            select_pairwise_candidates(
                target=target,
                candidate_sources=source_candidates,
                relations=relations,
                keep_margins=keep_margin_map,
                limit=args.pairwise_top_k,
                block_on_open_target=not (
                    args.global_support_guard
                    and args.global_support_sources == "regions"
                ),
            )
            if baseline_margin >= 0
            else []
        )
        pairwise_comparisons = engine.pairwise_comparisons(
            keep,
            target,
            pairwise_candidates,
            batch_size=args.margin_batch_size,
        )
        pairwise_certificate = pairwise_identity_certificate(
            baseline_margin=baseline_margin,
            target_keep_margin=keep_margin_map[target],
            comparisons=pairwise_comparisons,
        )
        required_sources = {
            "both": ("open", "closed"),
            "closed": ("closed",),
            "any": (),
        }[args.local_candidate_agreement]
        local_identity_proposal = robust_local_identity_proposal(
            pairwise_comparisons,
            required_sources=required_sources,
            require_directional_switch=(args.local_pairwise_gate == "directional"),
            allowed_relations=(
                ("C", "N")
                if args.local_relation_gate == "non_support"
                else ("C",)
            ),
        )
        local_target_support = robust_local_target_support(pairwise_comparisons)

        random_bbox = random_bbox_like(
            proposal.bbox,
            image.size,
            seed=args.seed
            + image_id * 101
            + int(record["question_id"]) * 17
            + region_index,
        )
        random_keep, random_drop = make_counterfactual_views(image, random_bbox)
        random_margins = engine.margins(
            [random_keep, random_drop],
            [target, target],
            batch_size=2,
            questions=[record["question"], record["question"]],
        )
        scores = {
            "target": source_score(
                target, [], relations, evidences, allow_support=False
            ),
            "random": {
                "score": evidence_margin(*random_margins),
                "support": evidence_margin(*random_margins),
                "alternative_penalty": 0.0,
                "support_count": 0,
                "alternative_count": 0,
            },
        }
        for source in ("closed", "oracle", "open"):
            if source in modes:
                scores[source] = source_score(
                    target,
                    source_candidates[source],
                    relations,
                    evidences,
                    allow_support=(source == "open"),
                )

        certificate = identity_switch_certificate(
            target=target,
            open_candidates=source_candidates["open"],
            closed_candidates=source_candidates["closed"],
            relations=relations,
            keep_margins=keep_margin_map,
            baseline_margin=baseline_margin,
        )

        region_rows.append(
            {
                "index": region_index,
                "bbox": list(proposal.bbox),
                "grid_bbox": list(proposal.grid_bbox),
                "attention_mass": proposal.attention_mass,
                "random_bbox": list(random_bbox),
                "candidate_sources": source_candidates,
                "open_candidate_raw": candidate_raw,
                "relations": relations,
                "relation_filter": relation_metadata,
                "margins": {
                    hypothesis: {"keep": keep_margin, "drop": drop_margin}
                    for hypothesis, keep_margin, drop_margin in zip(
                        hypotheses, keep_margins, drop_margins
                    )
                },
                "evidence": evidences,
                "scores": scores,
                "identity_switch_certificate": certificate,
                "pairwise_verification": {
                    "candidate_limit": args.pairwise_top_k,
                    "comparisons": pairwise_comparisons,
                    "certificate": pairwise_certificate,
                    "local_identity_proposal": local_identity_proposal,
                    "local_target_support": local_target_support,
                },
            }
        )
        keep.close()
        drop.close()
        random_keep.close()
        random_drop.close()

    aggregate_scores = {
        mode: max(region["scores"][mode]["score"] for region in region_rows)
        for mode in modes
        if mode in region_rows[0]["scores"]
    }
    oracle_related_present = any(
        region["scores"].get("oracle", {}).get("alternative_count", 0) > 0
        for region in region_rows
    )
    certificates = [
        region["identity_switch_certificate"]
        for region in region_rows
        if region["identity_switch_certificate"] is not None
    ]
    pairwise_comparisons = [
        {"region_index": region["index"], **comparison}
        for region in region_rows
        for comparison in region["pairwise_verification"]["comparisons"]
    ]
    pairwise_certificates = [
        {"region_index": region["index"], **certificate}
        for region in region_rows
        if (certificate := region["pairwise_verification"]["certificate"]) is not None
    ]
    local_identity_proposals = [
        {"region_index": region["index"], **proposal}
        for region in region_rows
        if (proposal := region["pairwise_verification"]["local_identity_proposal"])
        is not None
    ]
    regional_target_supports = [
        {"region_index": region["index"], **support}
        for region in region_rows
        if (support := region["pairwise_verification"]["local_target_support"])
        is not None
    ]
    global_inventory = []
    global_inventory_raw = ""
    global_inventories: list[list[str]] = []
    global_inventory_raw_outputs: list[str] = []
    global_support_votes: list[bool] = []
    global_relations: dict[str, str] = {}
    global_relation_metadata: dict = {}
    inventory_support_found = False
    complement_target_margin: float | None = None
    complement_support_found = False
    regional_support_found = False
    global_support_found = False
    global_certificate = None
    if args.global_support_guard and local_identity_proposals:
        best_local_proposal = min(
            local_identity_proposals,
            key=lambda item: item["target_preference"],
        )
        local_region = region_rows[int(best_local_proposal["region_index"])]
        if args.global_support_sources == "regions":
            regional_support_found = bool(regional_target_supports)
            global_support_found = regional_support_found
        else:
            complement_target_margin = float(
                local_region["margins"][target]["drop"]
            )
        if args.global_support_sources in {"inventories", "either"}:
            for variant in range(args.global_inventory_repeats):
                inventory, raw = global_candidates(
                    engine,
                    image,
                    limit=args.global_inventory_limit,
                    max_new_tokens=args.global_candidate_max_new_tokens,
                    variant=variant,
                )
                global_inventories.append(inventory)
                global_inventory_raw_outputs.append(raw)
            global_inventory = list(
                dict.fromkeys(
                    candidate
                    for inventory in global_inventories
                    for candidate in inventory
                )
            )
            global_inventory_raw = global_inventory_raw_outputs[0]
            if args.global_support_matcher == "relations":
                global_relations, global_relation_metadata = relation_filter.classify(
                    target, global_inventory
                )
                global_support_votes = [
                    inventory_supports_target(target, inventory, global_relations)
                    for inventory in global_inventories
                ]
            else:
                global_support_votes = [
                    inventory_lexically_supports_target(target, inventory)
                    for inventory in global_inventories
                ]
            inventory_support_found = all(global_support_votes)
        if args.global_support_sources != "regions":
            global_support_found, complement_support_found = (
                combine_global_target_support(
                    inventory_support_found,
                    complement_target_margin,
                    source=args.global_support_sources,
                )
            )
        if not global_support_found:
            global_certificate = best_local_proposal
    image.close()
    return {
        "schema_version": 6,
        "score_status": {
            "scores": "rejected-iteration1-keep-minus-drop",
            "current_diagnostic": "xev-v1-directional-alias-certificate",
        },
        "question_id": str(record["question_id"]),
        "image_id": image_id,
        "image": record["image"],
        "question": record["question"],
        "target": target,
        "label": str(record["label"]).lower(),
        "baseline_margin": baseline_margin,
        "baseline_prediction": "yes" if baseline_margin >= 0 else "no",
        "attention": attention_metadata,
        "regions": region_rows,
        "scores": aggregate_scores,
        "oracle_related_present": oracle_related_present,
        "identity_switch_certificate": (
            max(certificates, key=lambda item: item["excess"]) if certificates else None
        ),
        "pairwise_verification": {
            "candidate_limit": args.pairwise_top_k,
            "minimum_target_preference": (
                min(
                    comparison["target_preference"]
                    for comparison in pairwise_comparisons
                )
                if pairwise_comparisons
                else None
            ),
            "comparisons": pairwise_comparisons,
            "certificate": (
                min(
                    pairwise_certificates,
                    key=lambda item: item["target_preference"],
                )
                if pairwise_certificates
                else None
            ),
        },
        "global_support_verification": {
            "enabled": args.global_support_guard,
            "local_candidate_agreement": args.local_candidate_agreement,
            "local_pairwise_gate": args.local_pairwise_gate,
            "local_relation_gate": args.local_relation_gate,
            "global_support_matcher": args.global_support_matcher,
            "global_support_sources": args.global_support_sources,
            "evaluated": bool(args.global_support_guard and local_identity_proposals),
            "local_identity_proposals": local_identity_proposals,
            "inventory": global_inventory,
            "inventory_raw": global_inventory_raw,
            "inventories": global_inventories,
            "inventory_raw_outputs": global_inventory_raw_outputs,
            "support_votes": global_support_votes,
            "inventory_support_found": inventory_support_found,
            "complement_target_margin": complement_target_margin,
            "complement_support_found": complement_support_found,
            "regional_target_supports": regional_target_supports,
            "regional_support_found": regional_support_found,
            "relations": global_relations,
            "relation_filter": global_relation_metadata,
            "target_support_found": global_support_found,
            "certificate": global_certificate,
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    modes = {value.strip() for value in args.modes.split(",") if value.strip()}
    modes.update({"target", "random"})
    unknown = modes - {"target", "random", "closed", "oracle", "open"}
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    if args.global_support_guard and args.pairwise_top_k <= 0:
        raise ValueError("--global_support_guard requires --pairwise_top_k > 0")
    if args.global_inventory_repeats <= 0:
        raise ValueError("--global_inventory_repeats must be positive")
    if args.max_regions <= 0:
        raise ValueError("--max_regions must be positive")
    if not 0 < args.proposal_mass <= 1:
        raise ValueError("--proposal_mass must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    coco_root = Path(args.coco_root)
    knowledge = build_coco_knowledge(coco_root, Path(args.knowledge_cache))
    all_rows = list(
        iter_pope_records(
            args.pope_dir,
            coco_root,
            args.pope_split,
            shard_idx=0,
            num_shards=1,
            limit=0,
        )
    )
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.question_ids:
        requested_ids = {
            value.strip() for value in args.question_ids.split(",") if value.strip()
        }
        selected = [
            row for row in all_rows if str(row["question_id"]) in requested_ids
        ]
        found_ids = {str(row["question_id"]) for row in selected}
        if missing_ids := sorted(requested_ids - found_ids, key=int):
            raise ValueError(f"Unknown --question_ids: {missing_ids}")
    else:
        stop = args.start + args.limit if args.limit > 0 else None
        selected = all_rows[args.start : stop]
    rows = selected[args.shard_idx :: args.num_shards]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = existing_ids(output_path)

    engine = Qwen3VLEngine(args)
    relation_filter = RelationFilter(
        engine, max_new_tokens=args.relation_max_new_tokens
    )
    with output_path.open("a", encoding="utf-8") as output:
        iterator = tqdm(rows, desc=f"qwen3-xev:{args.pope_split}:shard{args.shard_idx}")
        for step, record in enumerate(iterator, 1):
            if str(record["question_id"]) in completed:
                continue
            payload = process_record(
                record, engine, relation_filter, knowledge, args, modes
            )
            payload["run"] = {
                "model_path": args.model_path,
                "split": args.pope_split,
                "start": args.start,
                "seed": args.seed,
                "proposal_mass": args.proposal_mass,
                "max_regions": args.max_regions,
                "proposal_strategy": args.proposal_strategy,
                "attention_map": args.attention_map,
                "closed_k": args.closed_k,
                "pairwise_top_k": args.pairwise_top_k,
                "global_support_guard": args.global_support_guard,
                "local_candidate_agreement": args.local_candidate_agreement,
                "local_pairwise_gate": args.local_pairwise_gate,
                "local_relation_gate": args.local_relation_gate,
                "global_support_matcher": args.global_support_matcher,
                "global_support_sources": args.global_support_sources,
                "global_inventory_limit": args.global_inventory_limit,
                "global_inventory_repeats": args.global_inventory_repeats,
                "modes": sorted(modes),
            }
            output.write(json.dumps(payload) + "\n")
            output.flush()
            iterator.set_postfix(seconds=f"{payload['runtime_seconds']:.1f}")
            if step % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
