"""Compute baseline scores from one teacher-forced LLaVA forward per image."""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

_EPS = 1e-12


def _nan_scores(n: int) -> dict[str, np.ndarray]:
    names = [
        "nll_hallu_score",
        "entropy_hallu_score",
        "pas_layer0_hallu_score",
        "svar_layer0_hallu_score",
        "ic_hallu_score",
        "glsim_hallu_score",
        "glsim_global_hallu_score",
        "glsim_local_hallu_score",
        "beyond_ads_hallu_score",
        "beyond_cgc_hallu_score",
        "beyond_adscgc_hallu_score",
    ]
    return {name: np.full(n, np.nan, dtype=np.float32) for name in names}


def _group_by_image(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["image_id"])].append(record)
    return dict(grouped)


def _get_lm_head(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.lm_head
    raise AttributeError("Cannot find language-model lm_head on this model")


def _normalize(vec):
    import torch

    return torch.nn.functional.normalize(vec.float(), p=2, dim=-1)


def _entropy_from_probs(probs):
    import torch

    probs = probs.float().clamp_min(_EPS)
    return -(probs * probs.log()).sum(dim=-1)


def _attention_entropy(row):
    import torch

    row = row.float().clamp_min(_EPS)
    row = row / row.sum().clamp_min(_EPS)
    return -(row * row.log()).sum()


def _safe_layer_index(num_hidden_layers: int, preferred: int) -> int:
    return max(0, min(preferred, num_hidden_layers - 1))


def compute_model_baselines(
    records: list[dict[str, Any]],
    generation_json: str | Path,
    model_path: str,
    coco_path: str | Path,
    device_index: int = 0,
    glsim_top_k: int = 32,
    glsim_w: float = 0.6,
    text_layer: int = 31,
    image_layer: int = 32,
    beyond_layer: int = 1,
) -> dict[str, np.ndarray]:
    import sys

    import torch
    from PIL import Image
    from tqdm import tqdm

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from sinkdetect.sink_utils import find_vis_bounds
    from sinkdetect.utils import build_caption_prompt, load_model_and_processor, partition_tokens

    n = len(records)
    scores = _nan_scores(n)
    if n == 0:
        return scores

    device = torch.device(f"cuda:{device_index}")
    model, processor = load_model_and_processor(model_path, device)
    lm_head = _get_lm_head(model)
    prompt_text = build_caption_prompt()
    grouped = _group_by_image(records)

    with open(generation_json, "r", encoding="utf-8") as f:
        generation = {int(x["image_id"]): x for x in json.load(f)}

    for image_id, image_records in tqdm(grouped.items(), desc="Computing baselines"):
        entry = generation.get(int(image_id))
        if entry is None:
            continue
        img_path = Path(coco_path) / "val2014" / f"COCO_val2014_{int(image_id):012d}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        full_text = prompt_text + " " + str(entry["caption"])
        inputs = processor(images=image, text=full_text, return_tensors="pt").to(device, dtype=torch.float16)
        img_token_id = getattr(processor, "image_token_id", None)
        if img_token_id is None:
            img_token_id = getattr(processor, "image_token_index", 32000)
        try:
            vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], img_token_id)
        except ValueError:
            continue

        with torch.inference_mode():
            outputs = model.forward(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
            )

        input_ids = inputs["input_ids"][0]
        prompt_end_idx = int(entry["prompt_end_idx"])
        token_masks = partition_tokens(input_ids, processor, prompt_end_idx)
        image_mask = token_masks["image_mask"]
        logits = outputs.logits[0].detach()
        hidden_states = tuple(h.detach() for h in outputs.hidden_states)
        attentions = tuple(a.detach() for a in outputs.attentions)
        num_attn_layers = len(attentions)
        num_hidden_layers = len(hidden_states)
        text_l = _safe_layer_index(num_hidden_layers, text_layer)
        image_l = _safe_layer_index(num_hidden_layers, image_layer)
        beyond_l = _safe_layer_index(num_attn_layers, beyond_layer)

        layer0_mean = attentions[0].mean(dim=1).squeeze(0)
        beyond_mean = attentions[beyond_l].mean(dim=1).squeeze(0)
        prompt_global = hidden_states[image_l][0, max(0, prompt_end_idx - 1), :]
        visual_hidden = hidden_states[image_l][0, vis_start:vis_end, :]

        for record in image_records:
            object_id = int(record["object_id"])
            token_pos = int(record["token_pos"])
            target_pos = token_pos + 1
            if token_pos < 0 or target_pos >= input_ids.numel() or token_pos >= logits.shape[0]:
                continue

            target_token = int(input_ids[target_pos].item())
            log_probs = torch.nn.functional.log_softmax(logits[token_pos].float(), dim=-1)
            probs = torch.nn.functional.softmax(logits[token_pos].float(), dim=-1)
            scores["nll_hallu_score"][object_id] = float(-log_probs[target_token].item())
            scores["entropy_hallu_score"][object_id] = float(_entropy_from_probs(probs).item())

            prelim_mass = layer0_mean[token_pos, prompt_end_idx:].sum()
            image_mass = layer0_mean[token_pos, image_mask].sum()
            scores["pas_layer0_hallu_score"][object_id] = float(prelim_mass.item())
            scores["svar_layer0_hallu_score"][object_id] = float(-image_mass.item())

            # IC: visual logit-lens confidence for the generated object token.
            # Grounded objects should be more visually predictable, so hallu_score is negated.
            visual_logits = lm_head(visual_hidden).float()
            visual_probs = torch.nn.functional.softmax(visual_logits, dim=-1)[:, target_token]
            visual_conf = visual_probs.max()
            scores["ic_hallu_score"][object_id] = float(-visual_conf.item())

            token_hidden = hidden_states[text_l][0, target_pos, :]
            token_norm = _normalize(token_hidden)
            global_cos = torch.dot(_normalize(prompt_global), token_norm)

            k = min(int(glsim_top_k), visual_hidden.shape[0])
            topk_idx = torch.topk(visual_probs, k=k).indices
            topk_hidden = visual_hidden[topk_idx]
            local_cos = torch.matmul(_normalize(topk_hidden), token_norm).mean()
            glsim_grounded = glsim_w * global_cos + (1.0 - glsim_w) * local_cos
            scores["glsim_global_hallu_score"][object_id] = float(-global_cos.item())
            scores["glsim_local_hallu_score"][object_id] = float(-local_cos.item())
            scores["glsim_hallu_score"][object_id] = float(-glsim_grounded.item())

            # Beyond Global Scores paper-level reimplementation.
            image_row = beyond_mean[token_pos, vis_start:vis_end].float()
            image_row = image_row / image_row.sum().clamp_min(_EPS)
            norm_entropy = _attention_entropy(image_row) / np.log(max(2, image_row.numel()))
            ads_grounded = 1.0 - norm_entropy
            top_attn_idx = torch.topk(image_row, k=k).indices
            cgc_grounded = torch.matmul(_normalize(visual_hidden[top_attn_idx]), token_norm).mean()
            adscgc_grounded = 0.5 * ads_grounded + 0.5 * cgc_grounded
            scores["beyond_ads_hallu_score"][object_id] = float(norm_entropy.item())
            scores["beyond_cgc_hallu_score"][object_id] = float(-cgc_grounded.item())
            scores["beyond_adscgc_hallu_score"][object_id] = float(-adscgc_grounded.item())

        del inputs, outputs, logits, hidden_states, attentions
        gc.collect()
        torch.cuda.empty_cache()

    return scores
