"""
Utility functions for model loading, token partitioning, and prompt construction.
"""

import json
from itertools import islice
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor


# ─── Prompt Construction ─────────────────────────────────────────────────────

LLAVA_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
LLAVA_USER_PROMPT = "Please help me describe the image in detail."


def build_caption_prompt() -> str:
    """Return the standard LLaVA-1.5 captioning prompt (matches PAS exactly).

    Format: "{system} USER: <image>\\n{user} ASSISTANT:"
    """
    return (
        f"{LLAVA_SYSTEM_PROMPT} "
        f"USER: <image>\n{LLAVA_USER_PROMPT} ASSISTANT:"
    )


# ─── Token Partitioning ──────────────────────────────────────────────────────

def find_prompt_end_idx(output_ids: Sequence[int]) -> int:
    """Find the index of the first generated (non-prompt) token in LLaVA output.

    Searches for the 5-token pattern (319, 1799, 9047, 13566, 29901)
    which corresponds to "_ASSISTANT:" in the LLaVA tokenizer.
    """
    pattern = (319, 1799, 9047, 13566, 29901)
    n = len(pattern)
    for j in range(len(output_ids) - n + 1):
        if tuple(output_ids[j : j + n]) == pattern:
            return j + n
    raise ValueError("Cannot find prompt end idx (ASSISTANT: pattern)")


def partition_tokens(
    output_ids: torch.Tensor,
    processor,
    prompt_end_idx: int,
    trunc_last: bool = False,
) -> dict:
    """Partition output tokens into BOS, image, instruction, and output groups.

    Args:
        output_ids: 1-D tensor of token IDs (length = seq_len of the forward pass).
        processor: LlavaProcessor instance.
        prompt_end_idx: Index of the first generated token.
        trunc_last: If True, drop the last token (legacy behavior). PAS uses False.

    Returns:
        Dict with bos_mask, image_mask, instruction_mask, output_mask — bool
        tensors on the same device as output_ids.
    """
    if trunc_last:
        output_ids = output_ids[:-1]

    output_ids_dev = output_ids.detach()

    # LLaVA processors may use either `image_token_id` (newer) or
    # `image_token_index` (older). Fall back gracefully.
    img_id = getattr(processor, "image_token_id", None)
    if img_id is None:
        img_id = getattr(processor, "image_token_index", 32000)

    image_mask = output_ids_dev == img_id
    bos_mask = output_ids_dev == processor.tokenizer.bos_token_id

    output_mask = torch.zeros_like(output_ids_dev, dtype=torch.bool)
    output_mask[prompt_end_idx:] = True

    instruction_mask = torch.ones_like(output_ids_dev, dtype=torch.bool)
    instruction_mask[image_mask | output_mask | bos_mask] = False

    return {
        "bos_mask": bos_mask,
        "image_mask": image_mask,
        "instruction_mask": instruction_mask,
        "output_mask": output_mask,
    }


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_processor(
    model_path: str,
    device: torch.device,
    attn_implementation: str = "eager",
    dtype: torch.dtype = torch.float16,
) -> tuple[LlavaForConditionalGeneration, LlavaProcessor]:
    """Load LlavaForConditionalGeneration and LlavaProcessor.

    ``model_path`` can be either an HF repo id (resolved via ``$HF_HOME``) or
    an absolute local path containing ``config.json`` directly. We do NOT pass
    ``local_files_only=True`` — that flag refuses to look up the snapshot from
    the HF cache hub-style layout in some transformers versions. Instead we
    rely on the cache; if you want to enforce offline mode export
    ``TRANSFORMERS_OFFLINE=1`` before running.
    """
    processor = LlavaProcessor.from_pretrained(model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    return model, processor


# ─── COCO Dataset Loading ────────────────────────────────────────────────────

def load_coco_val2014(
    coco_path: str,
    num_samples: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Load COCO val2014 image metadata.

    Args:
        coco_path: Root directory containing annotations/ and val2014/.
        num_samples: Number of images to sample (0 = all).
        seed: Random seed for sampling.

    Returns:
        List of dicts with image_id and file_name.
    """
    from pycocotools.coco import COCO

    anno_path = Path(coco_path) / "annotations" / "instances_val2014.json"
    coco = COCO(str(anno_path))
    img_ids = list(coco.getImgIds())

    if num_samples > 0 and num_samples < len(img_ids):
        import random
        rng = random.Random(seed)
        img_ids = sorted(rng.sample(img_ids, num_samples))

    imgs = coco.loadImgs(img_ids)
    return [{"image_id": img["id"], "file_name": img["file_name"]} for img in imgs]


# ─── Generation Data I/O ─────────────────────────────────────────────────────

def load_generation_json(path: str) -> list[dict]:
    """Load generation.json output from Stage 1.

    Each entry has: image_id, caption, output_ids, prompt_end_idx.
    """
    with open(path) as f:
        data = json.load(f)
    return data
