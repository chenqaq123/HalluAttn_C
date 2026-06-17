"""Shared OWLv2 image-object score cache helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


def encode_image_object_scores(
    model: Owlv2ForObjectDetection,
    processor: Owlv2Processor,
    images: list[Image.Image],
    prompts: list[str],
    object_names: list[str],
    device: torch.device,
) -> list[dict[str, float]]:
    text = [prompts for _ in images]
    inputs = processor(text=text, images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        query_scores = outputs.logits.sigmoid().amax(dim=1).detach().cpu()
    return [
        {obj: float(score) for obj, score in zip(object_names, image_scores)}
        for image_scores in query_scores
    ]


def load_score_cache(
    path: str | Path,
    required_image_keys: list[str],
    required_objects: list[str],
) -> dict[str, dict[str, float]]:
    data = np.load(path, allow_pickle=False)
    image_keys = [str(x) for x in data["image_keys"]]
    object_names = [str(x) for x in data["object_names"]]
    scores = data["scores"].astype(np.float32)

    image_index = {key: idx for idx, key in enumerate(image_keys)}
    object_index = {name: idx for idx, name in enumerate(object_names)}
    missing_images = sorted(set(required_image_keys) - set(image_index))
    missing_objects = sorted(set(required_objects) - set(object_index))
    if missing_images:
        raise KeyError(f"Image-score cache is missing {len(missing_images)} images; first={missing_images[:5]}")
    if missing_objects:
        raise KeyError(f"Image-score cache is missing objects: {missing_objects[:10]}")

    return {
        image_key: {
            obj: float(scores[image_index[image_key], object_index[obj]])
            for obj in required_objects
        }
        for image_key in required_image_keys
    }


def save_score_cache(
    path: str | Path,
    image_scores: dict[str, dict[str, float]],
    object_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_keys = sorted(image_scores)
    scores = np.asarray(
        [[image_scores[key][obj] for obj in object_names] for key in image_keys],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        image_keys=np.asarray(image_keys, dtype="U128"),
        object_names=np.asarray(object_names, dtype="U64"),
        scores=scores,
    )
