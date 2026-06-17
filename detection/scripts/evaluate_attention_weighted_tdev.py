#!/usr/bin/env python3
"""Evaluate attention-weighted CLIP TDEV scores on object mentions.

This is a detection-side TDEV audit. For each generated object mention, it
compares CLIP patch evidence for the mentioned target object against its COCO
co-occurrence neighbors, weighting CLIP's 7x7 patch scores by the cached LLaVA
object-token attention row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.data.build_object_cache import read_jsonl
from detection.baselines.metrics import compute_metric_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate attention-weighted CLIP TDEV on CHAIR object mentions")
    p.add_argument("--row_cache", default="experiments/coco_llava_7b_rows/attention_row_cache.npz")
    p.add_argument("--object_cache", default="detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--coco_path", default="/home/chenguanxu/common_dataset/coco-2014-dataset")
    p.add_argument("--clip_model_path", default="/home/chenguanxu/common_model/huggingface/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    p.add_argument("--output_dir", default="detection/baselines/results/attention_weighted_tdev")
    p.add_argument("--attention_key", default="orig_obj", choices=["orig_obj", "no_rope_obj"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_neighbors", type=int, default=10)
    return p.parse_args()


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def coco_image_path(coco_path: Path, image_id: int) -> Path:
    return coco_path / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    objects: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    prompts = [f"a photo of a {obj}" for obj in objects]
    with torch.inference_mode():
        for start in range(0, len(objects), batch_size):
            batch_objects = objects[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            inputs = processor(text=batch_prompts, padding=True, return_tensors="pt").to(device)
            feats = l2_normalize(model.get_text_features(**inputs)).detach().cpu()
            for obj, feat in zip(batch_objects, feats):
                embeddings[obj] = feat
    return embeddings


def encode_image_patches(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        if hasattr(model.vision_model, "post_layernorm"):
            patch_tokens = model.vision_model.post_layernorm(patch_tokens)
        patch_features = model.visual_projection(patch_tokens)
        return l2_normalize(patch_features).detach().cpu()


def resize_attention_to_clip_grid(attention_rows: np.ndarray, clip_grid: int = 7) -> np.ndarray:
    """Resize N x L x 576 LLaVA attention rows to N x L x 49 CLIP weights."""
    if attention_rows.ndim != 3 or attention_rows.shape[-1] != 576:
        raise ValueError(f"Expected attention rows with shape N x L x 576, got {attention_rows.shape}")
    tensor = torch.from_numpy(attention_rows.astype(np.float32)).reshape(-1, 1, 24, 24)
    resized = F.interpolate(tensor, size=(clip_grid, clip_grid), mode="bilinear", align_corners=False)
    weights = resized.reshape(attention_rows.shape[0], attention_rows.shape[1], clip_grid * clip_grid).numpy()
    weights = np.clip(weights, 0.0, None)
    denom = weights.sum(axis=-1, keepdims=True)
    return weights / np.clip(denom, 1e-12, None)


def read_neighbors(path: Path, top_k: int) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        obj: [item["object"] for item in items[:top_k]]
        for obj, items in raw.items()
    }


def score_rows(
    records: list[dict],
    patch_cache: dict[int, torch.Tensor],
    text_embeddings: dict[str, torch.Tensor],
    neighbors: dict[str, list[str]],
    attn_weights: np.ndarray,
    layer_indices: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    scores: dict[str, np.ndarray] = {
        f"tdev_attn_margin_layer_{int(layer_id)}": np.full(len(records), np.nan, dtype=np.float32)
        for layer_id in layer_indices
    }
    scores["tdev_attn_margin_mean"] = np.full(len(records), np.nan, dtype=np.float32)
    scores["tdev_attn_patch_margin_mean"] = np.full(len(records), np.nan, dtype=np.float32)

    rows: list[dict] = []
    for i, record in enumerate(records):
        target = str(record["word"])
        target_feat = text_embeddings.get(target)
        patch_feat = patch_cache[int(record["image_id"])]
        neighbor_names = [name for name in neighbors.get(target, []) if name in text_embeddings]
        neighbor_feats = torch.stack([text_embeddings[name] for name in neighbor_names]) if neighbor_names else None

        if target_feat is None:
            rows.append({**record, "neighbors": "", "best_neighbor": "", "tdev_margin_mean": math.nan})
            continue

        target_patch_scores = patch_feat @ target_feat
        if neighbor_feats is None:
            best_neighbor_patch_scores = torch.zeros_like(target_patch_scores)
            best_neighbor_name = ""
        else:
            neighbor_matrix = patch_feat @ neighbor_feats.T
            best_neighbor_patch_scores, best_neighbor_idx = torch.max(neighbor_matrix, dim=1)
            pooled_neighbor = torch.from_numpy(attn_weights[i]).float() @ neighbor_matrix
            pooled_neighbor_scores, pooled_neighbor_idx = torch.max(pooled_neighbor, dim=1)
            best_layer = int(torch.argmax(pooled_neighbor_scores).item())
            best_neighbor_name = neighbor_names[int(pooled_neighbor_idx[best_layer].item())]

        layer_margins = []
        layer_patch_margins = []
        weights = torch.from_numpy(attn_weights[i]).float()
        for layer_pos, layer_id in enumerate(layer_indices):
            target_score = torch.dot(weights[layer_pos], target_patch_scores)
            if neighbor_feats is None:
                neighbor_score = torch.tensor(0.0)
            else:
                neighbor_score = pooled_neighbor_scores[layer_pos]
            margin = float(target_score - neighbor_score)
            patch_margin = float(torch.dot(weights[layer_pos], target_patch_scores - best_neighbor_patch_scores))
            scores[f"tdev_attn_margin_layer_{int(layer_id)}"][i] = -margin
            layer_margins.append(margin)
            layer_patch_margins.append(patch_margin)

        mean_margin = float(np.mean(layer_margins))
        scores["tdev_attn_margin_mean"][i] = -mean_margin
        scores["tdev_attn_patch_margin_mean"][i] = -float(np.mean(layer_patch_margins))
        rows.append({
            **record,
            "neighbors": "|".join(neighbor_names),
            "best_neighbor": best_neighbor_name,
            "tdev_margin_mean": mean_margin,
            "tdev_patch_margin_mean": float(np.mean(layer_patch_margins)),
        })
    return rows, scores


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.object_cache)
    if args.limit > 0:
        records = records[: args.limit]

    row_data = np.load(args.row_cache)
    row_count = len(row_data["labels"])
    if len(records) > row_count:
        raise ValueError(f"object cache has {len(records)} rows but row cache has {row_count}")
    labels = row_data["labels"][: len(records)].astype(np.int32)
    image_ids = row_data["image_ids"][: len(records)].astype(np.int64)
    token_pos = row_data["token_pos"][: len(records)].astype(np.int32)
    layer_indices = row_data["layer_indices"].astype(np.int32)
    for idx, record in enumerate(records):
        if int(record["object_id"]) != idx:
            raise ValueError(f"object_id mismatch at row {idx}: {record['object_id']}")
        if int(record["image_id"]) != int(image_ids[idx]) or int(record["token_pos"]) != int(token_pos[idx]):
            raise ValueError(f"object cache and row cache are not aligned at row {idx}")

    attn_weights = resize_attention_to_clip_grid(row_data[args.attention_key][: len(records)])
    neighbors = read_neighbors(Path(args.neighbors_json), top_k=args.top_neighbors)
    objects = {str(record["word"]) for record in records}
    for word in list(objects):
        objects.update(neighbors.get(word, []))

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = CLIPModel.from_pretrained(args.clip_model_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_path, local_files_only=True)
    model.eval()

    text_embeddings = encode_texts(model, processor, sorted(objects), device, args.batch_size)
    patch_cache: dict[int, torch.Tensor] = {}
    unique_images = sorted({int(record["image_id"]) for record in records})
    coco_path = Path(args.coco_path)
    for start in range(0, len(unique_images), args.batch_size):
        batch_ids = unique_images[start : start + args.batch_size]
        images = [Image.open(coco_image_path(coco_path, image_id)).convert("RGB") for image_id in batch_ids]
        feats = encode_image_patches(model, processor, images, device)
        for image in images:
            image.close()
        for image_id, feat in zip(batch_ids, feats):
            patch_cache[image_id] = feat

    scored_rows, scores = score_rows(records, patch_cache, text_embeddings, neighbors, attn_weights, layer_indices)
    gen_pos = np.asarray([int(record["gen_pos"]) for record in records], dtype=np.int32)
    metrics = {
        name: compute_metric_bundle(labels, values, gen_pos)
        for name, values in scores.items()
    }
    payload = {
        "row_cache": args.row_cache,
        "object_cache": args.object_cache,
        "neighbors_json": args.neighbors_json,
        "clip_model_path": args.clip_model_path,
        "attention_key": args.attention_key,
        "top_neighbors": args.top_neighbors,
        "samples": len(records),
        "hallucinated": int(labels.sum()),
        "layer_indices": layer_indices.astype(int).tolist(),
        "score_direction": "larger score means more likely hallucinated; scores are negative TDEV margins",
        "metrics": metrics,
    }

    write_csv(output_dir / "attention_weighted_tdev_scores.csv", scored_rows)
    np.savez(output_dir / "attention_weighted_tdev_scores.npz", labels=labels, gen_pos=gen_pos, **scores)
    with (output_dir / "attention_weighted_tdev_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'attention_weighted_tdev_metrics.json'}")


if __name__ == "__main__":
    main()
