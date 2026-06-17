#!/usr/bin/env python3
"""Evaluate a CLIP-based TDEV-zero verifier on POPE.

TDEV-zero scores an object query by comparing CLIP image-text compatibility for
the target object against the strongest COCO co-occurrence neighbor:

    margin = sim(image, "a photo of a {target}") - max_c sim(image, "a photo of a {c}")

The default prediction is "yes" when the margin is positive. This is a
training-free target-discriminative verifier, not a calibrated classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_semantic_neighbor_audit import extract_image_id, iter_pope_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CLIP TDEV-zero on POPE semantic-neighbor subsets")
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--coco_path", required=True)
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--neighbors_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--clip_model_path", default="/home/chenguanxu/common_model/huggingface/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--margin_threshold", type=float, default=0.0)
    p.add_argument("--evidence_mode", choices=["global", "patch_max", "patch_margin_max"], default="global")
    return p.parse_args()


def read_audit(path: Path) -> dict[tuple[str, str], dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {(row["split"], str(row["question_id"])): row for row in csv.DictReader(f)}


def load_image(coco_path: Path, image_name: str) -> Image.Image:
    image_id = extract_image_id(image_name)
    if image_id is None:
        candidate = coco_path / "val2014" / image_name
    else:
        candidate = coco_path / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
    return Image.open(candidate).convert("RGB")


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_texts(model: CLIPModel, processor: CLIPProcessor, objects: list[str], device: torch.device, batch_size: int) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    prompts = [f"a photo of a {obj}" for obj in objects]
    with torch.inference_mode():
        for start in range(0, len(objects), batch_size):
            batch_objects = objects[start:start + batch_size]
            batch_prompts = prompts[start:start + batch_size]
            inputs = processor(text=batch_prompts, padding=True, return_tensors="pt").to(device)
            feats = l2_normalize(model.get_text_features(**inputs)).detach().cpu()
            for obj, feat in zip(batch_objects, feats):
                embeddings[obj] = feat
    return embeddings


def encode_images_global(model: CLIPModel, processor: CLIPProcessor, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        return l2_normalize(model.get_image_features(**inputs)).detach().cpu()


def encode_images_patch_max(model: CLIPModel, processor: CLIPProcessor, images: list[Image.Image], device: torch.device) -> list[torch.Tensor]:
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode():
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
        if hasattr(model.vision_model, "post_layernorm"):
            patch_tokens = model.vision_model.post_layernorm(patch_tokens)
        patch_features = model.visual_projection(patch_tokens)
        patch_features = l2_normalize(patch_features).detach().cpu()
    return [patch_features[idx] for idx in range(patch_features.shape[0])]


def score_evidence(image_feat: torch.Tensor, text_feat: torch.Tensor) -> tuple[float, int]:
    if image_feat.ndim == 1:
        return float(torch.dot(image_feat, text_feat)), -1
    patch_scores = image_feat @ text_feat
    value, index = torch.max(patch_scores, dim=0)
    return float(value), int(index)


def score_tdev_margin(
    image_feat: torch.Tensor,
    target_feat: torch.Tensor,
    neighbor_feats: list[tuple[str, torch.Tensor]],
    evidence_mode: str,
) -> tuple[float, int, str, float, int, float]:
    if not neighbor_feats:
        target_score, target_patch = score_evidence(image_feat, target_feat)
        return target_score, target_patch, "", float("-inf"), -1, float("inf")

    if evidence_mode != "patch_margin_max" or image_feat.ndim == 1:
        target_score, target_patch = score_evidence(image_feat, target_feat)
        neighbor_scores = [
            (neighbor, *score_evidence(image_feat, neighbor_feat))
            for neighbor, neighbor_feat in neighbor_feats
        ]
        best_neighbor, best_neighbor_score, best_neighbor_patch = max(neighbor_scores, key=lambda item: item[1])
        return target_score, target_patch, best_neighbor, best_neighbor_score, best_neighbor_patch, target_score - best_neighbor_score

    target_patch_scores = image_feat @ target_feat
    neighbor_matrix = torch.stack([image_feat @ feat for _, feat in neighbor_feats], dim=1)
    best_neighbor_scores, best_neighbor_indices = torch.max(neighbor_matrix, dim=1)
    patch_margins = target_patch_scores - best_neighbor_scores
    margin_value, patch_index = torch.max(patch_margins, dim=0)
    patch_idx = int(patch_index)
    neighbor_idx = int(best_neighbor_indices[patch_idx])
    best_neighbor, _ = neighbor_feats[neighbor_idx]
    return (
        float(target_patch_scores[patch_idx]),
        patch_idx,
        best_neighbor,
        float(best_neighbor_scores[patch_idx]),
        patch_idx,
        float(margin_value),
    )


def metrics(rows: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = row["prediction"]
        gold = row["label"]
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall_tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "positive":
        return [row for row in rows if row["label"] == "yes"]
    if subset == "negative":
        return [row for row in rows if row["label"] == "no"]
    return [row for row in rows if row.get("negative_type") == subset]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    audit = read_audit(Path(args.audit_csv))
    neighbors = json.loads(Path(args.neighbors_json).read_text())
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    rows: list[dict] = []
    objects: set[str] = set()
    for split in splits:
        for record in iter_pope_records(args.pope_dir, args.coco_path, split, limit=args.limit):
            key = (split, str(record["question_id"]))
            audit_row = audit[key]
            target = audit_row["target"]
            neighbor_names = [item["object"] for item in neighbors.get(target, [])]
            objects.add(target)
            objects.update(neighbor_names)
            rows.append({
                "split": split,
                "question_id": str(record["question_id"]),
                "image": record["image"],
                "label": record["label"],
                "target": target,
                "negative_type": audit_row["negative_type"],
                "neighbors": neighbor_names,
            })

    model = CLIPModel.from_pretrained(args.clip_model_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_path, local_files_only=True)
    model.eval()

    text_embeddings = encode_texts(model, processor, sorted(objects), device, args.batch_size)
    image_cache: dict[str, torch.Tensor] = {}
    image_keys = sorted({row["image"] for row in rows})
    for start in range(0, len(image_keys), args.batch_size):
        batch_keys = image_keys[start:start + args.batch_size]
        images = [load_image(Path(args.coco_path), key) for key in batch_keys]
        if args.evidence_mode == "global":
            feats = list(encode_images_global(model, processor, images, device))
        else:
            feats = encode_images_patch_max(model, processor, images, device)
        for image in images:
            image.close()
        for key, feat in zip(batch_keys, feats):
            image_cache[key] = feat

    scored = []
    for row in rows:
        image_feat = image_cache[row["image"]]
        neighbor_feats = [
            (neighbor, text_embeddings[neighbor])
            for neighbor in row["neighbors"]
            if neighbor in text_embeddings
        ]
        target_score, target_patch, best_neighbor, best_neighbor_score, best_neighbor_patch, margin = score_tdev_margin(
            image_feat,
            text_embeddings[row["target"]],
            neighbor_feats,
            args.evidence_mode,
        )
        prediction = "yes" if margin > args.margin_threshold else "no"
        scored.append({
            **row,
            "prediction": prediction,
            "evidence_mode": args.evidence_mode,
            "target_score": target_score,
            "target_patch": target_patch,
            "best_neighbor": best_neighbor,
            "best_neighbor_score": best_neighbor_score,
            "best_neighbor_patch": best_neighbor_patch,
            "tdev_margin": margin,
            "neighbors": "|".join(row["neighbors"]),
        })

    pred_path = output_dir / "clip_tdev_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    metric_rows = []
    summary = {
        "clip_model_path": args.clip_model_path,
        "evidence_mode": args.evidence_mode,
        "margin_threshold": args.margin_threshold,
        "subsets": {},
    }
    for split in splits + ["macro"]:
        split_rows = scored if split == "macro" else [row for row in scored if row["split"] == split]
        for subset in subsets:
            vals = metrics(subset_rows(split_rows, subset))
            summary["subsets"][f"{split}:{subset}"] = vals
            metric_rows.append({"split": split, "subset": subset, **vals})

    with (output_dir / "clip_tdev_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "clip_tdev_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)


if __name__ == "__main__":
    main()
