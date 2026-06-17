#!/usr/bin/env python3
"""Evaluate an OWLv2 region-evidence TDEV verifier on POPE.

For each image, this script queries OWLv2 with every COCO object name used by
the semantic-neighbor audit. It then scores each POPE question with a
target-discriminative region margin:

    margin = max_box score(target) - max_neighbor max_box score(neighbor)

The output schema intentionally matches the CLIP TDEV prototype so the existing
gate evaluator can reuse the produced CSV.
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
from transformers import Owlv2ForObjectDetection, Owlv2Processor

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_semantic_neighbor_audit import extract_image_id, iter_pope_records
from owlv2_cache_utils import encode_image_object_scores, load_score_cache, save_score_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OWLv2 region-evidence TDEV on POPE")
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--coco_path", required=True)
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--neighbors_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--owlv2_model_path", default="/home/chenguanxu/common_model/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d")
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top_neighbors", type=int, default=10)
    p.add_argument("--margin_threshold", type=float, default=0.0)
    p.add_argument("--target_score_threshold", type=float, default=0.0)
    p.add_argument("--image_score_cache", default="", help="Optional NPZ cache from OWLv2 image-object scoring")
    p.add_argument("--save_image_score_cache", default="", help="Optional NPZ path to save image-object scores")
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


def canonical_image_key(image_name: str) -> str:
    image_id = extract_image_id(image_name)
    if image_id is None:
        return str(image_name)
    return f"COCO_val2014_{image_id:012d}.jpg"


def read_neighbors(path: Path, top_k: int) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        obj: [item["object"] for item in items[:top_k]]
        for obj, items in raw.items()
    }


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
    neighbors = read_neighbors(Path(args.neighbors_json), top_k=args.top_neighbors)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    rows: list[dict] = []
    object_names: set[str] = set()
    for split in splits:
        for record in iter_pope_records(args.pope_dir, args.coco_path, split, limit=args.limit):
            key = (split, str(record["question_id"]))
            audit_row = audit[key]
            target = audit_row["target"]
            neighbor_names = neighbors.get(target, [])
            object_names.add(target)
            object_names.update(neighbor_names)
            rows.append({
                "split": split,
                "question_id": str(record["question_id"]),
                "image": canonical_image_key(record["image"]),
                "label": record["label"],
                "target": target,
                "negative_type": audit_row["negative_type"],
                "neighbors": neighbor_names,
            })

    object_list = sorted(object_names)
    prompts = [f"a photo of a {obj}" for obj in object_list]
    image_keys = sorted({row["image"] for row in rows})
    if args.image_score_cache:
        image_scores = load_score_cache(args.image_score_cache, image_keys, object_list)
    else:
        processor = Owlv2Processor.from_pretrained(args.owlv2_model_path, local_files_only=True)
        model = Owlv2ForObjectDetection.from_pretrained(args.owlv2_model_path, local_files_only=True).to(device)
        model.eval()

        image_scores: dict[str, dict[str, float]] = {}
        coco_path = Path(args.coco_path)
        for start in range(0, len(image_keys), args.batch_size):
            batch_keys = image_keys[start:start + args.batch_size]
            images = [load_image(coco_path, key) for key in batch_keys]
            score_batch = encode_image_object_scores(model, processor, images, prompts, object_list, device)
            for image in images:
                image.close()
            for key, scores in zip(batch_keys, score_batch):
                image_scores[key] = scores
        if args.save_image_score_cache:
            save_score_cache(args.save_image_score_cache, image_scores, object_list)

    scored = []
    for row in rows:
        scores = image_scores[row["image"]]
        target_score = scores[row["target"]]
        neighbor_scores = [(name, scores[name]) for name in row["neighbors"] if name in scores]
        if neighbor_scores:
            best_neighbor, best_neighbor_score = max(neighbor_scores, key=lambda item: item[1])
            margin = target_score - best_neighbor_score
        else:
            best_neighbor, best_neighbor_score = "", float("-inf")
            margin = float("inf")
        prediction = "yes" if (
            target_score > args.target_score_threshold and margin > args.margin_threshold
        ) else "no"
        scored.append({
            **row,
            "prediction": prediction,
            "target_score": target_score,
            "best_neighbor": best_neighbor,
            "best_neighbor_score": best_neighbor_score,
            "tdev_margin": margin,
            "neighbors": "|".join(row["neighbors"]),
        })

    pred_path = output_dir / "owlv2_tdev_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)

    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain"]
    metric_rows = []
    summary = {
        "owlv2_model_path": args.owlv2_model_path,
        "margin_threshold": args.margin_threshold,
        "target_score_threshold": args.target_score_threshold,
        "image_score_cache": args.image_score_cache,
        "save_image_score_cache": args.save_image_score_cache,
        "top_neighbors": args.top_neighbors,
        "object_count": len(object_list),
        "subsets": {},
    }
    for split in splits + ["macro"]:
        split_rows = scored if split == "macro" else [row for row in scored if row["split"] == split]
        for subset in subsets:
            vals = metrics(subset_rows(split_rows, subset))
            summary["subsets"][f"{split}:{subset}"] = vals
            metric_rows.append({"split": split, "subset": subset, **vals})

    with (output_dir / "owlv2_tdev_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    with (output_dir / "owlv2_tdev_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"Wrote {output_dir / 'owlv2_tdev_metrics.json'}")


if __name__ == "__main__":
    main()
