#!/usr/bin/env python3
"""Build POPE negative subsets based on COCO co-occurrence neighbors."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


QUESTION_RE = re.compile(
    r"^(?:is|are)\s+there\s+(?:any|an|a|the)?\s*(?P<object>.+?)\s+in\s+the\s+ima(?:ge|nge)\??$",
    re.IGNORECASE,
)
IMAGE_ID_RE = re.compile(r"(\d{12}|\d+)(?:\.[A-Za-z]+)?$")

ALIASES = {
    "airplanes": "airplane",
    "bicycles": "bicycle",
    "birds": "bird",
    "boats": "boat",
    "bottles": "bottle",
    "buses": "bus",
    "cars": "car",
    "cats": "cat",
    "chairs": "chair",
    "couches": "couch",
    "dogs": "dog",
    "donuts": "donut",
    "elephants": "elephant",
    "fire hydrants": "fire hydrant",
    "frisbees": "frisbee",
    "giraffes": "giraffe",
    "hair driers": "hair drier",
    "horses": "horse",
    "hot dogs": "hot dog",
    "keyboards": "keyboard",
    "knives": "knife",
    "laptops": "laptop",
    "microwaves": "microwave",
    "motorcycles": "motorcycle",
    "mouses": "mouse",
    "oranges": "orange",
    "ovens": "oven",
    "parking meters": "parking meter",
    "persons": "person",
    "people": "person",
    "pizzas": "pizza",
    "refrigerators": "refrigerator",
    "sandwiches": "sandwich",
    "scissors": "scissors",
    "sheep": "sheep",
    "skateboards": "skateboard",
    "skate boards": "skateboard",
    "skis": "skis",
    "snowboards": "snowboard",
    "sports balls": "sports ball",
    "stop signs": "stop sign",
    "suitcases": "suitcase",
    "surfboards": "surfboard",
    "teddy bears": "teddy bear",
    "tennis rackets": "tennis racket",
    "ties": "tie",
    "toilets": "toilet",
    "toothbrushes": "toothbrush",
    "traffic lights": "traffic light",
    "trains": "train",
    "trucks": "truck",
    "umbrellas": "umbrella",
    "vases": "vase",
    "wine glasses": "wine glass",
    "zebras": "zebra",
}


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pope_jsonl_path(question_dir: Path, split: str) -> Path:
    candidates = [
        question_dir / "coco" / f"coco_pope_{split}.jsonl",
        question_dir / f"coco_pope_{split}.jsonl",
        question_dir / f"coco_pope_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find COCO POPE {split!r} questions under {question_dir}")


def iter_pope_records(question_dir: str | Path, coco_path: str | Path, split: str, limit: int = 0) -> list[dict]:
    question_dir = Path(question_dir)
    coco_path = Path(coco_path)
    parquet_path = question_dir / "Full" / f"{split}-00000-of-00001.parquet"

    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyarrow is required for parquet-format POPE data") from exc
        rows = pq.read_table(
            parquet_path,
            columns=["question_id", "question", "answer", "image_source"],
        ).to_pylist()
        normalized = [
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "label": str(row["answer"]).lower(),
                "image": row.get("image_source", ""),
            }
            for row in rows
        ]
    else:
        rows = _read_jsonl(_pope_jsonl_path(question_dir, split))
        normalized = [
            {
                "question_id": row.get("question_id", row.get("qid")),
                "question": row.get("text", row.get("question", row.get("prompt"))),
                "label": str(row.get("label", row.get("answer"))).lower(),
                "image": row["image"],
            }
            for row in rows
        ]
    return normalized[:limit] if limit > 0 else normalized


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag POPE rows with COCO co-occurrence-neighbor negative subsets")
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--coco_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def normalize_object(text: str) -> str:
    obj = re.sub(r"\s+", " ", text.strip().lower())
    obj = obj.rstrip("?.")
    if obj in ALIASES:
        return ALIASES[obj]
    if obj.endswith("ies") and len(obj) > 3:
        candidate = obj[:-3] + "y"
        return ALIASES.get(candidate, candidate)
    if obj.endswith("es") and len(obj) > 3:
        candidate = obj[:-2]
        return ALIASES.get(candidate, candidate)
    if obj.endswith("s") and not obj.endswith("ss") and len(obj) > 3:
        candidate = obj[:-1]
        return ALIASES.get(candidate, candidate)
    return obj


def extract_target(question: str) -> str:
    match = QUESTION_RE.match(question.strip())
    if not match:
        return ""
    return normalize_object(match.group("object"))


def extract_image_id(image_name: str) -> int | None:
    stem = Path(str(image_name)).stem
    match = IMAGE_ID_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


def load_coco_index(coco_path: Path) -> tuple[dict[int, set[str]], dict[str, str], dict[str, list[str]]]:
    annotation_path = coco_path / "annotations" / "instances_val2014.json"
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    category_by_id = {cat["id"]: cat["name"].lower() for cat in data["categories"]}
    supercategory_by_name = {
        cat["name"].lower(): cat.get("supercategory", "").lower() for cat in data["categories"]
    }
    present_by_image: dict[int, set[str]] = defaultdict(set)
    for ann in data["annotations"]:
        present_by_image[int(ann["image_id"])].add(category_by_id[int(ann["category_id"])])

    categories_by_super: dict[str, list[str]] = defaultdict(list)
    for name, supercat in supercategory_by_name.items():
        categories_by_super[supercat].append(name)
    return dict(present_by_image), supercategory_by_name, dict(categories_by_super)


def build_neighbors(present_by_image: dict[int, set[str]], top_k: int) -> dict[str, list[dict]]:
    object_counts: Counter[str] = Counter()
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for objects in present_by_image.values():
        for obj in objects:
            object_counts[obj] += 1
        for obj in objects:
            for other in objects:
                if other != obj:
                    pair_counts[obj][other] += 1

    neighbors: dict[str, list[dict]] = {}
    for obj, counts in pair_counts.items():
        ranked = []
        for other, count in counts.items():
            union = object_counts[obj] + object_counts[other] - count
            ranked.append(
                {
                    "object": other,
                    "cooccur_count": count,
                    "target_count": object_counts[obj],
                    "neighbor_count": object_counts[other],
                    "jaccard": count / union if union else 0.0,
                }
            )
        ranked.sort(key=lambda row: (row["jaccard"], row["cooccur_count"], row["object"]), reverse=True)
        neighbors[obj] = ranked[:top_k]
    return neighbors


def label_row(
    label: str,
    target: str,
    present: set[str],
    neighbors: dict[str, list[dict]],
    supercategory_by_name: dict[str, str],
    categories_by_super: dict[str, list[str]],
) -> tuple[str, list[str], list[str]]:
    if label == "yes":
        return "positive_present" if target in present else "positive_missing_coco_label", [], []
    if target in present:
        return "negative_target_present_coco_label", [], []

    neighbor_names = [row["object"] for row in neighbors.get(target, [])]
    cooccur_present = sorted(set(neighbor_names) & present)
    supercat = supercategory_by_name.get(target, "")
    sibling_names = set(categories_by_super.get(supercat, [])) - {target} if supercat else set()
    sibling_present = sorted(sibling_names & present)

    related_present = sorted(set(cooccur_present) | set(sibling_present))
    if related_present:
        return "negative_related_present", related_present, neighbor_names
    return "negative_absent_plain", [], neighbor_names


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    present_by_image, supercategory_by_name, categories_by_super = load_coco_index(Path(args.coco_path))
    neighbors = build_neighbors(present_by_image, args.top_k)
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    summary = {
        "top_k": args.top_k,
        "splits": {},
        "unknown_targets": Counter(),
        "unparsed_questions": 0,
        "missing_image_ids": 0,
    }
    all_rows = []

    for split in splits:
        rows = []
        counts: Counter[str] = Counter()
        for record in iter_pope_records(args.pope_dir, args.coco_path, split, limit=args.limit):
            question = str(record["question"])
            target = extract_target(question)
            image_id = extract_image_id(str(record["image"]))
            if not target:
                summary["unparsed_questions"] += 1
            if image_id is None:
                summary["missing_image_ids"] += 1
            present = present_by_image.get(image_id or -1, set())
            label = str(record["label"]).lower()
            if target and target not in supercategory_by_name:
                summary["unknown_targets"][target] += 1
            row_type, related_present, neighbor_names = label_row(
                label,
                target,
                present,
                neighbors,
                supercategory_by_name,
                categories_by_super,
            )
            counts[row_type] += 1
            row = {
                "split": split,
                "question_id": record["question_id"],
                "image_id": image_id,
                "label": label,
                "target": target,
                "question": question,
                "negative_type": row_type,
                "present_objects": "|".join(sorted(present)),
                "related_present": "|".join(related_present),
                "top_cooccurrence_neighbors": "|".join(neighbor_names),
            }
            rows.append(row)
            all_rows.append(row)

        split_path = output_dir / f"{split}_semantic_neighbor_rows.csv"
        with split_path.open("w", encoding="utf-8", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        total_negatives = counts["negative_related_present"] + counts["negative_absent_plain"] + counts["negative_target_present_coco_label"]
        summary["splits"][split] = {
            "rows": len(rows),
            "counts": dict(counts),
            "negative_related_present_rate": (
                counts["negative_related_present"] / total_negatives if total_negatives else 0.0
            ),
            "csv": str(split_path),
        }

    if all_rows:
        merged_path = output_dir / "semantic_neighbor_rows.csv"
        with merged_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        summary["merged_csv"] = str(merged_path)

    summary["unknown_targets"] = dict(summary["unknown_targets"])
    write_json(output_dir / "semantic_neighbor_summary.json", summary)
    write_json(output_dir / "cooccurrence_neighbors.json", neighbors)


if __name__ == "__main__":
    main()
