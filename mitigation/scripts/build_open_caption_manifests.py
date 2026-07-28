#!/usr/bin/env python3
"""Build immutable JSONL manifests for the registered image-caption matrix."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco_root",
        default="/home/chenguanxu/common_dataset/coco-2014-dataset",
    )
    parser.add_argument(
        "--allpath_benchmarks",
        default="/tmp/sinkdetect-allpath-audit/playground/benchmarks.py",
    )
    parser.add_argument(
        "--amber_root", default="/tmp/sinkdetect_datasets/AMBER"
    )
    parser.add_argument(
        "--throne_root", default="/tmp/sinkdetect_datasets/THRONE"
    )
    parser.add_argument(
        "--output_root",
        default="/tmp/sinkdetect_datasets/open_caption_manifests",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _chair_ids(path: str | Path) -> list[int]:
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "CHAIR":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "fixed_500"
                for target in statement.targets
            ):
                continue
            result = ast.literal_eval(statement.value)
            if not isinstance(result, list) or len(result) != 500:
                raise ValueError("CHAIR.fixed_500 is not a 500-item list")
            ids = [int(value) for value in result]
            if len(set(ids)) != 500:
                raise ValueError("CHAIR.fixed_500 contains duplicates")
            return ids
    raise ValueError(f"Could not find CHAIR.fixed_500 in {path}")


def _atomic_jsonl(path: Path, records: Iterable[dict]) -> tuple[int, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, sha256_file(path)


def _chair_records(coco_root: Path, ids: list[int]) -> Iterable[dict]:
    for image_id in ids:
        image = coco_root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        yield {
            "benchmark": "chair",
            "sample_id": image_id,
            "image_path": str(image.resolve()),
            "prompt": "Please describe this image in detail.",
        }


def _amber_records(amber_root: Path) -> Iterable[dict]:
    annotations_path = amber_root / "official" / "data" / "annotations.json"
    annotations = json.loads(annotations_path.read_text())
    selected = [
        item for item in annotations if str(item.get("type")) == "generative"
    ]
    if len(selected) != 1004:
        raise ValueError(
            f"Expected 1,004 AMBER generative items, got {len(selected)}"
        )
    for item in sorted(selected, key=lambda value: int(value["id"])):
        image_id = int(item["id"])
        image = amber_root / "image" / f"AMBER_{image_id}.jpg"
        if not image.is_file():
            raise FileNotFoundError(image)
        yield {
            "benchmark": "amber-generative",
            "sample_id": image_id,
            "image_path": str(image.resolve()),
            "prompt": "Describe this image.",
        }


def _throne_records(throne_root: Path) -> Iterable[dict]:
    annotation_path = (
        throne_root / "metadata" / "zhiyuan_objv2_val_fixname.json"
    )
    subset_path = throne_root / "metadata" / "objects365_subset.txt"
    annotation = json.loads(annotation_path.read_text())
    subset = {
        int(value) for value in subset_path.read_text().split() if value.strip()
    }
    image_by_id = {
        int(image["id"]): image for image in annotation["images"]
    }
    if len(subset) != 5110 or not subset.issubset(image_by_id):
        raise ValueError("Invalid THRONE Objects365 subset")
    for image_id in sorted(subset):
        relative = Path(image_by_id[image_id]["file_name"])
        image = throne_root / "objects365" / relative
        if not image.is_file():
            raise FileNotFoundError(image)
        yield {
            "benchmark": "throne-objects365",
            "sample_id": image_id,
            "image_path": str(image.resolve()),
            "prompt": "Describe this image in detail.",
        }


def main() -> None:
    args = parse_args()
    coco_root = Path(args.coco_root)
    amber_root = Path(args.amber_root)
    throne_root = Path(args.throne_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ids = _chair_ids(args.allpath_benchmarks)
    outputs = {}
    for name, records in [
        ("chair", _chair_records(coco_root, ids)),
        ("amber-generative", _amber_records(amber_root)),
        ("throne-objects365", _throne_records(throne_root)),
    ]:
        path = output_root / f"{name}.jsonl"
        count, digest = _atomic_jsonl(path, records)
        outputs[name] = {
            "path": str(path.resolve()),
            "count": count,
            "sha256": digest,
        }

    source_files = {
        "chair_id_source": Path(args.allpath_benchmarks),
        "coco_instances": coco_root
        / "annotations"
        / "instances_val2014.json",
        "coco_captions": coco_root
        / "annotations"
        / "captions_val2014.json",
        "amber_annotations": amber_root
        / "official"
        / "data"
        / "annotations.json",
        "throne_annotations": throne_root
        / "metadata"
        / "zhiyuan_objv2_val_fixname.json",
        "throne_subset": throne_root
        / "metadata"
        / "objects365_subset.txt",
    }
    metadata = {
        "outputs": outputs,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in source_files.items()
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
