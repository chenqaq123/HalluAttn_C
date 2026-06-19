#!/usr/bin/env python3
"""Export POPE rows to AIR/LLaVA question JSONL format.

AIR's official runner expects LLaVA-style rows with question_id, image, and text.
Our local POPE copy is a HuggingFace/parquet layout, so this adapter keeps the
subset definition tied to the existing semantic-neighbor audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_semantic_neighbor_audit import extract_image_id, iter_pope_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a POPE subset for AIR official LLaVA runner")
    parser.add_argument("--pope_dir", required=True)
    parser.add_argument("--coco_path", required=True)
    parser.add_argument(
        "--audit_csv",
        default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv",
        help="CSV from build_semantic_neighbor_audit.py",
    )
    parser.add_argument("--split", default="adversarial")
    parser.add_argument(
        "--row_filter",
        default="all",
        choices=[
            "all",
            "positive",
            "negative",
            "negative_related_present",
            "negative_absent_plain",
            "negative_target_present_coco_label",
        ],
        help="Filter rows before applying --limit",
    )
    parser.add_argument("--limit", type=int, default=120, help="Maximum exported rows after filtering; 0 exports all")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--manifest_file", default="", help="Defaults to <output_file>.manifest.json")
    parser.add_argument("--skip_image_check", action="store_true")
    return parser.parse_args()


def read_audit_rows(path: Path, split: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows[str(row["question_id"])] = row
    if not rows:
        raise ValueError(f"No audit rows for split={split!r} in {path}")
    return rows


def canonical_coco_image(image_name: str) -> str:
    image_id = extract_image_id(image_name)
    if image_id is None:
        path = Path(str(image_name))
        return path.name if path.suffix else f"{path.name}.jpg"
    return f"COCO_val2014_{image_id:012d}.jpg"


def keep_row(record: dict, audit: dict[str, str], row_filter: str) -> bool:
    label = str(record["label"]).lower()
    if row_filter == "all":
        return True
    if row_filter == "positive":
        return label == "yes"
    if row_filter == "negative":
        return label == "no"
    return audit["negative_type"] == row_filter


def main() -> None:
    args = parse_args()
    output_file = Path(args.output_file)
    manifest_file = Path(args.manifest_file) if args.manifest_file else output_file.with_suffix(
        output_file.suffix + ".manifest.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    audit_rows = read_audit_rows(Path(args.audit_csv), args.split)
    records = iter_pope_records(args.pope_dir, args.coco_path, args.split)
    coco_val = Path(args.coco_path) / "val2014"

    exported = []
    missing_audit = []
    missing_images = []
    for record in records:
        question_id = str(record["question_id"])
        audit = audit_rows.get(question_id)
        if audit is None:
            missing_audit.append(question_id)
            continue
        if not keep_row(record, audit, args.row_filter):
            continue

        image = canonical_coco_image(str(record["image"]))
        if not args.skip_image_check and not (coco_val / image).exists():
            missing_images.append(image)
            continue

        exported.append(
            {
                "question_id": question_id,
                "image": image,
                "text": record["question"],
                "label": record["label"],
                "split": args.split,
                "target": audit["target"],
                "negative_type": audit["negative_type"],
                "source_image": record["image"],
            }
        )
        if args.limit > 0 and len(exported) >= args.limit:
            break

    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} COCO images, first={missing_images[:5]}")
    if not exported:
        raise ValueError(f"No rows exported for split={args.split!r}, row_filter={args.row_filter!r}")

    with output_file.open("w", encoding="utf-8") as f:
        for row in exported:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "pope_dir": args.pope_dir,
        "coco_path": args.coco_path,
        "audit_csv": args.audit_csv,
        "split": args.split,
        "row_filter": args.row_filter,
        "limit": args.limit,
        "output_file": str(output_file),
        "rows_exported": len(exported),
        "missing_audit_rows_seen_before_limit": len(missing_audit),
        "label_counts": dict(Counter(row["label"] for row in exported)),
        "negative_type_counts": dict(Counter(row["negative_type"] for row in exported)),
        "first_question_id": exported[0]["question_id"],
        "last_question_id": exported[-1]["question_id"],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(exported)} rows to {output_file}")
    print(f"Wrote manifest to {manifest_file}")


if __name__ == "__main__":
    main()
