#!/usr/bin/env python3
"""Audit where verified atomic-detail caption gains come from."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repaired_details_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_select_100/repaired_chair_details.json",
    )
    parser.add_argument(
        "--augmented_details_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_select_100/augmented_chair_details.json",
    )
    parser.add_argument(
        "--selected_details_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_select_100/selected_chair_details.json",
    )
    parser.add_argument(
        "--selection_examples_json",
        default="detection/baselines/results/tdev_caption_atomic_detail_select_100/verified_atomic_detail_selection_examples.json",
    )
    parser.add_argument("--output_dir", default="detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_items(items: list[str]) -> Counter[str]:
    return Counter(str(item) for item in items)


def grounded_counter(row: dict[str, Any]) -> Counter[str]:
    generated = count_items(row.get("generated_words", []))
    hallucinated = count_items(row.get("hallucinated_words", []))
    grounded = Counter(generated)
    grounded.subtract(hallucinated)
    return Counter({key: value for key, value in grounded.items() if value > 0})


def total(counter: Counter[str]) -> int:
    return sum(counter.values())


def positive_delta(left: Counter[str], right: Counter[str]) -> Counter[str]:
    delta = Counter(left)
    delta.subtract(right)
    return Counter({key: value for key, value in delta.items() if value > 0})


def word_count(text: str) -> int:
    return len(str(text).split())


def summarize_delta(base: dict[int, dict[str, Any]], variant: dict[int, dict[str, Any]]) -> dict[str, Any]:
    image_ids = sorted(set(base) & set(variant))
    totals = Counter()
    per_image = []
    for image_id in image_ids:
        base_row = base[image_id]
        variant_row = variant[image_id]
        base_generated = count_items(base_row.get("generated_words", []))
        variant_generated = count_items(variant_row.get("generated_words", []))
        base_hall = count_items(base_row.get("hallucinated_words", []))
        variant_hall = count_items(variant_row.get("hallucinated_words", []))
        base_grounded = grounded_counter(base_row)
        variant_grounded = grounded_counter(variant_row)
        row = {
            "image_id": image_id,
            "delta_words": word_count(variant_row.get("caption", "")) - word_count(base_row.get("caption", "")),
            "delta_object_mentions": total(variant_generated) - total(base_generated),
            "delta_grounded_mentions": total(variant_grounded) - total(base_grounded),
            "delta_hallucinated_mentions": total(variant_hall) - total(base_hall),
            "new_object_mentions": dict(positive_delta(variant_generated, base_generated)),
            "new_grounded_mentions": dict(positive_delta(variant_grounded, base_grounded)),
            "new_hallucinated_mentions": dict(positive_delta(variant_hall, base_hall)),
        }
        per_image.append(row)
        for key, value in row.items():
            if key.startswith("delta_"):
                totals[key] += int(value)
    return {
        "summary": {
            "num_images": len(image_ids),
            "delta_words": totals["delta_words"],
            "delta_object_mentions": totals["delta_object_mentions"],
            "delta_grounded_mentions": totals["delta_grounded_mentions"],
            "delta_hallucinated_mentions": totals["delta_hallucinated_mentions"],
            "images_with_more_objects": sum(int(row["delta_object_mentions"] > 0) for row in per_image),
            "images_with_more_grounded": sum(int(row["delta_grounded_mentions"] > 0) for row in per_image),
            "images_with_more_hallucinated": sum(int(row["delta_hallucinated_mentions"] > 0) for row in per_image),
        },
        "per_image": per_image,
    }


def main() -> None:
    args = parse_args()
    repaired = {int(row["image_id"]): row for row in read_json(Path(args.repaired_details_json))}
    augmented = {int(row["image_id"]): row for row in read_json(Path(args.augmented_details_json))}
    selected = {int(row["image_id"]): row for row in read_json(Path(args.selected_details_json))}
    examples = read_json(Path(args.selection_examples_json))
    accepted = [row for row in examples if row.get("selected_source") == "verified_atomic_additions"]
    rejected = [row for row in examples if row.get("selected_source") != "verified_atomic_additions"]
    payload = {
        "scope_note": (
            "Gain audit for verified atomic-detail captions. Deltas compare CHAIR-recognized "
            "object mentions before and after atomic additions; selection itself does not use CHAIR."
        ),
        "inputs": {
            "repaired_details_json": args.repaired_details_json,
            "augmented_details_json": args.augmented_details_json,
            "selected_details_json": args.selected_details_json,
            "selection_examples_json": args.selection_examples_json,
        },
        "selection_counts": {
            "accepted_atomic_addition_images": len(accepted),
            "repair_fallback_images": len(rejected),
            "accepted_atomic_spans": sum(len(row.get("atomic_detail_additions", [])) for row in accepted),
            "rejected_atomic_spans": sum(len(row.get("atomic_detail_additions", [])) for row in rejected),
        },
        "raw_augmented_vs_repaired": summarize_delta(repaired, augmented),
        "verified_selected_vs_repaired": summarize_delta(repaired, selected),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "atomic_detail_gain_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verified = payload["verified_selected_vs_repaired"]["summary"]
    raw = payload["raw_augmented_vs_repaired"]["summary"]
    counts = payload["selection_counts"]
    lines = [
        "# Atomic Detail Gain Audit",
        "",
        "| Metric | Raw augmented | Verified selected |",
        "|---|---:|---:|",
        f"| accepted/fallback images | -- | {counts['accepted_atomic_addition_images']} / {counts['repair_fallback_images']} |",
        f"| delta object mentions | {raw['delta_object_mentions']} | {verified['delta_object_mentions']} |",
        f"| delta grounded mentions | {raw['delta_grounded_mentions']} | {verified['delta_grounded_mentions']} |",
        f"| delta hallucinated mentions | {raw['delta_hallucinated_mentions']} | {verified['delta_hallucinated_mentions']} |",
        f"| images with more grounded mentions | {raw['images_with_more_grounded']} | {verified['images_with_more_grounded']} |",
        f"| images with more hallucinated mentions | {raw['images_with_more_hallucinated']} | {verified['images_with_more_hallucinated']} |",
    ]
    (output_dir / "atomic_detail_gain_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["selection_counts"] | {"verified_delta": verified, "raw_delta": raw}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
