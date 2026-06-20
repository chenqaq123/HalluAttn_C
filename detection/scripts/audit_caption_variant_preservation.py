#!/usr/bin/env python3
"""Audit content preservation for an arbitrary caption variant."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla_details_json", required=True)
    parser.add_argument("--gated_details_json", required=True)
    parser.add_argument("--variant_details_json", required=True)
    parser.add_argument("--variant_name", default="variant")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(text.split())


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


def overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum((left & right).values())


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    args = parse_args()
    rows_by_name = {
        "vanilla": {int(row["image_id"]): row for row in read_json(Path(args.vanilla_details_json))},
        "gated": {int(row["image_id"]): row for row in read_json(Path(args.gated_details_json))},
        "variant": {int(row["image_id"]): row for row in read_json(Path(args.variant_details_json))},
    }
    image_ids = sorted(set(rows_by_name["vanilla"]) & set(rows_by_name["gated"]) & set(rows_by_name["variant"]))
    totals = Counter()
    per_image = []
    for image_id in image_ids:
        rows = {name: rows_by_name[name][image_id] for name in rows_by_name}
        grounded = {name: grounded_counter(row) for name, row in rows.items()}
        hallucinated = {name: count_items(row.get("hallucinated_words", [])) for name, row in rows.items()}
        generated = {name: count_items(row.get("generated_words", [])) for name, row in rows.items()}
        words = {name: word_count(row.get("caption", "")) for name, row in rows.items()}
        row = {
            "image_id": image_id,
            "vanilla_words": words["vanilla"],
            "gated_words": words["gated"],
            "variant_words": words["variant"],
            "vanilla_object_mentions": total(generated["vanilla"]),
            "gated_object_mentions": total(generated["gated"]),
            "variant_object_mentions": total(generated["variant"]),
            "vanilla_grounded_mentions": total(grounded["vanilla"]),
            "gated_grounded_mentions": total(grounded["gated"]),
            "variant_grounded_mentions": total(grounded["variant"]),
            "vanilla_hallucinated_mentions": total(hallucinated["vanilla"]),
            "gated_hallucinated_mentions": total(hallucinated["gated"]),
            "variant_hallucinated_mentions": total(hallucinated["variant"]),
            "variant_retained_vanilla_grounded_mentions": overlap_count(grounded["variant"], grounded["vanilla"]),
            "variant_retained_gated_grounded_mentions": overlap_count(grounded["variant"], grounded["gated"]),
        }
        per_image.append(row)
        for key, value in row.items():
            if key != "image_id":
                totals[key] += int(value)

    n = len(image_ids)
    summary = {
        "variant_name": args.variant_name,
        "num_images": n,
        "mean_words": {
            "vanilla": rate(totals["vanilla_words"], n),
            "gated": rate(totals["gated_words"], n),
            "variant": rate(totals["variant_words"], n),
        },
        "object_mentions": {
            "vanilla": totals["vanilla_object_mentions"],
            "gated": totals["gated_object_mentions"],
            "variant": totals["variant_object_mentions"],
        },
        "grounded_mentions": {
            "vanilla": totals["vanilla_grounded_mentions"],
            "gated": totals["gated_grounded_mentions"],
            "variant": totals["variant_grounded_mentions"],
        },
        "hallucinated_mentions": {
            "vanilla": totals["vanilla_hallucinated_mentions"],
            "gated": totals["gated_hallucinated_mentions"],
            "variant": totals["variant_hallucinated_mentions"],
        },
        "variant_retained_vanilla_grounded_rate": rate(
            totals["variant_retained_vanilla_grounded_mentions"],
            totals["vanilla_grounded_mentions"],
        ),
        "variant_retained_gated_grounded_rate": rate(
            totals["variant_retained_gated_grounded_mentions"],
            totals["gated_grounded_mentions"],
        ),
        "variant_object_mention_retention_vs_vanilla": rate(
            totals["variant_object_mentions"],
            totals["vanilla_object_mentions"],
        ),
        "variant_object_mention_retention_vs_gated": rate(
            totals["variant_object_mentions"],
            totals["gated_object_mentions"],
        ),
        "variant_hallucination_reduction_vs_vanilla": rate(
            totals["vanilla_hallucinated_mentions"] - totals["variant_hallucinated_mentions"],
            totals["vanilla_hallucinated_mentions"],
        ),
        "variant_hallucination_reduction_vs_gated": rate(
            totals["gated_hallucinated_mentions"] - totals["variant_hallucinated_mentions"],
            totals["gated_hallucinated_mentions"],
        ),
    }
    payload = {"summary": summary, "per_image": per_image}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "caption_variant_preservation_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Caption Variant Preservation Audit",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| images | {n} |",
        f"| variant mean words | {summary['mean_words']['variant']:.2f} |",
        f"| retained vanilla grounded mentions | {summary['variant_retained_vanilla_grounded_rate']:.2%} |",
        f"| object retention vs vanilla | {summary['variant_object_mention_retention_vs_vanilla']:.2%} |",
        f"| hallucination reduction vs gated | {summary['variant_hallucination_reduction_vs_gated']:.2%} |",
    ]
    (output_dir / "caption_variant_preservation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
