#!/usr/bin/env python3
"""Oracle-select captions from existing repair/regeneration candidates.

This is an analysis script, not a deployable method. It uses CHAIR labels to
estimate whether the current candidate pool contains useful caption variants
that a stronger verifier could select.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vanilla_details_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/vanilla_chair_details.json",
    )
    parser.add_argument(
        "--gated_details_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/gated_chair_details.json",
    )
    parser.add_argument(
        "--repaired_details_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/repaired_chair_details.json",
    )
    parser.add_argument(
        "--concise_details_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_t80/regenerated_chair_details.json",
    )
    parser.add_argument(
        "--detail_details_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/regenerated_chair_details.json",
    )
    parser.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_caption_candidate_pool_oracle_100",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_items(items: list[str]) -> Counter[str]:
    return Counter(str(item) for item in items)


def word_count(text: str) -> int:
    return len(str(text).split())


def total(counter: Counter[str]) -> int:
    return sum(counter.values())


def grounded_counter(row: dict[str, Any]) -> Counter[str]:
    generated = count_items(row.get("generated_words", []))
    hallucinated = count_items(row.get("hallucinated_words", []))
    grounded = Counter(generated)
    grounded.subtract(hallucinated)
    return Counter({key: value for key, value in grounded.items() if value > 0})


def hall_count(row: dict[str, Any]) -> int:
    return len(row.get("hallucinated_words", []))


def object_count(row: dict[str, Any]) -> int:
    return len(row.get("generated_words", []))


def grounded_count(row: dict[str, Any]) -> int:
    return total(grounded_counter(row))


def overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum((left & right).values())


def rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def detail_rows_by_image(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["image_id"]): row for row in read_json(path)}


def choose_min_hallucination(candidates: dict[str, dict[str, Any]]) -> str:
    return min(
        candidates,
        key=lambda name: (
            hall_count(candidates[name]),
            -grounded_count(candidates[name]),
            -word_count(candidates[name].get("caption", "")),
            name,
        ),
    )


def choose_no_worse_than_repair(candidates: dict[str, dict[str, Any]]) -> str:
    repaired_hall = hall_count(candidates["claim-local repair"])
    allowed = {
        name: row
        for name, row in candidates.items()
        if hall_count(row) <= repaired_hall
    }
    return max(
        allowed,
        key=lambda name: (
            grounded_count(allowed[name]),
            word_count(allowed[name].get("caption", "")),
            -hall_count(allowed[name]),
            name,
        ),
    )


def build_variant(
    image_ids: list[int],
    variants: dict[str, dict[int, dict[str, Any]]],
    vanilla_rows: dict[int, dict[str, Any]],
    gated_rows: dict[int, dict[str, Any]],
    selector_name: str,
) -> dict[str, Any]:
    selected_rows = []
    per_image = []
    selected_counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for image_id in image_ids:
        candidates = {name: rows[image_id] for name, rows in variants.items()}
        if selector_name == "min_hallucination":
            selected_name = choose_min_hallucination(candidates)
        elif selector_name == "no_worse_than_repair":
            selected_name = choose_no_worse_than_repair(candidates)
        else:
            raise ValueError(selector_name)
        selected = candidates[selected_name]
        selected_counts[selected_name] += 1
        vanilla_grounded = grounded_counter(vanilla_rows[image_id])
        gated_grounded = grounded_counter(gated_rows[image_id])
        selected_grounded = grounded_counter(selected)
        row = {
            "image_id": image_id,
            "selected_candidate": selected_name,
            "selected_words": word_count(selected.get("caption", "")),
            "selected_object_mentions": object_count(selected),
            "selected_grounded_mentions": grounded_count(selected),
            "selected_hallucinated_mentions": hall_count(selected),
            "retained_vanilla_grounded_mentions": overlap_count(selected_grounded, vanilla_grounded),
            "retained_gated_grounded_mentions": overlap_count(selected_grounded, gated_grounded),
            "vanilla_grounded_mentions": grounded_count(vanilla_rows[image_id]),
            "gated_grounded_mentions": grounded_count(gated_rows[image_id]),
        }
        per_image.append(row)
        selected_rows.append(
            {
                "image_id": image_id,
                "caption": selected.get("caption", ""),
                "generated_words": selected.get("generated_words", []),
                "hallucinated_words": selected.get("hallucinated_words", []),
            }
        )
        for key, value in row.items():
            if key not in {"image_id", "selected_candidate"}:
                totals[key] += int(value)
    n = len(image_ids)
    summary = {
        "selector": selector_name,
        "num_images": n,
        "selected_candidate_counts": dict(selected_counts),
        "mean_words": rate(totals["selected_words"], n),
        "total_object_mentions": totals["selected_object_mentions"],
        "total_grounded_mentions": totals["selected_grounded_mentions"],
        "total_hallucinated_mentions": totals["selected_hallucinated_mentions"],
        "chairi": rate(totals["selected_hallucinated_mentions"], totals["selected_object_mentions"]),
        "chairs": rate(
            sum(int(row["selected_hallucinated_mentions"] > 0) for row in per_image),
            n,
        ),
        "retained_vanilla_grounded_rate": rate(
            totals["retained_vanilla_grounded_mentions"],
            totals["vanilla_grounded_mentions"],
        ),
        "retained_gated_grounded_rate": rate(
            totals["retained_gated_grounded_mentions"],
            totals["gated_grounded_mentions"],
        ),
        "object_mention_retention_vs_vanilla": rate(
            totals["selected_object_mentions"],
            sum(object_count(vanilla_rows[image_id]) for image_id in image_ids),
        ),
        "object_mention_retention_vs_gated": rate(
            totals["selected_object_mentions"],
            sum(object_count(gated_rows[image_id]) for image_id in image_ids),
        ),
    }
    return {"summary": summary, "per_image": per_image, "selected_rows": selected_rows}


def main() -> None:
    args = parse_args()
    vanilla_rows = detail_rows_by_image(Path(args.vanilla_details_json))
    gated_rows = detail_rows_by_image(Path(args.gated_details_json))
    variants = {
        "claim-local repair": detail_rows_by_image(Path(args.repaired_details_json)),
        "controlled regen concise": detail_rows_by_image(Path(args.concise_details_json)),
        "controlled regen detail": detail_rows_by_image(Path(args.detail_details_json)),
    }
    image_ids = sorted(set(vanilla_rows) & set(gated_rows) & set.intersection(*(set(rows) for rows in variants.values())))
    outputs = {
        selector: build_variant(image_ids, variants, vanilla_rows, gated_rows, selector)
        for selector in ("min_hallucination", "no_worse_than_repair")
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "scope_note": (
            "Oracle candidate-pool analysis. It uses CHAIR labels to select among already-generated "
            "repair/regeneration candidates and should be read as an upper bound for a future verifier, "
            "not as a deployable method."
        ),
        "inputs": {
            "vanilla_details_json": args.vanilla_details_json,
            "gated_details_json": args.gated_details_json,
            "repaired_details_json": args.repaired_details_json,
            "concise_details_json": args.concise_details_json,
            "detail_details_json": args.detail_details_json,
        },
        "summaries": {selector: payload["summary"] for selector, payload in outputs.items()},
    }
    (output_dir / "candidate_pool_oracle_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for selector, payload in outputs.items():
        prefix = selector.replace("_", "-")
        (output_dir / f"{prefix}_chair_details.json").write_text(
            json.dumps(payload["selected_rows"], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{prefix}_per_image.json").write_text(
            json.dumps(payload["per_image"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    lines = [
        "# Caption Candidate Pool Oracle",
        "",
        "This is an upper-bound analysis over existing claim-local repair and controlled-regeneration candidates.",
        "",
        "| Selector | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Selected candidates |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for selector, payload in outputs.items():
        summary = payload["summary"]
        counts = ", ".join(f"{name}: {count}" for name, count in sorted(summary["selected_candidate_counts"].items()))
        lines.append(
            f"| {selector.replace('_', ' ')} | {summary['chairi']:.4f} | "
            f"{summary['total_hallucinated_mentions']} | {summary['mean_words']:.2f} | "
            f"{summary['retained_vanilla_grounded_rate'] * 100:.2f}% | "
            f"{summary['object_mention_retention_vs_vanilla'] * 100:.2f}% | {counts} |"
        )
    (output_dir / "candidate_pool_oracle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
