#!/usr/bin/env python3
"""Audit concise-faithfulness for caption acceptance prototypes.

The goal is not to maximize caption length. This audit asks whether a shorter
caption still keeps supported object content while removing hallucinated object
claims.
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
        "--acceptance_dir",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance",
    )
    parser.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_concise_faithfulness",
    )
    parser.add_argument("--min_words_for_nonempty", type=int, default=8)
    parser.add_argument("--min_object_mentions_for_nonempty", type=int, default=1)
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


def overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum((left & right).values())


def total(counter: Counter[str]) -> int:
    return sum(counter.values())


def rates(num: int, den: int) -> float:
    return num / den if den else 0.0


def main() -> None:
    args = parse_args()
    root = Path(args.acceptance_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "vanilla": read_json(root / "vanilla_chair_details.json"),
        "gated": read_json(root / "gated_chair_details.json"),
        "accepted": read_json(root / "accepted_chair_details.json"),
    }
    examples = {int(row["image_id"]): row for row in read_json(root / "sentence_acceptance_examples.json")}
    by_variant = {
        name: {int(row["image_id"]): row for row in rows}
        for name, rows in variants.items()
    }
    image_ids = sorted(set(by_variant["vanilla"]) & set(by_variant["gated"]) & set(by_variant["accepted"]))

    per_image = []
    totals = {
        "vanilla_grounded_mentions": 0,
        "gated_grounded_mentions": 0,
        "accepted_grounded_mentions": 0,
        "accepted_retained_vanilla_grounded_mentions": 0,
        "accepted_retained_gated_grounded_mentions": 0,
        "vanilla_hallucinated_mentions": 0,
        "gated_hallucinated_mentions": 0,
        "accepted_hallucinated_mentions": 0,
        "vanilla_object_mentions": 0,
        "gated_object_mentions": 0,
        "accepted_object_mentions": 0,
        "vanilla_words": 0,
        "gated_words": 0,
        "accepted_words": 0,
        "generic_or_empty_accepted": 0,
    }

    for image_id in image_ids:
        rows = {name: by_variant[name][image_id] for name in variants}
        grounded = {name: grounded_counter(row) for name, row in rows.items()}
        hallucinated = {name: count_items(row.get("hallucinated_words", [])) for name, row in rows.items()}
        generated = {name: count_items(row.get("generated_words", [])) for name, row in rows.items()}
        words = {name: word_count(rows[name].get("caption", "")) for name in rows}
        accepted_object_mentions = total(generated["accepted"])
        accepted_is_generic_or_empty = int(
            words["accepted"] < args.min_words_for_nonempty
            or accepted_object_mentions < args.min_object_mentions_for_nonempty
        )

        row = {
            "image_id": image_id,
            "vanilla_words": words["vanilla"],
            "gated_words": words["gated"],
            "accepted_words": words["accepted"],
            "vanilla_object_mentions": total(generated["vanilla"]),
            "gated_object_mentions": total(generated["gated"]),
            "accepted_object_mentions": accepted_object_mentions,
            "vanilla_hallucinated_mentions": total(hallucinated["vanilla"]),
            "gated_hallucinated_mentions": total(hallucinated["gated"]),
            "accepted_hallucinated_mentions": total(hallucinated["accepted"]),
            "vanilla_grounded_mentions": total(grounded["vanilla"]),
            "gated_grounded_mentions": total(grounded["gated"]),
            "accepted_grounded_mentions": total(grounded["accepted"]),
            "accepted_retained_vanilla_grounded_mentions": overlap_count(grounded["accepted"], grounded["vanilla"]),
            "accepted_retained_gated_grounded_mentions": overlap_count(grounded["accepted"], grounded["gated"]),
            "accepted_generic_or_empty": accepted_is_generic_or_empty,
            "removed_words_by_acceptance": examples.get(image_id, {}).get("removed_words_by_acceptance", None),
        }
        per_image.append(row)
        for key in totals:
            if key in row:
                totals[key] += int(row[key])
        totals["generic_or_empty_accepted"] += accepted_is_generic_or_empty

    num_images = len(image_ids)
    summary = {
        "num_images": num_images,
        "mean_words": {
            name: rates(totals[f"{name}_words"], num_images) for name in ("vanilla", "gated", "accepted")
        },
        "object_mentions": {
            name: totals[f"{name}_object_mentions"] for name in ("vanilla", "gated", "accepted")
        },
        "grounded_mentions": {
            name: totals[f"{name}_grounded_mentions"] for name in ("vanilla", "gated", "accepted")
        },
        "hallucinated_mentions": {
            name: totals[f"{name}_hallucinated_mentions"] for name in ("vanilla", "gated", "accepted")
        },
        "accepted_retained_vanilla_grounded_rate": rates(
            totals["accepted_retained_vanilla_grounded_mentions"], totals["vanilla_grounded_mentions"]
        ),
        "accepted_retained_gated_grounded_rate": rates(
            totals["accepted_retained_gated_grounded_mentions"], totals["gated_grounded_mentions"]
        ),
        "accepted_hallucination_reduction_vs_gated": rates(
            totals["gated_hallucinated_mentions"] - totals["accepted_hallucinated_mentions"],
            totals["gated_hallucinated_mentions"],
        ),
        "accepted_hallucination_reduction_vs_vanilla": rates(
            totals["vanilla_hallucinated_mentions"] - totals["accepted_hallucinated_mentions"],
            totals["vanilla_hallucinated_mentions"],
        ),
        "accepted_object_mention_retention_vs_gated": rates(
            totals["accepted_object_mentions"], totals["gated_object_mentions"]
        ),
        "accepted_object_mention_retention_vs_vanilla": rates(
            totals["accepted_object_mentions"], totals["vanilla_object_mentions"]
        ),
        "generic_or_empty_accepted": totals["generic_or_empty_accepted"],
        "generic_or_empty_accepted_rate": rates(totals["generic_or_empty_accepted"], num_images),
        "definition": {
            "grounded_mentions": "CHAIR generated object mentions minus CHAIR hallucinated mentions.",
            "retained_grounded_rate": "Accepted grounded mention overlap with the baseline grounded mentions, counted with multiplicity.",
            "generic_or_empty": "Accepted caption has too few words or too few CHAIR-recognized object mentions under the configured thresholds.",
        },
    }
    payload = {
        "acceptance_dir": str(root),
        "summary": summary,
        "per_image": per_image,
        "thresholds": {
            "min_words_for_nonempty": args.min_words_for_nonempty,
            "min_object_mentions_for_nonempty": args.min_object_mentions_for_nonempty,
        },
    }
    (output_dir / "concise_faithfulness_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Concise-Faithfulness Audit",
        "",
        "This audit asks whether shorter accepted captions still keep supported object content while removing hallucinated object claims.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| images | {num_images} |",
        f"| accepted retained vanilla grounded mentions | {summary['accepted_retained_vanilla_grounded_rate']:.2%} |",
        f"| accepted retained gated grounded mentions | {summary['accepted_retained_gated_grounded_rate']:.2%} |",
        f"| accepted hallucination reduction vs gated | {summary['accepted_hallucination_reduction_vs_gated']:.2%} |",
        f"| accepted hallucination reduction vs vanilla | {summary['accepted_hallucination_reduction_vs_vanilla']:.2%} |",
        f"| accepted object mention retention vs gated | {summary['accepted_object_mention_retention_vs_gated']:.2%} |",
        f"| accepted object mention retention vs vanilla | {summary['accepted_object_mention_retention_vs_vanilla']:.2%} |",
        f"| generic/empty accepted captions | {summary['generic_or_empty_accepted']} |",
        f"| accepted mean words | {summary['mean_words']['accepted']:.2f} |",
        f"| accepted hallucinated mentions | {summary['hallucinated_mentions']['accepted']} |",
    ]
    (output_dir / "concise_faithfulness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
