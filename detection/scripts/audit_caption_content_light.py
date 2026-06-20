#!/usr/bin/env python3
"""Audit whether caption outputs are genuinely content-light.

CHAIR object mentions only cover COCO objects. A caption can therefore be
descriptive but CHAIR-objectless, e.g. road/sign/pole scenes. This audit keeps
that separate from truly short or content-light captions by counting simple
open-vocabulary content candidates in addition to CHAIR-recognized objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
if str(DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DETECTION_SRC))

from sinkdetect.open_vocab_claims import open_vocab_candidates

LOW_CONTENT_TERMS = {
    "appears",
    "background",
    "center",
    "depicts",
    "features",
    "image",
    "located",
    "near",
    "nearby",
    "overall",
    "scene",
    "situated",
    "surroundings",
    "visible",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chair_details_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant_name", default="caption")
    parser.add_argument("--min_words_for_nonempty", type=int, default=8)
    parser.add_argument("--min_open_vocab_content", type=int, default=3)
    parser.add_argument("--open_vocab_candidate_limit", type=int, default=64)
    parser.add_argument("--open_vocab_min_token_len", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def word_count(text: str) -> int:
    return len(text.split())


def content_candidates(text: str, min_len: int, limit: int) -> list[str]:
    candidates = open_vocab_candidates(text, min_len=min_len, limit=limit)
    filtered = []
    for candidate in candidates:
        tokens = candidate.lower().split()
        if not tokens:
            continue
        if all(token in LOW_CONTENT_TERMS for token in tokens):
            continue
        filtered.append(candidate)
    return filtered


def main() -> None:
    args = parse_args()
    rows = read_json(Path(args.chair_details_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    for row in rows:
        caption = str(row.get("caption", ""))
        words = word_count(caption)
        chair_objects = len(row.get("generated_words", []))
        candidates = content_candidates(
            caption,
            min_len=args.open_vocab_min_token_len,
            limit=args.open_vocab_candidate_limit,
        )
        chair_objectless = int(chair_objects < 1)
        content_light = int(
            words < args.min_words_for_nonempty
            or (chair_objectless and len(candidates) < args.min_open_vocab_content)
        )
        per_image.append(
            {
                "image_id": int(row["image_id"]),
                "caption": caption,
                "words": words,
                "chair_object_mentions": chair_objects,
                "chair_objectless": chair_objectless,
                "open_vocab_content_candidates": candidates,
                "num_open_vocab_content_candidates": len(candidates),
                "content_light": content_light,
            }
        )

    num_images = len(per_image)
    chair_objectless = sum(row["chair_objectless"] for row in per_image)
    content_light = sum(row["content_light"] for row in per_image)
    summary = {
        "variant_name": args.variant_name,
        "chair_details_json": args.chair_details_json,
        "num_images": num_images,
        "chair_objectless": chair_objectless,
        "chair_objectless_rate": chair_objectless / num_images if num_images else 0.0,
        "content_light": content_light,
        "content_light_rate": content_light / num_images if num_images else 0.0,
        "thresholds": {
            "min_words_for_nonempty": args.min_words_for_nonempty,
            "min_open_vocab_content": args.min_open_vocab_content,
            "open_vocab_candidate_limit": args.open_vocab_candidate_limit,
            "open_vocab_min_token_len": args.open_vocab_min_token_len,
        },
        "definition": (
            "content_light is true when the caption is very short, or when it has "
            "no CHAIR-recognized COCO object and too few open-vocabulary content candidates."
        ),
    }
    payload = {"summary": summary, "per_image": per_image}
    (output_dir / "caption_content_light_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Caption Content-Light Audit",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| images | {num_images} |",
        f"| CHAIR-objectless captions | {chair_objectless} |",
        f"| content-light captions | {content_light} |",
        f"| content-light rate | {summary['content_light_rate']:.2%} |",
    ]
    (output_dir / "caption_content_light_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
