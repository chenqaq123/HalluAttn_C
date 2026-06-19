#!/usr/bin/env python3
"""Audit open-vocabulary candidate mapping on cached generated captions.

This is a cache-only diagnostic. It maps open-vocabulary caption candidates to
canonical CHAIR/COCO object words and checks whether the mapped word was already
recognized by CHAIR for the same caption. It is not a human-labeled precision
estimate, but it helps catch obvious mapper overreach before scaling the decode
prototype.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
if str(DETECTION_SRC) not in sys.path:
    sys.path.insert(0, str(DETECTION_SRC))

from sinkdetect.open_vocab_claims import auto_map_candidate, open_vocab_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--object_cache",
        default="detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl",
    )
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/open_vocab_mapper_caption_cache_audit",
    )
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--candidate_limit", type=int, default=32)
    p.add_argument("--max_captions", type=int, default=0, help="0 means all captions in the cache.")
    p.add_argument("--example_limit", type=int, default=40)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_caption_records(rows: list[dict[str, Any]], max_captions: int) -> list[dict[str, Any]]:
    by_image: dict[int, dict[str, Any]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        record = by_image.setdefault(
            image_id,
            {
                "image_id": image_id,
                "caption": row["caption"],
                "chair_words": [],
                "hallucinated_words": [],
                "grounded_words": [],
            },
        )
        word = str(row["word"]).strip().lower()
        if word not in record["chair_words"]:
            record["chair_words"].append(word)
        if int(row.get("label", 0)) == 1:
            if word not in record["hallucinated_words"]:
                record["hallucinated_words"].append(word)
        else:
            if word not in record["grounded_words"]:
                record["grounded_words"].append(word)
    records = list(by_image.values())
    records.sort(key=lambda item: item["image_id"])
    if max_captions > 0:
        records = records[:max_captions]
    return records


def summarize_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return rows[:limit]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.object_cache))
    vocab = sorted({str(row["word"]).strip().lower() for row in rows})
    denied_items = [{"word": word} for word in vocab]
    records = build_caption_records(rows, args.max_captions)

    mapped_rows: list[dict[str, Any]] = []
    unmapped_count = 0
    total_candidates = 0
    mapped_word_counter: Counter[str] = Counter()
    extra_word_counter: Counter[str] = Counter()
    chair_aligned_counter: Counter[str] = Counter()

    for record in records:
        chair_words = set(record["chair_words"])
        candidates = open_vocab_candidates(
            record["caption"],
            limit=args.candidate_limit,
        )
        total_candidates += len(candidates)
        for candidate in candidates:
            mapped = auto_map_candidate(candidate, denied_items, args.threshold)
            mapped_word = str(mapped.get("mapped_word") or "")
            if not mapped_word:
                unmapped_count += 1
                continue
            is_chair_aligned = int(mapped_word in chair_words)
            is_hallucinated = int(mapped_word in set(record["hallucinated_words"]))
            is_grounded = int(mapped_word in set(record["grounded_words"]))
            mapped_word_counter[mapped_word] += 1
            if is_chair_aligned:
                chair_aligned_counter[mapped_word] += 1
            else:
                extra_word_counter[mapped_word] += 1
            mapped_rows.append(
                {
                    "image_id": record["image_id"],
                    "candidate": candidate,
                    "mapped_word": mapped_word,
                    "mapping_score": mapped.get("mapping_score"),
                    "chair_words": record["chair_words"],
                    "is_chair_aligned": is_chair_aligned,
                    "is_hallucinated_chair_word": is_hallucinated,
                    "is_grounded_chair_word": is_grounded,
                    "caption": record["caption"],
                }
            )

    mapped_count = len(mapped_rows)
    chair_aligned = [row for row in mapped_rows if row["is_chair_aligned"]]
    extra_maps = [row for row in mapped_rows if not row["is_chair_aligned"]]
    hallucinated_aligned = [row for row in mapped_rows if row["is_hallucinated_chair_word"]]
    grounded_aligned = [row for row in mapped_rows if row["is_grounded_chair_word"]]
    extra_maps_sorted = sorted(
        extra_maps,
        key=lambda row: (-float(row["mapping_score"] or 0.0), row["mapped_word"], row["candidate"]),
    )
    aligned_sorted = sorted(
        chair_aligned,
        key=lambda row: (-float(row["mapping_score"] or 0.0), row["mapped_word"], row["candidate"]),
    )
    payload = {
        "object_cache": args.object_cache,
        "threshold": args.threshold,
        "candidate_limit": args.candidate_limit,
        "max_captions": args.max_captions,
        "num_object_cache_rows": len(rows),
        "num_captions": len(records),
        "num_canonical_words": len(vocab),
        "total_open_vocab_candidates": total_candidates,
        "mapped_candidates": mapped_count,
        "unmapped_candidates": unmapped_count,
        "mapping_rate": mapped_count / total_candidates if total_candidates else 0.0,
        "chair_aligned_mapped_candidates": len(chair_aligned),
        "chair_aligned_rate_among_mapped": len(chair_aligned) / mapped_count if mapped_count else 0.0,
        "extra_mapped_candidates": len(extra_maps),
        "extra_map_rate_among_mapped": len(extra_maps) / mapped_count if mapped_count else 0.0,
        "hallucinated_chair_aligned_mapped_candidates": len(hallucinated_aligned),
        "grounded_chair_aligned_mapped_candidates": len(grounded_aligned),
        "top_mapped_words": mapped_word_counter.most_common(20),
        "top_chair_aligned_words": chair_aligned_counter.most_common(20),
        "top_extra_mapped_words": extra_word_counter.most_common(20),
        "extra_map_examples": summarize_examples(extra_maps_sorted, args.example_limit),
        "chair_aligned_examples": summarize_examples(aligned_sorted, args.example_limit),
        "scope_note": (
            "Cache-only CHAIR-alignment diagnostic over generated captions. Extra mapped candidates are "
            "mapped to a CHAIR/COCO word that CHAIR did not report for that caption; this is a conservative "
            "overreach signal, not a human precision label."
        ),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "open_vocab_mapper_caption_cache_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Captions | {len(records)} |",
        f"| Open-vocab candidates | {total_candidates} |",
        f"| Mapped candidates | {mapped_count} |",
        f"| Mapping rate | {payload['mapping_rate']:.4f} |",
        f"| CHAIR-aligned mapped candidates | {len(chair_aligned)} |",
        f"| CHAIR-aligned rate among mapped | {payload['chair_aligned_rate_among_mapped']:.4f} |",
        f"| Extra mapped candidates | {len(extra_maps)} |",
        f"| Extra map rate among mapped | {payload['extra_map_rate_among_mapped']:.4f} |",
    ]
    (output_dir / "open_vocab_mapper_caption_cache_audit.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
