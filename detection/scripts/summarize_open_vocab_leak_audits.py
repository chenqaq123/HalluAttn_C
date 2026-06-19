#!/usr/bin/env python3
"""Summarize open-vocabulary leak audits for closed-loop caption gates.

The table separates two questions that are easy to conflate:

1. Did an open-vocabulary pass discover the new object-like route?
2. Would raw phrase scoring reject it, or do we need to map the phrase back to a
   canonical verifier target such as a denied COCO object?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_AUDITS = [
    (
        "single_token_first",
        "detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_open_vocab_audit/closed_loop_example_audit.json",
    ),
    (
        "variant_alias_v1",
        "detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_open_vocab_audit/closed_loop_example_audit.json",
    ),
    (
        "variant_alias_v2",
        "detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v2_open_vocab_audit/closed_loop_example_audit.json",
    ),
    (
        "variant_alias_v3",
        "detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_open_vocab_audit/closed_loop_example_audit.json",
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audit",
        action="append",
        default=[],
        help="Label/path pair such as label=results/.../closed_loop_example_audit.json. Defaults to the four alias-chasing audits.",
    )
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_open_vocab_route_summary",
    )
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_specs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return [(label, Path(path)) for label, path in DEFAULT_AUDITS]
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--audit must be label=path, got {value!r}")
        label, raw_path = value.split("=", 1)
        specs.append((label.strip(), Path(raw_path.strip())))
    return specs


def raw_example_by_image(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    examples_path = Path(payload["examples_json"])
    raw_examples = read_json(examples_path)
    return {int(row["image_id"]): row for row in raw_examples}


def denied_item(raw_example: dict[str, Any], word: str) -> dict[str, Any]:
    for item in raw_example.get("denied_items", []):
        if str(item.get("word", "")).strip().lower() == word:
            return item
    return {}


def choose_route(example: dict[str, Any]) -> dict[str, str]:
    variant_leaks = example.get("introduced_variant_leaks", [])
    if variant_leaks:
        first = variant_leaks[0]
        return {"source": "variant_alias", "route": first["alias"], "mapped_word": first["word"]}
    root_leaks = example.get("introduced_root_leaks", [])
    if root_leaks:
        first = root_leaks[0]
        return {"source": "root", "route": first["token"], "mapped_word": first["word"]}
    introduced_words = example.get("introduced_words", [])
    if introduced_words:
        word = introduced_words[0]
        return {"source": "chair", "route": word, "mapped_word": word}
    open_vocab = example.get("introduced_open_vocab_candidates", [])
    if open_vocab:
        return {"source": "open_vocab", "route": open_vocab[0], "mapped_word": ""}
    return {"source": "none", "route": "", "mapped_word": ""}


def summarize_one(label: str, path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    raw_by_image = raw_example_by_image(payload)
    rows = []
    for example in payload.get("examples", []):
        image_id = int(example["image_id"])
        route = choose_route(example)
        raw_scores = example.get("introduced_open_vocab_claim_scores", {})
        route_score = raw_scores.get(route["route"], {})
        mapped = denied_item(raw_by_image.get(image_id, {}), route["mapped_word"])
        rows.append(
            {
                "run": label,
                "audit_json": str(path),
                "image_id": image_id,
                "route_source": route["source"],
                "route_phrase": route["route"],
                "open_vocab_detected_route": int(route["route"] in example.get("introduced_open_vocab_candidates", [])),
                "raw_phrase_target_score": route_score.get("target_score"),
                "raw_phrase_two_stage_present": route_score.get("two_stage_present"),
                "mapped_word": route["mapped_word"],
                "mapped_target_score": mapped.get("target_score"),
                "mapped_best_neighbor": mapped.get("best_neighbor"),
                "mapped_best_neighbor_score": mapped.get("best_neighbor_score"),
                "mapped_tdev_margin": mapped.get("tdev_margin"),
                "mapped_two_stage_present": mapped.get("two_stage_present"),
                "introduced_words": example.get("introduced_words", []),
                "introduced_variant_leaks": example.get("introduced_variant_leaks", []),
                "introduced_root_leaks": example.get("introduced_root_leaks", []),
                "unsupported_open_vocab_claims": example.get("unsupported_open_vocab_claims", []),
            }
        )
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "Run",
        "Route phrase",
        "Route source",
        "Open-vocab found",
        "Raw phrase present",
        "Raw score",
        "Mapped target",
        "Mapped present",
        "Mapped score",
        "Best neighbor",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["run"],
            row["route_phrase"],
            row["route_source"],
            row["open_vocab_detected_route"],
            row["raw_phrase_two_stage_present"],
            row["raw_phrase_target_score"],
            row["mapped_word"],
            row["mapped_two_stage_present"],
            row["mapped_target_score"],
            row["mapped_best_neighbor"],
        ]
        lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for label, path in audit_specs(args.audit):
        rows.extend(summarize_one(label, path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope_note": (
            "Single-image alias-chasing audit summary. Raw phrase verification is reported separately "
            "from mapped canonical-target verification because open-vocabulary candidates can be real "
            "object-like phrases while the original denied target is still absent."
        ),
        "rows": rows,
    }
    (output_dir / "open_vocab_leak_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(rows, output_dir / "open_vocab_leak_summary.md")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
