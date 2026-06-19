#!/usr/bin/env python3
"""Audit threshold sensitivity for open-vocabulary candidate-to-target mapping.

This script uses existing closed-loop caption-gate audit artifacts. It treats the
route phrase selected by the leak summary as the positive candidate for each run
and treats candidates that are unmapped at the reference threshold as a
conservative negative set for threshold-sweep diagnostics.
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

from sinkdetect.open_vocab_claims import auto_map_candidate, lexical_match_score

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
        "--thresholds",
        default="0.50,0.60,0.70,0.80,0.90,0.95,1.00",
        help="Comma-separated mapping thresholds to audit.",
    )
    p.add_argument("--reference_threshold", type=float, default=0.80)
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_decode_gate_open_vocab_mapping_thresholds",
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


def choose_route(example: dict[str, Any]) -> dict[str, str]:
    variant_leaks = example.get("introduced_variant_leaks", [])
    if variant_leaks:
        first = variant_leaks[0]
        return {"source": "variant_alias", "route": first["alias"], "expected_word": first["word"]}
    root_leaks = example.get("introduced_root_leaks", [])
    if root_leaks:
        first = root_leaks[0]
        return {"source": "root", "route": first["token"], "expected_word": first["word"]}
    introduced_words = example.get("introduced_words", [])
    if introduced_words:
        word = introduced_words[0]
        return {"source": "chair", "route": word, "expected_word": word}
    open_vocab = example.get("introduced_open_vocab_candidates", [])
    if open_vocab:
        return {"source": "open_vocab", "route": open_vocab[0], "expected_word": ""}
    return {"source": "none", "route": "", "expected_word": ""}


def best_word_and_score(candidate: str, denied_items: list[dict[str, Any]]) -> tuple[str, float]:
    best_word = ""
    best_score = 0.0
    for item in denied_items:
        word = str(item.get("word", "")).strip().lower()
        score = lexical_match_score(candidate, word)
        if score > best_score or (score == best_score and len(word) > len(best_word)):
            best_word = word
            best_score = score
    return best_word, best_score


def collect_candidates(specs: list[tuple[str, Path]], reference_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path in specs:
        payload = read_json(path)
        raw_by_image = raw_example_by_image(payload)
        for example in payload.get("examples", []):
            image_id = int(example["image_id"])
            raw_example = raw_by_image.get(image_id, {})
            denied_items = raw_example.get("denied_items", [])
            route = choose_route(example)
            for candidate in example.get("introduced_open_vocab_candidates", []):
                best_word, best_score = best_word_and_score(candidate, denied_items)
                ref_map = auto_map_candidate(candidate, denied_items, reference_threshold)
                is_route = int(candidate == route["route"])
                rows.append(
                    {
                        "run": label,
                        "image_id": image_id,
                        "candidate": candidate,
                        "route_phrase": route["route"],
                        "route_source": route["source"],
                        "expected_word": route["expected_word"] if is_route else "",
                        "is_route_positive": is_route,
                        "best_word": best_word,
                        "best_score": best_score,
                        "mapped_at_reference": int(bool(ref_map.get("mapped_word"))),
                        "mapped_word_at_reference": ref_map.get("mapped_word", ""),
                    }
                )
    return rows


def sweep_thresholds(rows: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    positives = [row for row in rows if row["is_route_positive"]]
    reference_unmapped = [row for row in rows if not row["is_route_positive"] and not row["mapped_at_reference"]]
    out = []
    for threshold in thresholds:
        positive_hits = [
            row for row in positives
            if row["best_score"] >= threshold and row["best_word"] == row["expected_word"]
        ]
        nonroute_maps = [row for row in reference_unmapped if row["best_score"] >= threshold]
        out.append(
            {
                "threshold": threshold,
                "route_positive_recall": len(positive_hits) / len(positives) if positives else 0.0,
                "route_positive_hits": len(positive_hits),
                "route_positive_total": len(positives),
                "reference_unmapped_nonroute_maps": len(nonroute_maps),
                "reference_unmapped_nonroute_total": len(reference_unmapped),
                "reference_unmapped_nonroute_map_rate": len(nonroute_maps) / len(reference_unmapped) if reference_unmapped else 0.0,
                "nonroute_maps": [
                    {"run": row["run"], "candidate": row["candidate"], "best_word": row["best_word"], "best_score": row["best_score"]}
                    for row in nonroute_maps
                ],
            }
        )
    return out


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "Threshold",
        "Route recall",
        "Route hits",
        "Reference-unmapped nonroute maps",
        "Nonroute map rate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            row["threshold"],
            row["route_positive_recall"],
            f"{row['route_positive_hits']}/{row['route_positive_total']}",
            f"{row['reference_unmapped_nonroute_maps']}/{row['reference_unmapped_nonroute_total']}",
            row["reference_unmapped_nonroute_map_rate"],
        ]
        lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    candidates = collect_candidates(audit_specs(args.audit), args.reference_threshold)
    sweep = sweep_thresholds(candidates, thresholds)
    positives = [row for row in candidates if row["is_route_positive"]]
    reference_unmapped = [row for row in candidates if not row["is_route_positive"] and not row["mapped_at_reference"]]
    payload = {
        "reference_threshold": args.reference_threshold,
        "thresholds": thresholds,
        "scope_note": (
            "Single-image threshold sensitivity over existing alias-chasing audits. "
            "Route positives are known leak route phrases; reference-unmapped nonroute "
            "candidates are a conservative diagnostic set, not a full human-labeled negative set."
        ),
        "num_route_positives": len(positives),
        "num_reference_unmapped_nonroute_candidates": len(reference_unmapped),
        "route_positive_candidates": positives,
        "reference_unmapped_nonroute_candidates": reference_unmapped,
        "threshold_sweep": sweep,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mapping_threshold_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(sweep, output_dir / "mapping_threshold_audit.md")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
