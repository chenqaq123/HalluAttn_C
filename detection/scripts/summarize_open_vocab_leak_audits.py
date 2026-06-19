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
import re
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
    p.add_argument("--mapping_threshold", type=float, default=0.80)
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


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9]*", text.lower())


def token_stem(token: str) -> str:
    token = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    return token


def char_ngrams(text: str, n: int = 3) -> set[str]:
    clean = "".join(tokens(text))
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[idx : idx + n] for idx in range(len(clean) - n + 1)}


def lexical_match_score(candidate: str, target: str) -> float:
    cand_tokens = tokens(candidate)
    target_tokens = tokens(target)
    if not cand_tokens or not target_tokens:
        return 0.0
    cand_stems = [token_stem(token) for token in cand_tokens]
    target_stems = [token_stem(token) for token in target_tokens]
    cand_joined = "".join(cand_stems)
    target_joined = "".join(target_stems)

    scores: list[float] = []
    for t_stem in target_stems:
        token_scores = []
        for c_stem in cand_stems:
            if c_stem == t_stem:
                token_scores.append(1.0)
            elif len(t_stem) >= 4 and c_stem.startswith(t_stem):
                token_scores.append(0.95)
            elif len(c_stem) >= 4 and t_stem.startswith(c_stem):
                token_scores.append(0.90)
            else:
                common = 0
                for left, right in zip(c_stem, t_stem):
                    if left != right:
                        break
                    common += 1
                token_scores.append(common / max(len(t_stem), 1))
        scores.append(max(token_scores) if token_scores else 0.0)

    token_score = sum(scores) / len(scores)
    if len(target_joined) >= 4 and target_joined in cand_joined:
        token_score = max(token_score, 0.95)

    cand_grams = char_ngrams(candidate)
    target_grams = char_ngrams(target)
    if cand_grams and target_grams:
        ngram_score = len(cand_grams & target_grams) / len(target_grams)
    else:
        ngram_score = 0.0
    return max(token_score, ngram_score)


def auto_map_candidate(
    candidate: str,
    denied_items: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    best_word = ""
    best_score = 0.0
    for item in denied_items:
        word = str(item.get("word", "")).strip().lower()
        score = lexical_match_score(candidate, word)
        if score > best_score:
            best_word = word
            best_score = score
    mapped = denied_item({"denied_items": denied_items}, best_word) if best_score >= threshold else {}
    return {
        "candidate": candidate,
        "mapped_word": best_word if mapped else "",
        "mapping_score": best_score,
        "mapped_target_score": mapped.get("target_score"),
        "mapped_best_neighbor": mapped.get("best_neighbor"),
        "mapped_best_neighbor_score": mapped.get("best_neighbor_score"),
        "mapped_tdev_margin": mapped.get("tdev_margin"),
        "mapped_two_stage_present": mapped.get("two_stage_present"),
    }


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


def summarize_one(label: str, path: Path, mapping_threshold: float) -> list[dict[str, Any]]:
    payload = read_json(path)
    raw_by_image = raw_example_by_image(payload)
    rows = []
    for example in payload.get("examples", []):
        image_id = int(example["image_id"])
        raw_example = raw_by_image.get(image_id, {})
        route = choose_route(example)
        raw_scores = example.get("introduced_open_vocab_claim_scores", {})
        route_score = raw_scores.get(route["route"], {})
        mapped = denied_item(raw_example, route["mapped_word"])
        auto_mappings = [
            auto_map_candidate(candidate, raw_example.get("denied_items", []), mapping_threshold)
            for candidate in example.get("introduced_open_vocab_candidates", [])
        ]
        route_auto = next(
            (item for item in auto_mappings if item["candidate"] == route["route"]),
            auto_map_candidate(route["route"], raw_example.get("denied_items", []), mapping_threshold),
        )
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
                "auto_mapped_word": route_auto.get("mapped_word"),
                "auto_mapping_score": route_auto.get("mapping_score"),
                "auto_mapped_target_score": route_auto.get("mapped_target_score"),
                "auto_mapped_best_neighbor": route_auto.get("mapped_best_neighbor"),
                "auto_mapped_tdev_margin": route_auto.get("mapped_tdev_margin"),
                "auto_mapped_two_stage_present": route_auto.get("mapped_two_stage_present"),
                "auto_candidate_mappings": auto_mappings,
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
        "Auto target",
        "Auto map score",
        "Auto present",
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
            row["auto_mapped_word"],
            row["auto_mapping_score"],
            row["auto_mapped_two_stage_present"],
            row["auto_mapped_best_neighbor"],
        ]
        lines.append("| " + " | ".join(fmt(value) for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for label, path in audit_specs(args.audit):
        rows.extend(summarize_one(label, path, args.mapping_threshold))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mapping_threshold": args.mapping_threshold,
        "scope_note": (
            "Single-image alias-chasing audit summary. Raw phrase verification is reported separately "
            "from mapped canonical-target verification because open-vocabulary candidates can be real "
            "object-like phrases while the original denied target is still absent. The auto mapping "
            "columns use a lexical candidate-to-denied-target matcher and do not read variant/root labels."
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
