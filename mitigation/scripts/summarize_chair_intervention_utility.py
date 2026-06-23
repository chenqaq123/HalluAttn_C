#!/usr/bin/env python3
"""Summarize CHAIR intervention quality and caption utility tradeoffs.

The intervention script already reports CHAIR deltas and mention-level
precision. This helper adds caption-preservation metrics so operating points
can be compared without over-claiming improvements from aggressive deletion.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize CHAIR caption intervention utility")
    p.add_argument(
        "--base_dir",
        default="mitigation/results/semantic_neighbor_audit/chair_internal_verifier_full",
        help="Directory containing intervention_answer_* result subdirectories.",
    )
    p.add_argument(
        "--output_csv",
        default="mitigation/results/semantic_neighbor_audit/chair_internal_verifier_full/intervention_utility_summary.csv",
    )
    p.add_argument(
        "--output_json",
        default="mitigation/results/semantic_neighbor_audit/chair_internal_verifier_full/intervention_utility_summary.json",
    )
    p.add_argument(
        "--dirs",
        nargs="*",
        default=None,
        help="Optional explicit intervention directories. Defaults to intervention_answer_* under base_dir.",
    )
    p.add_argument("--min_lcs_retention", type=float, default=0.995)
    p.add_argument("--min_object_retention", type=float, default=0.965)
    p.add_argument("--min_recall_delta", type=float, default=-0.011)
    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokens(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for item in a:
        cur = [0]
        for j, other in enumerate(b, start=1):
            if item == other:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def read_caption_pairs(path: Path) -> list[dict]:
    data = load_json(path)
    for row in data:
        if "original_caption" not in row or "caption" not in row:
            raise KeyError(f"{path} must contain original_caption and caption fields")
    return data


def summarize_dir(path: Path, args: argparse.Namespace) -> dict[str, object]:
    intervention = load_json(path / "caption_intervention_metrics.json")
    chair = load_json(path / "chair_metrics.json")
    pairs = read_caption_pairs(path / "caption_pairs.json")

    changed_caption_flags = []
    word_deltas = []
    abs_word_deltas = []
    lcs_ratios = []
    edit_ratios = []
    emptied = 0
    for pair in pairs:
        original = str(pair["original_caption"])
        edited = str(pair["caption"])
        original_tokens = tokens(original)
        edited_tokens = tokens(edited)
        changed = original != edited
        changed_caption_flags.append(changed)
        delta = len(edited_tokens) - len(original_tokens)
        word_deltas.append(delta)
        abs_word_deltas.append(abs(delta))
        if not edited_tokens:
            emptied += 1
        denom = max(1, len(original_tokens))
        lcs = lcs_len(original_tokens, edited_tokens)
        lcs_ratio = lcs / denom
        lcs_ratios.append(lcs_ratio)
        edit_ratios.append(1.0 - lcs_ratio)

    vanilla_chair = chair["vanilla"]["chair"]
    intervention_chair = chair["intervention"]["chair"]
    vanilla_stats = chair["vanilla"]["caption_stats"]
    intervention_stats = chair["intervention"]["caption_stats"]
    delta = chair["delta"]

    hallucinated_total = int(intervention["hallucinated_mentions"])
    grounded_total = int(intervention["mentions"]) - hallucinated_total
    changed_hall = int(intervention["changed_hallucinated"])
    changed_ground = int(intervention["changed_grounded"])
    chairi_reduction = -float(delta.get("delta_chair_CHAIRi", 0.0))
    chairs_reduction = -float(delta.get("delta_chair_CHAIRs", 0.0))
    recall_delta = float(delta.get("delta_chair_Recall", 0.0))
    object_retention = (
        float(intervention_stats["total_object_mentions"]) / float(vanilla_stats["total_object_mentions"])
        if vanilla_stats.get("total_object_mentions")
        else 0.0
    )
    hallucinated_retention = (
        float(intervention_stats["total_hallucinated_mentions"]) / float(vanilla_stats["total_hallucinated_mentions"])
        if vanilla_stats.get("total_hallucinated_mentions")
        else 0.0
    )
    grounded_loss_rate = changed_ground / grounded_total if grounded_total else 0.0
    hall_change_rate = changed_hall / hallucinated_total if hallucinated_total else 0.0
    precision = float(intervention["change_precision"])
    utility_score = chairi_reduction - grounded_loss_rate
    passes_preservation = (
        mean(lcs_ratios) >= args.min_lcs_retention
        and object_retention >= args.min_object_retention
        and recall_delta >= args.min_recall_delta
    )

    return {
        "run": path.name,
        "mode": intervention["mode"],
        "top_frac": float(intervention["top_frac"]),
        "selected_mentions": int(intervention["selected_mentions"]),
        "selected_precision": float(intervention["selected_precision"]),
        "changed_mentions": int(intervention["changed_mentions"]),
        "change_precision": precision,
        "changed_hallucinated": changed_hall,
        "changed_grounded": changed_ground,
        "hallucinated_claim_change_rate": hall_change_rate,
        "grounded_claim_change_rate": grounded_loss_rate,
        "changed_caption_rate": mean(changed_caption_flags),
        "empty_caption_rate": emptied / len(pairs) if pairs else 0.0,
        "mean_token_delta": mean(word_deltas),
        "mean_abs_token_delta": mean(abs_word_deltas),
        "mean_lcs_retention": mean(lcs_ratios),
        "mean_lcs_edit_rate": mean(edit_ratios),
        "vanilla_chairi": float(vanilla_chair["CHAIRi"]),
        "chairi": float(intervention_chair["CHAIRi"]),
        "chairi_reduction": chairi_reduction,
        "vanilla_chairs": float(vanilla_chair["CHAIRs"]),
        "chairs": float(intervention_chair["CHAIRs"]),
        "chairs_reduction": chairs_reduction,
        "precision_delta": float(delta.get("delta_chair_Precision", 0.0)),
        "recall_delta": recall_delta,
        "f1_delta": float(delta.get("delta_chair_F1", 0.0)),
        "object_mention_retention": object_retention,
        "hallucinated_mention_retention": hallucinated_retention,
        "mean_object_mentions": float(intervention_stats["mean_object_mentions"]),
        "mean_hallucinated_mentions": float(intervention_stats["mean_hallucinated_mentions"]),
        "utility_score_chairi_minus_grounded_loss": utility_score,
        "chairi_reduction_per_grounded_loss": chairi_reduction / grounded_loss_rate if grounded_loss_rate else 0.0,
        "hall_changed_per_ground_changed": changed_hall / changed_ground if changed_ground else 0.0,
        "passes_preservation_gate": int(passes_preservation),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    if args.dirs:
        dirs = [Path(item) for item in args.dirs]
    else:
        dirs = sorted(
            item for item in base.glob("intervention_answer_*")
            if (item / "caption_intervention_metrics.json").exists() and (item / "chair_metrics.json").exists()
        )
    rows = [summarize_dir(path, args) for path in dirs]
    rows.sort(key=lambda row: (str(row["mode"]), float(row["top_frac"])))

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    write_csv(output_csv, rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")

    preservation_rows = [row for row in rows if row["passes_preservation_gate"]]
    print(json.dumps({
        "rows": len(rows),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
        "best_chairi_reduction": max(rows, key=lambda row: row["chairi_reduction"])["run"] if rows else None,
        "best_utility_score": max(rows, key=lambda row: row["utility_score_chairi_minus_grounded_loss"])["run"] if rows else None,
        "preservation_gate": {
            "min_lcs_retention": args.min_lcs_retention,
            "min_object_retention": args.min_object_retention,
            "min_recall_delta": args.min_recall_delta,
        },
        "best_preserved_chairi_reduction": (
            max(preservation_rows, key=lambda row: row["chairi_reduction"])["run"]
            if preservation_rows else None
        ),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
