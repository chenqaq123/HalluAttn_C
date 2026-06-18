#!/usr/bin/env python3
"""Build a compact POPE internal-vs-external ablation table.

This script is intentionally post-hoc: it reads saved LH-Shape OOF predictions,
existing OWLv2/TDEV predictions, and simulates lightweight policies without
running either the VLM or OWLv2. It is meant to prevent paper-table drift by
placing LH-alone, LH->TDEV triage, prompt-only controls, and full TDEV under one
metric implementation and matched selection budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


SUBSETS = ["all", "negative_related_present"]
SPLITS = ["macro", "random", "popular", "adversarial"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build POPE internal/external ablation from saved predictions")
    p.add_argument("--lh_predictions_csv", default="mitigation/results/pope_lh_shape_transfer_full_imagecv/pope_lh_shape_transfer_predictions.csv")
    p.add_argument("--tdev_predictions_csv", default="mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_predictions.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--score_names", default="lh_shape_pope_layers_22_31,prompt_token_pos,target_char_len")
    p.add_argument("--selection_rates", default="0.25,0.50,0.75")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_rates(spec: str) -> list[float]:
    rates = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not rates:
        raise ValueError("selection rate list cannot be empty")
    for rate in rates:
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"selection rates must be in (0, 1], got {rate}")
    return rates


def metric(rows: Iterable[dict[str, str]]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = row["prediction"]
        gold = row["label"]
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1
        else:
            raise ValueError(f"Unexpected prediction/label pair: {pred}/{gold}")
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "precision": precision,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def subset(rows: list[dict[str, str]], name: str) -> list[dict[str, str]]:
    if name == "all":
        return rows
    return [row for row in rows if row.get("negative_type") == name]


def summarize(rows: list[dict[str, str]], method: str, score: str, selection_rate: float, detector_calls: int, selected_rows: int) -> list[dict[str, str | int | float]]:
    out = []
    for split in SPLITS:
        split_rows = rows if split == "macro" else [row for row in rows if row["split"] == split]
        if not split_rows:
            continue
        for subset_name in SUBSETS:
            items = subset(split_rows, subset_name)
            if not items:
                continue
            out.append({
                "method": method,
                "score": score,
                "selection_rate": selection_rate,
                "selected_rows": selected_rows,
                "detector_calls": detector_calls,
                "detector_call_savings": 1.0 - detector_calls / len(rows) if rows else 0.0,
                "split": split,
                "subset": subset_name,
                **metric(items),
            })
    return out


def top_keys(rows: list[dict[str, str]], score: str, rate: float, base_yes_only: bool) -> set[tuple[str, str]]:
    eligible = [row for row in rows if (not base_yes_only or row["base_prediction"] == "yes")]
    if not eligible:
        return set()
    k = max(1, min(math.ceil(len(eligible) * rate), len(eligible)))
    ranked = sorted(eligible, key=lambda row: float(row["absent_score"]), reverse=True)
    return {(row["split"], row["question_id"]) for row in ranked[:k]}


def markdown_table(rows: list[dict[str, str | int | float]]) -> str:
    keep = [row for row in rows if row["split"] == "macro" and row["subset"] == "all"]
    preferred = []
    order = [
        ("vanilla_base", "anchor"),
        ("lh_alone_base_yes_suppress", "lh_shape_pope_layers_22_31"),
        ("lh_to_tdev_base_yes", "lh_shape_pope_layers_22_31"),
        ("lh_to_tdev_all", "lh_shape_pope_layers_22_31"),
        ("lh_to_tdev_base_yes", "prompt_token_pos"),
        ("lh_to_tdev_base_yes", "target_char_len"),
        ("full_tdev", "anchor"),
    ]
    for method, score in order:
        candidates = [row for row in keep if row["method"] == method and row["score"] == score]
        if method not in {"vanilla_base", "full_tdev"}:
            candidates = [row for row in candidates if abs(float(row["selection_rate"]) - 0.5) < 1e-9]
        if candidates:
            preferred.append(candidates[0])
    lines = [
        "| Method | Score | Selection | Detector calls | MCC | TPR | FPR | Yes rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in preferred:
        lines.append(
            "| {method} | {score} | {selection_rate:.2f} | {detector_calls} | {mcc:.3f} | {tpr:.3f} | {fpr:.3f} | {yes_rate:.3f} |".format(
                method=row["method"],
                score=row["score"],
                selection_rate=float(row["selection_rate"]),
                detector_calls=int(row["detector_calls"]),
                mcc=float(row["mcc"]),
                tpr=float(row["tpr"]),
                fpr=float(row["fpr"]),
                yes_rate=float(row["yes_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lh_rows = read_csv(Path(args.lh_predictions_csv))
    tdev_rows = read_csv(Path(args.tdev_predictions_csv))
    tdev_by_key = {(row["split"], row["question_id"]): row for row in tdev_rows}
    score_names = parse_list(args.score_names)
    rates = parse_rates(args.selection_rates)
    all_summary = []

    for score_name in score_names:
        score_rows = [row for row in lh_rows if row["score"] == score_name]
        if not score_rows:
            raise ValueError(f"No LH rows found for score {score_name}")
        joined = []
        for row in score_rows:
            key = (row["split"], row["question_id"])
            if key not in tdev_by_key:
                raise KeyError(f"Missing TDEV prediction for {key}")
            tdev = tdev_by_key[key]
            joined.append({
                "split": row["split"],
                "question_id": row["question_id"],
                "target": row["target"],
                "negative_type": tdev["negative_type"],
                "label": tdev["label"],
                "base_prediction": tdev["base_prediction"],
                "tdev_prediction": tdev["prediction"],
                "absent_score": row["absent_score"],
            })
        if score_name == score_names[0]:
            base = [{**row, "prediction": row["base_prediction"]} for row in joined]
            full = [{**row, "prediction": row["tdev_prediction"]} for row in joined]
            all_summary.extend(summarize(base, "vanilla_base", "anchor", 0.0, 0, 0))
            all_summary.extend(summarize(full, "full_tdev", "anchor", 1.0, len(joined), len(joined)))
        for rate in rates:
            for base_yes_only, suffix in [(False, "all"), (True, "base_yes")]:
                selected = top_keys(joined, score_name, rate, base_yes_only=base_yes_only)
                triage = []
                lh_alone = []
                for row in joined:
                    key = (row["split"], row["question_id"])
                    is_selected = key in selected
                    triage.append({**row, "prediction": row["tdev_prediction"] if is_selected else row["base_prediction"]})
                    lh_alone_pred = "no" if is_selected and row["base_prediction"] == "yes" else row["base_prediction"]
                    lh_alone.append({**row, "prediction": lh_alone_pred})
                all_summary.extend(summarize(triage, f"lh_to_tdev_{suffix}", score_name, rate, len(selected), len(selected)))
                if base_yes_only:
                    all_summary.extend(summarize(lh_alone, "lh_alone_base_yes_suppress", score_name, rate, 0, len(selected)))

    csv_path = output_dir / "pope_internal_external_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_summary)
    json_path = output_dir / "pope_internal_external_ablation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "lh_predictions_csv": args.lh_predictions_csv,
            "tdev_predictions_csv": args.tdev_predictions_csv,
            "score_names": score_names,
            "selection_rates": rates,
            "metrics": all_summary,
            "caveat": "Post-hoc ablation from saved predictions. Detector-call counts are simulated selection budgets, not wall-clock runtime measurements.",
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = output_dir / "pope_internal_external_ablation.md"
    md_path.write_text("# POPE Internal vs External Ablation\n\n" + markdown_table(all_summary), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(markdown_table(all_summary))


if __name__ == "__main__":
    main()
