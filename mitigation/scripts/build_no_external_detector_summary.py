#!/usr/bin/env python3
"""Build the current no-external-detector method summary.

This script reads existing result artifacts and writes a compact Markdown table
so the paper-facing method rows do not drift from the validated numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path("mitigation/results/semantic_neighbor_audit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build no-external-detector summary tables")
    p.add_argument("--base_dir", default=str(DEFAULT_BASE))
    p.add_argument("--output_md", default="docs/no_external_detector_summary.md")
    p.add_argument("--output_json", default=str(DEFAULT_BASE / "no_external_detector_summary.json"))
    return p.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, Any], key: str) -> float:
    value = row[key]
    return float(value)


def find_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get(key)) == value:
            return row
    raise KeyError(f"Could not find {key}={value}")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def load_pope_summary(base: Path) -> dict[str, Any]:
    control = read_csv(base / "paper_control_table" / "semantic_neighbor_control_table.csv")
    vanilla = find_row(control, "method", "Vanilla")
    owlv2 = find_row(control, "method", "Hybrid gate+rescue")
    internal_dir = base / "answer_confidence_tdev_full" / "cross_split_hidden_answer6_standardized"
    internal_summary = read_json(internal_dir / "cross_split_summary.json")
    internal_split_rows = read_csv(internal_dir / "cross_split_metrics.csv")
    internal_best = find_row(internal_summary["macro"], "objective", "best_mcc")
    internal_low_fpr = find_row(internal_summary["macro"], "objective", "min_fpr_tpr0.80")
    best_adv_related = find_row(
        [
            row for row in internal_split_rows
            if row["objective"] == "best_mcc"
            and row["split"] == "adversarial"
            and row["subset"] == "negative_related_present"
        ],
        "subset",
        "negative_related_present",
    )
    low_fpr_adv_related = find_row(
        [
            row for row in internal_split_rows
            if row["objective"] == "min_fpr_tpr0.80"
            and row["split"] == "adversarial"
            and row["subset"] == "negative_related_present"
        ],
        "subset",
        "negative_related_present",
    )

    return {
        "vanilla": {
            "method": "Vanilla",
            "external_detector": "no",
            "mcc": as_float(vanilla, "macro_mcc"),
            "tpr": as_float(vanilla, "macro_tpr"),
            "fpr": as_float(vanilla, "macro_fpr"),
            "related_fpr": as_float(vanilla, "macro_related_fpr"),
            "plain_fpr": as_float(vanilla, "macro_plain_fpr"),
            "gap": as_float(vanilla, "macro_related_minus_plain_fpr"),
            "adv_related_fpr": as_float(vanilla, "adversarial_related_fpr"),
        },
        "internal_best": {
            "method": "Hidden+answer verifier",
            "external_detector": "no",
            **{key: float(internal_best[key]) for key in ("mcc", "tpr", "fpr", "related_fpr", "plain_fpr", "gap")},
            "adv_related_fpr": as_float(best_adv_related, "fpr"),
            "objective": "best_mcc",
        },
        "internal_low_fpr": {
            "method": "Hidden+answer verifier",
            "external_detector": "no",
            **{key: float(internal_low_fpr[key]) for key in ("mcc", "tpr", "fpr", "related_fpr", "plain_fpr", "gap")},
            "adv_related_fpr": as_float(low_fpr_adv_related, "fpr"),
            "objective": "min_fpr_tpr0.80",
        },
        "external_positive_control": {
            "method": "Hybrid gate+rescue",
            "external_detector": "yes / OWLv2",
            "mcc": as_float(owlv2, "macro_mcc"),
            "tpr": as_float(owlv2, "macro_tpr"),
            "fpr": as_float(owlv2, "macro_fpr"),
            "related_fpr": as_float(owlv2, "macro_related_fpr"),
            "plain_fpr": as_float(owlv2, "macro_plain_fpr"),
            "gap": as_float(owlv2, "macro_related_minus_plain_fpr"),
            "adv_related_fpr": as_float(owlv2, "adversarial_related_fpr"),
        },
    }


def load_chair_detection_summary(base: Path) -> dict[str, Any]:
    chair_base = base / "chair_internal_verifier_full"
    metrics = read_json(chair_base / "eval_answer_smallgrid" / "chair_verifier_metrics.json")
    controlled = read_json(chair_base / "controlled_analysis" / "controlled_metrics.json")
    answer_absence = find_row(metrics["baselines"], "score", "answer_absence_score")
    answer_only = metrics["macro"]
    gen_pos = find_row(metrics["baselines"], "score", "gen_pos")
    answer_control = controlled["scores"]["answer_absence_score"]
    gen_pos_control = controlled["scores"]["gen_pos"]

    return {
        "objects": controlled["objects"],
        "answer_absence": {
            "score": "answer_absence_score",
            "auroc": float(answer_absence["auroc"]),
            "mcc": float(answer_absence["mcc"]),
            "tpr": float(answer_absence["recall_tpr"]),
            "fpr": float(answer_absence["fpr"]),
            "within_bin_auroc": float(answer_control["within_bin_auroc"]),
            "matched_pair_auroc": float(answer_control["matched_pair_auroc"]),
            "residual_auroc": float(answer_control["residual_auroc"]),
        },
        "answer_only_5fold": {
            "score": "answer-only 5-fold verifier",
            "auroc": float(answer_only["auroc"]),
            "mcc": float(answer_only["mcc"]),
            "tpr": float(answer_only["recall_tpr"]),
            "fpr": float(answer_only["fpr"]),
        },
        "position_only": {
            "score": "gen_pos",
            "auroc": float(gen_pos["auroc"]),
            "mcc": float(gen_pos["mcc"]),
            "tpr": float(gen_pos["recall_tpr"]),
            "fpr": float(gen_pos["fpr"]),
            "within_bin_auroc": float(gen_pos_control["within_bin_auroc"]),
            "matched_pair_auroc": float(gen_pos_control["matched_pair_auroc"]),
            "residual_auroc": float(gen_pos_control["residual_auroc"]),
        },
    }


def load_chair_intervention_summary(base: Path) -> dict[str, Any]:
    rows = read_csv(base / "chair_internal_verifier_full" / "intervention_utility_summary.csv")
    delete_top10 = find_row(rows, "run", "intervention_answer_delete_top10")
    preserved = [row for row in rows if row["passes_preservation_gate"] == "1"]
    best_preserved = max(preserved, key=lambda row: as_float(row, "chairi_reduction"))

    return {
        "upper_bound": delete_top10,
        "main_preserved": best_preserved,
        "preservation_gate": {
            "mean_lcs_retention": ">=0.995",
            "object_mention_retention": ">=0.965",
            "recall_delta": ">=-0.011",
        },
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def build_markdown(summary: dict[str, Any]) -> str:
    pope = summary["pope"]
    chair_det = summary["chair_detection"]
    chair_int = summary["chair_intervention"]
    pope_rows = [
        pope["vanilla"],
        pope["internal_best"],
        pope["internal_low_fpr"],
        pope["external_positive_control"],
    ]
    pope_table = markdown_table(
        ["Method", "External detector", "Objective", "MCC", "TPR", "FPR", "Related FPR", "Plain FPR", "Gap", "Adv. related FPR"],
        [
            [
                row["method"],
                row["external_detector"],
                row.get("objective", "-"),
                fmt(row["mcc"]),
                fmt(row["tpr"]),
                fmt(row["fpr"]),
                fmt(row["related_fpr"]),
                fmt(row["plain_fpr"]),
                fmt(row["gap"]),
                fmt(row["adv_related_fpr"]),
            ]
            for row in pope_rows
        ],
    )
    chair_detection_table = markdown_table(
        ["Score", "AUROC", "Within-bin", "Matched", "Residual", "MCC", "TPR", "FPR"],
        [
            [
                "Position only (`gen_pos`)",
                fmt(chair_det["position_only"]["auroc"]),
                fmt(chair_det["position_only"]["within_bin_auroc"]),
                fmt(chair_det["position_only"]["matched_pair_auroc"]),
                fmt(chair_det["position_only"]["residual_auroc"]),
                fmt(chair_det["position_only"]["mcc"]),
                fmt(chair_det["position_only"]["tpr"]),
                fmt(chair_det["position_only"]["fpr"]),
            ],
            [
                "`answer_absence_score`",
                fmt(chair_det["answer_absence"]["auroc"]),
                fmt(chair_det["answer_absence"]["within_bin_auroc"]),
                fmt(chair_det["answer_absence"]["matched_pair_auroc"]),
                fmt(chair_det["answer_absence"]["residual_auroc"]),
                fmt(chair_det["answer_absence"]["mcc"]),
                fmt(chair_det["answer_absence"]["tpr"]),
                fmt(chair_det["answer_absence"]["fpr"]),
            ],
            [
                "Answer-only 5-fold verifier",
                fmt(chair_det["answer_only_5fold"]["auroc"]),
                "-",
                "-",
                "-",
                fmt(chair_det["answer_only_5fold"]["mcc"]),
                fmt(chair_det["answer_only_5fold"]["tpr"]),
                fmt(chair_det["answer_only_5fold"]["fpr"]),
            ],
        ],
    )
    main = chair_int["main_preserved"]
    upper = chair_int["upper_bound"]
    intervention_table = markdown_table(
        ["Role", "Run", "CHAIRi", "CHAIRs", "CHAIRi red.", "LCS retention", "Object retention", "Recall delta", "Pass gate"],
        [
            [
                "Main preserved",
                main["run"],
                fmt(as_float(main, "chairi")),
                fmt(as_float(main, "chairs")),
                fmt(as_float(main, "chairi_reduction")),
                fmt(as_float(main, "mean_lcs_retention"), 4),
                fmt(as_float(main, "object_mention_retention"), 4),
                fmt(as_float(main, "recall_delta"), 4),
                "yes",
            ],
            [
                "Upper bound",
                upper["run"],
                fmt(as_float(upper, "chairi")),
                fmt(as_float(upper, "chairs")),
                fmt(as_float(upper, "chairi_reduction")),
                fmt(as_float(upper, "mean_lcs_retention"), 4),
                fmt(as_float(upper, "object_mention_retention"), 4),
                fmt(as_float(upper, "recall_delta"), 4),
                "no",
            ],
        ],
    )

    return f"""# No-External-Detector Summary

Generated by `mitigation/scripts/build_no_external_detector_summary.py` from
the current result artifacts. This is the paper-facing selection table for the
internal method family; OWLv2 appears only as an external positive control.

## Selected Method

- **POPE:** use the leave-one-split-out **hidden+answer verifier** as the current
  no-external-detector main row. The best-MCC operating point is the default
  table row; the `min_fpr_tpr0.80` point is the stricter FPR ablation.
- **CHAIR detection:** use direct internal **answer absence**
  (`answer_absence_score = -target_yes_margin`).
- **CHAIR intervention:** use **generic rewrite top-10** as the preserved
  no-external-detector intervention. Keep delete top-10 as an upper-bound
  ablation, not the main row.

## POPE Semantic-Neighbor Gate

{pope_table}

Interpretation: the internal hidden+answer verifier improves vanilla without
using OWLv2, but it remains below the OWLv2 positive control. This supports
detector removal as a current method variant, not a claim that internal evidence
fully matches external region verification.

## CHAIR Object-Claim Detection

Objects: {chair_det["objects"]["total"]} mentions, {chair_det["objects"]["hallucinated"]} hallucinated.

{chair_detection_table}

Interpretation: CHAIR object-claim detection is best handled by answer absence,
not hidden-margin TDEV. It remains strong under generation-position control.

## CHAIR Caption Intervention

Preservation gate: LCS retention {chair_int["preservation_gate"]["mean_lcs_retention"]},
object retention {chair_int["preservation_gate"]["object_mention_retention"]},
Recall delta {chair_int["preservation_gate"]["recall_delta"]}.

{intervention_table}

Interpretation: generic rewrite top-10 is the current paper-facing CHAIR
intervention. Delete top-10 gives stronger CHAIR reduction, but fails the
preservation gate and should be reported only as an upper-bound ablation.
"""


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    summary = {
        "sources": {
            "pope_control_table": str(base / "paper_control_table" / "semantic_neighbor_control_table.csv"),
            "pope_internal": str(base / "answer_confidence_tdev_full" / "cross_split_hidden_answer6_standardized" / "cross_split_summary.json"),
            "chair_detection": str(base / "chair_internal_verifier_full" / "eval_answer_smallgrid" / "chair_verifier_metrics.json"),
            "chair_controlled": str(base / "chair_internal_verifier_full" / "controlled_analysis" / "controlled_metrics.json"),
            "chair_intervention": str(base / "chair_internal_verifier_full" / "intervention_utility_summary.csv"),
        },
        "pope": load_pope_summary(base),
        "chair_detection": load_chair_detection_summary(base),
        "chair_intervention": load_chair_intervention_summary(base),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
