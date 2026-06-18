#!/usr/bin/env python3
"""Build paper-facing TDEV ablation summary tables.

The TDEV evidence is spread across POPE mitigation outputs and CHAIR detection
outputs. This script gathers the committed metric artifacts into one markdown
summary so paper tables can be regenerated instead of copied by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


POPE_METHODS = [
    (
        "OWLv2 target score direct",
        "mitigation/results/semantic_neighbor_audit/owlv2_target_score_direct/direct_score_metrics.csv",
    ),
    (
        "OWLv2 margin direct",
        "mitigation/results/semantic_neighbor_audit/owlv2_margin_direct_cal/direct_score_metrics.csv",
    ),
    (
        "Two-stage direct",
        "mitigation/results/semantic_neighbor_audit/owlv2_two_stage_direct/two_stage_metrics.csv",
    ),
    (
        "Two-stage gate",
        "mitigation/results/semantic_neighbor_audit/owlv2_two_stage_gate/two_stage_metrics.csv",
    ),
    (
        "Hybrid gate+rescue",
        "mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_metrics.csv",
    ),
]


CHAIR_SCORES = [
    "owlv2_target_absence",
    "owlv2_margin_absence",
    "owlv2_two_stage_absence",
    "owlv2_hybrid_positive_branch_absence",
    "owlv2_hybrid_mcc_positive_branch_absence",
    "owlv2_target_absence_plus_neighbor_dominance_0.25",
    "owlv2_neighbor_presence",
]


CHAIR_LABELS = {
    "owlv2_target_absence": "OWLv2 target absence",
    "owlv2_margin_absence": "OWLv2 margin absence",
    "owlv2_two_stage_absence": "OWLv2 two-stage absence",
    "owlv2_hybrid_positive_branch_absence": "Hybrid positive branch absence",
    "owlv2_hybrid_mcc_positive_branch_absence": "Hybrid MCC positive branch absence",
    "owlv2_target_absence_plus_neighbor_dominance_0.25": "Target absence + 0.25 neighbor dominance",
    "owlv2_neighbor_presence": "OWLv2 neighbor presence",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], split: str, subset: str) -> dict[str, str]:
    for row in rows:
        if row["split"] == split and row["subset"] == subset:
            return row
    raise KeyError(f"Missing row split={split} subset={subset}")


def fmt(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def build_pope_rows() -> list[dict[str, str]]:
    out = []
    for name, rel_path in POPE_METHODS:
        rows = read_csv(ROOT / rel_path)
        macro = find_row(rows, "macro", "all")
        adv = find_row(rows, "adversarial", "all")
        adv_related = find_row(rows, "adversarial", "negative_related_present")
        adv_plain = find_row(rows, "adversarial", "negative_absent_plain")
        out.append({
            "method": name,
            "macro_mcc": fmt(macro["mcc"]),
            "macro_tpr": fmt(macro["recall_tpr"]),
            "macro_fpr": fmt(macro["fpr"]),
            "adv_mcc": fmt(adv["mcc"]),
            "adv_related_fpr": fmt(adv_related["fpr"]),
            "adv_plain_fpr": fmt(adv_plain["fpr"]),
            "adv_related_gap": fmt(float(adv_related["fpr"]) - float(adv_plain["fpr"])),
        })
    return out


def build_chair_rows() -> list[dict[str, str]]:
    rows = read_csv(ROOT / "detection/baselines/results/owlv2_region_posthoc_scores/owlv2_region_posthoc_metrics.csv")
    by_name = {row["score"]: row for row in rows}
    out = []
    for score in CHAIR_SCORES:
        row = by_name[score]
        out.append({
            "score": CHAIR_LABELS[score],
            "overall": fmt(row["overall_auroc"]),
            "within": fmt(row["within_bin_auroc"]),
            "matched": fmt(row["matched_pair_auroc"]),
            "residual": fmt(row["residual_auroc"]),
        })
    return out


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    pope_rows = build_pope_rows()
    chair_rows = build_chair_rows()

    pope_table = markdown_table(
        ["POPE variant", "Macro MCC", "TPR", "FPR", "Adv. MCC", "Adv. related FPR", "Adv. plain FPR", "Gap"],
        [
            [
                row["method"],
                row["macro_mcc"],
                row["macro_tpr"],
                row["macro_fpr"],
                row["adv_mcc"],
                row["adv_related_fpr"],
                row["adv_plain_fpr"],
                row["adv_related_gap"],
            ]
            for row in pope_rows
        ],
    )
    chair_table = markdown_table(
        ["CHAIR score", "Overall", "Within-bin", "Matched-pair", "Residual"],
        [
            [row["score"], row["overall"], row["within"], row["matched"], row["residual"]]
            for row in chair_rows
        ],
    )
    text = f"""# TDEV Ablation Summary

Generated by:

```bash
python scripts/build_tdev_ablation_summary.py
```

This table consolidates the current TDEV ablations from existing metric
artifacts. It is intended as a paper-facing checkpoint, not a replacement for
the full result files.

## POPE Semantic-Neighbor Ablation

{pope_table}

Interpretation: raw target detection has the best aggregate direct MCC but high
semantic-neighbor false positives. The two-stage and hybrid rules reduce
related-present FPR. Hybrid gate+rescue keeps the low-FPR gate behavior while
recovering some recall.

## CHAIR Object-Mention Detection Ablation

{chair_table}

Interpretation: target absence is the main CHAIR signal. The hybrid positive
branch transfers to object mentions, and a moderate neighbor-dominance penalty
improves position-residualized AUROC without collapsing the controlled AUROCs.
"""
    out_path = ROOT / "docs/tdev_ablation_summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
