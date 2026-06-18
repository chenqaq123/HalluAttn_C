#!/usr/bin/env python3
"""Build a paper-facing semantic-neighbor control table from saved POPE metrics.

The table keeps aggregate POPE behavior next to related-present and plain-absent
negative FPRs. It is designed to prevent paper claims from drifting back to
plain accuracy/MCC without checking the `looking is not grounding` failure mode:
models can answer yes when related visual evidence is present but the queried
target is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLITS = ("random", "popular", "adversarial")
METHOD_SPECS = [
    {
        "label": "Vanilla",
        "family": "base",
        "path": "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv",
        "method": "vanilla",
        "note": "Base LLaVA answerer.",
    },
    {
        "label": "PAI attention-only",
        "family": "attention intervention",
        "path": "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv",
        "method": "pai",
        "note": "Attention-only component under the local greedy runtime.",
    },
    {
        "label": "ClearSight VAF",
        "family": "attention intervention",
        "path": "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv",
        "method": "clearsight",
        "note": "Visual attention focus component.",
    },
    {
        "label": "VisAttnSink",
        "family": "attention intervention",
        "path": "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv",
        "method": "visattnsink",
        "note": "Visual attention sink redistribution component.",
    },
    {
        "label": "VCD-greedy",
        "family": "contrastive decoding",
        "path": "mitigation/results/semantic_neighbor_audit/vcd_greedy_subset_eval/semantic_neighbor_subset_metrics.csv",
        "method": "vcd",
        "note": "Controlled greedy VCD port.",
    },
    {
        "label": "OWLv2 target direct",
        "family": "region evidence",
        "path": "mitigation/results/semantic_neighbor_audit/owlv2_target_score_direct/direct_score_metrics.csv",
        "method": None,
        "note": "Raw target-region evidence calibrated on random split.",
    },
    {
        "label": "OWLv2 margin direct",
        "family": "target-vs-neighbor verifier",
        "path": "mitigation/results/semantic_neighbor_audit/owlv2_margin_direct_cal/direct_score_metrics.csv",
        "method": None,
        "note": "Strict target-minus-neighbor margin.",
    },
    {
        "label": "Two-stage direct",
        "family": "target-vs-neighbor verifier",
        "path": "mitigation/results/semantic_neighbor_audit/owlv2_two_stage_direct/two_stage_metrics.csv",
        "method": None,
        "note": "High target evidence or medium target evidence plus margin.",
    },
    {
        "label": "Two-stage gate",
        "family": "target-vs-neighbor verifier",
        "path": "mitigation/results/semantic_neighbor_audit/owlv2_two_stage_gate/two_stage_metrics.csv",
        "method": None,
        "note": "Two-stage verifier applied as a gate over vanilla yes answers.",
    },
    {
        "label": "Hybrid gate+rescue",
        "family": "target-vs-neighbor verifier",
        "path": "mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_metrics.csv",
        "method": None,
        "note": "Asymmetric positive gate with conservative no-answer rescue.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic-neighbor paper control table")
    parser.add_argument("--output_dir", default="mitigation/results/semantic_neighbor_audit/paper_control_table")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_rows(rows: list[dict[str, str]], method: str | None) -> list[dict[str, str]]:
    if method is None:
        return rows
    return [row for row in rows if row.get("method") == method]


def by_split_subset(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["split"], row["subset"]): row for row in rows}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def aggregate_counts(index: dict[tuple[str, str], dict[str, str]], subset: str) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "samples": 0, "invalid": 0}
    for split in SPLITS:
        row = index.get((split, subset))
        if row is None:
            raise KeyError(f"missing split={split} subset={subset}")
        for key in counts:
            if key in row and row[key] != "":
                counts[key] += int(float(row[key]))
    return counts


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "samples": float(n),
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall_tpr": tpr,
        "tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "mcc": ((tp * tn - fp * fn) / denom) if denom else 0.0,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def macro_value(index: dict[tuple[str, str], dict[str, str]], subset: str, key: str) -> float:
    # Always pool split-level confusion counts. Some source CSVs include macro
    # rows and others do not; recomputing here keeps table rows comparable.
    metrics = metrics_from_counts(aggregate_counts(index, subset))
    if key not in metrics:
        raise KeyError(f"unsupported pooled metric {key}")
    return metrics[key]


def split_value(index: dict[tuple[str, str], dict[str, str]], split: str, subset: str, key: str) -> float:
    row = index.get((split, subset))
    if row is None:
        raise KeyError(f"missing split={split} subset={subset}")
    return f(row, key)


def fmt(x: float) -> str:
    return f"{x:.3f}"


def build_summary(project_root: Path) -> list[dict[str, object]]:
    out = []
    for spec in METHOD_SPECS:
        path = project_root / str(spec["path"])
        rows = select_rows(read_csv(path), spec["method"])
        if not rows:
            raise ValueError(f"No rows for {spec['label']} from {path}")
        index = by_split_subset(rows)
        related = macro_value(index, "negative_related_present", "fpr")
        plain = macro_value(index, "negative_absent_plain", "fpr")
        adv_related = split_value(index, "adversarial", "negative_related_present", "fpr")
        adv_plain = split_value(index, "adversarial", "negative_absent_plain", "fpr")
        out.append({
            "method": spec["label"],
            "family": spec["family"],
            "source_csv": spec["path"],
            "source_method": spec["method"] or "",
            "macro_mcc": macro_value(index, "all", "mcc"),
            "macro_tpr": macro_value(index, "all", "recall_tpr"),
            "macro_fpr": macro_value(index, "all", "fpr"),
            "macro_yes_rate": macro_value(index, "all", "yes_rate"),
            "macro_related_fpr": related,
            "macro_plain_fpr": plain,
            "macro_related_minus_plain_fpr": related - plain,
            "adversarial_mcc": split_value(index, "adversarial", "all", "mcc"),
            "adversarial_related_fpr": adv_related,
            "adversarial_plain_fpr": adv_plain,
            "adversarial_related_minus_plain_fpr": adv_related - adv_plain,
            "note": spec["note"],
        })
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    cols = [
        ("method", "Method"),
        ("family", "Family"),
        ("macro_mcc", "Macro MCC"),
        ("macro_tpr", "TPR"),
        ("macro_fpr", "FPR"),
        ("macro_yes_rate", "Yes rate"),
        ("macro_related_fpr", "Related FPR"),
        ("macro_plain_fpr", "Plain FPR"),
        ("macro_related_minus_plain_fpr", "Gap"),
        ("adversarial_related_fpr", "Adv. related FPR"),
    ]
    lines = ["| " + " | ".join(name for _, name in cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        cells = []
        for key, _ in cols:
            value = row[key]
            cells.append(fmt(float(value)) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_summary(project_root)
    csv_path = output_dir / "semantic_neighbor_control_table.csv"
    json_path = output_dir / "semantic_neighbor_control_table.json"
    md_path = output_dir / "semantic_neighbor_control_table.md"

    write_csv(csv_path, rows)
    payload = {
        "description": "Paper-facing semantic-neighbor control table from saved POPE metrics.",
        "rows": rows,
        "interpretation": (
            "Attention and contrastive-decoding controls do not close the related-present false-positive gap. "
            "Raw target-region evidence improves aggregate MCC but over-fires on related-present negatives. "
            "Target-vs-neighbor verification reduces that failure mode, with hybrid gate+rescue retaining the best current tradeoff."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Semantic-Neighbor Control Table\n\n"
        "This table is built from saved POPE metric CSVs. It keeps aggregate behavior next to the semantic-neighbor stress test: related-present negative FPR versus plain-absent negative FPR.\n\n"
        + markdown_table(rows)
        + "\nInterpretation: attention and contrastive-decoding controls do not close the related-present false-positive gap. Raw target-region evidence improves aggregate MCC but over-fires on related-present negatives. Target-vs-neighbor verification reduces that failure mode, with hybrid gate+rescue retaining the best current tradeoff.\n",
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
