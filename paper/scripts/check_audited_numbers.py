#!/usr/bin/env python3
"""Check that paper tables match audited result artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper"


def _numbers_from_row(table_text: str, row_label: str) -> list[float]:
    pattern = re.compile(rf"^{re.escape(row_label)}\s*&(.+?)\\\\", re.MULTILINE)
    match = pattern.search(table_text)
    if not match:
        raise AssertionError(f"Missing table row: {row_label}")
    return [float(value) for value in re.findall(r"[+-]?\d+\.\d+", match.group(1))]


def _assert_rounded(actual: list[float], expected: list[float], digits: int, label: str) -> None:
    rounded = [round(value, digits) for value in expected]
    if actual != rounded:
        raise AssertionError(f"{label}: table={actual} expected={rounded}")


def check_detection_main() -> None:
    metrics = json.loads((PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_baselines/metrics.json").read_text())
    table = (PAPER_ROOT / "tables/table_detection_main.tex").read_text()
    mapping = {
        "NLL": "nll_hallu_score",
        "Entropy": "entropy_hallu_score",
        "IC": "ic_hallu_score",
        "GLSim": "glsim_hallu_score",
        "GLSim (global)": "glsim_global_hallu_score",
        "GLSim (local)": "glsim_local_hallu_score",
        "SVAR": "svar_layer0_hallu_score",
        "PAS": "pas_layer0_hallu_score",
        "Beyond-ADS": "beyond_ads_hallu_score",
        "Beyond-CGC": "beyond_cgc_hallu_score",
        "Beyond-ADS+CGC": "beyond_adscgc_hallu_score",
    }
    for row_label, key in mapping.items():
        score = metrics["scores"][key]
        expected = [
            score["overall_auroc"],
            score["within_bin_auroc"],
            score["matched_pair_auroc"],
            score["residual_auroc"],
        ]
        actual = _numbers_from_row(table, row_label)[-4:]
        _assert_rounded(actual, expected, 3, f"main:{row_label}")


def check_strong_controls() -> None:
    rows = {
        row["score"]: row
        for row in csv.DictReader(
            (PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_baselines/controlled_analysis/controlled_summary.csv").open()
        )
    }
    table = (PAPER_ROOT / "tables/table_strong_controls.tex").read_text()
    mapping = {
        "NLL": "nll_hallu_score",
        "Entropy": "entropy_hallu_score",
        "IC": "ic_hallu_score",
        "PAS": "pas_layer0_hallu_score",
        "SVAR": "svar_layer0_hallu_score",
        "GLSim-local": "glsim_local_hallu_score",
        "SinkDetect-global": "sinkdetect_best_global_hallu_score",
        "SinkDetect-CLC": "sinkdetect_clc_hallu_score",
    }
    for row_label, key in mapping.items():
        row = rows[key]
        expected = [
            float(row["same_word_matched_auroc"]),
            float(row["cubic_residual_auroc"]),
            float(row["retained_same_word_vs_overall"]),
        ]
        actual = _numbers_from_row(table, row_label)
        rounded = [round(expected[0], 3), round(expected[1], 3), round(expected[2], 2)]
        if actual != rounded:
            raise AssertionError(f"strong:{row_label}: table={actual} expected={rounded}")


def check_mitigation_behavior() -> None:
    pope_rows = {
        row["method"]: row
        for row in csv.DictReader((PROJECT_ROOT / "mitigation/results/coco_llava_7b_attention_only/audit_pope_macro.csv").open())
        if row["split"] == "macro"
    }
    chair_rows = {
        row["method"]: row
        for row in csv.DictReader((PROJECT_ROOT / "mitigation/results/coco_llava_7b_attention_only/audit_chair.csv").open())
    }
    table = (PAPER_ROOT / "tables/table_mitigation_behavior.tex").read_text()
    mapping = {
        "PAI-attn-only": "pai",
        "ClearSight": "clearsight",
        "VisAttnSink": "visattnsink",
    }
    for row_label, method in mapping.items():
        pope = pope_rows[method]
        chair = chair_rows[method]
        expected = [
            float(pope["delta_accuracy"]),
            float(pope["delta_mcc"]),
            float(pope["delta_yes_rate"]),
            float(pope["delta_recall_tpr"]),
            float(pope["delta_fpr"]),
            float(chair["delta_CHAIRi"]),
            float(chair["delta_mean_object_mentions"]),
            float(chair["delta_mean_hallucinated_mentions"]),
        ]
        actual = _numbers_from_row(table, row_label)
        rounded = [round(value, 3) for value in expected[:6]] + [round(value, 2) for value in expected[6:]]
        if actual != rounded:
            raise AssertionError(f"mitigation:{row_label}: table={actual} expected={rounded}")


def check_semantic_neighbor_fpr() -> None:
    rows = list(
        csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv").open()
        )
    )
    lookup = {(row["split"], row["method"], row["subset"]): float(row["fpr"]) * 100 for row in rows}
    table = (PAPER_ROOT / "tables/table_semantic_neighbor_fpr.tex").read_text()
    row_labels = {
        ("random", "vanilla"): "random & vanilla",
        ("random", "pai"): "random & PAI-attn-only",
        ("random", "clearsight"): "random & ClearSight",
        ("random", "visattnsink"): "random & VisAttnSink",
        ("popular", "vanilla"): "popular & vanilla",
        ("popular", "pai"): "popular & PAI-attn-only",
        ("popular", "clearsight"): "popular & ClearSight",
        ("popular", "visattnsink"): "popular & VisAttnSink",
        ("adversarial", "vanilla"): "adversarial & vanilla",
        ("adversarial", "pai"): "adversarial & PAI-attn-only",
        ("adversarial", "clearsight"): "adversarial & ClearSight",
        ("adversarial", "visattnsink"): "adversarial & VisAttnSink",
    }
    for (split, method), row_label in row_labels.items():
        expected = [
            lookup[(split, method, "negative")],
            lookup[(split, method, "negative_related_present")],
            lookup[(split, method, "negative_absent_plain")],
            lookup[(split, method, "negative_related_present")] - lookup[(split, method, "negative_absent_plain")],
        ]
        actual = _numbers_from_row(table, row_label)
        _assert_rounded(actual, expected, 1, f"semantic:{row_label}")


def main() -> None:
    check_detection_main()
    check_strong_controls()
    check_mitigation_behavior()
    check_semantic_neighbor_fpr()
    print("All audited paper numbers match current result artifacts.")


if __name__ == "__main__":
    main()
