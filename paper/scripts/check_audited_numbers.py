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
    return [float(value) for value in re.findall(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", match.group(1))]


def _numbers_from_markdown_row(table_text: str, row_label: str) -> list[float]:
    pattern = re.compile(rf"^\|\s*{re.escape(row_label)}\s*\|(.+)$", re.MULTILINE)
    match = pattern.search(table_text)
    if not match:
        raise AssertionError(f"Missing markdown table row: {row_label}")
    return [float(value) for value in re.findall(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", match.group(1))]


def _assert_rounded(actual: list[float], expected: list[float], digits: int, label: str) -> None:
    rounded = [round(value, digits) for value in expected]
    if actual != rounded:
        raise AssertionError(f"{label}: table={actual} expected={rounded}")


def _assert_contains(text: str, snippet: str, label: str) -> None:
    if snippet not in text:
        raise AssertionError(f"{label}: missing snippet {snippet!r}")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}"


def _control_rows_by_method() -> dict[str, dict[str, str]]:
    return {
        row["method"]: row
        for row in csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.csv").open()
        )
    }


def _control_value(rows: dict[str, dict[str, str]], method: str, key: str) -> float:
    return float(rows[method][key])


def _macro_all_from_subset_csv(metrics_path: str, method: str) -> dict[str, float]:
    rows = [
        row
        for row in csv.DictReader((PROJECT_ROOT / metrics_path).open())
        if row["method"] == method and row["subset"] == "all"
    ]
    if len(rows) != 3:
        raise AssertionError(f"Expected three all-split rows for {method} in {metrics_path}, found {len(rows)}")
    keys = ["accuracy", "mcc", "yes_rate", "recall_tpr", "fpr"]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


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
    vcd_rows = {
        row["method"]: row
        for row in csv.DictReader((PROJECT_ROOT / "mitigation/results/pope_full_vcd_greedy_audit/audit_pope_macro.csv").open())
        if row["split"] == "macro"
    }
    pope_rows.update({"vcd": vcd_rows["vcd"]})
    vanilla_macro = _macro_all_from_subset_csv(
        "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv",
        "vanilla",
    )
    nolan_macro = _macro_all_from_subset_csv(
        "mitigation/results/semantic_neighbor_audit/nolan_full_subset_eval/semantic_neighbor_subset_metrics.csv",
        "nolan",
    )
    pope_rows["nolan"] = {
        "delta_accuracy": nolan_macro["accuracy"] - vanilla_macro["accuracy"],
        "delta_mcc": nolan_macro["mcc"] - vanilla_macro["mcc"],
        "delta_yes_rate": nolan_macro["yes_rate"] - vanilla_macro["yes_rate"],
        "delta_recall_tpr": nolan_macro["recall_tpr"] - vanilla_macro["recall_tpr"],
        "delta_fpr": nolan_macro["fpr"] - vanilla_macro["fpr"],
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
        "VCD-greedy": "vcd",
        "NoLan-compatible": "nolan",
    }
    for row_label, method in mapping.items():
        pope = pope_rows[method]
        expected = [
            float(pope["delta_accuracy"]),
            float(pope["delta_mcc"]),
            float(pope["delta_yes_rate"]),
            float(pope["delta_recall_tpr"]),
            float(pope["delta_fpr"]),
        ]
        if method in chair_rows:
            chair = chair_rows[method]
            expected.extend([
                float(chair["delta_CHAIRi"]),
                float(chair["delta_mean_object_mentions"]),
                float(chair["delta_mean_hallucinated_mentions"]),
            ])
        actual = _numbers_from_row(table, row_label)
        rounded = [round(value, 3) for value in expected[:6]] + [round(value, 2) for value in expected[6:]]
        if actual != rounded:
            raise AssertionError(f"mitigation:{row_label}: table={actual} expected={rounded}")


def check_attention_audit() -> None:
    rows = {
        row["method"]: row
        for row in csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/coco_llava_7b_attention_audit/pope/adversarial/attention_audit_summary.csv").open()
        )
    }
    vanilla = rows["vanilla"]
    table = (PAPER_ROOT / "tables/table_attention_audit.tex").read_text()
    mapping = {
        "PAI": "pai",
        "ClearSight": "clearsight",
        "VisSink": "visattnsink",
    }
    for row_label, method in mapping.items():
        row = rows[method]
        expected = [
            float(row["matched_delta_active_visual_vs_vanilla"]),
            float(row["recall_tpr"]) - float(vanilla["recall_tpr"]),
            float(row["fpr"]) - float(vanilla["fpr"]),
            float(row["tp_mean_post_visual_mass"]),
            float(row["fp_mean_post_visual_mass"]),
        ]
        actual = _numbers_from_row(table, row_label)
        _assert_rounded(actual, expected, 3, f"attention-audit:{row_label}")


def check_semantic_neighbor_fpr() -> None:
    rows = list(
        csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/attention_only_subset_eval/semantic_neighbor_subset_metrics.csv").open()
        )
    )
    rows.extend(
        row
        for row in csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/vcd_greedy_subset_eval/semantic_neighbor_subset_metrics.csv").open()
        )
        if row["method"] == "vcd"
    )
    rows.extend(
        row
        for row in csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/nolan_full_subset_eval/semantic_neighbor_subset_metrics.csv").open()
        )
        if row["method"] == "nolan"
    )
    lookup = {(row["split"], row["method"], row["subset"]): float(row["fpr"]) * 100 for row in rows}
    table = (PAPER_ROOT / "tables/table_semantic_neighbor_fpr.tex").read_text()
    row_labels = {
        ("random", "vanilla"): "random & vanilla",
        ("random", "vcd"): "random & VCD-greedy",
        ("random", "nolan"): "random & NoLan-compatible",
        ("random", "pai"): "random & PAI-attn-only",
        ("random", "clearsight"): "random & ClearSight",
        ("random", "visattnsink"): "random & VisAttnSink",
        ("popular", "vanilla"): "popular & vanilla",
        ("popular", "vcd"): "popular & VCD-greedy",
        ("popular", "nolan"): "popular & NoLan-compatible",
        ("popular", "pai"): "popular & PAI-attn-only",
        ("popular", "clearsight"): "popular & ClearSight",
        ("popular", "visattnsink"): "popular & VisAttnSink",
        ("adversarial", "vanilla"): "adversarial & vanilla",
        ("adversarial", "vcd"): "adversarial & VCD-greedy",
        ("adversarial", "nolan"): "adversarial & NoLan-compatible",
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


def check_region_verifier_detection() -> None:
    baseline = json.loads((PROJECT_ROOT / "detection/baselines/results/coco_llava_7b_baselines/metrics.json").read_text())
    region = json.loads(
        (PROJECT_ROOT / "detection/baselines/results/owlv2_region_detection/owlv2_region_detection_metrics.json").read_text()
    )
    table = (PAPER_ROOT / "tables/table_region_verifier_detection.tex").read_text()
    mapping = {
        "IC": baseline["scores"]["ic_hallu_score"],
        "OWLv2 target absence": region["metrics"]["owlv2_target_absence"],
        "OWLv2 two-stage absence": region["metrics"]["owlv2_two_stage_absence"],
        "OWLv2 margin absence": region["metrics"]["owlv2_margin_absence"],
        "OWLv2 neighbor presence": region["metrics"]["owlv2_neighbor_presence"],
    }
    for row_label, score in mapping.items():
        expected = [
            score["overall_auroc"],
            score["within_bin_auroc"],
            score["matched_pair_auroc"],
            score["residual_auroc"],
        ]
        actual = _numbers_from_row(table, row_label)[-4:]
        _assert_rounded(actual, expected, 3, f"region-detection:{row_label}")


def _pope_metric_values(metrics_path: str) -> list[float]:
    rows = {
        (row["split"], row["subset"]): row
        for row in csv.DictReader((PROJECT_ROOT / metrics_path).open())
    }
    macro = rows[("macro", "all")]
    macro_related = rows[("macro", "negative_related_present")]
    adversarial_related = rows[("adversarial", "negative_related_present")]
    return [
        float(macro["mcc"]),
        float(macro["recall_tpr"]),
        float(macro["fpr"]),
        float(macro_related["fpr"]),
        float(adversarial_related["fpr"]),
    ]


def check_region_verifier_pope() -> None:
    table = (PAPER_ROOT / "tables/table_region_verifier_pope.tex").read_text()
    mapping = {
        "direct & target score": "mitigation/results/semantic_neighbor_audit/owlv2_target_score_direct/direct_score_metrics.csv",
        "direct & margin score": "mitigation/results/semantic_neighbor_audit/owlv2_margin_direct_cal/direct_score_metrics.csv",
        "direct & two-stage": "mitigation/results/semantic_neighbor_audit/owlv2_two_stage_direct/two_stage_metrics.csv",
        "gate & target score": "mitigation/results/semantic_neighbor_audit/owlv2_target_score_gate/tdev_gate_metrics.csv",
        "gate & two-stage": "mitigation/results/semantic_neighbor_audit/owlv2_two_stage_gate/two_stage_metrics.csv",
        "hybrid & gate+rescue": "mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_metrics.csv",
    }
    for row_label, metrics_path in mapping.items():
        expected = _pope_metric_values(metrics_path)
        actual = _numbers_from_row(table, row_label)
        _assert_rounded(actual, expected, 3, f"region-pope:{row_label}")


def check_semantic_neighbor_control_table() -> None:
    rows = list(
        csv.DictReader(
            (PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.csv").open()
        )
    )
    control_table = (
        PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md"
    ).read_text()
    status_note = (PROJECT_ROOT / "docs/current_result_baseline_comparison.md").read_text()
    keys = [
        "macro_mcc",
        "macro_tpr",
        "macro_fpr",
        "macro_yes_rate",
        "macro_related_fpr",
        "macro_plain_fpr",
        "macro_related_minus_plain_fpr",
        "adversarial_related_fpr",
    ]
    for row in rows:
        label = row["method"]
        expected = [float(row[key]) for key in keys]
        _assert_rounded(
            _numbers_from_markdown_row(control_table, label),
            expected,
            3,
            f"semantic-control-table:{label}",
        )
        _assert_rounded(
            _numbers_from_markdown_row(status_note, label),
            expected,
            3,
            f"current-result-baseline-table:{label}",
        )


def check_key_prose_claims() -> None:
    rows = _control_rows_by_method()
    intro = (PAPER_ROOT / "sections/01_introduction.tex").read_text()
    mitigation = (PAPER_ROOT / "sections/05_mitigation_findings.tex").read_text()
    associated = (PAPER_ROOT / "sections/06_associated_evidence_audit.tex").read_text()
    evidence = (PROJECT_ROOT / "docs/icml_evidence_matrix.md").read_text()
    mechanism = json.loads(
        (PROJECT_ROOT / "mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_summary.json").read_text()
    )

    vanilla_related = _control_value(rows, "Vanilla", "macro_related_fpr")
    vanilla_mcc = _control_value(rows, "Vanilla", "macro_mcc")
    hybrid_related = _control_value(rows, "Hybrid gate+rescue", "macro_related_fpr")
    hybrid_mcc = _control_value(rows, "Hybrid gate+rescue", "macro_mcc")

    _assert_contains(
        intro,
        f"from ${_fmt(vanilla_related)}$ to ${_fmt(hybrid_related)}$",
        "intro:TDEV-related-FPR",
    )
    _assert_contains(
        intro,
        f"from ${_fmt(vanilla_mcc)}$ to ${_fmt(hybrid_mcc)}$",
        "intro:TDEV-MCC",
    )

    nolan_related = _control_value(rows, "NoLan-compatible", "macro_related_fpr")
    nolan_tpr = _control_value(rows, "NoLan-compatible", "macro_tpr")
    nolan_yes = _control_value(rows, "NoLan-compatible", "macro_yes_rate")
    nolan_fpr = _control_value(rows, "NoLan-compatible", "macro_fpr")
    vanilla_fpr = _control_value(rows, "Vanilla", "macro_fpr")
    vanilla_tpr = _control_value(rows, "Vanilla", "macro_tpr")
    vanilla_yes = _control_value(rows, "Vanilla", "macro_yes_rate")
    vanilla_adv_related = _control_value(rows, "Vanilla", "adversarial_related_fpr")
    nolan_adv_related = _control_value(rows, "NoLan-compatible", "adversarial_related_fpr")

    _assert_contains(
        associated,
        f"from ${_fmt_pct(vanilla_adv_related)}\\%$ to ${_fmt_pct(nolan_adv_related)}\\%$",
        "associated:NoLan-adversarial-related-FPR",
    )
    _assert_contains(
        associated,
        f"from ${_fmt_pct(vanilla_related)}\\%$ to ${_fmt_pct(nolan_related)}\\%$",
        "associated:NoLan-macro-related-FPR",
    )

    _assert_contains(
        evidence,
        f"NoLan-compatible lowers related FPR to `{_fmt(nolan_related)}` but drops TPR to `{_fmt(nolan_tpr)}` and yes rate to `{_fmt(nolan_yes)}`",
        "evidence:NoLan-tradeoff",
    )
    _assert_contains(
        evidence,
        f"neighbor evidence exceeds target evidence in `{mechanism['summary_rates']['neighbor_dominance_rate_vanilla_related_fp'] * 100:.1f}%` of vanilla related FPs",
        "evidence:mechanism-rate",
    )

    # Mitigation prose uses point deltas, so compare against the same macro deltas
    # rounded to one decimal percentage points.
    _assert_contains(
        mitigation,
        f"lowers FPR by ${_fmt_pct(vanilla_fpr - nolan_fpr)}$ points",
        "mitigation:NoLan-FPR-delta",
    )
    _assert_contains(
        mitigation,
        f"lowers TPR by ${_fmt_pct(vanilla_tpr - nolan_tpr)}$ points",
        "mitigation:NoLan-TPR-delta",
    )
    _assert_contains(
        mitigation,
        f"yes rate by ${_fmt_pct(vanilla_yes - nolan_yes)}$ points",
        "mitigation:NoLan-yes-rate-delta",
    )

    direct_mcc = _control_value(rows, "OWLv2 target direct", "macro_mcc")
    direct_related = _control_value(rows, "OWLv2 target direct", "macro_related_fpr")
    direct_adv_related = _control_value(rows, "OWLv2 target direct", "adversarial_related_fpr")
    margin_related = _control_value(rows, "OWLv2 margin direct", "macro_related_fpr")
    margin_tpr = _control_value(rows, "OWLv2 margin direct", "macro_tpr")
    gate_related = _control_value(rows, "Two-stage gate", "macro_related_fpr")
    gate_adv_related = _control_value(rows, "Two-stage gate", "adversarial_related_fpr")
    gate_mcc = _control_value(rows, "Two-stage gate", "macro_mcc")
    hybrid_tpr = _control_value(rows, "Hybrid gate+rescue", "macro_tpr")

    _assert_contains(associated, f"macro MCC ${_fmt(direct_mcc)}$", "associated:direct-target-MCC")
    _assert_contains(
        associated,
        f"high related-present FPR (${_fmt(direct_related)}$ macro, ${_fmt(direct_adv_related)}$ on adversarial",
        "associated:direct-target-related-FPR",
    )
    _assert_contains(
        associated,
        f"nearly eliminates related-present false positives (${_fmt(margin_related)}$ macro) but loses most recall (TPR ${_fmt(margin_tpr)}$)",
        "associated:margin-tradeoff",
    )
    _assert_contains(
        associated,
        f"reduces related-present FPR to ${_fmt(gate_related)}$ macro and ${_fmt(gate_adv_related)}$ on adversarial related-present negatives while keeping macro MCC ${_fmt(gate_mcc)}$",
        "associated:gate-tradeoff",
    )
    _assert_contains(
        associated,
        f"recovers recall to ${_fmt(hybrid_tpr)}$, and improves macro MCC to ${_fmt(hybrid_mcc)}$",
        "associated:hybrid-tradeoff",
    )


def check_appendix_qwen() -> None:
    audit = json.loads((PROJECT_ROOT / "mitigation/results/semantic_neighbor_audit/qwen25vl_replication_audit.json").read_text())
    table = (PAPER_ROOT / "tables/table_appendix_qwen.tex").read_text()
    mapping = {
        "Qwen2.5-VL & vanilla": audit["macro"]["vanilla"],
        "Qwen2.5-VL & fixed TDEV hybrid": audit["macro"]["fixed_tdev"],
    }
    for row_label, row in mapping.items():
        expected = [
            row["mcc"],
            row["recall_tpr"],
            row["fpr"],
            row["related_fpr"],
            row["plain_fpr"],
        ]
        actual = _numbers_from_row(table, row_label)
        _assert_rounded(actual, expected, 3, f"appendix-qwen:{row_label}")


def check_appendix_tdev_lite() -> None:
    rows = list(csv.DictReader((PROJECT_ROOT / "mitigation/results/pope_internal_external_ablation_full/pope_internal_external_ablation.csv").open()))
    table = (PAPER_ROOT / "tables/table_appendix_tdev_lite.tex").read_text()

    def find_row(method: str, score: str, selection_rate: str, subset: str) -> dict[str, str]:
        for row in rows:
            if (
                row["split"] == "macro"
                and row["method"] == method
                and row["score"] == score
                and row["selection_rate"] == selection_rate
                and row["subset"] == subset
            ):
                return row
        raise AssertionError(f"Missing TDEV-lite row: {(method, score, selection_rate, subset)}")

    mapping = {
        "full TDEV hybrid": ("full_tdev", "anchor", "1.0"),
        "LH-alone suppress": ("lh_alone_base_yes_suppress", "lh_shape_pope_layers_22_31", "0.5"),
        "LH-routed TDEV": ("lh_to_tdev_base_yes", "lh_shape_pope_layers_22_31", "0.5"),
        "prompt-position routing": ("lh_to_tdev_base_yes", "prompt_token_pos", "0.5"),
        "target-length routing": ("lh_to_tdev_base_yes", "target_char_len", "0.5"),
    }
    for row_label, (method, score, selection_rate) in mapping.items():
        row = find_row(method, score, selection_rate, "all")
        related = find_row(method, score, selection_rate, "negative_related_present")
        expected = [
            float(row["detector_calls"]),
            float(row["mcc"]),
            float(row["tpr"]),
            float(row["fpr"]),
            float(related["fpr"]),
        ]
        actual = _numbers_from_row(table, row_label)
        _assert_rounded(actual, expected, 3, f"appendix-tdev-lite:{row_label}")


def check_appendix_caption_proxy() -> None:
    rewrite = json.loads((PROJECT_ROOT / "detection/baselines/results/tdev_caption_rewrite_neutral/chair_metrics.json").read_text())
    generic = json.loads((PROJECT_ROOT / "detection/baselines/results/tdev_caption_rewrite_generic/chair_metrics.json").read_text())
    deletion = json.loads((PROJECT_ROOT / "detection/baselines/results/tdev_caption_edit_hybrid_mcc_top10/chair_metrics.json").read_text())
    table = (PAPER_ROOT / "tables/table_appendix_caption_proxy.tex").read_text()
    mapping = {
        "vanilla": rewrite["vanilla"],
        "neutral rewrite top-5": rewrite["rewritten"],
        "generic noun rewrite top-5": generic["rewritten"],
        "deletion top-10": deletion["edited"],
    }
    for row_label, row in mapping.items():
        expected = [
            row["chair"]["CHAIRi"],
            row["chair"]["CHAIRs"],
            row["caption_stats"]["mean_words"],
        ]
        actual = _numbers_from_row(table, row_label)
        rounded = [round(expected[0], 4), round(expected[1], 4), round(expected[2], 2)]
        if actual != rounded:
            raise AssertionError(f"appendix-caption:{row_label}: table={actual} expected={rounded}")


def check_caption_route_summary() -> None:
    acceptance = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
        ).read_text()
    )
    repair = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json"
        ).read_text()
    )
    concise = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
        ).read_text()
    )
    claim_repair = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_claim_repair/claim_repair_metrics.json"
        ).read_text()
    )
    scaled_acceptance = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
        ).read_text()
    )
    scaled_concise = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
        ).read_text()
    )
    scaled_claim_repair = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_claim_repair/claim_repair_metrics.json"
        ).read_text()
    )
    scaled_acceptance_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    scaled_claim_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    expanded_acceptance = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
        ).read_text()
    )
    expanded_concise = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
        ).read_text()
    )
    expanded_claim_repair = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_claim_repair/claim_repair_metrics.json"
        ).read_text()
    )
    expanded_acceptance_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    expanded_claim_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    broad_acceptance = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json"
        ).read_text()
    )
    broad_concise = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json"
        ).read_text()
    )
    broad_claim_repair = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair/claim_repair_metrics.json"
        ).read_text()
    )
    broad_acceptance_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    broad_claim_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    regen_concise = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_controlled_regen_100_t80/controlled_regeneration_metrics.json"
        ).read_text()
    )
    regen_concise_preservation = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_controlled_regen_100_t80_preservation/caption_variant_preservation_metrics.json"
        ).read_text()
    )
    regen_concise_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_controlled_regen_100_t80_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    regen_detail = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_metrics.json"
        ).read_text()
    )
    regen_detail_preservation = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_preservation/caption_variant_preservation_metrics.json"
        ).read_text()
    )
    regen_detail_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    oracle = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_candidate_pool_oracle_100/candidate_pool_oracle_metrics.json"
        ).read_text()
    )
    oracle_min_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_candidate_pool_oracle_100_min_hallucination_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    oracle_no_worse_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_candidate_pool_oracle_100_no_worse_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    verified_select = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_candidate_select_100/verified_candidate_selection_metrics.json"
        ).read_text()
    )
    verified_select_preservation = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_candidate_select_100_preservation/caption_variant_preservation_metrics.json"
        ).read_text()
    )
    verified_select_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_candidate_select_100_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    verified_select_r10 = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_candidate_select_100_r10/verified_candidate_selection_metrics.json"
        ).read_text()
    )
    verified_select_r10_preservation = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_preservation/caption_variant_preservation_metrics.json"
        ).read_text()
    )
    verified_select_r10_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    local_add = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_local_additions_100/verified_local_addition_metrics.json"
        ).read_text()
    )
    local_add_preservation = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_local_additions_100_preservation/caption_variant_preservation_metrics.json"
        ).read_text()
    )
    local_add_content = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_local_additions_100_content_light/caption_content_light_metrics.json"
        ).read_text()
    )
    local_add_o04 = json.loads(
        (
            PROJECT_ROOT
            / "detection/baselines/results/tdev_caption_verified_local_additions_100_o04/verified_local_addition_metrics.json"
        ).read_text()
    )
    summary = (PROJECT_ROOT / "docs/caption_method_route_summary.md").read_text()
    evidence = (PROJECT_ROOT / "docs/icml_evidence_matrix.md").read_text()
    baseline_note = (PROJECT_ROOT / "docs/baseline_availability_refresh.md").read_text()

    generated_rows = {
        "t96 hard gate": [
            acceptance["num_examples"],
            acceptance["chair"]["gated"]["overall"]["CHAIRi"],
            acceptance["chair"]["gated"]["total_hallucinated_mentions"],
            acceptance["mean_gated_words"],
            0.0,
        ],
        "sentence repair": [
            repair["num_examples"],
            repair["chair"]["repaired"]["overall"]["CHAIRi"],
            repair["chair"]["repaired"]["total_hallucinated_mentions"],
            repair["mean_repaired_words"],
            repair["mean_removed_words_by_repair"],
        ],
        "sentence acceptance": [
            acceptance["num_examples"],
            acceptance["chair"]["accepted"]["overall"]["CHAIRi"],
            acceptance["chair"]["accepted"]["total_hallucinated_mentions"],
            acceptance["mean_accepted_words"],
            acceptance["mean_removed_words_by_acceptance"],
        ],
        "claim-local repair": [
            claim_repair["num_examples"],
            claim_repair["chair"]["repaired"]["overall"]["CHAIRi"],
            claim_repair["chair"]["repaired"]["total_hallucinated_mentions"],
            claim_repair["mean_repaired_words"],
            claim_repair["mean_removed_words_by_repair"],
        ],
    }
    for row_label, expected in generated_rows.items():
        actual = _numbers_from_markdown_row(summary, row_label)
        rounded = [
            round(expected[0]),
            round(expected[1], 4),
            round(expected[2]),
            round(expected[3], 2),
            round(expected[4], 2),
        ]
        if actual != rounded:
            raise AssertionError(f"caption-route-generated:{row_label}: table={actual} expected={rounded}")

    scaled_concise_summary = scaled_concise["summary"]
    scaled_claim_summary = scaled_claim_repair["preservation"]["summary"]
    expanded_concise_summary = expanded_concise["summary"]
    expanded_claim_summary = expanded_claim_repair["preservation"]["summary"]
    broad_concise_summary = broad_concise["summary"]
    broad_claim_summary = broad_claim_repair["preservation"]["summary"]
    scaled_acceptance_content_summary = scaled_acceptance_content["summary"]
    scaled_claim_content_summary = scaled_claim_content["summary"]
    expanded_acceptance_content_summary = expanded_acceptance_content["summary"]
    expanded_claim_content_summary = expanded_claim_content["summary"]
    broad_acceptance_content_summary = broad_acceptance_content["summary"]
    broad_claim_content_summary = broad_claim_content["summary"]
    regen_concise_preservation_summary = regen_concise_preservation["summary"]
    regen_concise_content_summary = regen_concise_content["summary"]
    regen_detail_preservation_summary = regen_detail_preservation["summary"]
    regen_detail_content_summary = regen_detail_content["summary"]
    oracle_min_summary = oracle["summaries"]["min_hallucination"]
    oracle_no_worse_summary = oracle["summaries"]["no_worse_than_repair"]
    oracle_min_content_summary = oracle_min_content["summary"]
    oracle_no_worse_content_summary = oracle_no_worse_content["summary"]
    verified_select_preservation_summary = verified_select_preservation["summary"]
    verified_select_content_summary = verified_select_content["summary"]
    verified_select_r10_preservation_summary = verified_select_r10_preservation["summary"]
    verified_select_r10_content_summary = verified_select_r10_content["summary"]
    local_add_preservation_summary = local_add_preservation["summary"]
    local_add_content_summary = local_add_content["summary"]

    _assert_contains(
        summary,
        f"| 20-image | t96 hard gate | {scaled_acceptance['num_examples']} | {scaled_acceptance['chair']['gated']['overall']['CHAIRi']:.4f} | {scaled_acceptance['chair']['gated']['total_hallucinated_mentions']} | {scaled_acceptance['mean_gated_words']:.2f} | -- | -- | -- | high-risk generated baseline |",
        "caption-route:scaled-20-gated-row",
    )
    _assert_contains(
        summary,
        f"| 20-image | sentence acceptance | {scaled_acceptance['num_examples']} | {scaled_acceptance['chair']['accepted']['overall']['CHAIRi']:.4f} | {scaled_acceptance['chair']['accepted']['total_hallucinated_mentions']} | {scaled_acceptance['mean_accepted_words']:.2f} | {scaled_concise_summary['accepted_retained_vanilla_grounded_rate'] * 100:.2f}% | {scaled_concise_summary['accepted_object_mention_retention_vs_gated'] * 100:.2f}% | {scaled_acceptance_content_summary['chair_objectless']} / {scaled_acceptance_content_summary['content_light']} | reduces hallucination but can delete too much |",
        "caption-route:scaled-20-acceptance-row",
    )
    _assert_contains(
        summary,
        f"| 20-image | claim-local repair | {scaled_claim_repair['num_examples']} | {scaled_claim_repair['chair']['repaired']['overall']['CHAIRi']:.4f} | {scaled_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {scaled_claim_repair['mean_repaired_words']:.2f} | {scaled_claim_summary['repaired_retained_vanilla_grounded_rate'] * 100:.2f}% | {scaled_claim_summary['repaired_object_mention_retention_vs_gated'] * 100:.2f}% | {scaled_claim_content_summary['chair_objectless']} / {scaled_claim_content_summary['content_light']} | preserves more supported content with the same hallucinated mention count |",
        "caption-route:scaled-20-claim-repair-row",
    )
    _assert_contains(
        summary,
        f"| 40-image | t96 hard gate | {expanded_acceptance['num_examples']} | {expanded_acceptance['chair']['gated']['overall']['CHAIRi']:.4f} | {expanded_acceptance['chair']['gated']['total_hallucinated_mentions']} | {expanded_acceptance['mean_gated_words']:.2f} | -- | -- | -- | maximum available iter2-prefilter set |",
        "caption-route:scaled-40-gated-row",
    )
    _assert_contains(
        summary,
        f"| 40-image | sentence acceptance | {expanded_acceptance['num_examples']} | {expanded_acceptance['chair']['accepted']['overall']['CHAIRi']:.4f} | {expanded_acceptance['chair']['accepted']['total_hallucinated_mentions']} | {expanded_acceptance['mean_accepted_words']:.2f} | {expanded_concise_summary['accepted_retained_vanilla_grounded_rate'] * 100:.2f}% | {expanded_concise_summary['accepted_object_mention_retention_vs_gated'] * 100:.2f}% | {expanded_acceptance_content_summary['chair_objectless']} / {expanded_acceptance_content_summary['content_light']} | same hallucination count as repair but less content retention |",
        "caption-route:scaled-40-acceptance-row",
    )
    _assert_contains(
        summary,
        f"| 40-image | claim-local repair | {expanded_claim_repair['num_examples']} | {expanded_claim_repair['chair']['repaired']['overall']['CHAIRi']:.4f} | {expanded_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {expanded_claim_repair['mean_repaired_words']:.2f} | {expanded_claim_summary['repaired_retained_vanilla_grounded_rate'] * 100:.2f}% | {expanded_claim_summary['repaired_object_mention_retention_vs_gated'] * 100:.2f}% | {expanded_claim_content_summary['chair_objectless']} / {expanded_claim_content_summary['content_light']} | best larger-scale prototype tradeoff |",
        "caption-route:scaled-40-claim-repair-row",
    )

    _assert_contains(
        summary,
        f"| 100-image | t96 hard gate | {broad_acceptance['num_examples']} | {broad_acceptance['chair']['gated']['overall']['CHAIRi']:.4f} | {broad_acceptance['chair']['gated']['total_hallucinated_mentions']} | {broad_acceptance['mean_gated_words']:.2f} | -- | -- | -- | broader high-risk generated baseline |",
        "caption-route:scaled-100-gated-row",
    )
    _assert_contains(
        summary,
        f"| 100-image | sentence acceptance | {broad_acceptance['num_examples']} | {broad_acceptance['chair']['accepted']['overall']['CHAIRi']:.4f} | {broad_acceptance['chair']['accepted']['total_hallucinated_mentions']} | {broad_acceptance['mean_accepted_words']:.2f} | {broad_concise_summary['accepted_retained_vanilla_grounded_rate'] * 100:.2f}% | {broad_concise_summary['accepted_object_mention_retention_vs_gated'] * 100:.2f}% | {broad_acceptance_content_summary['chair_objectless']} / {broad_acceptance_content_summary['content_light']} | strong hallucination reduction but visible-content loss remains |",
        "caption-route:scaled-100-acceptance-row",
    )
    _assert_contains(
        summary,
        f"| 100-image | claim-local repair | {broad_claim_repair['num_examples']} | {broad_claim_repair['chair']['repaired']['overall']['CHAIRi']:.4f} | {broad_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {broad_claim_repair['mean_repaired_words']:.2f} | {broad_claim_summary['repaired_retained_vanilla_grounded_rate'] * 100:.2f}% | {broad_claim_summary['repaired_object_mention_retention_vs_gated'] * 100:.2f}% | {broad_claim_content_summary['chair_objectless']} / {broad_claim_content_summary['content_light']} | slight CHAIR/content gain over acceptance; content-light cases are removed |",
        "caption-route:scaled-100-claim-repair-row",
    )

    _assert_contains(
        summary,
        f"| claim-local repair | {broad_claim_repair['num_examples']} | {broad_claim_repair['chair']['repaired']['overall']['CHAIRi']:.4f} | {broad_claim_repair['chair']['repaired']['total_hallucinated_mentions']} | {broad_claim_repair['mean_repaired_words']:.2f} | {broad_claim_summary['repaired_retained_vanilla_grounded_rate'] * 100:.2f}% | {broad_claim_summary['repaired_object_mention_retention_vs_vanilla'] * 100:.2f}% | {broad_claim_content_summary['content_light']} | current best deterministic fallback |",
        "caption-route:regen-probe-repair-row",
    )
    _assert_contains(
        summary,
        f"| controlled regen concise | {regen_concise['num_examples']} | {regen_concise['chair']['regenerated']['overall']['CHAIRi']:.4f} | {regen_concise['chair']['regenerated']['total_hallucinated_mentions']} | {regen_concise['mean_words']['regenerated']:.2f} | {regen_concise_preservation_summary['variant_retained_vanilla_grounded_rate'] * 100:.2f}% | {regen_concise_preservation_summary['variant_object_mention_retention_vs_vanilla'] * 100:.2f}% | {regen_concise_content_summary['content_light']} | lowers CHAIR by over-compressing content |",
        "caption-route:regen-probe-concise-row",
    )
    _assert_contains(
        summary,
        f"| controlled regen detail | {regen_detail['num_examples']} | {regen_detail['chair']['regenerated']['overall']['CHAIRi']:.4f} | {regen_detail['chair']['regenerated']['total_hallucinated_mentions']} | {regen_detail['mean_words']['regenerated']:.2f} | {regen_detail_preservation_summary['variant_retained_vanilla_grounded_rate'] * 100:.2f}% | {regen_detail_preservation_summary['variant_object_mention_retention_vs_vanilla'] * 100:.2f}% | {regen_detail_content_summary['content_light']} | recovers length but not the repair tradeoff |",
        "caption-route:regen-probe-detail-row",
    )
    _assert_contains(
        summary,
        f"| oracle min hallucination | {oracle_min_summary['num_images']} | {oracle_min_summary['chairi']:.4f} | {oracle_min_summary['total_hallucinated_mentions']} | {oracle_min_summary['mean_words']:.2f} | {oracle_min_summary['retained_vanilla_grounded_rate'] * 100:.2f}% | {oracle_min_summary['object_mention_retention_vs_vanilla'] * 100:.2f}% | {oracle_min_content_summary['content_light']} | repair {oracle_min_summary['selected_candidate_counts'].get('claim-local repair', 0)}, concise {oracle_min_summary['selected_candidate_counts'].get('controlled regen concise', 0)}, detail {oracle_min_summary['selected_candidate_counts'].get('controlled regen detail', 0)} | best possible hallucination control still costs visible content |",
        "caption-route:oracle-min-hallucination-row",
    )
    _assert_contains(
        summary,
        f"| oracle no-worse-than-repair | {oracle_no_worse_summary['num_images']} | {oracle_no_worse_summary['chairi']:.4f} | {oracle_no_worse_summary['total_hallucinated_mentions']} | {oracle_no_worse_summary['mean_words']:.2f} | {oracle_no_worse_summary['retained_vanilla_grounded_rate'] * 100:.2f}% | {oracle_no_worse_summary['object_mention_retention_vs_vanilla'] * 100:.2f}% | {oracle_no_worse_content_summary['content_light']} | repair {oracle_no_worse_summary['selected_candidate_counts'].get('claim-local repair', 0)}, concise {oracle_no_worse_summary['selected_candidate_counts'].get('controlled regen concise', 0)}, detail {oracle_no_worse_summary['selected_candidate_counts'].get('controlled regen detail', 0)} | small upper-bound gain: detail candidates help 20 images but do not change the conclusion |",
        "caption-route:oracle-no-worse-row",
    )
    _assert_contains(
        summary,
        "the gain is too small to justify a selector-only paper claim",
        "caption-route:oracle-small-gain",
    )
    _assert_contains(
        summary,
        f"| verified selector r0.75 | {verified_select['num_examples']} | {verified_select['selected_detail_regen']} | {verified_select['chair']['selected']['overall']['CHAIRi']:.4f} | {verified_select['chair']['selected']['total_hallucinated_mentions']} | {verified_select['mean_words']['selected']:.2f} | {verified_select_preservation_summary['variant_retained_vanilla_grounded_rate'] * 100:.2f}% | {verified_select_preservation_summary['variant_object_mention_retention_vs_vanilla'] * 100:.2f}% | {verified_select_content_summary['content_light']} | verifier admits too many compressed candidates; worse than repair |",
        "caption-route:verified-selector-r075-row",
    )
    _assert_contains(
        summary,
        f"| verified selector r1.00 | {verified_select_r10['num_examples']} | {verified_select_r10['selected_detail_regen']} | {verified_select_r10['chair']['selected']['overall']['CHAIRi']:.4f} | {verified_select_r10['chair']['selected']['total_hallucinated_mentions']} | {verified_select_r10['mean_words']['selected']:.2f} | {verified_select_r10_preservation_summary['variant_retained_vanilla_grounded_rate'] * 100:.2f}% | {verified_select_r10_preservation_summary['variant_object_mention_retention_vs_vanilla'] * 100:.2f}% | {verified_select_r10_content_summary['content_light']} | stricter guard collapses to repair-level behavior |",
        "caption-route:verified-selector-r100-row",
    )
    _assert_contains(
        summary,
        "The bottleneck is not just selection; the candidate generator must produce claim-local additions",
        "caption-route:verified-selector-conclusion",
    )
    _assert_contains(
        summary,
        f"| local additions o0.55 | {local_add['num_examples']} | {local_add['images_with_additions']} | {local_add['total_added_sentences']} | {local_add['chair']['selected']['overall']['CHAIRi']:.4f} | {local_add['chair']['selected']['total_hallucinated_mentions']} | {local_add['mean_words']['selected']:.2f} | {local_add_preservation_summary['variant_retained_vanilla_grounded_rate'] * 100:.2f}% | {local_add_preservation_summary['variant_object_mention_retention_vs_vanilla'] * 100:.2f}% | {local_add_content_summary['content_light']} | appends mostly paraphrases; adds one hallucinated mention |",
        "caption-route:local-add-o055-row",
    )
    _assert_contains(
        summary,
        f"| local additions o0.40 | {local_add_o04['num_examples']} | {local_add_o04['images_with_additions']} | {local_add_o04['total_added_sentences']} | {local_add_o04['chair']['selected']['overall']['CHAIRi']:.4f} | {local_add_o04['chair']['selected']['total_hallucinated_mentions']} | {local_add_o04['mean_words']['selected']:.2f} | {broad_claim_summary['repaired_retained_vanilla_grounded_rate'] * 100:.2f}% | {broad_claim_summary['repaired_object_mention_retention_vs_vanilla'] * 100:.2f}% | {broad_claim_content_summary['content_light']} | stricter overlap rejects all additions and returns to repair |",
        "caption-route:local-add-o040-row",
    )
    _assert_contains(
        summary,
        "The next generator must be explicitly trained or prompted to propose atomic missing-detail claims/spans",
        "caption-route:local-add-conclusion",
    )

    concise_summary = concise["summary"]
    concise_rows = {
        "retained vanilla grounded mentions": concise_summary["accepted_retained_vanilla_grounded_rate"] * 100,
        "hallucination reduction vs gated": concise_summary["accepted_hallucination_reduction_vs_gated"] * 100,
        "hallucination reduction vs vanilla": concise_summary["accepted_hallucination_reduction_vs_vanilla"] * 100,
        "object mention retention vs gated": concise_summary["accepted_object_mention_retention_vs_gated"] * 100,
        "generic/empty accepted captions": concise_summary["generic_or_empty_accepted"],
    }
    for row_label, expected in concise_rows.items():
        actual = _numbers_from_markdown_row(summary, row_label)
        expected_value = round(expected, 2)
        if actual != [expected_value]:
            raise AssertionError(f"caption-route-concise:{row_label}: table={actual} expected={[expected_value]}")

    _assert_contains(
        summary,
        "Its length reduction is not inherently bad: concise captions are preferable to long captions that keep inventing objects;",
        "caption-route:length-reduction-nuance",
    )
    _assert_contains(
        summary,
        "The old CHAIR-objectless counts",
        "caption-route:generic-risk",
    )
    claim_summary = claim_repair["preservation"]["summary"]
    _assert_contains(
        summary,
        "the remaining method gap is verification-in-loop candidate generation",
        "caption-route:scale-risk",
    )
    _assert_contains(
        summary,
        "propose atomic missing-detail spans",
        "caption-route:atomic-span-direction",
    )
    _assert_contains(
        summary,
        f"| retained vanilla grounded mentions | {claim_summary['repaired_retained_vanilla_grounded_rate'] * 100:.2f}% | improves content retention over sentence acceptance |",
        "caption-route:claim-repair-retention",
    )
    _assert_contains(
        summary,
        f"| sentence acceptance | {acceptance['num_examples']} | {acceptance['chair']['accepted']['overall']['CHAIRi']:.4f} | {acceptance['chair']['accepted']['total_hallucinated_mentions']} | {acceptance['mean_accepted_words']:.2f} | {acceptance['mean_removed_words_by_acceptance']:.2f} | strong hallucination reduction, but drops complete mixed sentences |",
        "caption-route:five-image-acceptance-row",
    )
    _assert_contains(
        summary,
        f"| retained vanilla grounded mentions | {concise_summary['accepted_retained_vanilla_grounded_rate'] * 100:.2f}% | accepted captions keep most supported object mentions |",
        "caption-route:accepted-concise-faithfulness",
    )
    _assert_contains(
        evidence,
        f"concise lowers CHAIRi to `{regen_concise['chair']['regenerated']['overall']['CHAIRi']:.4f}` by over-compressing",
        "evidence:caption-regen-concise-negative",
    )
    _assert_contains(
        evidence,
        f"detail reaches\n   `{regen_detail['mean_words']['regenerated']:.2f}` words but worsens CHAIRi to `{regen_detail['chair']['regenerated']['overall']['CHAIRi']:.4f}` and retains only `{regen_detail_preservation_summary['variant_retained_vanilla_grounded_rate'] * 100:.2f}%`",
        "evidence:caption-regen-detail-negative",
    )
    _assert_contains(
        evidence,
        f"no-worse-than-repair selection keeps\n   {oracle_no_worse_summary['total_hallucinated_mentions']} hallucinated mentions and only\n   raises retained vanilla grounded mentions to `{oracle_no_worse_summary['retained_vanilla_grounded_rate'] * 100:.2f}%`",
        "evidence:caption-oracle-small-gain",
    )
    _assert_contains(
        baseline_note,
        '"use a detector to revise captions" in general',
        "baseline-note:caption-boundary",
    )
    _assert_contains(
        baseline_note,
        "Shorter captions are acceptable when they stop unsupported object",
        "baseline-note:caption-short-ok",
    )
    _assert_contains(
        baseline_note,
        "content preservation, not length by",
        "baseline-note:caption-content-risk",
    )


def main() -> None:
    check_detection_main()
    check_strong_controls()
    check_mitigation_behavior()
    check_attention_audit()
    check_semantic_neighbor_fpr()
    check_region_verifier_detection()
    check_region_verifier_pope()
    check_semantic_neighbor_control_table()
    check_key_prose_claims()
    check_appendix_qwen()
    check_appendix_tdev_lite()
    check_appendix_caption_proxy()
    check_caption_route_summary()
    print("All audited paper numbers match current result artifacts.")


if __name__ == "__main__":
    main()
