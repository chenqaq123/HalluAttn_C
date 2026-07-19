#!/usr/bin/env python3
"""Analyze Qwen3-VL XEV JSONL shards without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mitigation.src.xev import (  # noqa: E402
    identity_switch_certificate,
    pairwise_identity_certificate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--matched_tpr", type=float, default=0.80)
    parser.add_argument(
        "--pairwise_evaluation",
        choices=("development", "confirmatory"),
        default="development",
    )
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict]:
    rows = []
    for value in paths:
        with Path(value).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    by_id = {str(row["question_id"]): row for row in rows}
    return [by_id[key] for key in sorted(by_id, key=lambda value: int(value))]


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = 1.0 - fpr
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": int(len(labels)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float((tp + tn) / len(labels)) if len(labels) else 0.0,
        "balanced_accuracy": float((tpr + tnr) / 2),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator else 0.0,
        "yes_rate": float((tp + fp) / len(labels)) if len(labels) else 0.0,
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "mean_yes": float(scores[labels == 1].mean()),
        "mean_no": float(scores[labels == 0].mean()),
    }


def threshold_for_tpr(
    labels: np.ndarray, scores: np.ndarray, target_tpr: float
) -> float:
    positive_scores = np.sort(scores[labels == 1])[::-1]
    if not len(positive_scores):
        return math.inf
    rank = min(
        len(positive_scores) - 1,
        max(0, math.ceil(target_tpr * len(positive_scores)) - 1),
    )
    return float(positive_scores[rank])


def fpr_at_matched_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    related_mask: np.ndarray,
    target_tpr: float,
) -> dict:
    threshold = threshold_for_tpr(labels, scores, target_tpr)
    predictions = scores >= threshold
    negatives = labels == 0
    related_negatives = negatives & related_mask
    return {
        "target_tpr": target_tpr,
        "threshold": threshold,
        "achieved_tpr": float(predictions[labels == 1].mean())
        if np.any(labels == 1)
        else None,
        "all_negative_fpr": float(predictions[negatives].mean())
        if np.any(negatives)
        else None,
        "related_present_count": int(related_negatives.sum()),
        "related_present_fpr": (
            float(predictions[related_negatives].mean())
            if np.any(related_negatives)
            else None
        ),
    }


def tune_lambda(
    labels: np.ndarray,
    baseline: np.ndarray,
    method_score: np.ndarray,
    dev_mask: np.ndarray,
) -> float:
    candidates = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    best = (float("-inf"), float("-inf"), 0.0)
    for value in candidates:
        predictions = (baseline[dev_mask] + value * method_score[dev_mask]) >= 0
        metrics = binary_metrics(labels[dev_mask], predictions)
        key = (metrics["balanced_accuracy"], metrics["mcc"], -value)
        if key > best:
            best = key
    return float(-best[2])


def candidate_diagnostics(rows: list[dict]) -> dict:
    open_counts = []
    label_counts: Counter[str] = Counter()
    relation_calls = relation_failures = 0
    related_rows = []
    source_coverage = {"closed": [], "open": []}
    source_dominance = {"closed": [], "open": []}
    open_target_exact = {"yes": [], "no": []}
    for row in rows:
        for region in row["regions"]:
            open_counts.append(len(region["candidate_sources"].get("open", [])))
            label_counts.update(region.get("relations", {}).values())
            metadata = region.get("relation_filter", {})
            if metadata.get("classified", 0):
                relation_calls += 1
                relation_failures += int(not metadata.get("parsed_ok", False))
            open_target_exact[row["label"]].append(
                row["target"] in region["candidate_sources"].get("open", [])
            )
            if row["label"] == "no" and row.get("oracle_related_present", False):
                oracle_alternatives = {
                    candidate
                    for candidate in region["candidate_sources"].get("oracle", [])
                    if region["relations"].get(candidate) == "C"
                }
                target_keep = float(region["margins"][row["target"]]["keep"])
                for source in source_coverage:
                    source_candidates = set(region["candidate_sources"].get(source, []))
                    source_coverage[source].append(
                        bool(source_candidates & oracle_alternatives)
                    )
                    alternative_margins = [
                        float(region["margins"][candidate]["keep"])
                        for candidate in source_candidates
                        if region["relations"].get(candidate) == "C"
                    ]
                    source_dominance[source].append(
                        bool(alternative_margins)
                        and max(alternative_margins) > target_keep
                    )
                related_rows.append(row["question_id"])
    return {
        "mean_open_candidates": float(np.mean(open_counts)) if open_counts else 0.0,
        "relation_labels": dict(label_counts),
        "relation_calls": relation_calls,
        "relation_parse_failures": relation_failures,
        "related_negative_count": len(related_rows),
        "related_alternative_coverage": {
            source: float(np.mean(values)) if values else None
            for source, values in source_coverage.items()
        },
        "related_competitor_dominance_rate": {
            source: float(np.mean(values)) if values else None
            for source, values in source_dominance.items()
        },
        "open_exact_target_rate": {
            label: float(np.mean(values)) if values else None
            for label, values in open_target_exact.items()
        },
    }


def certificate_diagnostics(
    rows: list[dict], labels: np.ndarray, baseline: np.ndarray
) -> dict:
    certificates = []
    trigger_mask = np.zeros(len(rows), dtype=bool)
    for row_index, row in enumerate(rows):
        certificate = row.get("identity_switch_certificate")
        if certificate is None:
            region_certificates = []
            for region in row["regions"]:
                keep_margins = {
                    candidate: float(values["keep"])
                    for candidate, values in region["margins"].items()
                }
                derived = identity_switch_certificate(
                    target=row["target"],
                    open_candidates=region["candidate_sources"].get("open", []),
                    closed_candidates=region["candidate_sources"].get("closed", []),
                    relations=region["relations"],
                    keep_margins=keep_margins,
                    baseline_margin=float(row["baseline_margin"]),
                )
                if derived is not None:
                    region_certificates.append(derived)
            if region_certificates:
                certificate = max(region_certificates, key=lambda item: item["excess"])
        if certificate is None:
            continue
        trigger_mask[row_index] = True
        certificates.append(
            {
                "question_id": row["question_id"],
                "image_id": row["image_id"],
                "target": row["target"],
                "label": row["label"],
                **certificate,
            }
        )

    baseline_predictions = baseline >= 0
    veto_predictions = baseline_predictions.copy()
    veto_predictions[trigger_mask] = False
    baseline_behavior = binary_metrics(labels, baseline_predictions)
    veto_behavior = binary_metrics(labels, veto_predictions)
    false_positive_corrections = int(np.sum(trigger_mask & (labels == 0)))
    true_positive_vetoes = int(np.sum(trigger_mask & (labels == 1)))
    trigger_count = int(trigger_mask.sum())
    correction_precision = (
        false_positive_corrections / trigger_count if trigger_count else None
    )
    usable = (
        trigger_count >= 3
        and correction_precision >= 2 / 3
        and veto_behavior["fpr"] < baseline_behavior["fpr"]
        and veto_behavior["tpr"] >= baseline_behavior["tpr"] - 0.005
        and veto_behavior["mcc"] >= baseline_behavior["mcc"]
    )
    return {
        "definition": {
            "baseline_prediction": "yes",
            "candidate_membership": "exact intersection(open, closed)",
            "semantic_relation": "C",
            "target_exactly_in_open": False,
            "alternative_margin_min": 0.0,
            "alternative_minus_target_min": 6.0,
            "action": "veto yes to no",
        },
        "trigger_count": trigger_count,
        "false_positive_corrections": false_positive_corrections,
        "true_positive_vetoes": true_positive_vetoes,
        "correction_precision": correction_precision,
        "baseline_behavior": baseline_behavior,
        "certificate_behavior": veto_behavior,
        "holdout_gate": {
            "minimum_triggers": 3,
            "minimum_correction_precision": 2 / 3,
            "maximum_tpr_drop": 0.005,
            "require_fpr_drop": True,
            "require_non_decreasing_mcc": True,
            "passed": usable,
        },
        "triggers": certificates,
    }


def pairwise_diagnostics(
    rows: list[dict],
    labels: np.ndarray,
    baseline: np.ndarray,
    evaluation: str,
) -> dict | None:
    preferences = np.full(len(rows), np.nan, dtype=np.float64)
    best_comparisons: list[dict | None] = [None] * len(rows)
    certificate_mask = np.zeros(len(rows), dtype=bool)
    certificates = []
    comparison_count = 0
    for index, row in enumerate(rows):
        payload = row.get("pairwise_verification", {})
        comparisons = payload.get("comparisons", [])
        comparison_count += len(comparisons)
        if not comparisons:
            continue
        best = min(comparisons, key=lambda item: item["target_preference"])
        preferences[index] = float(best["target_preference"])
        best_comparisons[index] = best
        certificate = payload.get("certificate")
        if certificate is None:
            region_certificates = []
            for region in row["regions"]:
                region_payload = region.get("pairwise_verification", {})
                derived = pairwise_identity_certificate(
                    baseline_margin=float(row["baseline_margin"]),
                    target_keep_margin=float(
                        region["margins"][row["target"]]["keep"]
                    ),
                    comparisons=region_payload.get("comparisons", []),
                )
                if derived is not None:
                    region_certificates.append(
                        {"region_index": region["index"], **derived}
                    )
            if region_certificates:
                certificate = min(
                    region_certificates,
                    key=lambda item: item["target_preference"],
                )
        if certificate is not None:
            certificate_mask[index] = True
            certificates.append(
                {
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "target": row["target"],
                    "label": row["label"],
                    **certificate,
                }
            )
    if comparison_count == 0:
        return None

    baseline_predictions = baseline >= 0
    baseline_behavior = binary_metrics(labels, baseline_predictions)

    def evaluate(threshold: float) -> dict:
        trigger_mask = np.isfinite(preferences) & (preferences < threshold)
        predictions = baseline_predictions.copy()
        predictions[trigger_mask] = False
        corrections = int(np.sum(trigger_mask & (labels == 0)))
        positive_vetoes = int(np.sum(trigger_mask & (labels == 1)))
        count = int(trigger_mask.sum())
        return {
            "threshold": threshold,
            "trigger_count": count,
            "false_positive_corrections": corrections,
            "true_positive_vetoes": positive_vetoes,
            "correction_precision": corrections / count if count else None,
            "behavior": binary_metrics(labels, predictions),
        }

    natural = evaluate(0.0)
    certificate_predictions = baseline_predictions.copy()
    certificate_predictions[certificate_mask] = False
    certificate_corrections = int(np.sum(certificate_mask & (labels == 0)))
    certificate_positive_vetoes = int(np.sum(certificate_mask & (labels == 1)))
    certificate_count = int(certificate_mask.sum())
    certificate_behavior = binary_metrics(labels, certificate_predictions)
    correction_precision = (
        certificate_corrections / certificate_count if certificate_count else None
    )
    passed = (
        certificate_count >= 3
        and correction_precision >= 2 / 3
        and certificate_behavior["fpr"] < baseline_behavior["fpr"]
        and certificate_behavior["tpr"] >= baseline_behavior["tpr"] - 0.005
        and certificate_behavior["mcc"] >= baseline_behavior["mcc"]
    )
    examples = []
    for index in np.where(np.isfinite(preferences) & (preferences < 0))[0]:
        comparison = best_comparisons[int(index)]
        examples.append(
            {
                "question_id": rows[index]["question_id"],
                "image_id": rows[index]["image_id"],
                "target": rows[index]["target"],
                "label": rows[index]["label"],
                **comparison,
            }
        )
    return {
        "status": evaluation,
        "definition": {
            "candidate_sources": ["open", "closed"],
            "target_named_in_open_guard": True,
            "candidate_keep_margin_min": 0.0,
            "candidate_limit": max(
                int(row.get("pairwise_verification", {}).get("candidate_limit", 0))
                for row in rows
            ),
            "target_preference": "0.5 * (target-first margin - alternative-first margin)",
            "natural_veto": "target_preference < 0",
        },
        "scored_rows": int(np.isfinite(preferences).sum()),
        "comparison_count": comparison_count,
        "baseline_behavior": baseline_behavior,
        "natural_zero": natural,
        "identity_certificate": {
            "definition": {
                "baseline_prediction": "yes",
                "target_keep_margin": "< 0",
                "candidate_relation": "C",
                "counterbalanced_target_preference": "< 0",
                "all_thresholds": 0.0,
                "action": "veto yes to no",
            },
            "trigger_count": certificate_count,
            "false_positive_corrections": certificate_corrections,
            "true_positive_vetoes": certificate_positive_vetoes,
            "correction_precision": correction_precision,
            "behavior": certificate_behavior,
            "holdout_gate": {
                "minimum_triggers": 3,
                "minimum_correction_precision": 2 / 3,
                "maximum_tpr_drop": 0.005,
                "require_fpr_drop": True,
                "require_non_decreasing_mcc": True,
                "passed": passed,
            },
            "triggers": certificates,
        },
        "threshold_sweep": [
            evaluate(threshold) for threshold in (-8.0, -6.0, -4.0, -2.0, 0.0)
        ],
        "natural_zero_triggers": examples,
    }


def global_support_diagnostics(
    rows: list[dict],
    labels: np.ndarray,
    baseline: np.ndarray,
    evaluation: str,
) -> dict | None:
    local_mask = np.zeros(len(rows), dtype=bool)
    inventory_support_mask = np.zeros(len(rows), dtype=bool)
    certificate_mask = np.zeros(len(rows), dtype=bool)
    examples = []
    inventories = []
    for index, row in enumerate(rows):
        payload = row.get("global_support_verification")
        if not payload or not payload.get("enabled", False):
            continue
        proposals = payload.get("local_identity_proposals", [])
        if not proposals:
            continue
        local_mask[index] = True
        inventory_support_mask[index] = bool(
            payload.get("target_support_found", False)
        )
        certificate = payload.get("certificate")
        if certificate is not None:
            certificate_mask[index] = True
        inventories.append(len(payload.get("inventory", [])))
        examples.append(
            {
                "question_id": row["question_id"],
                "image_id": row["image_id"],
                "target": row["target"],
                "label": row["label"],
                "target_support_found": bool(
                    payload.get("target_support_found", False)
                ),
                "inventory": payload.get("inventory", []),
                "inventories": payload.get(
                    "inventories", [payload.get("inventory", [])]
                ),
                "support_votes": payload.get(
                    "support_votes", [payload.get("target_support_found", False)]
                ),
                "support_relations": {
                    candidate: relation
                    for candidate, relation in payload.get("relations", {}).items()
                    if relation == "S"
                },
                "local_identity_proposals": proposals,
                "certificate": certificate,
            }
        )
    if not np.any(local_mask):
        return None

    baseline_predictions = baseline >= 0
    predictions = baseline_predictions.copy()
    predictions[certificate_mask] = False
    corrections = int(np.sum(certificate_mask & (labels == 0)))
    positive_vetoes = int(np.sum(certificate_mask & (labels == 1)))
    count = int(certificate_mask.sum())
    baseline_behavior = binary_metrics(labels, baseline_predictions)
    behavior = binary_metrics(labels, predictions)
    correction_precision = corrections / count if count else None
    passed = (
        count >= 3
        and correction_precision >= 2 / 3
        and behavior["fpr"] < baseline_behavior["fpr"]
        and behavior["tpr"] >= baseline_behavior["tpr"] - 0.005
        and behavior["mcc"] >= baseline_behavior["mcc"]
    )
    return {
        "status": evaluation,
        "definition": {
            "local_proposal": {
                "candidate_agreement": sorted(
                    {
                        row.get("global_support_verification", {}).get(
                            "local_candidate_agreement",
                            row.get("run", {}).get(
                                "local_candidate_agreement", "both"
                            ),
                        )
                        for row in rows
                        if row.get("global_support_verification", {}).get(
                            "enabled", False
                        )
                    }
                ),
                "conditions": "relation C and pairwise preference < 0",
            },
            "global_inventory": "two target-free full-image object lists",
            "support": "all inventories contain a lexical target equivalent or relation S",
            "action": "veto local proposal only when global target support is absent",
        },
        "local_proposal_count": int(local_mask.sum()),
        "inventory_support_count": int(np.sum(local_mask & inventory_support_mask)),
        "mean_inventory_size": float(np.mean(inventories)),
        "trigger_count": count,
        "false_positive_corrections": corrections,
        "true_positive_vetoes": positive_vetoes,
        "correction_precision": correction_precision,
        "baseline_behavior": baseline_behavior,
        "behavior": behavior,
        "holdout_gate": {
            "minimum_triggers": 3,
            "minimum_correction_precision": 2 / 3,
            "maximum_tpr_drop": 0.005,
            "require_fpr_drop": True,
            "require_non_decreasing_mcc": True,
            "passed": passed,
        },
        "examples": examples,
    }


def region_reformulation_scores(
    rows: list[dict], source: str | None = None
) -> dict[str, np.ndarray]:
    """Compute diagnostic scores from saved margins; these are post-hoc, not headline results."""

    keep_target = []
    remove_target = []
    direct_relative = []
    dominance_gated = []
    competitor_excess = []
    for row in rows:
        region = row["regions"][0]
        target = row["target"]
        target_keep = float(region["margins"][target]["keep"])
        keep_target.append(target_keep)
        remove_target.append(
            float(row["baseline_margin"] - region["margins"][target]["drop"])
        )
        if source is None:
            continue

        candidates = region["candidate_sources"].get(source, [])
        relations = region["relations"]
        support_margins = (
            [
                float(region["margins"][candidate]["keep"])
                for candidate in candidates
                if relations.get(candidate) == "S"
            ]
            if source == "open"
            else []
        )
        support = max([target_keep, *support_margins])
        alternative_margins = [
            float(region["margins"][candidate]["keep"])
            for candidate in candidates
            if relations.get(candidate) == "C"
        ]
        alternative = max(alternative_margins, default=float("-inf"))
        excess = max(0.0, alternative - support)
        direct_relative.append(support - max(0.0, alternative))
        dominance_gated.append(support - excess)
        competitor_excess.append(excess)

    output = {
        "keep_target": np.asarray(keep_target),
        "remove_target": np.asarray(remove_target),
    }
    if source is not None:
        output.update(
            {
                "direct_relative": np.asarray(direct_relative),
                "dominance_gated": np.asarray(dominance_gated),
                "competitor_excess": np.asarray(competitor_excess),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    rows = read_rows(args.inputs)
    if not rows:
        raise ValueError("No result rows found")
    labels = np.array([row["label"] == "yes" for row in rows], dtype=np.int64)
    baseline = np.array([row["baseline_margin"] for row in rows], dtype=np.float64)
    related = np.array(
        [row.get("oracle_related_present", False) for row in rows], dtype=bool
    )
    methods = sorted(set.intersection(*(set(row["scores"]) for row in rows)))
    scores = {
        method: np.array([row["scores"][method] for row in rows], dtype=np.float64)
        for method in methods
    }

    image_ids = np.array([int(row["image_id"]) for row in rows])
    dev_mask = image_ids % 2 == 0
    if dev_mask.all() or (~dev_mask).all() or len(np.unique(labels[dev_mask])) < 2:
        dev_mask = np.arange(len(rows)) % 2 == 0
    test_mask = ~dev_mask

    payload = {
        "inputs": args.inputs,
        "primary_score_status": "rejected-iteration1-keep-minus-drop",
        "posthoc_status": "diagnostic-only-not-a-method-gain",
        "samples": len(rows),
        "label_counts": {
            "yes": int(labels.sum()),
            "no": int((labels == 0).sum()),
        },
        "unique_images": int(len(set(image_ids.tolist()))),
        "mean_runtime_seconds": float(
            np.mean([row["runtime_seconds"] for row in rows])
        ),
        "baseline": {
            "ranking": ranking_metrics(labels, baseline),
            "behavior": binary_metrics(labels, baseline >= 0),
        },
        "candidate_diagnostics": candidate_diagnostics(rows),
        "identity_switch_certificate": certificate_diagnostics(rows, labels, baseline),
        "pairwise_verification": pairwise_diagnostics(
            rows, labels, baseline, args.pairwise_evaluation
        ),
        "global_support_verification": global_support_diagnostics(
            rows, labels, baseline, args.pairwise_evaluation
        ),
        "methods": {},
        "posthoc_region_diagnostics": {},
        "calibration_split": {
            "dev_samples": int(dev_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "grouped_by_image_parity": bool(
                np.array_equal(dev_mask, image_ids % 2 == 0)
            ),
        },
    }

    for method in methods:
        method_scores = scores[method]
        lambda_value = tune_lambda(labels, baseline, method_scores, dev_mask)
        corrected = baseline + lambda_value * method_scores
        payload["methods"][method] = {
            "ranking": ranking_metrics(labels, method_scores),
            "matched_tpr": fpr_at_matched_tpr(
                labels, method_scores, related, target_tpr=args.matched_tpr
            ),
            "lambda": lambda_value,
            "dev_behavior": binary_metrics(labels[dev_mask], corrected[dev_mask] >= 0),
            "test_behavior": binary_metrics(
                labels[test_mask], corrected[test_mask] >= 0
            ),
        }

    target_diagnostics = region_reformulation_scores(rows)
    for name, diagnostic_scores in target_diagnostics.items():
        payload["posthoc_region_diagnostics"][name] = {
            "ranking": ranking_metrics(labels, diagnostic_scores),
            "matched_tpr": fpr_at_matched_tpr(
                labels, diagnostic_scores, related, target_tpr=args.matched_tpr
            ),
        }
    for source in ("closed", "oracle", "open"):
        if source not in methods:
            continue
        source_diagnostics = region_reformulation_scores(rows, source=source)
        payload["posthoc_region_diagnostics"][source] = {
            name: {
                "ranking": ranking_metrics(labels, diagnostic_scores),
                "matched_tpr": fpr_at_matched_tpr(
                    labels, diagnostic_scores, related, target_tpr=args.matched_tpr
                ),
            }
            for name, diagnostic_scores in source_diagnostics.items()
            if name not in {"keep_target", "remove_target", "competitor_excess"}
        }
        excess = source_diagnostics["competitor_excess"]
        corrected = baseline - excess
        payload["posthoc_region_diagnostics"][source][
            "baseline_minus_competitor_excess"
        ] = {
            "dev_behavior": binary_metrics(labels[dev_mask], corrected[dev_mask] >= 0),
            "test_behavior": binary_metrics(
                labels[test_mask], corrected[test_mask] >= 0
            ),
            "nonzero_rate": float(np.mean(excess > 0)),
        }

    target_result = payload["posthoc_region_diagnostics"].get("keep_target")
    oracle_result = (
        payload["posthoc_region_diagnostics"].get("oracle", {}).get("dominance_gated")
    )
    if target_result and oracle_result:
        target_auc = target_result["ranking"]["auroc"]
        oracle_auc = oracle_result["ranking"]["auroc"]
        target_related_fpr = target_result["matched_tpr"]["related_present_fpr"]
        oracle_related_fpr = oracle_result["matched_tpr"]["related_present_fpr"]
        if target_related_fpr is None or oracle_related_fpr is None:
            verdict = "inconclusive-no-related-negatives"
        else:
            auc_gain = oracle_auc - target_auc
            related_fpr_drop = target_related_fpr - oracle_related_fpr
            verdict = (
                "continue"
                if auc_gain >= 0.02 or related_fpr_drop >= 0.05
                else "stop-candidate-expansion"
            )
        payload["oracle_kill_criterion"] = {
            "verdict": verdict,
            "comparison": "keep_target vs oracle_dominance_gated (post-hoc diagnostic)",
            "auroc_gain_over_target": None
            if target_auc is None
            else oracle_auc - target_auc,
            "related_fpr_drop_over_target": (
                None
                if target_related_fpr is None or oracle_related_fpr is None
                else target_related_fpr - oracle_related_fpr
            ),
            "pilot_thresholds": {"auroc_gain": 0.02, "related_fpr_drop": 0.05},
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
