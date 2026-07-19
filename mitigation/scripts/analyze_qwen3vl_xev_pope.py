#!/usr/bin/env python3
"""Analyze Qwen3-VL XEV JSONL shards without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--matched_tpr", type=float, default=0.80)
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
