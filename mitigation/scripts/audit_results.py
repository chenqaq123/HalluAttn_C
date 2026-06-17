#!/usr/bin/env python3
"""Paper-facing behavioral audit for attention mitigation results.

The audit is post-hoc: it reads merged POPE/CHAIR outputs and summarizes
whether an apparent gain reflects better discrimination or an answer/style shift.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))

from src.evaluation import normalize_yes_no, read_jsonl, write_json


POPE_FIELDS = [
    "accuracy",
    "f1",
    "mcc",
    "yes_rate",
    "invalid",
    "recall_tpr",
    "fpr",
    "balanced_accuracy",
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_stats(path: Path, text_key: str) -> dict:
    rows = read_jsonl(path)
    lengths = [len(str(row.get(text_key, "")).split()) for row in rows]
    stats = {
        "samples": len(rows),
        "mean_words": mean(lengths) if lengths else 0.0,
    }
    if text_key == "text":
        parsed = [normalize_yes_no(str(row.get("text", ""))) for row in rows]
        yes = sum(pred == "yes" for pred in parsed)
        invalid = sum(pred == "invalid" for pred in parsed)
        stats["normalized_yes_rate"] = yes / len(rows) if rows else 0.0
        stats["invalid_answers"] = invalid
    return stats


def _pope_split_rows(result_root: Path, methods: list[str]) -> list[dict]:
    rows = []
    pope_root = result_root / "pope"
    if not pope_root.exists():
        return rows
    for split_dir in sorted(p for p in pope_root.iterdir() if p.is_dir()):
        comparison_path = split_dir / "comparison.json"
        if not comparison_path.exists():
            continue
        comparison = _load_json(comparison_path)
        vanilla = comparison["vanilla"]
        vanilla_text = _text_stats(split_dir / "vanilla" / "predictions.jsonl", "text")
        for method in methods:
            if method == "vanilla":
                row = {
                    "task": "pope",
                    "split": split_dir.name,
                    "method": method,
                    "scope": "anchor",
                }
                for key in POPE_FIELDS:
                    row[key] = vanilla.get(key, "")
                    row[f"delta_{key}"] = 0.0
                row.update({
                    "mean_answer_words": vanilla_text["mean_words"],
                    "invalid_answers": vanilla_text.get("invalid_answers", 0),
                    "delta_mean_answer_words": 0.0,
                })
            else:
                payload = comparison["methods"][method]
                metrics = payload["metrics"]
                delta = payload["delta_vs_vanilla"]
                text_stats = _text_stats(split_dir / method / "predictions.jsonl", "text")
                scope = {
                    "pai": "attention_component",
                    "clearsight": "ported_attention_intervention",
                    "visattnsink": "ported_attention_intervention",
                    "vcd": "ported_decoding_intervention",
                }.get(method, "ported_mitigation")
                row = {
                    "task": "pope",
                    "split": split_dir.name,
                    "method": method,
                    "scope": scope,
                }
                for key in POPE_FIELDS:
                    row[key] = metrics.get(key, "")
                    if key == "recall_tpr":
                        row[f"delta_{key}"] = delta.get("delta_tpr", "")
                    elif key == "fpr":
                        row[f"delta_{key}"] = delta.get("delta_fpr", "")
                    elif key in {"mcc", "invalid"}:
                        row[f"delta_{key}"] = metrics.get(key, 0.0) - vanilla.get(key, 0.0)
                    else:
                        row[f"delta_{key}"] = delta.get(f"delta_{key}", "")
                row.update({
                    "delta_tpr_minus_delta_fpr": delta.get("delta_tpr_minus_delta_fpr", ""),
                    "mean_answer_words": text_stats["mean_words"],
                    "invalid_answers": text_stats.get("invalid_answers", 0),
                    "delta_invalid_answers": text_stats.get("invalid_answers", 0) - vanilla_text.get("invalid_answers", 0),
                    "delta_mean_answer_words": text_stats["mean_words"] - vanilla_text["mean_words"],
                })
            rows.append(row)
    return rows


def _macro_rows(split_rows: list[dict], methods: list[str]) -> list[dict]:
    macro_rows = []
    for method in methods:
        rows = [row for row in split_rows if row["method"] == method]
        if not rows:
            continue
        macro: dict = {"task": "pope", "split": "macro", "method": method}
        numeric_keys = [
            "accuracy",
            "f1",
            "mcc",
            "yes_rate",
            "invalid",
            "recall_tpr",
            "fpr",
            "balanced_accuracy",
            "delta_accuracy",
            "delta_f1",
            "delta_mcc",
            "delta_yes_rate",
            "delta_invalid",
            "invalid_answers",
            "delta_invalid_answers",
            "delta_recall_tpr",
            "delta_fpr",
            "delta_tpr_minus_delta_fpr",
            "mean_answer_words",
            "delta_mean_answer_words",
        ]
        for key in numeric_keys:
            values = [row[key] for row in rows if key in row and isinstance(row[key], (int, float))]
            macro[key] = mean(values) if values else ""
        macro["interpretation"] = _interpret_pope_shift(macro)
        macro_rows.append(macro)
    return macro_rows


def _interpret_pope_shift(row: dict) -> str:
    if row["method"] == "vanilla":
        return "anchor"
    delta_tpr = row.get("delta_recall_tpr", 0.0)
    delta_fpr = row.get("delta_fpr", 0.0)
    delta_mcc = row.get("delta_mcc", 0.0)
    if isinstance(delta_tpr, (int, float)) and isinstance(delta_fpr, (int, float)):
        if delta_tpr > 0 and delta_fpr >= delta_tpr:
            return "yes-prior shift: recall gain is matched or exceeded by FPR gain"
        if delta_tpr <= 0 and delta_fpr <= 0:
            return "no recall gain; minor answer-prior movement"
    if isinstance(delta_mcc, (int, float)) and delta_mcc < 0:
        return "discrimination weakens"
    return "mixed"


def _chair_rows(result_root: Path, methods: list[str]) -> list[dict]:
    comparison_path = result_root / "chair" / "caption" / "comparison.json"
    if not comparison_path.exists():
        return []
    comparison = _load_json(comparison_path)
    vanilla = comparison["vanilla"]
    rows = []
    for method in methods:
        if method == "vanilla":
            metrics = vanilla
            delta = {
                "delta_CHAIRi": 0.0,
                "delta_CHAIRs": 0.0,
                "delta_mean_words": 0.0,
                "delta_mean_object_mentions": 0.0,
                "delta_mean_hallucinated_mentions": 0.0,
            }
        else:
            payload = comparison["methods"][method]
            metrics = payload["metrics"]
            delta = payload["delta_vs_vanilla"]
        chair = metrics["chair"]
        stats = metrics["caption_stats"]
        rows.append({
            "task": "chair",
            "split": "caption",
            "method": method,
            "CHAIRi": chair["CHAIRi"],
            "CHAIRs": chair["CHAIRs"],
            "mean_words": stats["mean_words"],
            "mean_object_mentions": stats["mean_object_mentions"],
            "mean_hallucinated_mentions": stats["mean_hallucinated_mentions"],
            **delta,
            "interpretation": _interpret_chair_shift(delta, method),
        })
    return rows


def _interpret_chair_shift(delta: dict, method: str) -> str:
    if method == "vanilla":
        return "anchor"
    d_ci = delta.get("delta_CHAIRi", 0.0)
    d_hall = delta.get("delta_mean_hallucinated_mentions", 0.0)
    d_obj = delta.get("delta_mean_object_mentions", 0.0)
    if d_ci <= 0 and d_hall <= 0:
        return "possible caption-level mitigation"
    if d_obj > 0 and d_hall > 0:
        return "richer captions with more hallucinated object mentions"
    if d_obj < 0 and d_hall <= 0 and d_ci >= 0:
        return "shorter/sparser captions without CHAIR improvement"
    return "no clear mitigation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize POPE/CHAIR mitigation outputs for paper tables")
    p.add_argument("--result_root", default=str(PROJECT_ROOT / "mitigation/results/coco_llava_7b_attention_only"))
    p.add_argument("--methods", default="vanilla,pai,clearsight,visattnsink")
    p.add_argument("--output_prefix", default="audit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    split_rows = _pope_split_rows(result_root, methods)
    macro_rows = _macro_rows(split_rows, methods)
    chair_rows = _chair_rows(result_root, methods)
    payload = {
        "result_root": str(result_root),
        "methods": methods,
        "pope_by_split": split_rows,
        "pope_macro": macro_rows,
        "chair": chair_rows,
    }
    write_json(result_root / f"{args.output_prefix}.json", payload)

    for name, rows in [
        ("pope_by_split", split_rows),
        ("pope_macro", macro_rows),
        ("chair", chair_rows),
    ]:
        path = result_root / f"{args.output_prefix}_{name}.csv"
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")
    print(f"Wrote {result_root / f'{args.output_prefix}.json'}")


if __name__ == "__main__":
    main()
