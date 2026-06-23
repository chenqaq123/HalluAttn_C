#!/usr/bin/env python3
"""Cross-split validation for cached D-lite hidden-margin verifier formulas.

The script does not run the VLM. It tunes a low-dimensional linear formula on
all but one POPE split, then evaluates the selected operating point on the held-
out split. This tests whether cached internal hidden-margin evidence can support
a deployable no-external-detector verifier without selecting thresholds on the
evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Iterable

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)

DEFAULT_COMPONENTS = [
    "hidden_align_margin",
    "hidden_cross_margin",
    "hidden_obj_separation",
    "hidden_vis_separation",
]
SUBSETS = [
    "all",
    "positive",
    "negative",
    "negative_related_present",
    "negative_absent_plain",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-split tune/evaluate cached hidden-margin verifier formulas")
    p.add_argument("--score_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--components", default=",".join(DEFAULT_COMPONENTS))
    p.add_argument("--weight_grid", default="-1,-0.5,0,0.5,1")
    p.add_argument("--mode", choices=["gate", "direct"], default="gate")
    p.add_argument("--result_root", default="", help="Root containing pope/<split>/<method>/predictions.jsonl for gate base predictions")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--objectives", default="best_mcc,best_mcc_tpr0.80,best_mcc_tpr0.75,min_fpr_tpr0.80,min_fpr_tpr0.75")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_yes_no(text: str) -> str:
    match = YES_NO_RE.search(text or "")
    return match.group(1).lower() if match else "invalid"


def attach_base_predictions(rows: list[dict], result_root: Path, method: str) -> list[dict]:
    splits = sorted({row["split"] for row in rows})
    base = {}
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for pred in read_jsonl(pred_path):
            base[(split, str(pred["question_id"]))] = normalize_yes_no(pred.get("text", pred.get("prediction", "")))
    out = []
    for row in rows:
        key = (row["split"], str(row["question_id"]))
        if key in base:
            out.append({**row, "base_prediction": base[key]})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(spec: str) -> list[float]:
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_str_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def metric_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {
        "samples": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall_tpr": tpr,
        "fpr": fpr,
        "tnr": tnr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": mcc,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def metrics(rows: list[dict]) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for r in rows:
        pred = r["prediction"].lower()
        gold = r["label"].lower()
        if pred == "yes" and gold == "yes":
            tp += 1
        elif pred == "yes" and gold == "no":
            fp += 1
        elif pred == "no" and gold == "no":
            tn += 1
        elif pred == "no" and gold == "yes":
            fn += 1
    return metric_counts(tp, fp, tn, fn)


def subset_rows(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return rows
    if subset == "positive":
        return [r for r in rows if r["label"].lower() == "yes"]
    if subset == "negative":
        return [r for r in rows if r["label"].lower() == "no"]
    return [r for r in rows if r.get("negative_type") == subset]


def summarize(rows: list[dict], split_name: str) -> list[dict]:
    return [{"split": split_name, "subset": subset, **metrics(subset_rows(rows, subset))} for subset in SUBSETS]


def score_row(row: dict, weights: tuple[float, ...], components: list[str]) -> float:
    return sum(w * float(row[c]) for w, c in zip(weights, components))


def eligible(row: dict, mode: str) -> bool:
    if mode == "direct":
        return True
    return row.get("base_prediction", row.get("prediction", "")).lower() == "yes"


def objective_spec(name: str) -> tuple[str, float | None]:
    if name == "best_mcc":
        return "best_mcc", None
    if name.startswith("best_mcc_tpr"):
        return "best_mcc", float(name.replace("best_mcc_tpr", ""))
    if name.startswith("min_fpr_tpr"):
        return "min_fpr", float(name.replace("min_fpr_tpr", ""))
    raise ValueError(f"Unknown objective: {name}")


def better(candidate: dict, incumbent: dict | None, objective: str, min_tpr: float | None) -> bool:
    if min_tpr is not None and candidate["recall_tpr"] < min_tpr:
        return False
    if incumbent is None:
        return True
    if objective == "min_fpr":
        key = (-candidate["fpr"], candidate["mcc"], candidate["recall_tpr"], -candidate["yes_rate"])
        old = (-incumbent["fpr"], incumbent["mcc"], incumbent["recall_tpr"], -incumbent["yes_rate"])
    else:
        key = (candidate["mcc"], -candidate["fpr"], candidate["recall_tpr"], -candidate["yes_rate"])
        old = (incumbent["mcc"], -incumbent["fpr"], incumbent["recall_tpr"], -incumbent["yes_rate"])
    return key > old


def scan_thresholds(rows: list[dict], weights: tuple[float, ...], components: list[str], mode: str) -> Iterable[dict]:
    scored = [(score_row(r, weights, components), r) for r in rows]
    scored.sort(key=lambda x: x[0], reverse=True)
    total_pos = sum(1 for _, r in scored if r["label"].lower() == "yes")
    total_neg = len(scored) - total_pos
    tp = fp = 0
    yield {"threshold": scored[0][0] + 1e-8 if scored else 0.0, **metric_counts(0, 0, total_neg, total_pos)}
    i = 0
    n = len(scored)
    while i < n:
        value = scored[i][0]
        while i < n and scored[i][0] == value:
            _, row = scored[i]
            if eligible(row, mode):
                if row["label"].lower() == "yes":
                    tp += 1
                else:
                    fp += 1
            i += 1
        tn = total_neg - fp
        fn = total_pos - tp
        next_value = scored[i][0] if i < n else value - 1e-8
        threshold = (value + next_value) / 2.0 if i < n else value - 1e-8
        yield {"threshold": threshold, **metric_counts(tp, fp, tn, fn)}


def tune(train_rows: list[dict], components: list[str], weights_values: list[float], mode: str, objectives: list[str]) -> dict[str, dict]:
    selected: dict[str, dict | None] = {name: None for name in objectives}
    specs = {name: objective_spec(name) for name in objectives}
    for weights in itertools.product(weights_values, repeat=len(components)):
        if all(w == 0.0 for w in weights):
            continue
        for m in scan_thresholds(train_rows, weights, components, mode):
            for name, (objective, min_tpr) in specs.items():
                cand = {**m, "weights": weights, "mode": mode}
                if better(cand, selected[name], objective, min_tpr):
                    selected[name] = cand
    return {name: cand for name, cand in selected.items() if cand is not None}


def apply_formula(rows: list[dict], weights: tuple[float, ...], threshold: float, components: list[str], mode: str) -> list[dict]:
    out = []
    for row in rows:
        score = score_row(row, weights, components)
        score_yes = score > threshold
        if mode == "direct":
            pred = "yes" if score_yes else "no"
        else:
            pred = "yes" if eligible(row, mode) and score_yes else "no"
        out.append({**row, "prediction": pred, "verifier_score": score})
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    components = parse_str_list(args.components)
    weights_values = parse_float_list(args.weight_grid)
    objectives = parse_str_list(args.objectives)
    rows = read_rows(Path(args.score_csv))
    rows = [{**r, "label": r["label"].lower()} for r in rows]
    if args.mode == "gate":
        if args.result_root:
            rows = attach_base_predictions(rows, Path(args.result_root), args.base_method)
        else:
            rows = [{**r, "base_prediction": r.get("prediction", "").lower()} for r in rows]
    if not rows:
        raise ValueError("No rows available after attaching base predictions")
    splits = sorted({r["split"] for r in rows})

    fold_rows = []
    metric_rows = []
    prediction_rows = []
    for heldout in splits:
        train = [r for r in rows if r["split"] != heldout]
        test = [r for r in rows if r["split"] == heldout]
        selected = tune(train, components, weights_values, args.mode, objectives)
        for objective, cand in selected.items():
            weights = tuple(cand["weights"])
            pred = apply_formula(test, weights, float(cand["threshold"]), components, args.mode)
            test_all = metrics(pred)
            rel = metrics(subset_rows(pred, "negative_related_present"))
            plain = metrics(subset_rows(pred, "negative_absent_plain"))
            fold_rows.append({
                "objective": objective,
                "heldout_split": heldout,
                "mode": args.mode,
                "weights": "|".join(f"{w:g}" for w in weights),
                "threshold": cand["threshold"],
                "train_mcc": cand["mcc"],
                "train_tpr": cand["recall_tpr"],
                "train_fpr": cand["fpr"],
                "test_mcc": test_all["mcc"],
                "test_tpr": test_all["recall_tpr"],
                "test_fpr": test_all["fpr"],
                "test_related_fpr": rel["fpr"],
                "test_plain_fpr": plain["fpr"],
                "test_gap": rel["fpr"] - plain["fpr"],
                "test_yes_rate": test_all["yes_rate"],
            })
            metric_rows.extend({"objective": objective, "heldout_split": heldout, **m} for m in summarize(pred, heldout))
            prediction_rows.extend({"objective": objective, "heldout_split": heldout, **r} for r in pred)

    macro_rows = []
    for objective in objectives:
        obj_preds = [r for r in prediction_rows if r["objective"] == objective]
        if not obj_preds:
            continue
        metric_rows.extend({"objective": objective, "heldout_split": "macro", **m} for m in summarize(obj_preds, "macro"))
        all_m = metrics(obj_preds)
        rel = metrics(subset_rows(obj_preds, "negative_related_present"))
        plain = metrics(subset_rows(obj_preds, "negative_absent_plain"))
        macro_rows.append({
            "objective": objective,
            "mcc": all_m["mcc"],
            "tpr": all_m["recall_tpr"],
            "fpr": all_m["fpr"],
            "related_fpr": rel["fpr"],
            "plain_fpr": plain["fpr"],
            "gap": rel["fpr"] - plain["fpr"],
            "yes_rate": all_m["yes_rate"],
        })

    write_csv(out_dir / "cross_split_folds.csv", fold_rows)
    write_csv(out_dir / "cross_split_metrics.csv", metric_rows)
    write_csv(out_dir / "cross_split_macro.csv", macro_rows)
    write_csv(out_dir / "cross_split_predictions.csv", prediction_rows)
    summary = {
        "score_csv": args.score_csv,
        "components": components,
        "weight_grid": weights_values,
        "mode": args.mode,
        "objectives": objectives,
        "splits": splits,
        "macro": macro_rows,
    }
    (out_dir / "cross_split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
