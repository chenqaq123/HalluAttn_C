#!/usr/bin/env python3
"""Cross-split logistic regression verifier for richer hidden feature sets.

Replaces the grid-search approach in cross_split_hidden_margin_verifier.py with
leave-one-split-out logistic regression, which scales to 30+ feature columns.

Usage:
  python cross_split_logreg_verifier.py \
    --score_csv .../hidden_margin_v2_predictions.csv \
    --feature_columns l16_align_margin,l16_cross_margin,...,answer_support_score \
    --result_root .../coco_llava_7b_attention_only \
    --output_dir .../cross_split_logreg_out

The script:
  1. Loads the cached feature CSV (no VLM needed).
  2. For each held-out POPE split, trains a logistic regression on the other two
     splits (z-score normalized using training rows only).
  3. Gates vanilla 'yes' predictions: suppresses if the verifier score falls
     below the calibrated threshold.
  4. Reports the same subset metrics as cross_split_hidden_margin_verifier.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

YES_NO_RE = re.compile(r"^[\s\W_]*(yes|no)\b", re.IGNORECASE)

SUBSETS = [
    "all",
    "positive",
    "negative",
    "negative_related_present",
    "negative_absent_plain",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-split logreg verifier (cache-only)")
    p.add_argument("--score_csv", required=True, help="Merged per-row feature CSV from v2 extractor")
    p.add_argument("--feature_columns", required=True, help="Comma-separated feature column names")
    p.add_argument("--result_root", default="", help="Root with pope/<split>/vanilla/predictions.jsonl")
    p.add_argument("--base_method", default="vanilla")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--objectives", default="best_mcc,min_fpr_tpr0.80,best_mcc_tpr0.75")
    p.add_argument("--l2", type=float, default=0.1, help="L2 regularization strength (lambda/n)")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_yes_no(text: str) -> str:
    m = YES_NO_RE.search(text or "")
    return m.group(1).lower() if m else "invalid"


def attach_base_predictions(rows: list[dict], result_root: Path, method: str) -> list[dict]:
    splits = sorted({r["split"] for r in rows})
    base: dict[tuple[str, str], str] = {}
    for split in splits:
        pred_path = result_root / "pope" / split / method / "predictions.jsonl"
        for pred in read_jsonl(pred_path):
            base[(split, str(pred["question_id"]))] = normalize_yes_no(
                pred.get("text", pred.get("prediction", ""))
            )
    return [{**r, "base_prediction": base.get((r["split"], str(r["question_id"])), "invalid")} for r in rows]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (fp + tn) if fp + tn else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "samples": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall_tpr": tpr, "fpr": fpr, "tnr": tnr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "balanced_accuracy": (tpr + tnr) / 2,
        "mcc": mcc,
        "yes_rate": (tp + fp) / n if n else 0.0,
    }


def metrics(rows: list[dict]) -> dict:
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
    return [{"split": split_name, "subset": s, **metrics(subset_rows(rows, s))} for s in SUBSETS]


# ---------------------------------------------------------------------------
# Logistic regression (pure numpy, no scipy)
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def logreg_train(X: list[list[float]], y: list[int], l2: float, lr: float, epochs: int) -> tuple[list[float], float]:
    import numpy as np
    Xn = np.array(X, dtype=np.float64)
    yn = np.array(y, dtype=np.float64)
    n, d = Xn.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - yn
        w -= lr * ((Xn.T @ g) / n + l2 * w)
        b -= lr * float(g.mean())
    return w.tolist(), float(b)


def logreg_score(X: list[list[float]], w: list[float], b: float) -> list[float]:
    import numpy as np
    Xn = np.array(X, dtype=np.float64)
    wn = np.array(w, dtype=np.float64)
    return (Xn @ wn + b).tolist()


def standardize_fit(X: list[list[float]]) -> tuple[list[float], list[float]]:
    import numpy as np
    Xn = np.array(X, dtype=np.float64)
    mean = Xn.mean(axis=0).tolist()
    std = (Xn.std(axis=0) + 1e-8).tolist()
    return mean, std


def standardize_apply(X: list[list[float]], mean: list[float], std: list[float]) -> list[list[float]]:
    import numpy as np
    Xn = np.array(X, dtype=np.float64)
    mn = np.array(mean, dtype=np.float64)
    sn = np.array(std, dtype=np.float64)
    return ((Xn - mn) / sn).tolist()


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def choose_threshold(scores: list[float], labels: list[int], base_preds: list[str], objective: str, min_tpr: float | None) -> tuple[float, dict]:
    """Find threshold maximising the objective, gating only vanilla 'yes' rows."""
    eligible_idx = [i for i, bp in enumerate(base_preds) if bp == "yes"]
    eligible_scores = sorted({scores[i] for i in eligible_idx})
    thresholds = [eligible_scores[0] - 1e-8] if eligible_scores else [-1e-8]
    thresholds += [(a + b) / 2 for a, b in zip(eligible_scores, eligible_scores[1:])]
    thresholds += [eligible_scores[-1] + 1e-8] if eligible_scores else [1e-8]

    best_t, best_m = thresholds[0], None
    for t in thresholds:
        preds = _apply_gate(scores, base_preds, t)
        m = metrics([{"prediction": p, "label": "yes" if lab == 1 else "no"} for p, lab in zip(preds, labels)])
        if min_tpr is not None and m["recall_tpr"] < min_tpr:
            continue
        if best_m is None:
            best_t, best_m = t, m
            continue
        if objective == "min_fpr":
            key = (-m["fpr"], m["mcc"], m["recall_tpr"])
            old = (-best_m["fpr"], best_m["mcc"], best_m["recall_tpr"])
        else:
            key = (m["mcc"], -m["fpr"], m["recall_tpr"])
            old = (best_m["mcc"], -best_m["fpr"], best_m["recall_tpr"])
        if key > old:
            best_t, best_m = t, m
    return best_t, best_m or {}


def _apply_gate(scores: list[float], base_preds: list[str], threshold: float) -> list[str]:
    preds = []
    for score, bp in zip(scores, base_preds):
        if bp == "yes" and score <= threshold:
            preds.append("no")
        else:
            preds.append(bp)
    return preds


def objective_spec(name: str) -> tuple[str, float | None]:
    if name == "best_mcc":
        return "best_mcc", None
    if name.startswith("best_mcc_tpr"):
        return "best_mcc", float(name.replace("best_mcc_tpr", ""))
    if name.startswith("min_fpr_tpr"):
        return "min_fpr", float(name.replace("min_fpr_tpr", ""))
    raise ValueError(f"Unknown objective: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
    objectives = [o.strip() for o in args.objectives.split(",") if o.strip()]

    rows = read_csv(Path(args.score_csv))
    rows = [{**r, "label": r["label"].lower()} for r in rows]

    if args.result_root:
        rows = attach_base_predictions(rows, Path(args.result_root), args.base_method)
    else:
        rows = [{**r, "base_prediction": r.get("prediction", "").lower()} for r in rows]

    missing = [c for c in feature_cols if c not in rows[0]]
    if missing:
        raise ValueError(f"Feature columns not found in CSV: {missing}")

    splits = sorted({r["split"] for r in rows})

    fold_rows: list[dict] = []
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for heldout in splits:
        train_rows = [r for r in rows if r["split"] != heldout]
        test_rows = [r for r in rows if r["split"] == heldout]

        X_train = [[float(r[c]) for c in feature_cols] for r in train_rows]
        y_train = [1 if r["label"] == "yes" else 0 for r in train_rows]
        X_test = [[float(r[c]) for c in feature_cols] for r in test_rows]
        y_test = [1 if r["label"] == "yes" else 0 for r in test_rows]
        base_test = [r.get("base_prediction", "yes") for r in test_rows]

        mean, std = standardize_fit(X_train)
        X_train_z = standardize_apply(X_train, mean, std)
        X_test_z = standardize_apply(X_test, mean, std)

        w, b = logreg_train(X_train_z, y_train, l2=args.l2, lr=args.lr, epochs=args.epochs)
        test_scores = logreg_score(X_test_z, w, b)

        for obj_name in objectives:
            objective, min_tpr = objective_spec(obj_name)
            threshold, train_check = choose_threshold(
                logreg_score(X_train_z, w, b), y_train,
                [r.get("base_prediction", "yes") for r in train_rows],
                objective, min_tpr,
            )
            test_preds = _apply_gate(test_scores, base_test, threshold)
            pred_with_meta = [
                {**r, "prediction": p, "verifier_score": s, "objective": obj_name, "heldout_split": heldout}
                for r, p, s in zip(test_rows, test_preds, test_scores)
            ]
            prediction_rows.extend(pred_with_meta)
            m_all = metrics(pred_with_meta)
            m_rel = metrics(subset_rows(pred_with_meta, "negative_related_present"))
            m_plain = metrics(subset_rows(pred_with_meta, "negative_absent_plain"))
            fold_rows.append({
                "objective": obj_name,
                "heldout_split": heldout,
                "threshold": threshold,
                "test_mcc": m_all["mcc"],
                "test_tpr": m_all["recall_tpr"],
                "test_fpr": m_all["fpr"],
                "test_related_fpr": m_rel["fpr"],
                "test_plain_fpr": m_plain["fpr"],
                "test_gap": m_rel["fpr"] - m_plain["fpr"],
                "test_yes_rate": m_all["yes_rate"],
                "n_features": len(feature_cols),
                "l2": args.l2,
            })
            metric_rows.extend({"objective": obj_name, "heldout_split": heldout, **m}
                                for m in summarize(pred_with_meta, heldout))

    macro_rows = []
    for obj_name in objectives:
        obj_preds = [r for r in prediction_rows if r["objective"] == obj_name]
        if not obj_preds:
            continue
        m_all = metrics(obj_preds)
        m_rel = metrics(subset_rows(obj_preds, "negative_related_present"))
        m_plain = metrics(subset_rows(obj_preds, "negative_absent_plain"))
        adv_preds = [r for r in obj_preds if r.get("split") == "adversarial"]
        m_adv_rel = metrics(subset_rows(adv_preds, "negative_related_present")) if adv_preds else {}
        macro_rows.append({
            "objective": obj_name,
            "mcc": m_all["mcc"],
            "tpr": m_all["recall_tpr"],
            "fpr": m_all["fpr"],
            "related_fpr": m_rel["fpr"],
            "plain_fpr": m_plain["fpr"],
            "gap": m_rel["fpr"] - m_plain["fpr"],
            "yes_rate": m_all["yes_rate"],
            "adv_related_fpr": m_adv_rel.get("fpr", float("nan")),
        })
        metric_rows.extend({"objective": obj_name, "heldout_split": "macro", **m}
                            for m in summarize(obj_preds, "macro"))

    write_csv(out_dir / "logreg_folds.csv", fold_rows)
    write_csv(out_dir / "logreg_metrics.csv", metric_rows)
    write_csv(out_dir / "logreg_macro.csv", macro_rows)
    write_csv(out_dir / "logreg_predictions.csv", prediction_rows)

    summary = {
        "score_csv": args.score_csv,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "objectives": objectives,
        "l2": args.l2,
        "lr": args.lr,
        "epochs": args.epochs,
        "splits": splits,
        "macro": macro_rows,
    }
    (out_dir / "logreg_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"macro": macro_rows, "n_features": len(feature_cols)}, indent=2))


if __name__ == "__main__":
    main()
