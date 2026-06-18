#!/usr/bin/env python3
"""Evaluate calibrated LH-Shape transfer on POPE per-head caches.

The cache labels use target_absent=1. This script trains a small L2 logistic
readout on one or more calibration splits, chooses a threshold on those splits,
and reports whether the score detects absent targets on held-out POPE splits and
semantic-neighbor subsets. It is an evaluation harness for
cache_pope_per_head_rows.py outputs; it does not run the VLM.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LH-Shape transfer on cached POPE per-head rows")
    p.add_argument("--cache", default="experiments/pope_llava_7b_per_head/pope_per_head_row_cache.npz")
    p.add_argument("--cache_glob", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layer_sets", default="31;22,31")
    p.add_argument("--train_splits", default="random")
    p.add_argument("--eval_splits", default="random,popular,adversarial")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--image_cv_folds", type=int, default=0, help="If >1, use image-grouped out-of-fold evaluation instead of fixed train_splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_prompt_baselines", action="store_true", help="Also evaluate prompt-only token-position/length baselines")
    return p.parse_args()


def read_npz_files(cache: str, cache_glob: str | None) -> list[Path]:
    if cache_glob:
        files = [Path(p) for p in sorted(glob.glob(cache_glob))]
        if not files:
            raise FileNotFoundError(cache_glob)
        return files
    return [Path(cache)]


def load_cache(cache: str, cache_glob: str | None) -> dict:
    files = read_npz_files(cache, cache_glob)
    parts = [np.load(path, allow_pickle=True) for path in files]
    first_layers = parts[0]["layer_indices"].astype(np.int32)
    first_features = [str(x) for x in parts[0]["feature_names"]]
    for path, part in zip(files, parts):
        if not np.array_equal(part["layer_indices"].astype(np.int32), first_layers):
            raise ValueError(f"Layer mismatch in {path}")
        if [str(x) for x in part["feature_names"]] != first_features:
            raise ValueError(f"Feature-name mismatch in {path}")
    return {
        "files": [str(path) for path in files],
        "feats": np.concatenate([part["feats"] for part in parts], axis=0).astype(np.float64),
        "labels": np.concatenate([part["labels"] for part in parts], axis=0).astype(np.int32),
        "splits": np.concatenate([part["splits"].astype(str) for part in parts], axis=0),
        "negative_types": np.concatenate([part["negative_types"].astype(str) for part in parts], axis=0),
        "targets": np.concatenate([part["targets"].astype(str) for part in parts], axis=0),
        "question_ids": np.concatenate([part["question_ids"].astype(str) for part in parts], axis=0),
        "image_ids": np.concatenate([part["image_ids"] for part in parts], axis=0).astype(np.int64),
        "token_pos": np.concatenate([part["token_pos"] for part in parts], axis=0).astype(np.int32),
        "target_token_start": np.concatenate([part["target_token_start"] for part in parts], axis=0).astype(np.int32),
        "target_token_end": np.concatenate([part["target_token_end"] for part in parts], axis=0).astype(np.int32),
        "layer_indices": first_layers,
        "feature_names": first_features,
    }


def train_logreg(X: np.ndarray, y: np.ndarray, l2: float, lr: float, epochs: int) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        w -= lr * ((X.T @ g) / n + l2 * w / n)
        b -= lr * float(g.mean())
    return w, b


def roc_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    labels = labels[valid]
    values = values[valid]
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics_from_pred(labels: np.ndarray, scores: np.ndarray, pred_absent: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    pred_absent = np.asarray(pred_absent, dtype=bool)
    tp = int(((labels == 1) & pred_absent).sum())
    fp = int(((labels == 0) & pred_absent).sum())
    tn = int(((labels == 0) & ~pred_absent).sum())
    fn = int(((labels == 1) & ~pred_absent).sum())
    n = tp + fp + tn + fn
    absent_tpr = tp / (tp + fn) if tp + fn else 0.0
    present_fpr = fp / (fp + tn) if fp + tn else 0.0
    present_tnr = tn / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": n,
        "absent": int(labels.sum()),
        "present": int(n - labels.sum()),
        "tp_absent": tp,
        "fp_present": fp,
        "tn_present": tn,
        "fn_absent": fn,
        "absent_tpr": absent_tpr,
        "present_fpr": present_fpr,
        "balanced_accuracy": 0.5 * (absent_tpr + present_tnr),
        "precision_absent": precision,
        "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0,
        "predicted_absent_rate": (tp + fp) / n if n else 0.0,
        "auroc": roc_auc(labels, scores),
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return binary_metrics_from_pred(labels, scores, np.asarray(scores, dtype=np.float64) > threshold)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    values = np.asarray(scores, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite calibration scores")
    unique = np.unique(finite)
    candidates = [float(unique[0] - 1e-6), float(unique[-1] + 1e-6)]
    candidates.extend(float((a + b) / 2.0) for a, b in zip(unique, unique[1:]))
    best_threshold = candidates[0]
    best_metrics = binary_metrics(labels, scores, best_threshold)
    for threshold in candidates[1:]:
        metrics = binary_metrics(labels, scores, threshold)
        if metrics["mcc"] > best_metrics["mcc"]:
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def parse_layers(layer_indices: np.ndarray, spec: str) -> list[int]:
    requested = [int(item.strip()) for item in spec.split(",") if item.strip()]
    mapping = {int(layer): idx for idx, layer in enumerate(layer_indices)}
    missing = [layer for layer in requested if layer not in mapping]
    if missing:
        raise ValueError(f"Missing requested layers {missing}; available={list(map(int, layer_indices))}")
    return [mapping[layer] for layer in requested]


def split_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def image_folds(image_ids: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    unique_images = np.asarray(sorted(set(int(x) for x in image_ids.tolist())))
    rng.shuffle(unique_images)
    fold_images = np.array_split(unique_images, folds)
    output = []
    for items in fold_images:
        test = np.isin(image_ids, items)
        output.append((~test, test))
    return output


def oof_linear_scores(
    X: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    folds: int,
    seed: int,
    l2: float,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    pred_absent = np.zeros(labels.shape[0], dtype=bool)
    details = []
    for fold_id, (train, test) in enumerate(image_folds(image_ids, folds, seed), start=1):
        if np.unique(labels[train]).size < 2:
            raise ValueError(f"Fold {fold_id} training rows do not contain both classes")
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0) + 1e-8
        X_train = (X[train] - mu) / sd
        X_test = (X[test] - mu) / sd
        w, b = train_logreg(X_train, labels[train], l2=l2, lr=lr, epochs=epochs)
        train_scores = X_train @ w + b
        threshold, train_metrics = choose_threshold(labels[train], train_scores)
        test_scores = X_test @ w + b
        scores[test] = test_scores
        pred_absent[test] = test_scores > threshold
        details.append({
            "fold": fold_id,
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "threshold": float(threshold),
            "train_mcc": float(train_metrics["mcc"]),
            "train_absent_tpr": float(train_metrics["absent_tpr"]),
            "train_present_fpr": float(train_metrics["present_fpr"]),
            "weight_l2_norm": float(np.linalg.norm(w)),
            "bias": float(b),
        })
    return scores, pred_absent, details


def oof_threshold_scores(
    values: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    raw = np.asarray(values, dtype=np.float64)
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    pred_absent = np.zeros(labels.shape[0], dtype=bool)
    details = []
    for fold_id, (train, test) in enumerate(image_folds(image_ids, folds, seed), start=1):
        best = None
        for direction, train_scores, test_scores in (("positive", raw[train], raw[test]), ("negative", -raw[train], -raw[test])):
            threshold, train_metrics = choose_threshold(labels[train], train_scores)
            candidate = (train_metrics["mcc"], direction, threshold, train_scores, test_scores, train_metrics)
            if best is None or candidate[0] > best[0]:
                best = candidate
        assert best is not None
        _mcc, direction, threshold, _train_scores, test_scores, train_metrics = best
        scores[test] = test_scores
        pred_absent[test] = test_scores > threshold
        details.append({
            "fold": fold_id,
            "direction": direction,
            "threshold": float(threshold),
            "train_mcc": float(train_metrics["mcc"]),
            "train_absent_tpr": float(train_metrics["absent_tpr"]),
            "train_present_fpr": float(train_metrics["present_fpr"]),
        })
    return scores, pred_absent, details


def fixed_threshold_scores(values: np.ndarray, labels: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.asarray(values, dtype=np.float64)
    best = None
    for direction, candidate_scores in (("positive", raw), ("negative", -raw)):
        threshold, train_metrics = choose_threshold(labels[train_mask], candidate_scores[train_mask])
        candidate = (train_metrics["mcc"], direction, threshold, candidate_scores, train_metrics)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    _mcc, direction, threshold, scores, train_metrics = best
    return scores, scores > threshold, {
        "direction": direction,
        "threshold": float(threshold),
        "calibration_metrics": train_metrics,
    }


def prompt_baseline_values(data: dict) -> dict[str, np.ndarray]:
    return {
        "prompt_token_pos": data["token_pos"].astype(np.float64),
        "target_span_len": (data["target_token_end"] - data["target_token_start"]).astype(np.float64),
        "target_char_len": np.asarray([len(str(target)) for target in data["targets"]], dtype=np.float64),
    }


def subset_mask(labels: np.ndarray, negative_types: np.ndarray, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(labels.shape[0], dtype=bool)
    if subset == "present":
        return labels == 0
    if subset == "absent":
        return labels == 1
    return negative_types == subset


def main() -> None:
    args = parse_args()
    data = load_cache(args.cache, args.cache_glob)
    labels = data["labels"]
    splits = data["splits"]
    negative_types = data["negative_types"]
    train_splits = set(split_list(args.train_splits))
    eval_splits = split_list(args.eval_splits)
    train_mask = np.isin(splits, list(train_splits))
    if train_mask.sum() == 0:
        raise ValueError(f"No rows found for train_splits={sorted(train_splits)}")
    if np.unique(labels[train_mask]).size < 2:
        raise ValueError("Training rows must contain both present and absent targets")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    model_payload = {}
    subsets = ["all", "present", "absent", "negative_related_present", "negative_absent_plain", "negative_target_present_coco_label"]

    for layer_spec in [item.strip() for item in args.layer_sets.split(";") if item.strip()]:
        layer_ids = parse_layers(data["layer_indices"], layer_spec)
        name = "lh_shape_pope_layers_" + "_".join(str(int(data["layer_indices"][idx])) for idx in layer_ids)
        X = data["feats"][:, layer_ids, :, :].reshape(labels.shape[0], -1)
        if args.image_cv_folds > 1:
            scores, pred_absent, fold_details = oof_linear_scores(
                X,
                labels,
                data["image_ids"],
                folds=args.image_cv_folds,
                seed=args.seed,
                l2=args.l2,
                lr=args.lr,
                epochs=args.epochs,
            )
            model_payload[name] = {
                "mode": "image_grouped_oof",
                "layers": [int(data["layer_indices"][idx]) for idx in layer_ids],
                "dims": int(X.shape[1]),
                "folds": int(args.image_cv_folds),
                "fold_details": fold_details,
            }
        else:
            mu = X[train_mask].mean(axis=0)
            sd = X[train_mask].std(axis=0) + 1e-8
            X_train = (X[train_mask] - mu) / sd
            w, b = train_logreg(X_train, labels[train_mask], l2=args.l2, lr=args.lr, epochs=args.epochs)
            scores = ((X - mu) / sd) @ w + b
            threshold, calibration_metrics = choose_threshold(labels[train_mask], scores[train_mask])
            pred_absent = scores > threshold
            model_payload[name] = {
                "mode": "fixed_train_splits",
                "layers": [int(data["layer_indices"][idx]) for idx in layer_ids],
                "dims": int(X.shape[1]),
                "threshold": threshold,
                "calibration_metrics": calibration_metrics,
                "weight_l2_norm": float(np.linalg.norm(w)),
                "bias": float(b),
            }
        for split in eval_splits + ["macro"]:
            split_mask = np.ones(labels.shape[0], dtype=bool) if split == "macro" else splits == split
            for subset in subsets:
                mask = split_mask & subset_mask(labels, negative_types, subset)
                if mask.sum() == 0:
                    continue
                metric_rows.append({
                    "score": name,
                    "split": split,
                    "subset": subset,
                    **binary_metrics_from_pred(labels[mask], scores[mask], pred_absent[mask]),
                })

    if args.include_prompt_baselines:
        for name, values in prompt_baseline_values(data).items():
            if args.image_cv_folds > 1:
                scores, pred_absent, details = oof_threshold_scores(
                    values, labels, data["image_ids"], args.image_cv_folds, args.seed
                )
                model_payload[name] = {
                    "mode": "prompt_only_image_grouped_oof",
                    "folds": int(args.image_cv_folds),
                    "fold_details": details,
                }
            else:
                scores, pred_absent, details = fixed_threshold_scores(values, labels, train_mask)
                model_payload[name] = {"mode": "prompt_only_fixed_train_splits", **details}
            for split in eval_splits + ["macro"]:
                split_mask = np.ones(labels.shape[0], dtype=bool) if split == "macro" else splits == split
                for subset in subsets:
                    mask = split_mask & subset_mask(labels, negative_types, subset)
                    if mask.sum() == 0:
                        continue
                    metric_rows.append({
                        "score": name,
                        "split": split,
                        "subset": subset,
                        **binary_metrics_from_pred(labels[mask], scores[mask], pred_absent[mask]),
                    })

    with (output_dir / "pope_lh_shape_transfer_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)
    payload = {
        "cache": data["files"],
        "num_rows": int(labels.size),
        "train_splits": sorted(train_splits),
        "eval_splits": eval_splits,
        "label_name": "target_absent",
        "l2": float(args.l2),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "image_cv_folds": int(args.image_cv_folds),
        "seed": int(args.seed),
        "include_prompt_baselines": bool(args.include_prompt_baselines),
        "models": model_payload,
        "metrics": metric_rows,
        "caveat": "This evaluates target-absence detection from cached POPE question-token features; it does not by itself apply a yes/no gate to generated model outputs.",
    }
    with (output_dir / "pope_lh_shape_transfer_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    best = max([row for row in metric_rows if row["split"] == "macro" and row["subset"] == "all"], key=lambda row: row["mcc"])
    print(f"Wrote {output_dir / 'pope_lh_shape_transfer_metrics.json'}")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
