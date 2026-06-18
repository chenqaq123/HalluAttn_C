#!/usr/bin/env python3
"""Evaluate calibrated LH-Shape linear readouts against CHAIR baselines.

This is the method-oriented TDEV-lite audit after the split-selected LH-Shape
ablation. It keeps the deployment shape simple: cached late-layer per-head
attention-shape features, a small L2-regularized linear/logistic readout trained
only on calibration images, and image-grouped held-out evaluation.

The readout is supervised, so it should be compared as a calibrated TDEV-lite
variant, not as a training-free baseline. The output table also recomputes core
training-free baseline metrics on the same rows and metric implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_lh_shape import (  # noqa: E402
    image_folds,
    json_ready,
    load_cache,
    load_gen_pos,
    metric_bundle,
)


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


def layer_indices_from_spec(layer_indices: np.ndarray, spec: str) -> list[int]:
    requested = [int(item.strip()) for item in spec.split(",") if item.strip()]
    mapping = {int(layer): idx for idx, layer in enumerate(layer_indices)}
    missing = [layer for layer in requested if layer not in mapping]
    if missing:
        raise ValueError(f"Missing requested layers {missing}; available={list(map(int, layer_indices))}")
    return [mapping[layer] for layer in requested]


def image_group_linear_scores(
    X: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    folds: int,
    seed: int,
    l2: float,
    lr: float,
    epochs: int,
    name: str,
) -> tuple[np.ndarray, list[dict]]:
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    fold_rows = []
    for fold_id, (train, test) in enumerate(image_folds(image_ids, folds, seed), start=1):
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0) + 1e-8
        X_train = (X[train] - mu) / sd
        X_test = (X[test] - mu) / sd
        print(f"{name} fold {fold_id}/{folds}: train={int(train.sum())} test={int(test.sum())} dims={X.shape[1]} epochs={epochs}", flush=True)
        w, b = train_logreg(X_train, labels[train], l2=l2, lr=lr, epochs=epochs)
        scores[test] = X_test @ w + b
        fold_rows.append({
            "fold": fold_id,
            "train_mentions": int(train.sum()),
            "test_mentions": int(test.sum()),
            "weight_l2_norm": float(np.linalg.norm(w)),
            "bias": float(b),
        })
    return scores, fold_rows


def read_baseline_scores(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    gen_pos = np.asarray([int(row["gen_pos"]) for row in rows], dtype=np.int32)
    return rows, labels, image_ids, gen_pos


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def baseline_metric_rows(rows: list[dict[str, str]], labels: np.ndarray, gen_pos: np.ndarray, score_cols: list[str], bin_width: int) -> list[dict]:
    output = []
    for col in score_cols:
        values = np.asarray([float(row[col]) for row in rows], dtype=np.float64)
        output.append(metric_bundle(col, values, labels, gen_pos, bin_width=bin_width))
    return output


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate calibrated LH-Shape linear readouts")
    p.add_argument("--cache", default="experiments/coco_llava_7b_rows/per_head_row_cache.npz")
    p.add_argument("--cache_glob", default=None)
    p.add_argument("--generation_json", default="experiments/coco_llava_7b/generation.json")
    p.add_argument("--baseline_scores_csv", default="detection/baselines/results/coco_llava_7b_baselines/baseline_scores.csv")
    p.add_argument("--output_dir", default="detection/baselines/results/lh_shape_linear")
    p.add_argument("--layer_sets", default="31;22,31")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument(
        "--baseline_cols",
        default="pas_layer0_hallu_score,svar_layer0_hallu_score,ic_hallu_score,glsim_local_hallu_score,entropy_hallu_score,nll_hallu_score,beyond_adscgc_hallu_score",
    )
    args = p.parse_args()

    feats, labels, token_pos, image_ids, layer_indices, feature_names = load_cache(args.cache, args.cache_glob)
    gen_pos = load_gen_pos(args.generation_json, image_ids, token_pos)
    rows, csv_labels, csv_image_ids, csv_gen_pos = read_baseline_scores(Path(args.baseline_scores_csv))
    if not (np.array_equal(labels, csv_labels) and np.array_equal(image_ids, csv_image_ids) and np.array_equal(gen_pos, csv_gen_pos)):
        raise ValueError("Per-head cache rows do not align with baseline_scores.csv labels/image_ids/gen_pos")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    fold_payload = {}

    baseline_cols = [item.strip() for item in args.baseline_cols.split(",") if item.strip()]
    metrics.extend(baseline_metric_rows(rows, labels, gen_pos, baseline_cols, args.bin_width))

    for layer_spec in [item.strip() for item in args.layer_sets.split(";") if item.strip()]:
        li = layer_indices_from_spec(layer_indices, layer_spec)
        X = feats[:, li, :, :].reshape(feats.shape[0], -1).astype(np.float64)
        name = "lh_shape_linear_layers_" + "_".join(str(int(layer_indices[idx])) for idx in li)
        scores, fold_rows = image_group_linear_scores(
            X,
            labels,
            image_ids,
            folds=args.folds,
            seed=args.seed,
            l2=args.l2,
            lr=args.lr,
            epochs=args.epochs,
            name=name,
        )
        metrics.append(metric_bundle(name, scores, labels, gen_pos, bin_width=args.bin_width))
        fold_payload[name] = fold_rows

    write_metrics_csv(output_dir / "lh_shape_linear_metrics.csv", metrics)
    payload = {
        "cache": args.cache_glob or args.cache,
        "generation_json": args.generation_json,
        "baseline_scores_csv": args.baseline_scores_csv,
        "num_mentions": int(labels.size),
        "num_images": int(len(set(image_ids.tolist()))),
        "hallucinated_mentions": int(labels.sum()),
        "grounded_mentions": int(labels.size - labels.sum()),
        "layer_indices": [int(x) for x in layer_indices],
        "feature_names": feature_names,
        "folds": int(args.folds),
        "seed": int(args.seed),
        "l2": float(args.l2),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "bin_width": int(args.bin_width),
        "baseline_cols": baseline_cols,
        "metrics": metrics,
        "folds_detail": fold_payload,
        "caveat": "LH-Shape linear readouts are supervised image-grouped calibration baselines; they are not training-free and should be reported separately from unsupervised baselines",
    }
    with (output_dir / "lh_shape_linear_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    best = max(metrics, key=lambda item: item["within_bin_auroc"] or -1)
    print(f"Wrote {output_dir / 'lh_shape_linear_metrics.json'}")
    print(json.dumps(json_ready(best), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
