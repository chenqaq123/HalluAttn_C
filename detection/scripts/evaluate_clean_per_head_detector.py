#!/usr/bin/env python3
"""Evaluate locked label-free and position-conditional clean-head detectors."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import (  # noqa: E402
    compute_metric_bundle,
    roc_auc,
    within_bin_auc,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--features_glob",
        default="experiments/position_free_detector/clean_per_head_features_shard*.npz",
    )
    p.add_argument(
        "--baseline_scores",
        default="detection/baselines/results/coco_llava_7b_baselines/baseline_scores.npz",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_npz", required=True)
    return p.parse_args()


def _load_features(pattern: str) -> dict[str, np.ndarray]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    parts = [np.load(path, allow_pickle=True) for path in paths]
    fixed = {"layer_indices", "feature_names"}
    arrays: dict[str, np.ndarray] = {}
    for key in parts[0].files:
        if key in fixed or np.asarray(parts[0][key]).ndim == 0:
            arrays[key] = np.asarray(parts[0][key])
        else:
            arrays[key] = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
    order = np.argsort(arrays["object_id"])
    for key, value in list(arrays.items()):
        if key not in fixed and value.ndim > 0 and value.shape[0] == order.shape[0]:
            arrays[key] = value[order]
    if np.unique(arrays["object_id"]).size != arrays["object_id"].size:
        raise ValueError("Duplicate object IDs across feature shards")
    return arrays


def _image_folds(image_ids: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    unique_images = np.asarray(sorted(set(image_ids.tolist())), dtype=np.int64)
    rng.shuffle(unique_images)
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(unique_images, folds)]


def _empirical_cdf(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    train = np.sort(np.asarray(train, dtype=np.float64))
    if train.size == 0:
        return np.full_like(values, 0.5, dtype=np.float64)
    return np.searchsorted(train, values, side="right") / train.size


def _unsupervised_score(
    train_features: np.ndarray,
    values: np.ndarray,
    top_heads: int,
    local_gain: bool,
    aggregate: str,
) -> np.ndarray:
    heads = values.shape[1]
    evidence = np.empty((values.shape[0], heads), dtype=np.float64)
    for head in range(heads):
        if local_gain:
            train_a = train_features[:, head, 4]
            value_a = values[:, head, 4]
            train_b = np.maximum(0.0, train_features[:, head, 6])
            value_b = np.maximum(0.0, values[:, head, 6])
        else:
            train_a = -train_features[:, head, 0]
            value_a = -values[:, head, 0]
            train_b = train_features[:, head, 2]
            value_b = values[:, head, 2]
        evidence[:, head] = 0.5 * _empirical_cdf(train_a, value_a)
        evidence[:, head] += 0.5 * _empirical_cdf(train_b, value_b)
    if aggregate == "mean":
        grounding = evidence.mean(axis=1)
    elif aggregate == "top":
        top = np.partition(evidence, heads - top_heads, axis=1)[:, -top_heads:]
        grounding = top.mean(axis=1)
    else:
        raise ValueError(aggregate)
    return -grounding


def _bin_means(
    train_scores: np.ndarray,
    train_positions: np.ndarray,
    bin_width: int,
) -> dict[int, float]:
    result: dict[int, float] = {}
    bins = np.floor_divide(train_positions, bin_width)
    for bin_id in np.unique(bins):
        result[int(bin_id)] = float(train_scores[bins == bin_id].mean())
    return result


def _apply_bin_residual(
    scores: np.ndarray,
    positions: np.ndarray,
    means: dict[int, float],
    bin_width: int,
) -> np.ndarray:
    fallback = float(np.mean(list(means.values())))
    return np.asarray(
        [score - means.get(int(pos // bin_width), fallback) for score, pos in zip(scores, positions)],
        dtype=np.float64,
    )


def _oof_unsupervised(
    features: np.ndarray,
    image_ids: np.ndarray,
    positions: np.ndarray,
    fold_images: list[np.ndarray],
    top_heads: int = 8,
    local_gain: bool = True,
    aggregate: str = "top",
    bin_width: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.full(features.shape[0], np.nan, dtype=np.float64)
    residuals = np.full(features.shape[0], np.nan, dtype=np.float64)
    for test_images in fold_images:
        test = np.isin(image_ids, test_images)
        train = ~test
        test_scores = _unsupervised_score(
            features[train], features[test], top_heads, local_gain, aggregate
        )
        train_scores = _unsupervised_score(
            features[train], features[train], top_heads, local_gain, aggregate
        )
        scores[test] = test_scores
        residuals[test] = _apply_bin_residual(
            test_scores,
            positions[test],
            _bin_means(train_scores, positions[train], bin_width),
            bin_width,
        )
    return scores, residuals


def _matched_difference_rows(
    X: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    image_ids: np.ndarray,
    bin_width: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.RandomState(seed)
    differences = []
    bins = np.floor_divide(positions, bin_width)
    pair_count = 0
    for bin_id in np.unique(bins):
        in_bin = bins == bin_id
        hallu = np.where(in_bin & (labels == 1))[0]
        grounded = np.where(in_bin & (labels == 0))[0]
        if hallu.size == 0 or grounded.size == 0:
            continue
        hallu = hallu[rng.permutation(hallu.size)]
        usage = np.zeros(grounded.size, dtype=np.int32)
        tie_noise = rng.uniform(0.0, 1e-6, size=grounded.size)
        for hi in hallu:
            eligible = image_ids[grounded] != image_ids[hi]
            if not eligible.any():
                continue
            cost = np.abs(positions[grounded] - positions[hi]).astype(np.float64)
            cost += usage * (bin_width + 1.0) + tie_noise
            cost[~eligible] = np.inf
            gi_local = int(np.argmin(cost))
            gi = grounded[gi_local]
            usage[gi_local] += 1
            differences.append(X[hi] - X[gi])
            pair_count += 1
    if not differences:
        raise RuntimeError("No matched within-position training pairs")
    positive = np.stack(differences, axis=0)
    D = np.concatenate([positive, -positive], axis=0)
    y = np.concatenate(
        [np.ones(pair_count, dtype=np.int32), np.zeros(pair_count, dtype=np.int32)]
    )
    return D, y, pair_count


def _oof_conditional_probe(
    features: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    positions: np.ndarray,
    fold_images: list[np.ndarray],
    bin_width: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X = features.reshape(features.shape[0], -1).astype(np.float64)
    scores = np.full(X.shape[0], np.nan, dtype=np.float64)
    residuals = np.full(X.shape[0], np.nan, dtype=np.float64)
    fold_pairs: list[int] = []
    for fold_idx, test_images in enumerate(fold_images):
        test = np.isin(image_ids, test_images)
        train = ~test
        mean = X[train].mean(axis=0)
        std = X[train].std(axis=0)
        std[std < 1e-8] = 1.0
        X_train = (X[train] - mean) / std
        X_test = (X[test] - mean) / std
        D, pair_labels, pair_count = _matched_difference_rows(
            X_train,
            labels[train],
            positions[train],
            image_ids[train],
            bin_width,
            seed + fold_idx,
        )
        classifier = LogisticRegression(
            penalty="l2",
            C=1.0,
            fit_intercept=False,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )
        classifier.fit(D, pair_labels)
        train_scores = X_train @ classifier.coef_[0]
        test_scores = X_test @ classifier.coef_[0]
        scores[test] = test_scores
        residuals[test] = _apply_bin_residual(
            test_scores,
            positions[test],
            _bin_means(train_scores, positions[train], bin_width),
            bin_width,
        )
        fold_pairs.append(pair_count)
    return scores, residuals, fold_pairs


def _oof_baseline_residual(
    scores: np.ndarray,
    positions: np.ndarray,
    image_ids: np.ndarray,
    fold_images: list[np.ndarray],
    bin_width: int,
) -> np.ndarray:
    residuals = np.full(scores.shape[0], np.nan, dtype=np.float64)
    for test_images in fold_images:
        test = np.isin(image_ids, test_images)
        train = ~test
        residuals[test] = _apply_bin_residual(
            scores[test],
            positions[test],
            _bin_means(scores[train], positions[train], bin_width),
            bin_width,
        )
    return residuals


def _metric_bundle(
    labels: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    positions: np.ndarray,
    bin_width: int,
) -> dict:
    metrics = compute_metric_bundle(labels, scores, positions, bin_width=bin_width)
    metrics["residual_auroc"] = roc_auc(labels, residuals)
    valid_bins = [item for item in metrics["within_bins"] if item["auc"] is not None]
    metrics["valid_position_bins"] = len(valid_bins)
    metrics["fraction_bins_above_055"] = (
        float(np.mean([item["auc"] > 0.55 for item in valid_bins])) if valid_bins else None
    )
    metrics["spearman_position"] = float(spearmanr(scores, positions).statistic)
    grounded = labels == 0
    metrics["spearman_position_grounded"] = float(
        spearmanr(scores[grounded], positions[grounded]).statistic
    )
    return metrics


def _bootstrap_difference(
    labels: np.ndarray,
    positions: np.ndarray,
    image_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    iterations: int,
    seed: int,
    bin_width: int,
) -> dict:
    if iterations <= 0:
        return {"iterations": 0, "low": None, "median": None, "high": None}
    rng = np.random.RandomState(seed)
    unique_images = np.asarray(sorted(set(image_ids.tolist())))
    by_image = {image_id: np.where(image_ids == image_id)[0] for image_id in unique_images}
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(unique_images, size=unique_images.size, replace=True)
        idx = np.concatenate([by_image[image_id] for image_id in sampled])
        left_auc = within_bin_auc(
            labels[idx], left[idx], positions[idx], bin_width=bin_width
        )["auc"]
        right_auc = within_bin_auc(
            labels[idx], right[idx], positions[idx], bin_width=bin_width
        )["auc"]
        if left_auc is not None and right_auc is not None:
            differences.append(left_auc - right_auc)
    if not differences:
        return {"iterations": 0, "low": None, "median": None, "high": None}
    low, median, high = np.percentile(differences, [2.5, 50.0, 97.5])
    return {
        "iterations": len(differences),
        "low": float(low),
        "median": float(median),
        "high": float(high),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    args = parse_args()
    data = _load_features(args.features_glob)
    labels = data["label"].astype(np.int32)
    positions = data["gen_pos"].astype(np.int32)
    image_ids = data["image_id"].astype(np.int64)
    layer_indices = [int(x) for x in data["layer_indices"]]
    feature_names = [str(x) for x in data["feature_names"]]
    if feature_names != [
        "entropy_obj",
        "top1_obj",
        "top5_obj",
        "max_over_mean_obj",
        "jsd_obj_local",
        "delta_top1",
        "delta_top5",
        "delta_entropy",
    ]:
        raise ValueError(f"Unexpected feature schema: {feature_names}")
    layer_lookup = {layer: idx for idx, layer in enumerate(layer_indices)}
    if 22 not in layer_lookup or 31 not in layer_lookup:
        raise ValueError(f"Expected layers 22 and 31, found {layer_indices}")

    fold_images = _image_folds(image_ids, args.folds, args.seed)
    scores: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    fold_pair_counts: dict[str, list[int]] = {}

    unsupervised_specs = {
        "unsup_raw_l31_top8_local": ("raw", 31, True, "top"),
        "unsup_no_rope_l31_top8_local": ("no_rope", 31, True, "top"),
        "unsup_clean_l31_top8_local": ("clean", 31, True, "top"),
        "unsup_clean_l31_mean_local": ("clean", 31, True, "mean"),
        "unsup_clean_l31_top8_object": ("clean", 31, False, "top"),
        "unsup_clean_l22_top8_local": ("clean", 22, True, "top"),
    }
    for name, (branch, layer, local_gain, aggregate) in unsupervised_specs.items():
        scores[name], residuals[name] = _oof_unsupervised(
            data[branch][:, layer_lookup[layer]],
            image_ids,
            positions,
            fold_images,
            top_heads=8,
            local_gain=local_gain,
            aggregate=aggregate,
            bin_width=args.bin_width,
        )

    supervised_specs = {
        "probe_raw_l31_all": ("raw", slice(None)),
        "probe_no_rope_l31_all": ("no_rope", slice(None)),
        "probe_clean_l31_all": ("clean", slice(None)),
        "probe_clean_l31_object": ("clean", slice(0, 4)),
    }
    for name, (branch, feature_slice) in supervised_specs.items():
        features = data[branch][:, layer_lookup[31], :, feature_slice]
        scores[name], residuals[name], fold_pair_counts[name] = _oof_conditional_probe(
            features,
            labels,
            image_ids,
            positions,
            fold_images,
            args.bin_width,
            args.seed,
        )

    baseline = np.load(args.baseline_scores)
    object_ids = data["object_id"].astype(np.int64)
    if object_ids.min(initial=0) < 0 or object_ids.max(initial=-1) >= baseline["labels"].size:
        raise ValueError("Feature object IDs are outside the baseline score cache")
    if not np.array_equal(baseline["labels"][object_ids].astype(np.int32), labels):
        raise ValueError("Feature labels do not align with the baseline score cache")
    if not np.array_equal(baseline["gen_pos"][object_ids].astype(np.int32), positions):
        raise ValueError("Feature positions do not align with the baseline score cache")
    baseline_keys = {
        "baseline_ic": "ic_hallu_score",
        "baseline_entropy": "entropy_hallu_score",
        "baseline_nll": "nll_hallu_score",
        "baseline_position": "gen_pos",
    }
    for name, key in baseline_keys.items():
        scores[name] = np.asarray(baseline[key][object_ids], dtype=np.float64)
        residuals[name] = _oof_baseline_residual(
            scores[name], positions, image_ids, fold_images, args.bin_width
        )

    metrics = {
        name: _metric_bundle(labels, value, residuals[name], positions, args.bin_width)
        for name, value in scores.items()
    }
    intervals = {
        "unsup_clean_minus_ic": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["unsup_clean_l31_top8_local"],
            scores["baseline_ic"],
            args.bootstrap,
            args.seed,
            args.bin_width,
        ),
        "unsup_clean_minus_raw": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["unsup_clean_l31_top8_local"],
            scores["unsup_raw_l31_top8_local"],
            args.bootstrap,
            args.seed + 1,
            args.bin_width,
        ),
        "unsup_clean_minus_mean": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["unsup_clean_l31_top8_local"],
            scores["unsup_clean_l31_mean_local"],
            args.bootstrap,
            args.seed + 2,
            args.bin_width,
        ),
        "probe_clean_minus_raw": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["probe_clean_l31_all"],
            scores["probe_raw_l31_all"],
            args.bootstrap,
            args.seed + 3,
            args.bin_width,
        ),
        "probe_clean_minus_object": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["probe_clean_l31_all"],
            scores["probe_clean_l31_object"],
            args.bootstrap,
            args.seed + 4,
            args.bin_width,
        ),
    }

    unsup = metrics["unsup_clean_l31_top8_local"]
    ic = metrics["baseline_ic"]
    unsup_gate = {
        "within_at_least_070": unsup["within_bin_auroc"] >= 0.70,
        "beats_ic_by_001": unsup["within_bin_auroc"] >= ic["within_bin_auroc"] + 0.01,
        "beats_ic_matched": unsup["matched_pair_auroc"] > ic["matched_pair_auroc"],
        "beats_ic_residual": unsup["residual_auroc"] > ic["residual_auroc"],
        "beats_raw_by_001": unsup["within_bin_auroc"]
        >= metrics["unsup_raw_l31_top8_local"]["within_bin_auroc"] + 0.01,
        "beats_mean_by_001": unsup["within_bin_auroc"]
        >= metrics["unsup_clean_l31_mean_local"]["within_bin_auroc"] + 0.01,
        "no_rope_positive": metrics["unsup_no_rope_l31_top8_local"]["within_bin_auroc"]
        > metrics["unsup_raw_l31_top8_local"]["within_bin_auroc"],
        "sink_positive": unsup["within_bin_auroc"]
        > metrics["unsup_no_rope_l31_top8_local"]["within_bin_auroc"],
    }
    unsup_gate["pass"] = all(unsup_gate.values())

    probe = metrics["probe_clean_l31_all"]
    probe_gate = {
        "within_at_least_075": probe["within_bin_auroc"] >= 0.75,
        "beats_frozen_raw_probe_by_002": probe["within_bin_auroc"] >= 0.746,
        "beats_matched_raw_by_002": probe["within_bin_auroc"]
        >= metrics["probe_raw_l31_all"]["within_bin_auroc"] + 0.02,
        "matched_at_least_072": probe["matched_pair_auroc"] >= 0.72,
        "residual_at_least_068": probe["residual_auroc"] >= 0.68,
        "beats_object_only": probe["within_bin_auroc"]
        > metrics["probe_clean_l31_object"]["within_bin_auroc"],
        "no_rope_positive": metrics["probe_no_rope_l31_all"]["within_bin_auroc"]
        > metrics["probe_raw_l31_all"]["within_bin_auroc"],
        "sink_positive": probe["within_bin_auroc"]
        > metrics["probe_no_rope_l31_all"]["within_bin_auroc"],
    }
    probe_gate["pass"] = all(probe_gate.values())

    payload = {
        "protocol": "docs/iteration_protocols/position_free_detector_iteration_1.md",
        "feature_glob": args.features_glob,
        "num_mentions": int(labels.size),
        "num_images": int(np.unique(image_ids).size),
        "hallucinated": int(labels.sum()),
        "grounded": int(labels.size - labels.sum()),
        "layers": layer_indices,
        "feature_names": feature_names,
        "folds": args.folds,
        "seed": args.seed,
        "bin_width": args.bin_width,
        "fold_pair_counts": fold_pair_counts,
        "metrics": metrics,
        "bootstrap_within_bin_differences": intervals,
        "gates": {"unsupervised": unsup_gate, "supervised": probe_gate},
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        object_id=object_ids,
        image_id=image_ids,
        gen_pos=positions,
        label=labels,
        **{f"score__{name}": value for name, value in scores.items()},
        **{f"residual__{name}": value for name, value in residuals.items()},
    )

    print(f"N={labels.size} images={np.unique(image_ids).size}")
    for name in [
        "baseline_ic",
        "unsup_raw_l31_top8_local",
        "unsup_no_rope_l31_top8_local",
        "unsup_clean_l31_top8_local",
        "unsup_clean_l31_mean_local",
        "probe_raw_l31_all",
        "probe_no_rope_l31_all",
        "probe_clean_l31_all",
        "probe_clean_l31_object",
    ]:
        metric = metrics[name]
        print(
            f"{name:<36} overall={metric['overall_auroc']:.4f} "
            f"within={metric['within_bin_auroc']:.4f} "
            f"matched={metric['matched_pair_auroc']:.4f} "
            f"residual={metric['residual_auroc']:.4f}"
        )
    print(f"unsupervised gate: {unsup_gate['pass']}")
    print(f"supervised gate:   {probe_gate['pass']}")
    print(f"Wrote {output_json} and {output_npz}")


if __name__ == "__main__":
    main()
