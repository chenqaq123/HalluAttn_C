#!/usr/bin/env python3
"""Evaluate Visual Value Evidence with image-grouped, position-controlled OOF."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import compute_metric_bundle, roc_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_glob",
        default="experiments/visual_value_evidence/features_shard*.npz",
    )
    parser.add_argument(
        "--attention_features_glob",
        default="experiments/position_free_detector/clean_per_head_features_shard*.npz",
    )
    parser.add_argument(
        "--baseline_scores",
        default="detection/baselines/results/coco_llava_7b_baselines/baseline_scores.npz",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--category_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_npz", required=True)
    return parser.parse_args()


def _load_npz_parts(pattern: str) -> dict[str, np.ndarray]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    parts = [np.load(path, allow_pickle=True) for path in paths]
    fixed = {
        "layer_indices",
        "step_names",
        "per_head_feature_names",
        "joint_feature_names",
        "confidence_feature_names",
        "feature_names",
    }
    arrays: dict[str, np.ndarray] = {}
    for key in parts[0].files:
        value = np.asarray(parts[0][key])
        if key in fixed or value.ndim == 0:
            arrays[key] = value
        elif all(key in part.files for part in parts):
            arrays[key] = np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
    order = np.argsort(arrays["object_id"])
    for key, value in list(arrays.items()):
        if key not in fixed and value.ndim > 0 and value.shape[0] == order.size:
            arrays[key] = value[order]
    if np.unique(arrays["object_id"]).size != arrays["object_id"].size:
        raise ValueError(f"Duplicate object IDs in {pattern}")
    return arrays


def _image_folds(image_ids: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    images = np.asarray(sorted(set(image_ids.tolist())), dtype=np.int64)
    rng.shuffle(images)
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(images, folds)]


def _category_fold_masks(words: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    categories = np.asarray(sorted(set(words.tolist())))
    rng.shuffle(categories)
    return [np.isin(words, part) for part in np.array_split(categories, folds)]


def _bin_means(scores: np.ndarray, positions: np.ndarray, width: int) -> dict[int, float]:
    means = {}
    for bin_id in np.unique(positions // width):
        mask = positions // width == bin_id
        means[int(bin_id)] = float(scores[mask].mean())
    return means


def _apply_bin_means(
    scores: np.ndarray, positions: np.ndarray, means: dict[int, float], width: int
) -> np.ndarray:
    fallback = float(np.mean(list(means.values())))
    return np.asarray(
        [score - means.get(int(pos // width), fallback) for score, pos in zip(scores, positions)],
        dtype=np.float64,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _conditional_category_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    positions: np.ndarray,
    words: np.ndarray,
    width: int,
) -> dict:
    aucs = []
    weights = []
    groups = 0
    for bin_id in np.unique(positions // width):
        for word in np.unique(words[positions // width == bin_id]):
            mask = (positions // width == bin_id) & (words == word)
            if mask.sum() < 2 or np.unique(labels[mask]).size < 2:
                continue
            value = roc_auc(labels[mask], scores[mask])
            if value is not None:
                aucs.append(value)
                weights.append(int(mask.sum()))
                groups += 1
    return {
        "auc": float(np.average(aucs, weights=weights)) if aucs else None,
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "groups": groups,
        "mentions": int(np.sum(weights)),
    }


def _metric_bundle(
    labels: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    positions: np.ndarray,
    words: np.ndarray,
    width: int,
) -> dict:
    result = compute_metric_bundle(labels, scores, positions, bin_width=width)
    result["residual_auroc"] = roc_auc(labels, residuals)
    valid_bins = [item for item in result["within_bins"] if item["auc"] is not None]
    result["macro_bin_auroc"] = (
        float(np.mean([item["auc"] for item in valid_bins])) if valid_bins else None
    )
    grounded = labels == 0
    result["spearman_position"] = _spearman(scores, positions)
    result["spearman_position_grounded"] = _spearman(
        scores[grounded], positions[grounded]
    )
    result["position_category_control"] = _conditional_category_auc(
        labels, scores, positions, words, width
    )
    return result


def _label_free_oof(
    joint: np.ndarray,
    layer_indices: np.ndarray,
    image_ids: np.ndarray,
    positions: np.ndarray,
    folds: list[np.ndarray],
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    gain_index = 4
    selected = np.asarray(layer_indices) >= 20
    values = joint[:, selected, 0, gain_index].astype(np.float64)
    scores = np.full(values.shape[0], np.nan, dtype=np.float64)
    residuals = np.full_like(scores, np.nan)
    for test_images in folds:
        test = np.isin(image_ids, test_images)
        train = ~test
        median = np.median(values[train], axis=0)
        mad = np.median(np.abs(values[train] - median), axis=0)
        scale = 1.4826 * mad
        scale[scale < 1e-8] = 1.0
        train_scores = -np.median((values[train] - median) / scale, axis=1)
        test_scores = -np.median((values[test] - median) / scale, axis=1)
        scores[test] = test_scores
        residuals[test] = _apply_bin_means(
            test_scores,
            positions[test],
            _bin_means(train_scores, positions[train], width),
            width,
        )
    return scores, residuals


def _conditional_weights(labels: np.ndarray, positions: np.ndarray, width: int) -> np.ndarray:
    weights = np.zeros(labels.size, dtype=np.float32)
    bins = positions // width
    for bin_id in np.unique(bins):
        in_bin = bins == bin_id
        for label in (0, 1):
            mask = in_bin & (labels == label)
            count = int(mask.sum())
            if count:
                weights[mask] = float(in_bin.sum()) / (2.0 * count)
    positive = weights > 0
    weights[positive] /= weights[positive].mean()
    return weights


def _feature_moments(X: np.ndarray, indices: np.ndarray, batch_size: int = 512):
    total = np.zeros(X.shape[1], dtype=np.float64)
    total_sq = np.zeros_like(total)
    for start in range(0, indices.size, batch_size):
        batch = X[indices[start : start + batch_size]].astype(np.float64)
        total += batch.sum(axis=0)
        total_sq += np.square(batch).sum(axis=0)
    mean = total / indices.size
    variance = np.maximum(total_sq / indices.size - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class _IndexedDataset(Dataset):
    def __init__(self, X, labels, weights, indices):
        self.X = X
        self.labels = labels
        self.weights = weights
        self.indices = indices

    def __len__(self):
        return self.indices.size

    def __getitem__(self, item):
        index = self.indices[item]
        return self.X[index], self.labels[index], self.weights[index]


class _Probe(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _predict(model, X, indices, mean, std, device, batch_size):
    output = np.empty(indices.size, dtype=np.float64)
    model.eval()
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    with torch.inference_mode():
        for start in range(0, indices.size, batch_size):
            batch_indices = indices[start : start + batch_size]
            features = torch.from_numpy(X[batch_indices]).to(device)
            features = (features - mean_tensor) / std_tensor
            output[start : start + batch_indices.size] = (
                model(features).float().cpu().numpy()
            )
    return output


def _mlp_oof(
    X: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    test_masks: list[np.ndarray],
    args: argparse.Namespace,
    model_seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    residuals = np.full_like(scores, np.nan)
    device = torch.device(args.device)
    for fold_index, test in enumerate(test_masks):
        test = np.asarray(test, dtype=bool)
        train = ~test
        train_indices = np.where(train)[0]
        test_indices = np.where(test)[0]
        mean, std = _feature_moments(X, train_indices)
        weights = _conditional_weights(labels[train], positions[train], args.bin_width)
        full_weights = np.zeros(labels.size, dtype=np.float32)
        full_weights[train] = weights
        dataset = _IndexedDataset(X, labels.astype(np.float32), full_weights, train_indices)
        generator = torch.Generator().manual_seed(args.seed + model_seed_offset + fold_index)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        torch.manual_seed(args.seed + model_seed_offset + fold_index)
        model = _Probe(X.shape[1], args.hidden_size).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        mean_tensor = torch.from_numpy(mean).to(device)
        std_tensor = torch.from_numpy(std).to(device)
        model.train()
        for _ in range(args.epochs):
            for features, targets, sample_weights in loader:
                features = features.to(device)
                targets = targets.to(device)
                sample_weights = sample_weights.to(device)
                features = (features - mean_tensor) / std_tensor
                logits = model(features)
                losses = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                loss = (losses * sample_weights).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        train_scores = _predict(
            model, X, train_indices, mean, std, device, args.batch_size * 4
        )
        test_scores = _predict(
            model, X, test_indices, mean, std, device, args.batch_size * 4
        )
        scores[test_indices] = test_scores
        residuals[test_indices] = _apply_bin_means(
            test_scores,
            positions[test_indices],
            _bin_means(train_scores, positions[train_indices], args.bin_width),
            args.bin_width,
        )
        del model, optimizer, loader, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores, residuals


def _bootstrap_difference(
    labels: np.ndarray,
    positions: np.ndarray,
    image_ids: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    width: int,
    iterations: int,
    seed: int,
) -> dict:
    rng = np.random.RandomState(seed)
    unique_images = np.asarray(sorted(set(image_ids.tolist())))
    by_image = {image_id: np.where(image_ids == image_id)[0] for image_id in unique_images}
    differences = []
    for _ in range(iterations):
        sampled = rng.choice(unique_images, size=unique_images.size, replace=True)
        indices = np.concatenate([by_image[image_id] for image_id in sampled])
        left_metric = compute_metric_bundle(
            labels[indices], left[indices], positions[indices], bin_width=width
        )["within_bin_auroc"]
        right_metric = compute_metric_bundle(
            labels[indices], right[indices], positions[indices], bin_width=width
        )["within_bin_auroc"]
        if left_metric is not None and right_metric is not None:
            differences.append(left_metric - right_metric)
    if not differences:
        return {
            "iterations": 0,
            "low": None,
            "median": None,
            "high": None,
        }
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


def _format_metric(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    data = _load_npz_parts(args.features_glob)
    labels = data["label"].astype(np.int32)
    positions = data["gen_pos"].astype(np.int32)
    image_ids = data["image_id"].astype(np.int64)
    words = data["word"].astype(str)
    object_ids = data["object_id"].astype(np.int64)
    if list(data["per_head_feature_names"].astype(str)) != [
        "visual_mass",
        "value_norm",
        "target_dot",
        "target_cosine",
        "target_gain",
    ]:
        raise ValueError("Unexpected VVE per-head schema")
    folds = _image_folds(image_ids, args.folds, args.seed)
    image_test_masks = [np.isin(image_ids, test_images) for test_images in folds]

    confidence = data["confidence"].reshape(labels.size, -1).astype(np.float32)
    no_rope = data["no_rope"].reshape(labels.size, -1).astype(np.float32)
    raw = data["raw"].reshape(labels.size, -1).astype(np.float32)
    gain_only = data["no_rope"][..., 4].reshape(labels.size, -1).astype(np.float32)
    no_rope_pre = data["no_rope"][:, :, 0].reshape(labels.size, -1).astype(np.float32)
    no_rope_post = data["no_rope"][:, :, 1].reshape(labels.size, -1).astype(np.float32)
    gain_pre = data["no_rope"][:, :, 0, :, 4].reshape(labels.size, -1).astype(np.float32)
    gain_post = data["no_rope"][:, :, 1, :, 4].reshape(labels.size, -1).astype(np.float32)
    feature_sets = {
        "probe_confidence": confidence,
        "probe_vve_no_rope_gain": gain_only,
        "probe_vve_no_rope_gain_pre": gain_pre,
        "probe_vve_no_rope_gain_post": gain_post,
        "probe_vve_no_rope": no_rope,
        "probe_vve_no_rope_pre": no_rope_pre,
        "probe_vve_no_rope_post": no_rope_post,
        "probe_vve_raw": raw,
        "probe_vve_no_rope_confidence": np.concatenate([no_rope, confidence], axis=1),
        "probe_vve_raw_confidence": np.concatenate([raw, confidence], axis=1),
    }

    attention_paths = sorted(glob.glob(args.attention_features_glob))
    if attention_paths:
        attention = _load_npz_parts(args.attention_features_glob)
        lookup = {int(object_id): index for index, object_id in enumerate(attention["object_id"])}
        indices = np.asarray([lookup[int(object_id)] for object_id in object_ids], dtype=np.int64)
        if not np.array_equal(attention["label"][indices].astype(np.int32), labels):
            raise ValueError("Attention cache labels do not align with VVE cache")
        attention_raw = attention["raw"][indices].reshape(labels.size, -1).astype(np.float32)
        feature_sets["probe_attention_confidence"] = np.concatenate(
            [attention_raw, confidence], axis=1
        )

    scores = {}
    residuals = {}
    scores["unsup_vve_no_rope"], residuals["unsup_vve_no_rope"] = _label_free_oof(
        data["joint_no_rope"],
        data["layer_indices"],
        image_ids,
        positions,
        folds,
        args.bin_width,
    )
    for spec_index, (name, features) in enumerate(feature_sets.items()):
        scores[name], residuals[name] = _mlp_oof(
            features,
            labels,
            positions,
            image_test_masks,
            args,
            model_seed_offset=100 * (spec_index + 1),
        )

    category_masks = _category_fold_masks(words, args.category_folds, args.seed + 1000)
    category_specs = {
        "category_oof_confidence": confidence,
        "category_oof_vve_no_rope_gain": gain_only,
        "category_oof_vve_no_rope": no_rope,
    }
    if "probe_attention_confidence" in feature_sets:
        category_specs["category_oof_attention_confidence"] = feature_sets[
            "probe_attention_confidence"
        ]
    for spec_index, (name, features) in enumerate(category_specs.items()):
        scores[name], residuals[name] = _mlp_oof(
            features,
            labels,
            positions,
            category_masks,
            args,
            model_seed_offset=5000 + 100 * spec_index,
        )

    baseline = np.load(args.baseline_scores)
    if object_ids.min(initial=0) < 0 or object_ids.max(initial=-1) >= baseline["labels"].size:
        raise ValueError("VVE object IDs are outside the baseline cache")
    if not np.array_equal(baseline["labels"][object_ids].astype(np.int32), labels):
        raise ValueError("Baseline labels do not align with VVE cache")
    scores["baseline_ic"] = baseline["ic_hallu_score"][object_ids].astype(np.float64)
    residuals["baseline_ic"] = np.full(labels.size, np.nan, dtype=np.float64)
    for test_images in folds:
        test = np.isin(image_ids, test_images)
        train = ~test
        residuals["baseline_ic"][test] = _apply_bin_means(
            scores["baseline_ic"][test],
            positions[test],
            _bin_means(scores["baseline_ic"][train], positions[train], args.bin_width),
            args.bin_width,
        )

    metrics = {
        name: _metric_bundle(
            labels, value, residuals[name], positions, words, args.bin_width
        )
        for name, value in scores.items()
    }
    intervals = {
        "vve_no_rope_minus_raw": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["probe_vve_no_rope"],
            scores["probe_vve_raw"],
            args.bin_width,
            500,
            args.seed + 1,
        ),
        "vve_no_rope_minus_confidence": _bootstrap_difference(
            labels,
            positions,
            image_ids,
            scores["probe_vve_no_rope"],
            scores["probe_confidence"],
            args.bin_width,
            500,
            args.seed + 2,
        ),
    }
    if "probe_attention_confidence" in scores:
        intervals["vve_no_rope_minus_attention_confidence"] = (
            _bootstrap_difference(
                labels,
                positions,
                image_ids,
                scores["probe_vve_no_rope"],
                scores["probe_attention_confidence"],
                args.bin_width,
                500,
                args.seed,
            )
        )
    if "category_oof_attention_confidence" in scores:
        intervals["category_oof_vve_minus_attention_confidence"] = (
            _bootstrap_difference(
                labels,
                positions,
                image_ids,
                scores["category_oof_vve_no_rope"],
                scores["category_oof_attention_confidence"],
                args.bin_width,
                500,
                args.seed + 3,
            )
        )
    output = {
        "protocol": "docs/iteration_protocols/visual_value_evidence_iteration_1.md",
        "n": int(labels.size),
        "images": int(np.unique(image_ids).size),
        "hallucinated": int(labels.sum()),
        "folds": args.folds,
        "category_folds": args.category_folds,
        "seed": args.seed,
        "bin_width": args.bin_width,
        "epochs": args.epochs,
        "metrics": metrics,
        "bootstrap_within_bin_differences": intervals,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(output), indent=2) + "\n")
    npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "object_id": object_ids,
        "image_id": image_ids,
        "gen_pos": positions,
        "label": labels,
    }
    arrays.update({f"score__{name}": value for name, value in scores.items()})
    arrays.update({f"residual__{name}": value for name, value in residuals.items()})
    np.savez_compressed(npz_path, **arrays)
    for name, metric in metrics.items():
        print(
            f"{name:36s} overall={_format_metric(metric['overall_auroc'])} "
            f"within={_format_metric(metric['within_bin_auroc'])} "
            f"matched={_format_metric(metric['matched_pair_auroc'])} "
            f"residual={_format_metric(metric['residual_auroc'])}"
        )


if __name__ == "__main__":
    main()
