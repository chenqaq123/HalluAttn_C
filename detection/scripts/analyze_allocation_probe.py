#!/usr/bin/env python3
"""Mechanism audit for the all-layer No-RoPE visual-allocation probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import roc_auc, within_bin_auc  # noqa: E402
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _conditional_weights,
    _feature_moments,
    _image_folds,
    _IndexedDataset,
    _load_npz_parts,
    _Probe,
)

LAYERS = 32
ROWS = 2
HEADS = 32
PAIR_COUNT = LAYERS * HEADS
BANDS = ((0, 7), (8, 15), (16, 23), (24, 31))
SPARSE_K = (8, 16, 32, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_glob",
        default="experiments/visual_value_evidence/features_shard*.npz",
    )
    parser.add_argument(
        "--reference_oof",
        default=(
            "experiments/visual_value_evidence/"
            "iteration2_size_shared_seed_oof.npz"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_json",
        default="experiments/allocation_probe_audit/iteration1_metrics.json",
    )
    parser.add_argument(
        "--output_npz",
        default="experiments/allocation_probe_audit/iteration1_scores.npz",
    )
    return parser.parse_args()


def pair_feature_indices(pairs: np.ndarray) -> np.ndarray:
    indices = []
    for pair in np.asarray(pairs, dtype=np.int64):
        layer = int(pair // HEADS)
        head = int(pair % HEADS)
        indices.extend((layer * ROWS * HEADS + head, layer * ROWS * HEADS + HEADS + head))
    return np.asarray(indices, dtype=np.int64)


def _fit_model(
    X: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    model_seed: int,
) -> tuple[_Probe, np.ndarray, np.ndarray]:
    X = np.ascontiguousarray(X, dtype=np.float32)
    mean, std = _feature_moments(X, train_indices)
    fold_weights = _conditional_weights(
        labels[train_indices], positions[train_indices], args.bin_width, mode="size"
    )
    weights = np.zeros(labels.size, dtype=np.float32)
    weights[train_indices] = fold_weights
    fit_indices = train_indices[fold_weights > 0]
    dataset = _IndexedDataset(
        X, labels.astype(np.float32), weights, fit_indices
    )
    generator = torch.Generator().manual_seed(model_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    device = torch.device(args.device)
    torch.manual_seed(model_seed)
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
            logits = model((features - mean_tensor) / std_tensor)
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            )
            loss = (losses * sample_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model, mean, std


def _predict_standardized(
    model: _Probe,
    standardized: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(standardized.shape[0], dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, standardized.shape[0], batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(standardized[start : start + batch_size])
            ).to(device)
            output[start : start + batch.shape[0]] = (
                model(batch).float().cpu().numpy()
            )
    return output


def _heldout_local_effect(
    model: _Probe,
    standardized: np.ndarray,
    positions: np.ndarray,
    bin_width: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    total = np.zeros(standardized.shape[1], dtype=np.float64)
    count = 0
    bin_totals: dict[int, np.ndarray] = {}
    bin_counts: dict[int, int] = {}
    model.eval()
    for start in range(0, standardized.shape[0], batch_size):
        stop = min(start + batch_size, standardized.shape[0])
        batch = torch.from_numpy(np.ascontiguousarray(standardized[start:stop])).to(device)
        batch.requires_grad_(True)
        logits = model(batch)
        gradients = torch.autograd.grad(logits.sum(), batch)[0]
        effect = torch.abs(gradients * batch).detach().float().cpu().numpy()
        total += effect.sum(axis=0)
        count += effect.shape[0]
        batch_bins = positions[start:stop] // bin_width
        for bin_id in np.unique(batch_bins):
            mask = batch_bins == bin_id
            bin_totals.setdefault(
                int(bin_id), np.zeros(standardized.shape[1], dtype=np.float64)
            )
            bin_totals[int(bin_id)] += effect[mask].sum(axis=0)
            bin_counts[int(bin_id)] = bin_counts.get(int(bin_id), 0) + int(mask.sum())
    return total / count, {
        bin_id: values / bin_counts[bin_id] for bin_id, values in bin_totals.items()
    }


def _balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size < 2:
        return None
    prediction = scores >= 0.0
    tpr = float(np.mean(prediction[labels == 1]))
    tnr = float(np.mean(~prediction[labels == 0]))
    return 0.5 * (tpr + tnr)


def _score_metrics(
    labels: np.ndarray, positions: np.ndarray, scores: np.ndarray, bin_width: int
) -> dict:
    within = within_bin_auc(labels, scores, positions, bin_width=bin_width)
    bins = []
    for item in within["bins"]:
        mask = (positions >= item["bin_start"]) & (positions <= item["bin_end"])
        bins.append(
            {
                **item,
                "grounded": int(np.sum(mask & (labels == 0))),
                "balanced_accuracy_at_zero": _balanced_accuracy(
                    labels[mask], scores[mask]
                ),
            }
        )
    valid = [item for item in bins if item["auc"] is not None]
    return {
        "overall_auroc": roc_auc(labels, scores),
        "within_bin_auroc": within["auc"],
        "macro_bin_auroc": (
            float(np.mean([item["auc"] for item in valid])) if valid else None
        ),
        "bins": bins,
    }


def _rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.int32)
    ranks[order] = np.arange(1, order.size + 1)
    return ranks


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(_rank_descending(left), _rank_descending(right))[0, 1])


def _pairwise_stability(scores: np.ndarray, top_k: int) -> dict:
    correlations = []
    jaccards = []
    for left in range(scores.shape[0]):
        for right in range(left + 1, scores.shape[0]):
            correlations.append(_rank_correlation(scores[left], scores[right]))
            left_top = set(np.argsort(-scores[left])[:top_k].tolist())
            right_top = set(np.argsort(-scores[right])[:top_k].tolist())
            jaccards.append(len(left_top & right_top) / len(left_top | right_top))
    return {
        "mean_rank_spearman": float(np.mean(correlations)),
        "min_rank_spearman": float(np.min(correlations)),
        "mean_top_jaccard": float(np.mean(jaccards)),
        "min_top_jaccard": float(np.min(jaccards)),
    }


def _representation_oof(
    X: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    test_masks: list[np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    device = torch.device(args.device)
    for fold_index, test in enumerate(test_masks):
        train_indices = np.where(~test)[0]
        test_indices = np.where(test)[0]
        model, mean, std = _fit_model(
            X,
            labels,
            positions,
            train_indices,
            args,
            args.seed + fold_index,
        )
        standardized = (X[test_indices].astype(np.float32) - mean) / std
        scores[test_indices] = _predict_standardized(
            model, standardized, device, args.batch_size * 4
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    data = _load_npz_parts(args.features_glob)
    labels = data["label"].astype(np.int32)
    positions = data["gen_pos"].astype(np.int32)
    image_ids = data["image_id"].astype(np.int64)
    object_ids = data["object_id"].astype(np.int64)
    allocation = data["no_rope"][..., 0].astype(np.float32)
    if allocation.shape[1:] != (LAYERS, ROWS, HEADS):
        raise ValueError(f"Unexpected allocation shape: {allocation.shape}")
    X = np.ascontiguousarray(allocation.reshape(labels.size, -1))
    layer_mean = allocation.mean(axis=-1).reshape(labels.size, -1).astype(np.float32)
    head_deviation = (
        allocation - allocation.mean(axis=-1, keepdims=True)
    ).reshape(labels.size, -1).astype(np.float32)

    folds = _image_folds(image_ids, args.folds, args.seed)
    test_masks = [np.isin(image_ids, test_images) for test_images in folds]
    device = torch.device(args.device)
    full_scores = np.full(labels.size, np.nan, dtype=np.float64)
    layer_remove = np.full((LAYERS, labels.size), np.nan, dtype=np.float64)
    layer_keep = np.full_like(layer_remove, np.nan)
    row_remove = np.full((ROWS, labels.size), np.nan, dtype=np.float64)
    band_keep = np.full((len(BANDS), labels.size), np.nan, dtype=np.float64)
    weight_pair_scores = np.zeros((args.folds, PAIR_COUNT), dtype=np.float64)
    gradient_pair_scores = np.zeros_like(weight_pair_scores)
    gradient_pair_by_bin = np.zeros((args.folds, 15, PAIR_COUNT), dtype=np.float64)
    sparse_scores = {
        f"top_{k}": np.full(labels.size, np.nan, dtype=np.float64) for k in SPARSE_K
    }
    sparse_scores.update(
        {
            f"random_{k}": np.full(labels.size, np.nan, dtype=np.float64)
            for k in SPARSE_K
        }
    )
    selected_pairs: dict[str, list[list[int]]] = {
        f"top_{k}": [] for k in SPARSE_K
    }
    selected_pairs.update({f"random_{k}": [] for k in SPARSE_K})

    for fold_index, test in enumerate(test_masks):
        train_indices = np.where(~test)[0]
        test_indices = np.where(test)[0]
        model, mean, std = _fit_model(
            X,
            labels,
            positions,
            train_indices,
            args,
            args.seed + fold_index,
        )
        standardized = (X[test_indices] - mean) / std
        full_scores[test_indices] = _predict_standardized(
            model, standardized, device, args.batch_size * 4
        )

        first_weight = model.network[0].weight.detach().float().cpu().numpy()
        feature_norm = np.linalg.norm(first_weight, axis=0).reshape(
            LAYERS, ROWS, HEADS
        )
        weight_pair_scores[fold_index] = np.sqrt(
            np.square(feature_norm).sum(axis=1)
        ).reshape(-1)
        local_effect, effect_by_bin = _heldout_local_effect(
            model,
            standardized,
            positions[test_indices],
            args.bin_width,
            device,
            args.batch_size * 4,
        )
        gradient_pair_scores[fold_index] = np.sqrt(
            np.square(local_effect.reshape(LAYERS, ROWS, HEADS)).sum(axis=1)
        ).reshape(-1)
        for bin_id, values in effect_by_bin.items():
            if 0 <= bin_id < gradient_pair_by_bin.shape[1]:
                gradient_pair_by_bin[fold_index, bin_id] = np.sqrt(
                    np.square(values.reshape(LAYERS, ROWS, HEADS)).sum(axis=1)
                ).reshape(-1)

        for layer in range(LAYERS):
            start = layer * ROWS * HEADS
            stop = start + ROWS * HEADS
            removed = standardized.copy()
            removed[:, start:stop] = 0.0
            layer_remove[layer, test_indices] = _predict_standardized(
                model, removed, device, args.batch_size * 4
            )
            kept = np.zeros_like(standardized)
            kept[:, start:stop] = standardized[:, start:stop]
            layer_keep[layer, test_indices] = _predict_standardized(
                model, kept, device, args.batch_size * 4
            )

        shaped = standardized.reshape(-1, LAYERS, ROWS, HEADS)
        for row in range(ROWS):
            removed = shaped.copy()
            removed[:, :, row, :] = 0.0
            row_remove[row, test_indices] = _predict_standardized(
                model,
                removed.reshape(standardized.shape),
                device,
                args.batch_size * 4,
            )
        for band_index, (lo, hi) in enumerate(BANDS):
            kept = np.zeros_like(shaped)
            kept[:, lo : hi + 1] = shaped[:, lo : hi + 1]
            band_keep[band_index, test_indices] = _predict_standardized(
                model,
                kept.reshape(standardized.shape),
                device,
                args.batch_size * 4,
            )

        pair_order = np.argsort(-weight_pair_scores[fold_index])
        random_order = np.random.default_rng(1701 + fold_index).permutation(PAIR_COUNT)
        del standardized, shaped
        for prefix, order in (("top", pair_order), ("random", random_order)):
            for k in SPARSE_K:
                pairs = order[:k]
                selected_pairs[f"{prefix}_{k}"].append(pairs.tolist())
                feature_indices = pair_feature_indices(pairs)
                subset = np.ascontiguousarray(X[:, feature_indices])
                sparse_model, sparse_mean, sparse_std = _fit_model(
                    subset,
                    labels,
                    positions,
                    train_indices,
                    args,
                    args.seed + fold_index,
                )
                sparse_standardized = (
                    subset[test_indices] - sparse_mean
                ) / sparse_std
                sparse_scores[f"{prefix}_{k}"][test_indices] = _predict_standardized(
                    sparse_model,
                    sparse_standardized,
                    device,
                    args.batch_size * 4,
                )
                del subset, sparse_model, sparse_standardized
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"completed fold {fold_index + 1}/{args.folds}", flush=True)

    print("fitting fixed mean/deviation representations", flush=True)
    representation_scores = {
        "layer_mean": _representation_oof(
            layer_mean, labels, positions, test_masks, args
        ),
        "head_deviation": _representation_oof(
            head_deviation, labels, positions, test_masks, args
        ),
    }

    reference = np.load(args.reference_oof)
    if not np.array_equal(reference["object_id"], object_ids):
        raise ValueError("Reference OOF object IDs do not align")
    if not np.array_equal(reference["label"].astype(np.int32), labels):
        raise ValueError("Reference OOF labels do not align")
    if not np.array_equal(reference["gen_pos"].astype(np.int32), positions):
        raise ValueError("Reference OOF positions do not align")
    reference_scores = reference["score__probe_allocation_mass"]
    baseline_metrics = _score_metrics(labels, positions, full_scores, args.bin_width)
    reference_metrics = _score_metrics(labels, positions, reference_scores, args.bin_width)
    max_score_difference = float(np.max(np.abs(full_scores - reference_scores)))

    all_metrics = {"full": baseline_metrics}
    all_metrics.update(
        {
            name: _score_metrics(labels, positions, values, args.bin_width)
            for name, values in representation_scores.items()
        }
    )
    all_metrics.update(
        {
            name: _score_metrics(labels, positions, values, args.bin_width)
            for name, values in sparse_scores.items()
        }
    )
    layer_remove_metrics = [
        _score_metrics(labels, positions, values, args.bin_width)
        for values in layer_remove
    ]
    layer_keep_metrics = [
        _score_metrics(labels, positions, values, args.bin_width)
        for values in layer_keep
    ]
    row_remove_metrics = [
        _score_metrics(labels, positions, values, args.bin_width)
        for values in row_remove
    ]
    band_keep_metrics = [
        _score_metrics(labels, positions, values, args.bin_width)
        for values in band_keep
    ]

    weight_ranks = np.stack([_rank_descending(row) for row in weight_pair_scores])
    gradient_ranks = np.stack(
        [_rank_descending(row) for row in gradient_pair_scores]
    )
    top32_frequency = np.sum(weight_ranks <= 32, axis=0)
    mean_gradient_rank = gradient_ranks.mean(axis=0)
    stable_pairs = np.where((top32_frequency >= 4) & (mean_gradient_rank <= 256))[0]
    pair_table = []
    for pair in np.argsort(weight_ranks.mean(axis=0))[:64]:
        pair_table.append(
            {
                "layer": int(pair // HEADS),
                "head": int(pair % HEADS),
                "mean_weight_rank": float(weight_ranks[:, pair].mean()),
                "top32_fold_count": int(top32_frequency[pair]),
                "mean_gradient_rank": float(mean_gradient_rank[pair]),
                "stable": bool(pair in stable_pairs),
            }
        )

    bin_top_pairs = {}
    for bin_id in range(15):
        valid_fold = np.any(gradient_pair_by_bin[:, bin_id] > 0, axis=1)
        values = gradient_pair_by_bin[valid_fold, bin_id].mean(axis=0)
        bin_top_pairs[f"{bin_id * args.bin_width}-{(bin_id + 1) * args.bin_width - 1}"] = [
            {"layer": int(pair // HEADS), "head": int(pair % HEADS)}
            for pair in np.argsort(-values)[:10]
        ]

    stable_bin_keys = [
        f"{item['bin_start']}-{item['bin_end']}"
        for item in baseline_metrics["bins"]
        if min(item["hallucinated"], item["grounded"]) >= 30
    ]

    def bin_lookup(metrics: dict) -> dict[str, float]:
        return {
            f"{item['bin_start']}-{item['bin_end']}": item["auc"]
            for item in metrics["bins"]
            if item["auc"] is not None
        }

    baseline_bins = bin_lookup(baseline_metrics)

    def retention_count(metrics: dict, fraction: float) -> int:
        candidate = bin_lookup(metrics)
        return sum(
            candidate[key] - 0.5 >= fraction * (baseline_bins[key] - 0.5)
            for key in stable_bin_keys
        )

    layer_weight_scores = weight_pair_scores.reshape(args.folds, LAYERS, HEADS).sum(axis=2)
    layer_top_quartile_count = np.sum(
        np.stack([_rank_descending(row) for row in layer_weight_scores]) <= 8,
        axis=0,
    )
    necessary_layers = []
    layer_diagnostics = []
    for layer, metrics in enumerate(layer_remove_metrics):
        removed_bins = bin_lookup(metrics)
        drop_bins = [
            key for key in stable_bin_keys if baseline_bins[key] - removed_bins[key] >= 0.02
        ]
        necessary = len(drop_bins) >= 3 and layer_top_quartile_count[layer] >= 4
        if necessary:
            necessary_layers.append(layer)
        layer_diagnostics.append(
            {
                "layer": layer,
                "attribution_top_quartile_folds": int(layer_top_quartile_count[layer]),
                "drop_ge_0_02_bins": drop_bins,
                "necessary": necessary,
                "remove": metrics,
                "keep_only": layer_keep_metrics[layer],
            }
        )

    band_diagnostics = []
    for (lo, hi), metrics in zip(BANDS, band_keep_metrics):
        retained = retention_count(metrics, 0.90)
        band_diagnostics.append(
            {
                "layers": [lo, hi],
                "retention_90pct_bins": retained,
                "sufficient": retained >= 11,
                "metrics": metrics,
            }
        )

    sparse_top32 = all_metrics["top_32"]
    sparse_random32 = all_metrics["random_32"]
    top32_bins = bin_lookup(sparse_top32)
    random32_bins = bin_lookup(sparse_random32)
    top32_retained = retention_count(sparse_top32, 0.90)
    top32_random_wins = sum(
        top32_bins[key] - random32_bins[key] >= 0.03 for key in stable_bin_keys
    )
    stability = _pairwise_stability(weight_pair_scores, 32)
    sparse_gate = (
        stability["mean_top_jaccard"] >= 0.20
        and top32_retained >= 11
        and top32_random_wins >= 8
    )

    decisions = {
        "reproduction_pass": (
            abs(
                baseline_metrics["within_bin_auroc"]
                - reference_metrics["within_bin_auroc"]
            )
            <= 0.002
        ),
        "stable_pair_count": int(stable_pairs.size),
        "necessary_layers": necessary_layers,
        "sufficient_bands": [
            item["layers"] for item in band_diagnostics if item["sufficient"]
        ],
        "layer_mean_retention_bins": retention_count(
            all_metrics["layer_mean"], 0.90
        ),
        "mean_trajectory_dominated": (
            retention_count(all_metrics["layer_mean"], 0.90) >= 11
        ),
        "head_deviation_retention_bins": retention_count(
            all_metrics["head_deviation"], 0.75
        ),
        "meaningful_head_identity": (
            retention_count(all_metrics["head_deviation"], 0.75) >= 11
        ),
        "top32_retention_bins": top32_retained,
        "top32_random_win_bins": top32_random_wins,
        "sparse_head_gate": sparse_gate,
    }

    output = {
        "protocol": "docs/iteration_protocols/allocation_probe_mechanism_audit.md",
        "n": int(labels.size),
        "seed": args.seed,
        "folds": args.folds,
        "bin_width": args.bin_width,
        "reproduction": {
            "reference": reference_metrics,
            "refit": baseline_metrics,
            "max_absolute_score_difference": max_score_difference,
        },
        "parameter_stability": {
            "weight": stability,
            "gradient": _pairwise_stability(gradient_pair_scores, 32),
            "stable_pairs": [
                {"layer": int(pair // HEADS), "head": int(pair % HEADS)}
                for pair in stable_pairs
            ],
            "top_pair_table": pair_table,
            "position_top_gradient_pairs": bin_top_pairs,
        },
        "representation_metrics": all_metrics,
        "layer_diagnostics": layer_diagnostics,
        "row_remove_metrics": row_remove_metrics,
        "band_diagnostics": band_diagnostics,
        "selected_pairs": selected_pairs,
        "decisions": decisions,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(output), indent=2) + "\n")
    npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        object_id=object_ids,
        image_id=image_ids,
        gen_pos=positions,
        label=labels,
        score__full=full_scores,
        score__layer_mean=representation_scores["layer_mean"],
        score__head_deviation=representation_scores["head_deviation"],
        layer_remove_scores=layer_remove,
        layer_keep_scores=layer_keep,
        row_remove_scores=row_remove,
        band_keep_scores=band_keep,
        weight_pair_scores=weight_pair_scores,
        gradient_pair_scores=gradient_pair_scores,
        gradient_pair_by_bin=gradient_pair_by_bin,
        **{f"score__{name}": values for name, values in sparse_scores.items()},
    )
    print(json.dumps(decisions, indent=2), flush=True)
    print(f"wrote {output_path} and {npz_path}", flush=True)


if __name__ == "__main__":
    main()
