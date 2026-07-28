#!/usr/bin/env python3
"""Capacity-controlled layer-breadth x head-treatment allocation ablation.

The expensive grid is intentionally shardable. A typical execution is:

    python detection/scripts/evaluate_layer_head_factorial.py --prepare_compact
    CUDA_VISIBLE_DEVICES=0 python ... --shard_index 0 --num_shards 8

The shard outputs contain aligned OOF logits for three model seeds. Statistical
analysis and nested training-fold layer selection are performed by the companion
``analyze_layer_head_factorial.py`` script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import roc_auc, within_bin_auc  # noqa: E402
from detection.scripts.analyze_allocation_probe import (  # noqa: E402
    HEADS,
    LAYERS,
    ROWS,
    _fit_model,
    _predict_standardized,
)
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _feature_moments,
    _image_folds,
    _load_npz_parts,
)

OUTPUT_DIM = 64
MODEL_SEEDS = (17, 927, 1701)
BANDS = ((0, 7), (8, 15), (16, 23), (24, 31))
HALVES = ((0, 15), (16, 31))
TREATMENTS = ("head_mean", "per_head", "head_deviation")


@dataclass(frozen=True)
class Cell:
    name: str
    layers: tuple[int, ...]
    treatment: str
    breadth: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_glob",
        default="experiments/visual_value_evidence/features_shard*.npz",
    )
    parser.add_argument(
        "--compact_cache",
        default="experiments/allocation_layer_head_factorial/allocation_mass.npz",
    )
    parser.add_argument("--prepare_compact", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=17)
    parser.add_argument(
        "--model_seeds",
        default=",".join(str(value) for value in MODEL_SEEDS),
    )
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pca_iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_dir",
        default="experiments/allocation_layer_head_factorial",
    )
    parser.add_argument(
        "--only_cells",
        default="",
        help="Optional comma-separated exact cell names for debugging.",
    )
    return parser.parse_args()


def enumerate_cells() -> list[Cell]:
    breadths: list[tuple[str, str, tuple[int, ...]]] = []
    breadths.extend(
        (f"single_l{layer:02d}", "single", (layer,)) for layer in range(LAYERS)
    )
    breadths.extend(
        (
            f"band_{lo:02d}_{hi:02d}",
            "band8",
            tuple(range(lo, hi + 1)),
        )
        for lo, hi in BANDS
    )
    breadths.extend(
        (
            f"half_{lo:02d}_{hi:02d}",
            "half16",
            tuple(range(lo, hi + 1)),
        )
        for lo, hi in HALVES
    )
    breadths.append(("all_00_31", "all", tuple(range(LAYERS))))
    return [
        Cell(
            name=f"{breadth_name}__{treatment}",
            layers=layers,
            treatment=treatment,
            breadth=breadth_type,
        )
        for breadth_name, breadth_type, layers in breadths
        for treatment in TREATMENTS
    ]


def make_representation(allocation: np.ndarray, cell: Cell) -> np.ndarray:
    """Construct one raw factorial representation.

    ``allocation`` has shape [mentions, layers, rows, heads].
    """

    selected = np.asarray(allocation[:, cell.layers, :, :], dtype=np.float32)
    if cell.treatment == "head_mean":
        selected = selected.mean(axis=-1)
    elif cell.treatment == "per_head":
        pass
    elif cell.treatment == "head_deviation":
        selected = selected - selected.mean(axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unknown treatment: {cell.treatment}")
    return np.ascontiguousarray(selected.reshape(selected.shape[0], -1))


def capacity_transform(
    raw: np.ndarray,
    train_indices: np.ndarray,
    output_dim: int,
    device: torch.device,
    seed: int,
    pca_iterations: int,
) -> tuple[np.ndarray, dict]:
    """Fit standardization/PCA on ``train_indices`` and transform all rows."""

    raw = np.ascontiguousarray(raw, dtype=np.float32)
    mean, std = _feature_moments(raw, train_indices)
    standardized = (raw - mean) / std
    raw_dim = int(raw.shape[1])
    if raw_dim <= output_dim:
        transformed = np.zeros((raw.shape[0], output_dim), dtype=np.float32)
        transformed[:, :raw_dim] = standardized
        return transformed, {
            "raw_dim": raw_dim,
            "output_dim": output_dim,
            "method": "identity_zero_pad",
            "explained_variance_fraction": 1.0,
        }

    torch.manual_seed(seed)
    train_tensor = torch.from_numpy(
        np.ascontiguousarray(standardized[train_indices])
    ).to(device)
    _, singular_values, components = torch.pca_lowrank(
        train_tensor,
        q=output_dim,
        center=False,
        niter=pca_iterations,
    )
    del train_tensor
    transformed = np.empty((raw.shape[0], output_dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, raw.shape[0], 1024):
            batch = torch.from_numpy(
                np.ascontiguousarray(standardized[start : start + 1024])
            ).to(device)
            transformed[start : start + batch.shape[0]] = (
                batch @ components
            ).float().cpu().numpy()
    train_variance = float(np.square(standardized[train_indices]).sum())
    retained_variance = float(np.square(singular_values.float().cpu().numpy()).sum())
    del components
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return transformed, {
        "raw_dim": raw_dim,
        "output_dim": output_dim,
        "method": "train_fold_pca",
        "explained_variance_fraction": (
            retained_variance / train_variance if train_variance > 0 else None
        ),
    }


def best_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Select the train-only threshold maximizing balanced accuracy."""

    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("Threshold selection requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels == 1)
    false_positive = np.cumsum(sorted_labels == 0)
    positives = int(true_positive[-1])
    negatives = int(false_positive[-1])
    balanced = 0.5 * (
        true_positive / positives + (negatives - false_positive) / negatives
    )
    best = int(np.argmax(balanced))
    if best + 1 == scores.size:
        return float(np.nextafter(scores[order[best]], -np.inf))
    return float(0.5 * (scores[order[best]] + scores[order[best + 1]]))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("Average precision requires a positive example")
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked == 1].sum() / positives)


def _prepare_compact(args: argparse.Namespace) -> None:
    data = _load_npz_parts(args.features_glob)
    allocation = data["no_rope"][..., 0].astype(np.float32)
    if allocation.shape[1:] != (LAYERS, ROWS, HEADS):
        raise ValueError(f"Unexpected allocation shape: {allocation.shape}")
    output = Path(args.compact_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        allocation=allocation,
        object_id=data["object_id"].astype(np.int64),
        image_id=data["image_id"].astype(np.int64),
        token_pos=data["token_pos"].astype(np.int64),
        gen_pos=data["gen_pos"].astype(np.int32),
        label=data["label"].astype(np.int32),
        word=data["word"].astype(str),
        source_features_glob=np.asarray(args.features_glob),
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "wrote": str(output),
                "n": int(allocation.shape[0]),
                "shape": list(allocation.shape),
            },
            indent=2,
        ),
        flush=True,
    )


def _cell_oof(
    raw: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    test_masks: list[np.ndarray],
    cell: Cell,
    model_seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    n = labels.size
    seed_scores = np.full((len(model_seeds), n), np.nan, dtype=np.float64)
    seed_decisions = np.full((len(model_seeds), n), np.nan, dtype=np.float32)
    transform_metadata = []
    device = torch.device(args.device)
    cell_seed = zlib.crc32(cell.name.encode("utf-8"))
    for fold_index, test_mask in enumerate(test_masks):
        train_indices = np.where(~test_mask)[0]
        test_indices = np.where(test_mask)[0]
        transformed, metadata = capacity_transform(
            raw,
            train_indices,
            OUTPUT_DIM,
            device,
            seed=3100 + fold_index + cell_seed,
            pca_iterations=args.pca_iterations,
        )
        transform_metadata.append({"fold": fold_index, **metadata})
        for seed_index, model_seed in enumerate(model_seeds):
            fit_seed = model_seed + fold_index
            model, mean, std = _fit_model(
                transformed,
                labels,
                positions,
                train_indices,
                args,
                fit_seed,
            )
            train_standardized = (
                transformed[train_indices].astype(np.float32) - mean
            ) / std
            test_standardized = (
                transformed[test_indices].astype(np.float32) - mean
            ) / std
            train_scores = _predict_standardized(
                model, train_standardized, device, args.batch_size * 4
            )
            test_scores = _predict_standardized(
                model, test_standardized, device, args.batch_size * 4
            )
            threshold = best_balanced_threshold(labels[train_indices], train_scores)
            seed_scores[seed_index, test_indices] = test_scores
            seed_decisions[seed_index, test_indices] = (
                test_scores >= threshold
            ).astype(np.float32)
            del model, train_standardized, test_standardized
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del transformed
        print(
            f"{cell.name}: completed fold {fold_index + 1}/{len(test_masks)}",
            flush=True,
        )
    if not np.isfinite(seed_scores).all() or not np.isfinite(seed_decisions).all():
        raise ValueError(f"Non-finite OOF output for {cell.name}")
    return seed_scores, seed_decisions, transform_metadata


def _metric_summary(
    labels: np.ndarray,
    positions: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    bin_width: int,
) -> dict:
    within = within_bin_auc(labels, scores, positions, bin_width=bin_width)
    return {
        "overall_auroc": roc_auc(labels, scores),
        "within_bin_auroc": within["auc"],
        "average_precision": average_precision(labels, scores),
        "balanced_accuracy_train_threshold": float(
            0.5
            * (
                np.mean(decisions[labels == 1] >= 0.5)
                + np.mean(decisions[labels == 0] < 0.5)
            )
        ),
        "within_bins": within["bins"],
    }


def main() -> None:
    args = parse_args()
    if args.prepare_compact:
        _prepare_compact(args)
        return
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    model_seeds = tuple(
        int(value.strip()) for value in args.model_seeds.split(",") if value.strip()
    )
    if not model_seeds:
        raise ValueError("At least one model seed is required")
    compact = np.load(args.compact_cache)
    allocation = compact["allocation"].astype(np.float32)
    labels = compact["label"].astype(np.int32)
    positions = compact["gen_pos"].astype(np.int32)
    image_ids = compact["image_id"].astype(np.int64)
    object_ids = compact["object_id"].astype(np.int64)
    words = compact["word"].astype(str)
    folds = _image_folds(image_ids, args.folds, args.fold_seed)
    test_masks = [np.isin(image_ids, fold_images) for fold_images in folds]

    cells = enumerate_cells()
    if args.only_cells:
        requested = set(args.only_cells.split(","))
        cells = [cell for cell in cells if cell.name in requested]
        missing = requested - {cell.name for cell in cells}
        if missing:
            raise ValueError(f"Unknown cells: {sorted(missing)}")
    cells = [
        cell
        for index, cell in enumerate(cells)
        if index % args.num_shards == args.shard_index
    ]
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(cells)} cells",
        flush=True,
    )
    arrays: dict[str, np.ndarray] = {
        "object_id": object_ids,
        "image_id": image_ids,
        "gen_pos": positions,
        "label": labels,
        "word": words,
        "model_seeds": np.asarray(model_seeds, dtype=np.int64),
    }
    cell_results = {}
    started = time.time()
    for cell_index, cell in enumerate(cells):
        raw = make_representation(allocation, cell)
        cell_started = time.time()
        seed_scores, seed_decisions, transform_metadata = _cell_oof(
            raw,
            labels,
            positions,
            test_masks,
            cell,
            model_seeds,
            args,
        )
        scores = seed_scores.mean(axis=0)
        decisions = seed_decisions.mean(axis=0)
        arrays[f"score__{cell.name}"] = scores
        arrays[f"score_seed__{cell.name}"] = seed_scores
        arrays[f"decision_seed__{cell.name}"] = seed_decisions
        cell_results[cell.name] = {
            "layers": list(cell.layers),
            "breadth": cell.breadth,
            "treatment": cell.treatment,
            "raw_dim": int(raw.shape[1]),
            "capacity_dim": OUTPUT_DIM,
            "metrics": _metric_summary(
                labels, positions, scores, decisions, args.bin_width
            ),
            "seed_metrics": [
                _metric_summary(
                    labels,
                    positions,
                    seed_scores[index],
                    seed_decisions[index],
                    args.bin_width,
                )
                for index in range(len(model_seeds))
            ],
            "transform": transform_metadata,
            "elapsed_seconds": time.time() - cell_started,
        }
        del raw, seed_scores, seed_decisions
        print(
            f"completed cell {cell_index + 1}/{len(cells)}: {cell.name} "
            f"within={cell_results[cell.name]['metrics']['within_bin_auroc']:.6f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"grid_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **arrays)
    result = {
        "protocol": "docs/iteration_protocols/allocation_layer_head_factorial.md",
        "compact_cache": args.compact_cache,
        "n": int(labels.size),
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "model_seeds": list(model_seeds),
        "bin_width": args.bin_width,
        "capacity_dim": OUTPUT_DIM,
        "pca_iterations": args.pca_iterations,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device": args.device,
        "elapsed_seconds": time.time() - started,
        "cells": cell_results,
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {json_path} and {npz_path}", flush=True)


if __name__ == "__main__":
    main()
