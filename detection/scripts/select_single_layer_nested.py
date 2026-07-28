#!/usr/bin/env python3
"""Select the best single layer strictly inside each outer training fold."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import within_bin_auc  # noqa: E402
from detection.scripts.analyze_allocation_probe import (  # noqa: E402
    _fit_model,
    _predict_standardized,
)
from detection.scripts.evaluate_layer_head_factorial import (  # noqa: E402
    Cell,
    LAYERS,
    OUTPUT_DIM,
    capacity_transform,
    make_representation,
)
from detection.scripts.evaluate_visual_value_evidence import _image_folds  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact_cache",
        default="experiments/allocation_layer_head_factorial/allocation_mass.npz",
    )
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=17)
    parser.add_argument("--model_seeds", default="17,927,1701")
    parser.add_argument("--inner_validation_fraction", type=float, default=0.2)
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
        default="experiments/allocation_layer_head_factorial/nested_selection",
    )
    return parser.parse_args()


def inner_image_split(
    outer_train_images: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(sorted(set(outer_train_images.tolist())), dtype=np.int64)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("inner_validation_fraction must be in (0, 1)")
    rng = np.random.RandomState(seed)
    rng.shuffle(images)
    count = max(1, int(round(images.size * validation_fraction)))
    return images[count:], images[:count]


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard_index < num_shards")
    model_seeds = tuple(
        int(value.strip()) for value in args.model_seeds.split(",") if value.strip()
    )
    compact = np.load(args.compact_cache)
    allocation = compact["allocation"].astype(np.float32)
    labels = compact["label"].astype(np.int32)
    positions = compact["gen_pos"].astype(np.int32)
    image_ids = compact["image_id"].astype(np.int64)
    outer_folds = _image_folds(image_ids, args.folds, args.fold_seed)
    tasks = [
        (fold_index, model_seed)
        for fold_index in range(args.folds)
        for model_seed in model_seeds
    ]
    tasks = [
        task
        for index, task in enumerate(tasks)
        if index % args.num_shards == args.shard_index
    ]
    device = torch.device(args.device)
    output_tasks = []
    started = time.time()
    for task_index, (fold_index, model_seed) in enumerate(tasks):
        outer_test_images = outer_folds[fold_index]
        outer_train_images = np.setdiff1d(
            np.unique(image_ids), outer_test_images, assume_unique=True
        )
        inner_train_images, inner_val_images = inner_image_split(
            outer_train_images,
            args.inner_validation_fraction,
            seed=8100 + fold_index,
        )
        inner_train = np.where(np.isin(image_ids, inner_train_images))[0]
        inner_val = np.where(np.isin(image_ids, inner_val_images))[0]
        if np.intersect1d(inner_train_images, inner_val_images).size:
            raise AssertionError("Inner train/validation images overlap")
        if np.intersect1d(inner_val_images, outer_test_images).size:
            raise AssertionError("Inner validation/outer test images overlap")
        layer_metrics = []
        for layer in range(LAYERS):
            cell = Cell(
                name=f"single_l{layer:02d}__per_head",
                layers=(layer,),
                treatment="per_head",
                breadth="single",
            )
            raw = make_representation(allocation, cell)
            transformed, transform = capacity_transform(
                raw,
                inner_train,
                OUTPUT_DIM,
                device,
                seed=9100 + fold_index,
                pca_iterations=args.pca_iterations,
            )
            model, mean, std = _fit_model(
                transformed,
                labels,
                positions,
                inner_train,
                args,
                model_seed + fold_index,
            )
            standardized = (
                transformed[inner_val].astype(np.float32) - mean
            ) / std
            scores = _predict_standardized(
                model, standardized, device, args.batch_size * 4
            )
            within = within_bin_auc(
                labels[inner_val],
                scores,
                positions[inner_val],
                bin_width=args.bin_width,
            )["auc"]
            layer_metrics.append(
                {
                    "layer": layer,
                    "within_bin_auroc": within,
                    "raw_dim": int(raw.shape[1]),
                    "transform": transform,
                }
            )
            del raw, transformed, model, standardized
            if device.type == "cuda":
                torch.cuda.empty_cache()
        valid = [
            item for item in layer_metrics if item["within_bin_auroc"] is not None
        ]
        if not valid:
            raise ValueError("No single layer has a valid inner validation AUROC")
        selected = max(
            valid,
            key=lambda item: (item["within_bin_auroc"], -item["layer"]),
        )
        output_tasks.append(
            {
                "outer_fold": fold_index,
                "model_seed": model_seed,
                "outer_test_images": outer_test_images.tolist(),
                "inner_train_image_count": int(inner_train_images.size),
                "inner_validation_image_count": int(inner_val_images.size),
                "inner_train_mention_count": int(inner_train.size),
                "inner_validation_mention_count": int(inner_val.size),
                "selected_layer": int(selected["layer"]),
                "selected_inner_within_bin_auroc": float(
                    selected["within_bin_auroc"]
                ),
                "layer_metrics": layer_metrics,
            }
        )
        print(
            f"task {task_index + 1}/{len(tasks)} outer_fold={fold_index} "
            f"seed={model_seed} selected=L{selected['layer']:02d} "
            f"inner_within={selected['within_bin_auroc']:.6f}",
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        f"nested_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    )
    output = {
        "protocol": "docs/iteration_protocols/allocation_layer_head_factorial.md",
        "fold_seed": args.fold_seed,
        "model_seeds": list(model_seeds),
        "inner_validation_fraction": args.inner_validation_fraction,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device": args.device,
        "elapsed_seconds": time.time() - started,
        "tasks": output_tasks,
    }
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

