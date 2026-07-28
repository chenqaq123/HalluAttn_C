#!/usr/bin/env python3
"""Refit a caption sensor after excluding the registered CHAIR test IDs."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.scripts.evaluate_visual_value_evidence import (
    _IndexedDataset,
    _Probe,
    _conditional_weights,
    _feature_moments,
    _image_folds,
    _load_npz_parts,
    _predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_glob", required=True)
    parser.add_argument("--exclude_manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--grounded_trigger_rate", type=float, default=0.02)
    return parser.parse_args()


def _excluded_ids(path: str | Path) -> set[int]:
    with Path(path).open() as handle:
        return {
            int(json.loads(line)["sample_id"])
            for line in handle
            if line.strip()
        }


def main() -> None:
    args = parse_args()
    if not glob.glob(args.features_glob):
        raise FileNotFoundError(args.features_glob)
    data = _load_npz_parts(args.features_glob)
    image_ids = np.asarray(data["image_id"], dtype=np.int64)
    labels = np.asarray(data["label"], dtype=np.int32)
    positions = np.asarray(data["gen_pos"], dtype=np.int32)
    no_rope = np.asarray(data["no_rope"], dtype=np.float32)
    features = np.ascontiguousarray(
        no_rope[:, :, 1, :, 0].reshape(no_rope.shape[0], -1),
        dtype=np.float32,
    )
    excluded = _excluded_ids(args.exclude_manifest)
    overlap = sorted(set(image_ids.tolist()) & excluded)
    keep = ~np.isin(image_ids, np.asarray(sorted(excluded)))
    image_ids = image_ids[keep]
    labels = labels[keep]
    positions = positions[keep]
    features = features[keep]

    unique_images = np.unique(image_ids)
    generator = np.random.default_rng(args.seed)
    unique_images = unique_images[generator.permutation(len(unique_images))]
    calibration_count = max(
        1, int(round(len(unique_images) * args.calibration_fraction))
    )
    calibration_images = unique_images[:calibration_count]
    training_images = unique_images[calibration_count:]
    calibration = np.isin(image_ids, calibration_images)
    training = np.isin(image_ids, training_images)
    folds = _image_folds(training_images, args.folds, args.seed)

    device = torch.device(args.device)
    fold_records = []
    models = []
    for fold_index, heldout_images in enumerate(folds):
        fit = training & ~np.isin(image_ids, heldout_images)
        fit_indices = np.where(fit)[0]
        mean, std = _feature_moments(features, fit_indices)
        weights = _conditional_weights(
            labels[fit],
            positions[fit],
            args.bin_width,
            mode="size",
        )
        full_weights = np.zeros(labels.size, dtype=np.float32)
        full_weights[fit] = weights
        nonzero = fit_indices[weights > 0]
        dataset = _IndexedDataset(
            features,
            labels.astype(np.float32),
            full_weights,
            nonzero,
        )
        torch_generator = torch.Generator().manual_seed(
            args.seed + fold_index
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch_generator,
            num_workers=0,
        )
        torch.manual_seed(args.seed + fold_index)
        model = _Probe(features.shape[1], args.hidden_size).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        mean_t = torch.from_numpy(mean).to(device)
        std_t = torch.from_numpy(std).to(device)
        model.train()
        for _ in range(args.epochs):
            for batch_features, targets, sample_weights in loader:
                batch_features = batch_features.to(device)
                targets = targets.to(device)
                sample_weights = sample_weights.to(device)
                logits = model((batch_features - mean_t) / std_t)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
                loss = (losses * sample_weights).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        record = {
            "state_dict": {
                name: value.cpu()
                for name, value in model.state_dict().items()
            },
            "mean": mean,
            "std": std,
            "test_images": np.asarray(heldout_images, dtype=np.int64),
        }
        fold_records.append(record)
        models.append((model, mean, std))

    calibration_indices = np.where(calibration)[0]
    calibration_scores = np.mean(
        [
            _predict(
                model,
                features,
                calibration_indices,
                mean,
                std,
                device,
                args.batch_size * 4,
            )
            for model, mean, std in models
        ],
        axis=0,
    )
    calibration_labels = labels[calibration]
    grounded = calibration_scores[calibration_labels == 0]
    if grounded.size == 0:
        raise ValueError("Calibration split contains no grounded mentions")
    threshold = float(
        np.percentile(
            grounded, 100 * (1.0 - args.grounded_trigger_rate)
        )
    )
    trigger_rate = float(np.mean(calibration_scores >= threshold))

    payload = {
        "protocol": "docs/iteration_protocols/image_caption_benchmark_matrix.md",
        "feature": "no_rope__mass_post",
        "input_size": int(features.shape[1]),
        "hidden_size": args.hidden_size,
        "folds": fold_records,
        "config": vars(args),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    thresholds = {
        "tau_risk": threshold,
        "target_grounded_trigger_rate": args.grounded_trigger_rate,
        "realized_grounded_trigger_rate": float(
            np.mean(grounded >= threshold)
        ),
        "all_candidate_trigger_rate": trigger_rate,
        "source_mentions": int(keep.sum()),
        "source_images": int(len(unique_images)),
        "excluded_eval_images_with_features": len(overlap),
        "excluded_eval_ids": overlap,
        "training_images": int(len(training_images)),
        "training_mentions": int(training.sum()),
        "calibration_images": int(len(calibration_images)),
        "calibration_mentions": int(calibration.sum()),
        "calibration_grounded_mentions": int(grounded.size),
        "seed": args.seed,
    }
    Path(args.thresholds).write_text(
        json.dumps(thresholds, indent=2) + "\n"
    )
    print(json.dumps(thresholds, indent=2))


if __name__ == "__main__":
    main()
