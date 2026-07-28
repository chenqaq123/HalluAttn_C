#!/usr/bin/env python3
"""Train and calibrate one model-specific OOF Vript-HAL sensor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.baselines.metrics import compute_metric_bundle
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
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grounded_trigger_rate", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_npz_parts(args.features_glob)
    labels = np.asarray(data["label"], dtype=np.int32)
    positions = np.asarray(data["gen_pos"], dtype=np.int32)
    sample_ids = np.asarray(data["image_id"], dtype=np.int64)
    object_ids = np.asarray(data["object_id"], dtype=np.int64)
    no_rope = np.asarray(data["no_rope"], dtype=np.float32)
    features = np.ascontiguousarray(
        no_rope[:, :, 1, :, 0].reshape(no_rope.shape[0], -1),
        dtype=np.float32,
    )
    folds = _image_folds(sample_ids, args.folds, args.seed)
    device = torch.device(args.device)
    scores = np.full(labels.size, np.nan, dtype=np.float64)
    records = []
    for fold_index, heldout_samples in enumerate(folds):
        test = np.isin(sample_ids, heldout_samples)
        train = ~test
        train_indices = np.where(train)[0]
        test_indices = np.where(test)[0]
        mean, std = _feature_moments(features, train_indices)
        weights = _conditional_weights(
            labels[train],
            positions[train],
            args.bin_width,
            mode="size",
        )
        full_weights = np.zeros(labels.size, dtype=np.float32)
        full_weights[train] = weights
        fit_indices = train_indices[weights > 0]
        dataset = _IndexedDataset(
            features,
            labels.astype(np.float32),
            full_weights,
            fit_indices,
        )
        generator = torch.Generator().manual_seed(
            args.seed + fold_index
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        torch.manual_seed(args.seed + fold_index)
        model = _Probe(features.shape[1], args.hidden_size).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        mean_tensor = torch.from_numpy(mean).to(device)
        std_tensor = torch.from_numpy(std).to(device)
        model.train()
        for _ in range(args.epochs):
            for batch, targets, sample_weights in loader:
                batch = batch.to(device)
                targets = targets.to(device)
                sample_weights = sample_weights.to(device)
                logits = model(
                    (batch - mean_tensor) / std_tensor
                )
                losses = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                )
                loss = (losses * sample_weights).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        scores[test_indices] = _predict(
            model,
            features,
            test_indices,
            mean,
            std,
            device,
            args.batch_size * 4,
        )
        records.append(
            {
                "state_dict": {
                    name: value.cpu()
                    for name, value in model.state_dict().items()
                },
                "mean": mean,
                "std": std,
                "test_images": np.asarray(
                    heldout_samples, dtype=np.int64
                ),
            }
        )
        del model, optimizer, loader, dataset

    metrics = compute_metric_bundle(
        labels, scores, positions, bin_width=args.bin_width
    )
    grounded = scores[labels == 0]
    if grounded.size == 0:
        raise ValueError("Vript-HAL features contain no grounded mentions")
    tau_risk = float(
        np.percentile(
            grounded,
            100.0 * (1.0 - args.grounded_trigger_rate),
        )
    )
    sensor_path = Path(args.sensor)
    sensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": (
                "docs/iteration_protocols/"
                "video_caption_benchmark_matrix.md"
            ),
            "feature": "no_rope__mass_post",
            "input_size": int(features.shape[1]),
            "hidden_size": args.hidden_size,
            "folds": records,
            "config": vars(args),
        },
        sensor_path,
    )
    np.savez(
        args.oof,
        object_id=object_ids,
        image_id=sample_ids,
        label=labels,
        gen_pos=positions,
        score=scores,
    )
    threshold_payload = {
        "tau_risk": tau_risk,
        "target_grounded_trigger_rate": args.grounded_trigger_rate,
        "realized_grounded_trigger_rate": float(
            np.mean(grounded >= tau_risk)
        ),
        "all_candidate_trigger_rate": float(
            np.mean(scores >= tau_risk)
        ),
        "mentions": int(labels.size),
        "samples": int(np.unique(sample_ids).size),
        "hallucination_rate": float(labels.mean()),
        "overall_auroc": metrics["overall_auroc"],
        "within_bin_auroc": metrics["within_bin_auroc"],
        "matched_pair_auroc": metrics["matched_pair_auroc"],
        "folds": args.folds,
        "seed": args.seed,
    }
    Path(args.thresholds).write_text(
        json.dumps(threshold_payload, indent=2) + "\n"
    )
    print(json.dumps(threshold_payload, indent=2))


if __name__ == "__main__":
    main()
