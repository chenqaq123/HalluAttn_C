#!/usr/bin/env python3
"""Recompute and audit a registered disjoint caption sensor artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.scripts.evaluate_visual_value_evidence import (
    _Probe,
    _image_folds,
    _load_npz_parts,
    _predict,
)
from mitigation.scripts.train_disjoint_caption_sensor import _excluded_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_glob", required=True)
    parser.add_argument("--exclude_manifest", required=True)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--promote_sensor")
    parser.add_argument("--promote_thresholds")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            f"{name} mismatch: recorded={actual}, recomputed={expected}"
        )


def audit_sensor(
    features_glob: str,
    exclude_manifest: Path,
    sensor_path: Path,
    thresholds_path: Path,
) -> dict[str, Any]:
    feature_paths = sorted(glob.glob(features_glob))
    if not feature_paths:
        raise FileNotFoundError(features_glob)
    payload = torch.load(
        sensor_path, map_location="cpu", weights_only=False
    )
    thresholds = json.loads(thresholds_path.read_text())
    required_payload = {
        "protocol",
        "feature",
        "input_size",
        "hidden_size",
        "folds",
        "config",
    }
    missing_payload = sorted(required_payload - set(payload))
    if missing_payload:
        raise RuntimeError(
            f"Sensor payload lacks keys {missing_payload}"
        )
    config = payload["config"]
    required_config = {
        "seed",
        "calibration_fraction",
        "folds",
        "epochs",
        "hidden_size",
        "grounded_trigger_rate",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise RuntimeError(
            f"Sensor config lacks keys {missing_config}"
        )

    data = _load_npz_parts(features_glob)
    image_ids = np.asarray(data["image_id"], dtype=np.int64)
    labels = np.asarray(data["label"], dtype=np.int32)
    no_rope = np.asarray(data["no_rope"], dtype=np.float32)
    if no_rope.ndim != 5:
        raise RuntimeError(f"no_rope must be rank 5, got {no_rope.shape}")
    features = np.ascontiguousarray(
        no_rope[:, :, 1, :, 0].reshape(no_rope.shape[0], -1),
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise RuntimeError("Sensor inputs contain non-finite values")
    input_size = int(payload["input_size"])
    hidden_size = int(payload["hidden_size"])
    if features.shape[1] != input_size:
        raise RuntimeError(
            f"Sensor input size {input_size} != {features.shape[1]}"
        )
    if hidden_size != int(config["hidden_size"]):
        raise RuntimeError("Payload/config hidden sizes differ")

    excluded = _excluded_ids(exclude_manifest)
    overlap = sorted(set(map(int, image_ids)) & excluded)
    if overlap:
        raise RuntimeError(
            f"Excluded evaluation images occur in features: {overlap[:10]}"
        )
    keep = ~np.isin(image_ids, np.asarray(sorted(excluded)))
    image_ids = image_ids[keep]
    labels = labels[keep]
    features = features[keep]

    seed = int(config["seed"])
    unique_images = np.unique(image_ids)
    generator = np.random.default_rng(seed)
    unique_images = unique_images[
        generator.permutation(len(unique_images))
    ]
    calibration_count = max(
        1,
        int(
            round(
                len(unique_images)
                * float(config["calibration_fraction"])
            )
        ),
    )
    calibration_images = unique_images[:calibration_count]
    training_images = unique_images[calibration_count:]
    calibration = np.isin(image_ids, calibration_images)
    training = np.isin(image_ids, training_images)
    expected_folds = _image_folds(
        training_images, int(config["folds"]), seed
    )
    records = payload["folds"]
    if len(records) != len(expected_folds):
        raise RuntimeError(
            f"Sensor fold count {len(records)} != {len(expected_folds)}"
        )

    predictions = []
    fold_sizes = []
    for fold_index, (record, expected_test_images) in enumerate(
        zip(records, expected_folds)
    ):
        actual_test_images = np.asarray(
            record["test_images"], dtype=np.int64
        )
        if not np.array_equal(actual_test_images, expected_test_images):
            raise RuntimeError(
                f"Fold {fold_index} held-out images differ"
            )
        mean = np.asarray(record["mean"], dtype=np.float32)
        std = np.asarray(record["std"], dtype=np.float32)
        if mean.shape != (input_size,) or std.shape != (input_size,):
            raise RuntimeError(
                f"Fold {fold_index} moments have invalid shapes"
            )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise RuntimeError(
                f"Fold {fold_index} moments are not finite/positive"
            )
        model = _Probe(input_size, hidden_size)
        model.load_state_dict(record["state_dict"], strict=True)
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(
                    f"Fold {fold_index} has non-finite parameters"
                )
        model.eval()
        calibration_indices = np.where(calibration)[0]
        predictions.append(
            _predict(
                model,
                features,
                calibration_indices,
                mean,
                std,
                torch.device("cpu"),
                512,
            )
        )
        fold_sizes.append(int(actual_test_images.size))

    calibration_scores = np.mean(predictions, axis=0)
    calibration_labels = labels[calibration]
    grounded = calibration_scores[calibration_labels == 0]
    if grounded.size == 0:
        raise RuntimeError("Calibration split has no grounded mentions")
    target_rate = float(config["grounded_trigger_rate"])
    tau = float(
        np.percentile(grounded, 100 * (1.0 - target_rate))
    )
    grounded_rate = float(np.mean(grounded >= tau))
    candidate_rate = float(np.mean(calibration_scores >= tau))
    _close(float(thresholds["tau_risk"]), tau, "tau_risk")
    _close(
        float(thresholds["target_grounded_trigger_rate"]),
        target_rate,
        "target_grounded_trigger_rate",
    )
    _close(
        float(thresholds["realized_grounded_trigger_rate"]),
        grounded_rate,
        "realized_grounded_trigger_rate",
    )
    _close(
        float(thresholds["all_candidate_trigger_rate"]),
        candidate_rate,
        "all_candidate_trigger_rate",
    )
    exact_counts = {
        "source_mentions": int(keep.sum()),
        "source_images": int(len(unique_images)),
        "training_images": int(len(training_images)),
        "training_mentions": int(training.sum()),
        "calibration_images": int(len(calibration_images)),
        "calibration_mentions": int(calibration.sum()),
        "calibration_grounded_mentions": int(grounded.size),
        "excluded_eval_images_with_features": 0,
    }
    mismatched_counts = {
        key: {
            "recorded": int(thresholds[key]),
            "recomputed": value,
        }
        for key, value in exact_counts.items()
        if int(thresholds[key]) != value
    }
    if mismatched_counts:
        raise RuntimeError(
            f"Threshold count mismatch: {mismatched_counts}"
        )
    if thresholds.get("excluded_eval_ids", []) != []:
        raise RuntimeError("Threshold artifact reports excluded eval IDs")

    return {
        "valid": True,
        "protocol": payload["protocol"],
        "feature": payload["feature"],
        "features": [
            {
                "path": path,
                "sha256": _sha256(Path(path)),
            }
            for path in feature_paths
        ],
        "sensor_sha256": _sha256(sensor_path),
        "thresholds_sha256": _sha256(thresholds_path),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "folds": len(records),
        "fold_heldout_image_counts": fold_sizes,
        "epochs": int(config["epochs"]),
        "seed": seed,
        "source_mentions": exact_counts["source_mentions"],
        "source_images": exact_counts["source_images"],
        "training_images": exact_counts["training_images"],
        "calibration_images": exact_counts["calibration_images"],
        "excluded_eval_overlap": [],
        "tau_risk": tau,
        "target_grounded_trigger_rate": target_rate,
        "realized_grounded_trigger_rate": grounded_rate,
        "all_candidate_trigger_rate": candidate_rate,
        "recomputed_threshold_matches_within_tolerance": True,
    }


def main() -> None:
    args = parse_args()
    sensor_path = Path(args.sensor)
    thresholds_path = Path(args.thresholds)
    report = audit_sensor(
        args.features_glob,
        Path(args.exclude_manifest),
        sensor_path,
        thresholds_path,
    )
    promote_requested = bool(
        args.promote_sensor or args.promote_thresholds
    )
    if promote_requested and not (
        args.promote_sensor and args.promote_thresholds
    ):
        raise ValueError(
            "Both --promote_sensor and --promote_thresholds are required"
        )
    if promote_requested:
        final_sensor = Path(args.promote_sensor)
        final_thresholds = Path(args.promote_thresholds)
        if final_sensor.exists() or final_thresholds.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing promoted artifact"
            )
        os.replace(sensor_path, final_sensor)
        os.replace(thresholds_path, final_thresholds)
        report["sensor"] = str(final_sensor)
        report["thresholds"] = str(final_thresholds)
        report["promoted"] = True
    else:
        report["sensor"] = str(sensor_path)
        report["thresholds"] = str(thresholds_path)
        report["promoted"] = False
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
