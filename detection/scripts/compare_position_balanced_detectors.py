#!/usr/bin/env python3
"""Compare reproduced detectors on identical position-balanced draws."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import compute_metric_bundle, roc_auc  # noqa: E402
from detection.scripts.evaluate_position_balanced_oof import (  # noqa: E402
    _summary,
    position_balanced_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal_oof",
        default="experiments/spatial_allocation/temporal_row_oof.npz",
    )
    parser.add_argument(
        "--vve_oof",
        default=(
            "experiments/visual_value_evidence/"
            "iteration2_size_shared_seed_oof.npz"
        ),
    )
    parser.add_argument(
        "--baseline_scores",
        default=(
            "detection/baselines/results/coco_llava_7b_baselines/"
            "baseline_scores.npz"
        ),
    )
    parser.add_argument(
        "--beyond_glob", default="experiments/beyond_svar_v3/shard*.npz"
    )
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--min_bin_count", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_npz", required=True)
    return parser.parse_args()


def _load_beyond_parts(pattern: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    parts = [np.load(path) for path in paths]
    keys = (
        "object_id",
        "image_id",
        "gen_pos",
        "label",
        "image_mass",
        "ads",
        "cgc",
        "tver",
        "sae_seg",
    )
    raw = {
        key: np.concatenate([np.asarray(part[key]) for part in parts], axis=0)
        for key in keys
    }
    order = np.argsort(raw["object_id"])
    return raw, order


def _fit_adscgc_logistic_oof(raw: dict[str, np.ndarray], folds: int = 5) -> np.ndarray:
    features = np.concatenate((raw["ads"], raw["cgc"]), axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    labels = raw["label"].astype(np.int32)
    image_ids = raw["image_id"].astype(np.int64)
    output = np.zeros(labels.size, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train, test in splitter.split(features, labels, groups=image_ids):
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=2000, C=1.0).fit(
            scaler.transform(features[train]), labels[train]
        )
        output[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    return output


def _assert_aligned(
    name: str,
    object_ids: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    reference: dict[str, np.ndarray],
) -> None:
    checks = {
        "object_id": (object_ids, reference["object_id"]),
        "label": (labels, reference["label"]),
        "gen_pos": (positions, reference["gen_pos"]),
    }
    for key, (value, expected) in checks.items():
        if not np.array_equal(value, expected):
            raise ValueError(f"{name} does not align on {key}")


def _check_reproduction(name: str, metrics: dict, expected: tuple[float, float]) -> None:
    actual = (metrics["overall_auroc"], metrics["within_bin_auroc"])
    if any(abs(value - target) > 5e-4 for value, target in zip(actual, expected)):
        raise AssertionError(
            f"{name} reproduction drift: actual={actual}, expected={expected}"
        )


def main() -> None:
    args = parse_args()
    temporal = np.load(args.temporal_oof)
    reference = {
        "object_id": np.asarray(temporal["object_id"], dtype=np.int64),
        "label": np.asarray(temporal["label"], dtype=np.int32),
        "gen_pos": np.asarray(temporal["gen_pos"], dtype=np.int32),
        "image_id": np.asarray(temporal["image_id"], dtype=np.int64),
    }
    if not np.array_equal(reference["object_id"], np.arange(reference["label"].size)):
        raise ValueError("Temporal OOF object IDs must be the complete sorted range")

    vve = np.load(args.vve_oof)
    _assert_aligned(
        "VVE OOF",
        np.asarray(vve["object_id"]),
        np.asarray(vve["label"]),
        np.asarray(vve["gen_pos"]),
        reference,
    )
    baseline = np.load(args.baseline_scores)
    if not np.array_equal(np.asarray(baseline["labels"]), reference["label"]):
        raise ValueError("Baseline scores do not align on labels")
    if not np.array_equal(np.asarray(baseline["gen_pos"]), reference["gen_pos"]):
        raise ValueError("Baseline scores do not align on generation position")

    beyond_raw, beyond_order = _load_beyond_parts(args.beyond_glob)
    adscgc_raw = _fit_adscgc_logistic_oof(beyond_raw)
    beyond = {key: value[beyond_order] for key, value in beyond_raw.items()}
    _assert_aligned(
        "Beyond v3",
        beyond["object_id"],
        beyond["label"],
        beyond["gen_pos"],
        reference,
    )
    adscgc = adscgc_raw[beyond_order]

    scores = {
        "pas": np.asarray(baseline["pas_layer0_hallu_score"], dtype=np.float64),
        "svar": -np.asarray(beyond["image_mass"][:, 5:19], dtype=np.float64).sum(axis=1),
        "ic": np.asarray(vve["score__baseline_ic"], dtype=np.float64),
        "tver": np.asarray(beyond["tver"][:, 0], dtype=np.float64),
        "sae": -np.asarray(beyond["sae_seg"][:, 10], dtype=np.float64),
        "adscgc": adscgc,
        "haloprobe_style": np.asarray(
            vve["score__probe_attention_confidence"], dtype=np.float64
        ),
        "mass_pre": np.asarray(temporal["score__no_rope__mass_pre"], dtype=np.float64),
        "mass_post": np.asarray(temporal["score__no_rope__mass_post"], dtype=np.float64),
        "mass_full": np.asarray(temporal["score__no_rope__mass"], dtype=np.float64),
    }
    if any(values.shape != reference["label"].shape for values in scores.values()):
        raise ValueError("At least one detector score has the wrong length")
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("All detector scores must be finite")

    metrics = {
        name: compute_metric_bundle(
            reference["label"], values, reference["gen_pos"], bin_width=args.bin_width
        )
        for name, values in scores.items()
    }
    expected = {
        "pas": (0.8349, 0.5928),
        "svar": (0.8053, 0.5678),
        "ic": (0.7761, 0.6856),
        "tver": (0.8303, 0.5655),
        "sae": (0.7833, 0.5747),
        "adscgc": (0.8661, 0.7185),
        "haloprobe_style": (0.8561, 0.7988),
        "mass_pre": (0.8773, 0.8425),
        "mass_post": (0.9099, 0.8859),
        "mass_full": (0.9053, 0.8812),
    }
    for name, target in expected.items():
        _check_reproduction(name, metrics[name], target)

    aggregate = {name: [] for name in scores}
    per_bin = {name: {} for name in scores}
    rng = np.random.default_rng(args.seed)
    selected_count = None
    valid_bins = None
    for _ in range(args.repeats):
        indices, current_bins = position_balanced_indices(
            reference["label"],
            reference["gen_pos"],
            args.bin_width,
            args.min_bin_count,
            rng,
        )
        if selected_count is None:
            selected_count = int(indices.size)
            valid_bins = current_bins
        for name, values in scores.items():
            value = roc_auc(reference["label"][indices], values[indices])
            if value is None:
                raise ValueError(f"Balanced AUROC is undefined for {name}")
            aggregate[name].append(float(value))
            for item in current_bins:
                key = f"{item['bin_start']}-{item['bin_end']}"
                selected = indices[
                    (reference["gen_pos"][indices] >= item["bin_start"])
                    & (reference["gen_pos"][indices] <= item["bin_end"])
                ]
                value = roc_auc(reference["label"][selected], values[selected])
                if value is None:
                    raise ValueError(f"Per-bin AUROC is undefined for {name} in {key}")
                per_bin[name].setdefault(key, []).append(float(value))

    metadata = {
        "pas": {"supervision": "label-free", "definition": "PAS layer 0"},
        "svar": {"supervision": "label-free", "definition": "SVAR layers 5-18"},
        "ic": {"supervision": "label-free", "definition": "IC"},
        "tver": {"supervision": "label-free", "definition": "TVER layer 0"},
        "sae": {
            "supervision": "label-free",
            "definition": "Mask2Former SAE, previously selected best layer 10",
        },
        "adscgc": {"supervision": "supervised", "definition": "ADS+CGC logistic OOF"},
        "haloprobe_style": {
            "supervision": "supervised",
            "definition": "local HaloProbe-style attention+confidence MLP OOF",
        },
        "mass_pre": {"supervision": "supervised", "definition": "No-RoPE mass pre"},
        "mass_post": {"supervision": "supervised", "definition": "No-RoPE mass post"},
        "mass_full": {"supervision": "supervised", "definition": "No-RoPE mass pre+post"},
    }
    output = {
        "protocol": "docs/iteration_protocols/position_balanced_detector_comparison.md",
        "n": int(reference["label"].size),
        "images": int(np.unique(reference["image_id"]).size),
        "repeats": args.repeats,
        "seed": args.seed,
        "bin_width": args.bin_width,
        "selected_count_per_repeat": selected_count,
        "valid_bins": valid_bins,
        "detectors": metadata,
        "natural_metrics": metrics,
        "balanced_metrics": {name: _summary(values) for name, values in aggregate.items()},
        "per_bin_auroc": {
            name: {key: _summary(values) for key, values in bins.items()}
            for name, bins in per_bin.items()
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    arrays = {
        "object_id": reference["object_id"],
        "image_id": reference["image_id"],
        "label": reference["label"],
        "gen_pos": reference["gen_pos"],
    }
    arrays.update({f"score__{name}": values for name, values in scores.items()})
    npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    for name in scores:
        natural = metrics[name]
        balanced = output["balanced_metrics"][name]
        print(
            f"{name:18s} overall={natural['overall_auroc']:.4f} "
            f"within={natural['within_bin_auroc']:.4f} "
            f"balanced={balanced['mean']:.4f}"
        )
    print(f"balanced mentions={selected_count}, repeats={args.repeats}")
    print(f"wrote {output_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
