#!/usr/bin/env python3
"""Evaluate no-RoPE spatial allocation and sink-deletion ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import roc_auc  # noqa: E402
from detection.scripts.evaluate_position_balanced_oof import (  # noqa: E402
    _summary,
    position_balanced_indices,
)
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _bootstrap_difference,
    _image_folds,
    _load_npz_parts,
    _metric_bundle,
    _mlp_oof,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_glob",
        default="experiments/spatial_allocation/features_shard*.npz",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--weighting", choices=("size", "uniform_bin"), default="size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap_iterations", type=int, default=500)
    parser.add_argument("--balanced_repeats", type=int, default=1000)
    parser.add_argument("--balanced_seed", type=int, default=1701)
    parser.add_argument("--min_bin_count", type=int, default=30)
    parser.add_argument(
        "--only_feature_sets",
        default="",
        help="Optional comma-separated subset of registered feature sets.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_npz", required=True)
    return parser.parse_args()


def _registered_feature_sets(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = [str(value) for value in data["feature_names"].tolist()]
    expected = ["visual_mass", "top10_mass", "top10_neighbor_coherence"]
    if names != expected:
        raise ValueError(f"Unexpected spatial features: {names}")
    feature_sets: dict[str, np.ndarray] = {}
    definitions = {
        "mass": [0],
        "top10": [1],
        "connectivity": [2],
        "mass_top10": [0, 1],
        "mass_connectivity": [0, 2],
    }
    for branch in ("no_rope", "sink_clean"):
        values = np.asarray(data[branch], dtype=np.float32)
        if values.ndim != 5 or values.shape[1:] != (32, 2, 32, 3):
            raise ValueError(f"Unexpected {branch} shape: {values.shape}")
        for suffix, indices in definitions.items():
            selected = values[..., indices]
            feature_sets[f"{branch}__{suffix}"] = np.ascontiguousarray(
                selected.reshape(selected.shape[0], -1), dtype=np.float32
            )
    return feature_sets


def _bin_comparison(left: dict, right: dict, minimum_per_class: int = 30) -> dict:
    right_bins = {
        (item["bin_start"], item["bin_end"]): item for item in right["within_bins"]
    }
    rows = []
    for item in left["within_bins"]:
        key = (item["bin_start"], item["bin_end"])
        other = right_bins.get(key)
        if item["auc"] is None or other is None or other["auc"] is None:
            continue
        item_pos = int(item["hallucinated"])
        other_pos = int(other["hallucinated"])
        stable = min(
            item_pos,
            int(item["n"]) - item_pos,
            other_pos,
            int(other["n"]) - other_pos,
        ) >= minimum_per_class
        rows.append(
            {
                "bin_start": int(item["bin_start"]),
                "bin_end": int(item["bin_end"]),
                "n": int(item["n"]),
                "stable": bool(stable),
                "left_auc": float(item["auc"]),
                "right_auc": float(other["auc"]),
                "difference": float(item["auc"] - other["auc"]),
            }
        )
    stable_rows = [row for row in rows if row["stable"]]
    left_within = left["within_bin_auroc"]
    right_within = right["within_bin_auroc"]
    return {
        "left_minus_right_within_bin": (
            float(left_within - right_within)
            if left_within is not None and right_within is not None
            else None
        ),
        "stable_bin_wins": int(sum(row["difference"] > 0.0 for row in stable_rows)),
        "stable_bin_count": len(stable_rows),
        "min_stable_bin_difference": (
            float(min(row["difference"] for row in stable_rows)) if stable_rows else None
        ),
        "per_bin": rows,
    }


def _balanced_metrics(
    labels: np.ndarray,
    positions: np.ndarray,
    scores: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict:
    if args.balanced_repeats <= 0:
        return {}
    rng = np.random.default_rng(args.balanced_seed)
    aggregate = {name: {"auroc": [], "accuracy_at_zero": []} for name in scores}
    per_bin: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {} for name in scores
    }
    selected_count = None
    bin_summary = None
    for _ in range(args.balanced_repeats):
        indices, current_summary = position_balanced_indices(
            labels,
            positions,
            args.bin_width,
            args.min_bin_count,
            rng,
        )
        if selected_count is None:
            selected_count = int(indices.size)
            bin_summary = current_summary
        for name, values in scores.items():
            auc = roc_auc(labels[indices], values[indices])
            if auc is None:
                raise ValueError(f"Balanced AUROC is undefined for {name}")
            aggregate[name]["auroc"].append(float(auc))
            aggregate[name]["accuracy_at_zero"].append(
                float(np.mean((values[indices] >= 0.0) == labels[indices]))
            )
            for item in current_summary:
                key = f"{item['bin_start']}-{item['bin_end']}"
                selected = indices[
                    (positions[indices] >= item["bin_start"])
                    & (positions[indices] <= item["bin_end"])
                ]
                bin_auc = roc_auc(labels[selected], values[selected])
                if bin_auc is None:
                    raise ValueError(f"Balanced AUROC is undefined for {name} in {key}")
                target = per_bin[name].setdefault(
                    key, {"auroc": [], "accuracy_at_zero": []}
                )
                target["auroc"].append(float(bin_auc))
                target["accuracy_at_zero"].append(
                    float(np.mean((values[selected] >= 0.0) == labels[selected]))
                )
    return {
        "repeats": args.balanced_repeats,
        "seed": args.balanced_seed,
        "selected_count_per_repeat": selected_count,
        "valid_bins": bin_summary,
        "metrics": {
            name: {metric: _summary(values) for metric, values in methods.items()}
            for name, methods in aggregate.items()
        },
        "per_bin_metrics": {
            name: {
                key: {metric: _summary(values) for metric, values in methods.items()}
                for key, methods in bins.items()
            }
            for name, bins in per_bin.items()
        },
    }


def main() -> None:
    args = parse_args()
    data = _load_npz_parts(args.features_glob)
    labels = np.asarray(data["label"], dtype=np.int32)
    positions = np.asarray(data["gen_pos"], dtype=np.int32)
    image_ids = np.asarray(data["image_id"], dtype=np.int64)
    words = np.asarray(data["word"])
    object_ids = np.asarray(data["object_id"], dtype=np.int64)
    if np.unique(object_ids).size != object_ids.size:
        raise ValueError("Object IDs must be unique")
    if not np.array_equal(np.asarray(data["layer_indices"]), np.arange(32)):
        raise ValueError("The registered audit requires all 32 layers")

    feature_sets = _registered_feature_sets(data)
    if args.only_feature_sets.strip():
        requested = {name.strip() for name in args.only_feature_sets.split(",") if name.strip()}
        unknown = requested - feature_sets.keys()
        if unknown:
            raise ValueError(f"Unknown feature sets: {sorted(unknown)}")
        feature_sets = {name: value for name, value in feature_sets.items() if name in requested}
    folds = _image_folds(image_ids, args.folds, args.seed)
    test_masks = [np.isin(image_ids, fold) for fold in folds]

    scores: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}

    def format_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    for name, features in feature_sets.items():
        score, residual = _mlp_oof(
            features,
            labels,
            positions,
            test_masks,
            args,
            model_seed_offset=0,
        )
        if not np.isfinite(score).all() or not np.isfinite(residual).all():
            raise ValueError(f"Non-finite OOF values for {name}")
        scores[name] = score
        residuals[name] = residual
        metrics[name] = _metric_bundle(
            labels, score, residual, positions, words, args.bin_width
        )
        print(
            f"{name:36s} overall={format_metric(metrics[name]['overall_auroc'])} "
            f"within={format_metric(metrics[name]['within_bin_auroc'])} "
            f"macro={format_metric(metrics[name]['macro_bin_auroc'])}"
        )

    comparison_pairs = {
        "connectivity_vs_top10_no_rope": (
            "no_rope__mass_connectivity",
            "no_rope__mass_top10",
        ),
        "connectivity_vs_top10_sink_clean": (
            "sink_clean__mass_connectivity",
            "sink_clean__mass_top10",
        ),
        "sink_mass": ("sink_clean__mass", "no_rope__mass"),
        "sink_top10": ("sink_clean__top10", "no_rope__top10"),
        "sink_connectivity": (
            "sink_clean__connectivity",
            "no_rope__connectivity",
        ),
        "sink_mass_connectivity": (
            "sink_clean__mass_connectivity",
            "no_rope__mass_connectivity",
        ),
        "candidate_no_rope_vs_mass": (
            "no_rope__mass_connectivity",
            "no_rope__mass",
        ),
        "candidate_sink_clean_vs_mass": (
            "sink_clean__mass_connectivity",
            "no_rope__mass",
        ),
    }
    comparisons = {}
    for comparison_name, (left_name, right_name) in comparison_pairs.items():
        if left_name not in scores or right_name not in scores:
            continue
        comparison = _bin_comparison(metrics[left_name], metrics[right_name])
        if args.bootstrap_iterations > 0:
            comparison["image_bootstrap"] = _bootstrap_difference(
                labels,
                positions,
                image_ids,
                scores[left_name],
                scores[right_name],
                args.bin_width,
                args.bootstrap_iterations,
                args.seed + 1700,
            )
        comparisons[comparison_name] = comparison

    balanced = _balanced_metrics(labels, positions, scores, args)
    sink_mass = np.asarray(data["sink_mass"], dtype=np.float32)
    sink_count = np.asarray(data["sink_count"], dtype=np.float32)
    diagnostics = {
        "mean_sink_count_by_layer": sink_count.mean(axis=0).tolist(),
        "mean_sink_attention_mass": float(sink_mass.mean()),
        "mean_sink_attention_mass_pre": float(sink_mass[:, :, 0].mean()),
        "mean_sink_attention_mass_post": float(sink_mass[:, :, 1].mean()),
    }
    output = {
        "protocol": "docs/iteration_protocols/no_rope_spatial_allocation_audit.md",
        "features_glob": args.features_glob,
        "n": int(labels.size),
        "images": int(np.unique(image_ids).size),
        "labels": {"grounded": int((labels == 0).sum()), "hallucinated": int((labels == 1).sum())},
        "folds": args.folds,
        "seed": args.seed,
        "bin_width": args.bin_width,
        "shared_model_seed_offset": 0,
        "metrics": metrics,
        "comparisons": comparisons,
        "position_balanced": balanced,
        "diagnostics": diagnostics,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    score_arrays = {
        "object_id": object_ids,
        "image_id": image_ids,
        "label": labels,
        "gen_pos": positions,
    }
    score_arrays.update({f"score__{name}": value for name, value in scores.items()})
    score_arrays.update(
        {f"residual__{name}": value for name, value in residuals.items()}
    )
    npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **score_arrays)
    print(f"wrote {output_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
