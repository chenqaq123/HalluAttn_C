#!/usr/bin/env python3
"""Matched width-10 comparison of VAR-MLP and the SinkDetect sensor.

Protocol: docs/iteration_protocols/var_mlp_matched_within_bin.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import (  # noqa: E402
    matched_pair_auc,
    roc_auc,
    within_bin_auc,
)
from detection.scripts.analyze_allocation_probe import (  # noqa: E402
    _fit_model,
    _predict_standardized,
)
from detection.scripts.analyze_layer_head_factorial import (  # noqa: E402
    _bootstrap_contrasts,
    _caption_length_proxy,
    _within_bin_spearman,
    sha256_file,
)
from detection.scripts.evaluate_layer_head_factorial import (  # noqa: E402
    average_precision,
    best_balanced_threshold,
    capacity_transform,
)
from detection.scripts.evaluate_position_balanced_oof import (  # noqa: E402
    _summary,
    position_balanced_indices,
)
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _apply_bin_means,
    _bin_means,
    _image_folds,
    _load_npz_parts,
    _spearman,
)


REPRESENTATIONS = (
    "native_l05_18",
    "no_rope_l05_18",
    "native_l00_31",
    "no_rope_l00_31",
)
MODEL_SEEDS = (17, 927, 1701)
CAPACITY_MODES = ("native", "pca64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features_glob",
        default="experiments/visual_value_evidence/features_shard*.npz",
    )
    parser.add_argument(
        "--compact_cache",
        default=(
            "experiments/allocation_layer_head_factorial/"
            "allocation_mass.npz"
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=17)
    parser.add_argument(
        "--model_seeds",
        default=",".join(str(value) for value in MODEL_SEEDS),
    )
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--min_bin_count", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--pca_seed", type=int, default=20260811)
    parser.add_argument("--pca_iterations", type=int, default=4)
    parser.add_argument("--balanced_repeats", type=int, default=1000)
    parser.add_argument("--balanced_seed", type=int, default=1701)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--capacity_modes",
        default=",".join(CAPACITY_MODES),
        help="Comma-separated subset of: native,pca64.",
    )
    parser.add_argument(
        "--output_dir", default="experiments/var_mlp_matched"
    )
    return parser.parse_args()


def build_representations(
    raw_mass: np.ndarray,
    no_rope_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the four frozen source-by-breadth post-row views."""

    expected = (32, 2, 32)
    if raw_mass.ndim != 4 or raw_mass.shape[1:] != expected:
        raise ValueError(f"Unexpected raw mass shape: {raw_mass.shape}")
    if no_rope_mass.shape != raw_mass.shape:
        raise ValueError(
            f"Raw/No-RoPE shapes differ: {raw_mass.shape} "
            f"versus {no_rope_mass.shape}"
        )

    def view(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
        selected = values[:, lo : hi + 1, 1, :]
        return np.ascontiguousarray(
            selected.reshape(selected.shape[0], -1), dtype=np.float32
        )

    return {
        "native_l05_18": view(raw_mass, 5, 18),
        "no_rope_l05_18": view(no_rope_mass, 5, 18),
        "native_l00_31": view(raw_mass, 0, 31),
        "no_rope_l00_31": view(no_rope_mass, 0, 31),
    }


def _align_features(
    features_glob: str,
    compact_cache: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    paths = sorted(glob.glob(features_glob))
    if not paths:
        raise FileNotFoundError(features_glob)
    shard_audits = []
    for path in paths:
        shard = np.load(path, allow_pickle=False)
        feature_names = list(shard["per_head_feature_names"].astype(str))
        if feature_names != [
            "visual_mass",
            "value_norm",
            "target_dot",
            "target_cosine",
            "target_gain",
        ]:
            raise ValueError(f"Unexpected feature names in {path}: {feature_names}")
        if list(shard["step_names"].astype(str)) != ["pre", "post"]:
            raise ValueError(f"Unexpected temporal rows in {path}")
        if float(shard["max_raw_mass_error"]) != 0.0:
            raise ValueError(f"Native VAR reconstruction failed in {path}")
        shard_audits.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "n": int(shard["object_id"].size),
                "max_raw_mass_error": float(shard["max_raw_mass_error"]),
            }
        )
        shard.close()

    data = _load_npz_parts(features_glob)
    order = np.argsort(data["object_id"], kind="mergesort")
    object_ids = np.asarray(data["object_id"])[order].astype(np.int64)
    if not np.array_equal(object_ids, np.arange(object_ids.size)):
        raise ValueError("VVE object IDs are not the complete sorted range")
    reference = {
        key: np.asarray(data[key])[order]
        for key in (
            "object_id",
            "image_id",
            "token_pos",
            "gen_pos",
            "label",
            "word",
        )
    }
    raw_mass = np.asarray(data["raw"])[order, ..., 0].astype(np.float32)
    no_rope_mass = (
        np.asarray(data["no_rope"])[order, ..., 0].astype(np.float32)
    )
    del data

    compact = np.load(compact_cache, allow_pickle=False)
    compact_order = np.argsort(compact["object_id"], kind="mergesort")
    for key in ("object_id", "image_id", "token_pos", "gen_pos", "label", "word"):
        value = np.asarray(compact[key])[compact_order]
        if not np.array_equal(reference[key], value):
            raise ValueError(f"Compact allocation cache differs on {key}")
    compact_allocation = np.asarray(compact["allocation"])[compact_order]
    max_no_rope_error = float(
        np.max(np.abs(no_rope_mass - compact_allocation))
    )
    if max_no_rope_error > 1e-6:
        raise ValueError(
            "No-RoPE feature identity failed: "
            f"max error {max_no_rope_error:.3e}"
        )
    compact.close()

    provenance = {
        "feature_shards": shard_audits,
        "compact_cache": compact_cache,
        "compact_cache_sha256": sha256_file(compact_cache),
        "max_no_rope_cache_error": max_no_rope_error,
    }
    return (
        reference,
        build_representations(raw_mass, no_rope_mass),
        provenance,
    )


def _fit_representation(
    raw: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    test_masks: list[np.ndarray],
    model_seeds: tuple[int, ...],
    capacity_mode: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    n = labels.size
    seed_scores = np.full((len(model_seeds), n), np.nan, dtype=np.float64)
    seed_residuals = np.full_like(seed_scores, np.nan)
    seed_decisions = np.full((len(model_seeds), n), np.nan, dtype=np.float32)
    seed_thresholds = np.full(
        (len(model_seeds), len(test_masks)), np.nan, dtype=np.float64
    )
    transforms = []
    device = torch.device(args.device)
    for fold_index, test_mask in enumerate(test_masks):
        train_indices = np.where(~test_mask)[0]
        test_indices = np.where(test_mask)[0]
        if capacity_mode == "native":
            transformed = raw
            transform = {
                "fold": fold_index,
                "raw_dim": int(raw.shape[1]),
                "output_dim": int(raw.shape[1]),
                "method": "identity",
                "explained_variance_fraction": 1.0,
            }
        elif capacity_mode == "pca64":
            transformed, transform = capacity_transform(
                raw,
                train_indices,
                args.pca_dim,
                device,
                seed=args.pca_seed + fold_index,
                pca_iterations=args.pca_iterations,
            )
            transform = {"fold": fold_index, **transform}
        else:
            raise ValueError(f"Unknown capacity mode: {capacity_mode}")
        transforms.append(transform)

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
                model,
                train_standardized,
                device,
                args.batch_size * 4,
            )
            test_scores = _predict_standardized(
                model,
                test_standardized,
                device,
                args.batch_size * 4,
            )
            threshold = best_balanced_threshold(
                labels[train_indices], train_scores
            )
            seed_scores[seed_index, test_indices] = test_scores
            seed_residuals[seed_index, test_indices] = _apply_bin_means(
                test_scores,
                positions[test_indices],
                _bin_means(
                    train_scores,
                    positions[train_indices],
                    args.bin_width,
                ),
                args.bin_width,
            )
            seed_decisions[seed_index, test_indices] = (
                test_scores >= threshold
            ).astype(np.float32)
            seed_thresholds[seed_index, fold_index] = threshold
            del model, train_standardized, test_standardized
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if capacity_mode == "pca64":
            del transformed
        print(
            f"{capacity_mode}: completed fold {fold_index + 1}/"
            f"{len(test_masks)} for {raw.shape[1]} raw features",
            flush=True,
        )

    for name, values in (
        ("scores", seed_scores),
        ("residuals", seed_residuals),
        ("decisions", seed_decisions),
        ("thresholds", seed_thresholds),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} for {capacity_mode}")
    return (
        seed_scores,
        seed_residuals,
        seed_decisions,
        seed_thresholds,
        transforms,
    )


def _metric_summary(
    labels: np.ndarray,
    positions: np.ndarray,
    scores: np.ndarray,
    residuals: np.ndarray,
    decisions: np.ndarray,
    bin_width: int,
) -> dict:
    within = within_bin_auc(
        labels, scores, positions, bin_width=bin_width
    )
    matched = matched_pair_auc(labels, scores, positions, delta=5)
    valid_bins = [
        item for item in within["bins"] if item["auc"] is not None
    ]
    late_bins = [
        item
        for item in valid_bins
        if item["bin_start"] >= 80 and item["bin_end"] <= 139
    ]
    return {
        "overall_auroc": roc_auc(labels, scores),
        "within_bin_auroc": within["auc"],
        "average_precision": average_precision(labels, scores),
        "matched_pair_auroc": matched["auc"],
        "matched_pair_count": matched["pairs"],
        "train_fold_residual_auroc": roc_auc(labels, residuals),
        "balanced_accuracy_train_threshold": float(
            0.5
            * (
                np.mean(decisions[labels == 1] >= 0.5)
                + np.mean(decisions[labels == 0] < 0.5)
            )
        ),
        "macro_bin_auroc": float(
            np.mean([item["auc"] for item in valid_bins])
        ),
        "late_80_139_auroc": float(
            np.average(
                [item["auc"] for item in late_bins],
                weights=[item["n"] for item in late_bins],
            )
        ),
        "within_bins": within["bins"],
    }


def _position_balanced(
    labels: np.ndarray,
    positions: np.ndarray,
    score_map: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, dict], int, list[dict]]:
    rng = np.random.default_rng(args.balanced_seed)
    values = {
        name: {"auroc": [], "accuracy_at_zero": []}
        for name in score_map
    }
    selected_count = None
    bin_summary = None
    for repeat in range(args.balanced_repeats):
        indices, current_bins = position_balanced_indices(
            labels,
            positions,
            args.bin_width,
            args.min_bin_count,
            rng,
        )
        if selected_count is None:
            selected_count = int(indices.size)
            bin_summary = current_bins
        for name, scores in score_map.items():
            auc = roc_auc(labels[indices], scores[indices])
            if auc is None:
                raise ValueError(f"Undefined balanced AUROC for {name}")
            values[name]["auroc"].append(float(auc))
            values[name]["accuracy_at_zero"].append(
                float(np.mean((scores[indices] >= 0.0) == labels[indices]))
            )
        if (repeat + 1) % 100 == 0:
            print(
                f"literal position balance {repeat + 1}/"
                f"{args.balanced_repeats}",
                flush=True,
            )
    return (
        {
            name: {
                metric: _summary(metric_values)
                for metric, metric_values in method_values.items()
            }
            for name, method_values in values.items()
        },
        int(selected_count),
        list(bin_summary),
    )


def _contrast_weights() -> dict[str, dict[str, float]]:
    return {
        "sinkdetect_minus_var_mlp": {
            "no_rope_l00_31": 1.0,
            "native_l05_18": -1.0,
        },
        "no_rope_minus_native_l05_18": {
            "no_rope_l05_18": 1.0,
            "native_l05_18": -1.0,
        },
        "no_rope_minus_native_l00_31": {
            "no_rope_l00_31": 1.0,
            "native_l00_31": -1.0,
        },
        "all_minus_middle_native": {
            "native_l00_31": 1.0,
            "native_l05_18": -1.0,
        },
        "all_minus_middle_no_rope": {
            "no_rope_l00_31": 1.0,
            "no_rope_l05_18": -1.0,
        },
        "source_by_breadth_interaction": {
            "no_rope_l00_31": 1.0,
            "native_l00_31": -1.0,
            "no_rope_l05_18": -1.0,
            "native_l05_18": 1.0,
        },
    }


def _markdown(output: dict) -> str:
    lines = [
        "# VAR-MLP versus SinkDetect matched comparison",
        "",
    ]
    for mode in CAPACITY_MODES:
        if mode not in output["metrics"]:
            continue
        lines.extend(
            [
                f"## {mode}",
                "",
                "| representation | dim | overall | within-bin | macro-bin | balanced AUROC | accuracy@0 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in REPRESENTATIONS:
            value = output["metrics"][mode][name]
            balanced = value["literal_position_balanced"]
            lines.append(
                f"| {name} | {value['input_dim']} | "
                f"{value['overall_auroc']:.4f} | "
                f"{value['within_bin_auroc']:.4f} | "
                f"{value['macro_bin_auroc']:.4f} | "
                f"{balanced['auroc']['mean']:.4f} | "
                f"{balanced['accuracy_at_zero']['mean']:.4f} |"
            )
        contrast = output["contrasts"][mode][
            "sinkdetect_minus_var_mlp"
        ]
        lines.extend(
            [
                "",
                "Named contrast, SinkDetect minus VAR-MLP: "
                f"{contrast['median']:+.4f} "
                f"[{contrast['low']:+.4f}, {contrast['high']:+.4f}], "
                f"Holm p={contrast['p_holm']:.4g}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- Reproduction gate: **{output['decisions']['reproduction_gate']}**",
            f"- Native named contrast: **{output['decisions']['native_named_positive']}**",
            f"- 64-D named contrast: **{output['decisions']['pca64_named_positive']}**",
            f"- Feature-advantage gate: **{output['decisions']['feature_advantage_gate']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    capacity_modes = tuple(
        value.strip()
        for value in args.capacity_modes.split(",")
        if value.strip()
    )
    if not capacity_modes or any(
        value not in CAPACITY_MODES for value in capacity_modes
    ):
        raise ValueError(f"Invalid capacity modes: {capacity_modes}")
    model_seeds = tuple(
        int(value.strip())
        for value in args.model_seeds.split(",")
        if value.strip()
    )
    if model_seeds != MODEL_SEEDS:
        raise ValueError(
            f"Registered model seeds are {MODEL_SEEDS}, got {model_seeds}"
        )

    reference, representations, provenance = _align_features(
        args.features_glob, args.compact_cache
    )
    labels = reference["label"].astype(np.int32)
    positions = reference["gen_pos"].astype(np.int32)
    image_ids = reference["image_id"].astype(np.int64)
    token_positions = reference["token_pos"].astype(np.int64)
    if labels.size != 16426 or int(labels.sum()) != 4009:
        raise ValueError(
            f"Frozen population mismatch: n={labels.size}, hallu={labels.sum()}"
        )
    if np.unique(image_ids).size != 4977:
        raise ValueError("Frozen image count mismatch")
    folds = _image_folds(image_ids, args.folds, args.fold_seed)
    test_masks = [np.isin(image_ids, fold_images) for fold_images in folds]
    heldout_count = np.stack(test_masks).sum(axis=0)
    if not np.array_equal(heldout_count, np.ones(labels.size, dtype=np.int64)):
        raise ValueError("Every mention must occur in exactly one held-out fold")
    caption_proxy = _caption_length_proxy(image_ids, token_positions)

    metrics: dict[str, dict] = {}
    score_maps: dict[str, dict[str, np.ndarray]] = {}
    score_arrays: dict[str, np.ndarray] = {}
    fit_metadata: dict[str, dict] = {}
    for mode in capacity_modes:
        metrics[mode] = {}
        score_maps[mode] = {}
        fit_metadata[mode] = {}
        for name in REPRESENTATIONS:
            print(f"fitting {mode}/{name}", flush=True)
            raw = representations[name]
            (
                seed_scores,
                seed_residuals,
                seed_decisions,
                seed_thresholds,
                transforms,
            ) = _fit_representation(
                raw,
                labels,
                positions,
                test_masks,
                model_seeds,
                mode,
                args,
            )
            scores = seed_scores.mean(axis=0)
            residuals = seed_residuals.mean(axis=0)
            decisions = seed_decisions.mean(axis=0)
            value = _metric_summary(
                labels,
                positions,
                scores,
                residuals,
                decisions,
                args.bin_width,
            )
            value["input_dim"] = int(
                raw.shape[1] if mode == "native" else args.pca_dim
            )
            value["raw_input_dim"] = int(raw.shape[1])
            value["seed_metrics"] = [
                _metric_summary(
                    labels,
                    positions,
                    seed_scores[index],
                    seed_residuals[index],
                    seed_decisions[index],
                    args.bin_width,
                )
                for index in range(len(model_seeds))
            ]
            value["confounds"] = {
                "spearman_position": _spearman(scores, positions),
                "within_bin_spearman_position": _within_bin_spearman(
                    scores,
                    positions,
                    labels,
                    positions,
                    args.bin_width,
                ),
                "spearman_caption_length_proxy": _spearman(
                    scores, caption_proxy
                ),
                "within_bin_spearman_caption_length_proxy": (
                    _within_bin_spearman(
                        scores,
                        caption_proxy,
                        labels,
                        positions,
                        args.bin_width,
                    )
                ),
            }
            metrics[mode][name] = value
            score_maps[mode][name] = scores
            score_arrays[f"score__{mode}__{name}"] = scores
            score_arrays[f"residual__{mode}__{name}"] = residuals
            score_arrays[f"seed_score__{mode}__{name}"] = seed_scores
            fit_metadata[mode][name] = {
                "transforms": transforms,
                "thresholds": seed_thresholds.tolist(),
            }

    balanced_input = {
        f"{mode}__{name}": score_maps[mode][name]
        for mode in capacity_modes
        for name in REPRESENTATIONS
    }
    balanced, balanced_count, balanced_bins = _position_balanced(
        labels, positions, balanced_input, args
    )
    for mode in capacity_modes:
        for name in REPRESENTATIONS:
            metrics[mode][name]["literal_position_balanced"] = balanced[
                f"{mode}__{name}"
            ]

    contrasts = {}
    for mode in capacity_modes:
        print(f"bootstrapping {mode} contrasts", flush=True)
        contrasts[mode] = _bootstrap_contrasts(
            labels,
            positions,
            image_ids,
            score_maps[mode],
            _contrast_weights(),
            args,
        )

    seed17_ours = metrics["native"]["no_rope_l00_31"][
        "seed_metrics"
    ][0]["within_bin_auroc"]
    reproduction_error = abs(float(seed17_ours) - 0.8859)
    reproduction_gate = reproduction_error <= 0.002

    def positive(mode: str) -> bool:
        if mode not in contrasts:
            return False
        value = contrasts[mode]["sinkdetect_minus_var_mlp"]
        return bool(value["low"] > 0.0 and value["p_holm"] < 0.05)

    decisions = {
        "reproduction_gate": bool(reproduction_gate),
        "seed17_ours_within_bin": float(seed17_ours),
        "seed17_reproduction_abs_error": float(reproduction_error),
        "native_named_positive": positive("native"),
        "pca64_named_positive": positive("pca64"),
        "feature_advantage_gate": bool(
            reproduction_gate
            and positive("native")
            and positive("pca64")
        ),
    }
    if "native" not in capacity_modes:
        raise ValueError("Registered reproduction gate requires native mode")
    if set(capacity_modes) != set(CAPACITY_MODES):
        raise ValueError("Formal run requires both registered capacity modes")

    output = {
        "protocol": (
            "docs/iteration_protocols/var_mlp_matched_within_bin.md"
        ),
        "n": int(labels.size),
        "images": int(np.unique(image_ids).size),
        "hallucinated": int(labels.sum()),
        "grounded": int((labels == 0).sum()),
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "model_seeds": list(model_seeds),
        "bin_width": args.bin_width,
        "capacity_modes": list(capacity_modes),
        "balanced": {
            "repeats": args.balanced_repeats,
            "seed": args.balanced_seed,
            "selected_count_per_repeat": balanced_count,
            "valid_bins": balanced_bins,
        },
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "holm_family_size": len(_contrast_weights()),
        },
        "provenance": provenance,
        "representations": {
            name: {
                "raw_dim": int(representations[name].shape[1]),
                "source": "no_rope" if name.startswith("no_rope") else "native",
                "layers": "5--18" if name.endswith("l05_18") else "0--31",
                "row": "post",
            }
            for name in REPRESENTATIONS
        },
        "fit_metadata": fit_metadata,
        "metrics": metrics,
        "contrasts": contrasts,
        "decisions": decisions,
        "elapsed_seconds": time.time() - started,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "summary.json"
    output_npz = output_dir / "oof_scores.npz"
    output_markdown = output_dir / "summary.md"
    output_json.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(
        output_npz,
        object_id=reference["object_id"],
        image_id=image_ids,
        token_pos=token_positions,
        gen_pos=positions,
        label=labels,
        word=reference["word"],
        caption_length_proxy=caption_proxy,
        model_seeds=np.asarray(model_seeds, dtype=np.int64),
        **score_arrays,
    )
    output_markdown.write_text(_markdown(output) + "\n")
    print(json.dumps(decisions, indent=2), flush=True)
    print(
        f"wrote {output_json}, {output_npz}, and {output_markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
