#!/usr/bin/env python3
"""Merge, audit, and test the registered layer/head factorial experiment."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import roc_auc, within_bin_auc  # noqa: E402
from detection.scripts.evaluate_layer_head_factorial import (  # noqa: E402
    _cell_oof,
    _metric_summary,
    enumerate_cells,
)
from detection.scripts.evaluate_position_balanced_oof import (  # noqa: E402
    _summary,
    position_balanced_indices,
)
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _image_folds,
    _spearman,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid_npz_glob",
        default=(
            "experiments/allocation_layer_head_factorial/"
            "grid_shard_??_of_08.npz"
        ),
    )
    parser.add_argument(
        "--grid_json_glob",
        default=(
            "experiments/allocation_layer_head_factorial/"
            "grid_shard_??_of_08.json"
        ),
    )
    parser.add_argument(
        "--nested_json_glob",
        default=(
            "experiments/allocation_layer_head_factorial/nested_selection/"
            "nested_shard_??_of_08.json"
        ),
    )
    parser.add_argument(
        "--compact_cache",
        default="experiments/allocation_layer_head_factorial/allocation_mass.npz",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=17)
    parser.add_argument("--model_seeds", default="17,927,1701")
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--min_bin_count", type=int, default=30)
    parser.add_argument("--balanced_repeats", type=int, default=1000)
    parser.add_argument("--balanced_seed", type=int, default=20260728)
    parser.add_argument("--bootstrap_draws", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260728)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pca_iterations", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_json",
        default=(
            "experiments/allocation_layer_head_factorial/"
            "factorial_analysis.json"
        ),
    )
    parser.add_argument(
        "--output_npz",
        default=(
            "experiments/allocation_layer_head_factorial/"
            "factorial_scores.npz"
        ),
    )
    parser.add_argument(
        "--output_markdown",
        default=(
            "experiments/allocation_layer_head_factorial/"
            "factorial_summary.md"
        ),
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""

    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def _load_grid(
    npz_pattern: str,
    json_pattern: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    npz_paths = sorted(glob.glob(npz_pattern))
    json_paths = sorted(glob.glob(json_pattern))
    if not npz_paths or not json_paths:
        raise FileNotFoundError(
            f"Missing grid shards: npz={npz_pattern}, json={json_pattern}"
        )
    arrays: dict[str, np.ndarray] = {}
    reference = None
    for path in npz_paths:
        shard = np.load(path)
        current = {
            key: shard[key]
            for key in ("object_id", "image_id", "gen_pos", "label", "model_seeds")
        }
        if reference is None:
            reference = current
            arrays.update(current)
            if "word" in shard.files:
                arrays["word"] = shard["word"]
        else:
            for key, value in current.items():
                if not np.array_equal(reference[key], value):
                    raise ValueError(f"{path} does not align on {key}")
        for key in shard.files:
            if key.startswith(("score__", "score_seed__", "decision_seed__")):
                if key in arrays:
                    raise ValueError(f"Duplicate grid key: {key}")
                arrays[key] = shard[key]

    cells = {}
    grid_metadata = []
    for path in json_paths:
        shard = json.loads(Path(path).read_text())
        grid_metadata.append(
            {
                key: shard[key]
                for key in (
                    "shard_index",
                    "num_shards",
                    "device",
                    "elapsed_seconds",
                )
            }
        )
        for name, value in shard["cells"].items():
            if name in cells:
                raise ValueError(f"Duplicate cell metadata: {name}")
            cells[name] = value
    expected = {cell.name for cell in enumerate_cells()}
    if set(cells) != expected:
        raise ValueError(
            f"Grid incomplete: missing={sorted(expected - set(cells))}, "
            f"extra={sorted(set(cells) - expected)}"
        )
    return arrays, cells, {"shards": grid_metadata, "paths": npz_paths + json_paths}


def _load_nested(pattern: str, folds: int, model_seeds: tuple[int, ...]) -> dict:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    tasks = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for task in payload["tasks"]:
            key = (int(task["outer_fold"]), int(task["model_seed"]))
            if key in tasks:
                raise ValueError(f"Duplicate nested-selection task: {key}")
            tasks[key] = task
    expected = {
        (fold_index, model_seed)
        for fold_index in range(folds)
        for model_seed in model_seeds
    }
    if set(tasks) != expected:
        raise ValueError(
            f"Nested selection incomplete: missing={sorted(expected - set(tasks))}"
        )
    return {"tasks": tasks, "paths": paths}


def _selected_layer_scores(
    arrays: dict[str, np.ndarray],
    nested_tasks: dict[tuple[int, int], dict],
    folds: list[np.ndarray],
    model_seeds: tuple[int, ...],
    treatment: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    image_ids = arrays["image_id"]
    n = image_ids.size
    seed_scores = np.full((len(model_seeds), n), np.nan, dtype=np.float64)
    seed_decisions = np.full((len(model_seeds), n), np.nan, dtype=np.float32)
    selections = []
    for fold_index, test_images in enumerate(folds):
        test_mask = np.isin(image_ids, test_images)
        for seed_index, model_seed in enumerate(model_seeds):
            task = nested_tasks[(fold_index, model_seed)]
            layer = int(task["selected_layer"])
            cell = f"single_l{layer:02d}__{treatment}"
            seed_scores[seed_index, test_mask] = arrays[
                f"score_seed__{cell}"
            ][seed_index, test_mask]
            seed_decisions[seed_index, test_mask] = arrays[
                f"decision_seed__{cell}"
            ][seed_index, test_mask]
            selections.append(
                {
                    "outer_fold": fold_index,
                    "model_seed": model_seed,
                    "selected_layer": layer,
                    "treatment": treatment,
                }
            )
    if not np.isfinite(seed_scores).all() or not np.isfinite(seed_decisions).all():
        raise ValueError(f"Incomplete selected-layer output for {treatment}")
    return seed_scores, seed_decisions, selections


def _caption_length_proxy(
    image_ids: np.ndarray, token_positions: np.ndarray
) -> np.ndarray:
    output = np.zeros(image_ids.size, dtype=np.float32)
    for image_id in np.unique(image_ids):
        mask = image_ids == image_id
        output[mask] = float(token_positions[mask].max())
    return output


def _fit_controls(
    labels: np.ndarray,
    positions: np.ndarray,
    image_ids: np.ndarray,
    caption_proxy: np.ndarray,
    test_masks: list[np.ndarray],
    model_seeds: tuple[int, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    controls = {
        "control_position_only": positions[:, None].astype(np.float32),
        "control_position_length": np.stack(
            (positions.astype(np.float32), caption_proxy), axis=1
        ),
    }
    scores = {}
    metadata = {}
    for name, raw in controls.items():
        # _cell_oof uses the Cell only for stable PCA seeding and messages.
        pseudo_cell = type(
            "ControlCell",
            (),
            {"name": name},
        )()
        seed_scores, seed_decisions, transform = _cell_oof(
            raw,
            labels,
            positions,
            test_masks,
            pseudo_cell,
            model_seeds,
            args,
        )
        scores[name] = seed_scores.mean(axis=0)
        scores[f"{name}__seed"] = seed_scores
        metadata[name] = {
            "metrics": _metric_summary(
                labels,
                positions,
                scores[name],
                seed_decisions.mean(axis=0),
                args.bin_width,
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
            "transform": transform,
        }
    return scores, metadata


def _within_bin_spearman(
    scores: np.ndarray,
    other: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    bin_width: int,
) -> float | None:
    total = 0
    weighted = 0.0
    for bin_id in np.unique(positions // bin_width):
        mask = positions // bin_width == bin_id
        if mask.sum() < 30 or np.unique(labels[mask]).size < 2:
            continue
        value = _spearman(scores[mask], other[mask])
        if value is not None:
            total += int(mask.sum())
            weighted += float(mask.sum()) * value
    return weighted / total if total else None


def _literal_position_balanced(
    labels: np.ndarray,
    positions: np.ndarray,
    score_map: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, dict], int, list[dict]]:
    rng = np.random.default_rng(args.balanced_seed)
    values = {name: [] for name in score_map}
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
            value = roc_auc(labels[indices], scores[indices])
            if value is None:
                raise ValueError(f"Undefined balanced AUROC for {name}")
            values[name].append(value)
        if (repeat + 1) % 100 == 0:
            print(
                f"literal position balance {repeat + 1}/{args.balanced_repeats}",
                flush=True,
            )
    return (
        {name: _summary(metric_values) for name, metric_values in values.items()},
        int(selected_count),
        list(bin_summary),
    )


def _bootstrap_contrasts(
    labels: np.ndarray,
    positions: np.ndarray,
    image_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    contrast_weights: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, dict]:
    rng = np.random.RandomState(args.bootstrap_seed)
    unique_images = np.asarray(sorted(set(image_ids.tolist())))
    by_image = {
        image_id: np.where(image_ids == image_id)[0] for image_id in unique_images
    }
    distributions = {
        name: np.empty(args.bootstrap_draws, dtype=np.float64)
        for name in contrast_weights
    }
    needed = sorted(
        {
            score_name
            for weights in contrast_weights.values()
            for score_name in weights
        }
    )
    for draw in range(args.bootstrap_draws):
        sampled = rng.choice(unique_images, size=unique_images.size, replace=True)
        indices = np.concatenate([by_image[image_id] for image_id in sampled])
        metrics = {}
        for score_name in needed:
            value = within_bin_auc(
                labels[indices],
                scores[score_name][indices],
                positions[indices],
                bin_width=args.bin_width,
            )["auc"]
            if value is None:
                raise ValueError(f"Undefined bootstrap metric for {score_name}")
            metrics[score_name] = value
        for contrast_name, weights in contrast_weights.items():
            distributions[contrast_name][draw] = sum(
                weight * metrics[score_name]
                for score_name, weight in weights.items()
            )
        if (draw + 1) % 1000 == 0:
            print(
                f"cluster bootstrap {draw + 1}/{args.bootstrap_draws}",
                flush=True,
            )
    output = {}
    p_values = {}
    for name, distribution in distributions.items():
        low, median, high = np.percentile(distribution, [2.5, 50.0, 97.5])
        below = (np.sum(distribution <= 0.0) + 1) / (distribution.size + 1)
        above = (np.sum(distribution >= 0.0) + 1) / (distribution.size + 1)
        p_value = min(1.0, 2.0 * min(below, above))
        p_values[name] = p_value
        output[name] = {
            "draws": int(distribution.size),
            "low": float(low),
            "median": float(median),
            "high": float(high),
            "p_two_sided": float(p_value),
        }
    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        output[name]["p_holm"] = float(value)
    return output


def _markdown_table(
    metrics: dict[str, dict],
    selected_name: str,
    decisions: dict,
) -> str:
    names = [
        "control_position_only",
        "control_position_length",
        selected_name,
        "band_00_07__per_head",
        "band_08_15__per_head",
        "band_16_23__per_head",
        "band_24_31__per_head",
        "half_00_15__per_head",
        "half_16_31__per_head",
        "all_00_31__head_mean",
        "all_00_31__head_deviation",
        "all_00_31__per_head",
    ]
    lines = [
        "# Layer/head factorial summary",
        "",
        "| representation | within-bin AUROC | global AUROC | AP | balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        value = metrics[name]
        lines.append(
            f"| {name} | {value['within_bin_auroc']:.4f} | "
            f"{value['overall_auroc']:.4f} | {value['average_precision']:.4f} | "
            f"{value['balanced_accuracy_train_threshold']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Registered decisions",
            "",
            f"- Cross-layer gate: **{decisions['cross_layer_gate']}**",
            f"- Cross-head gate: **{decisions['cross_head_gate']}**",
            f"- Positive interaction gate: **{decisions['positive_interaction_gate']}**",
            f"- Full mechanism gate: **{decisions['full_mechanism_gate']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    model_seeds = tuple(
        int(value.strip()) for value in args.model_seeds.split(",") if value.strip()
    )
    arrays, cell_metadata, grid_provenance = _load_grid(
        args.grid_npz_glob, args.grid_json_glob
    )
    nested = _load_nested(args.nested_json_glob, args.folds, model_seeds)
    labels = arrays["label"].astype(np.int32)
    positions = arrays["gen_pos"].astype(np.int32)
    image_ids = arrays["image_id"].astype(np.int64)
    object_ids = arrays["object_id"].astype(np.int64)
    if not np.array_equal(arrays["model_seeds"], np.asarray(model_seeds)):
        raise ValueError("Grid model seeds do not match analysis model seeds")
    folds = _image_folds(image_ids, args.folds, args.fold_seed)
    test_masks = [np.isin(image_ids, fold_images) for fold_images in folds]

    compact = np.load(args.compact_cache)
    for key in ("object_id", "image_id", "gen_pos", "label"):
        if not np.array_equal(compact[key], arrays[key]):
            raise ValueError(f"Compact cache does not align on {key}")
    if "token_pos" not in compact.files:
        raise ValueError("Compact cache must be regenerated with token_pos")
    caption_proxy = _caption_length_proxy(image_ids, compact["token_pos"])

    selected_per_head_seed, selected_per_head_decisions, selections = (
        _selected_layer_scores(
            arrays, nested["tasks"], folds, model_seeds, "per_head"
        )
    )
    selected_head_mean_seed, selected_head_mean_decisions, mean_selections = (
        _selected_layer_scores(
            arrays, nested["tasks"], folds, model_seeds, "head_mean"
        )
    )
    selected_name = "train_selected_single__per_head"
    selected_mean_name = "train_selected_single__head_mean_same_layer"
    score_map = {
        key.removeprefix("score__"): value
        for key, value in arrays.items()
        if key.startswith("score__")
    }
    score_map[selected_name] = selected_per_head_seed.mean(axis=0)
    score_map[selected_mean_name] = selected_head_mean_seed.mean(axis=0)

    control_scores, control_metadata = _fit_controls(
        labels,
        positions,
        image_ids,
        caption_proxy,
        test_masks,
        model_seeds,
        args,
    )
    for name in ("control_position_only", "control_position_length"):
        score_map[name] = control_scores[name]

    metrics = {
        name: value["metrics"] for name, value in cell_metadata.items()
    }
    metrics[selected_name] = _metric_summary(
        labels,
        positions,
        score_map[selected_name],
        selected_per_head_decisions.mean(axis=0),
        args.bin_width,
    )
    metrics[selected_mean_name] = _metric_summary(
        labels,
        positions,
        score_map[selected_mean_name],
        selected_head_mean_decisions.mean(axis=0),
        args.bin_width,
    )
    for name, value in control_metadata.items():
        metrics[name] = value["metrics"]

    balanced_names = {
        name: score_map[name]
        for name in sorted(score_map)
        if name in metrics
    }
    balanced_metrics, balanced_count, balanced_bins = _literal_position_balanced(
        labels, positions, balanced_names, args
    )
    for name, value in balanced_metrics.items():
        metrics[name]["literal_position_balanced_auroc"] = value

    contrast_weights = {
        "all_per_head_minus_selected_single": {
            "all_00_31__per_head": 1.0,
            selected_name: -1.0,
        },
        "all_per_head_minus_band_00_07": {
            "all_00_31__per_head": 1.0,
            "band_00_07__per_head": -1.0,
        },
        "all_per_head_minus_band_08_15": {
            "all_00_31__per_head": 1.0,
            "band_08_15__per_head": -1.0,
        },
        "all_per_head_minus_band_16_23": {
            "all_00_31__per_head": 1.0,
            "band_16_23__per_head": -1.0,
        },
        "all_per_head_minus_band_24_31": {
            "all_00_31__per_head": 1.0,
            "band_24_31__per_head": -1.0,
        },
        "all_per_head_minus_half_00_15": {
            "all_00_31__per_head": 1.0,
            "half_00_15__per_head": -1.0,
        },
        "all_per_head_minus_half_16_31": {
            "all_00_31__per_head": 1.0,
            "half_16_31__per_head": -1.0,
        },
        "all_per_head_minus_all_head_mean": {
            "all_00_31__per_head": 1.0,
            "all_00_31__head_mean": -1.0,
        },
        "all_head_deviation_minus_all_per_head": {
            "all_00_31__head_deviation": 1.0,
            "all_00_31__per_head": -1.0,
        },
        "layer_by_head_interaction": {
            "all_00_31__per_head": 1.0,
            selected_name: -1.0,
            "all_00_31__head_mean": -1.0,
            selected_mean_name: 1.0,
        },
    }
    contrasts = _bootstrap_contrasts(
        labels,
        positions,
        image_ids,
        score_map,
        contrast_weights,
        args,
    )

    cross_layer_names = [
        "all_per_head_minus_selected_single",
        "all_per_head_minus_band_00_07",
        "all_per_head_minus_band_08_15",
        "all_per_head_minus_band_16_23",
        "all_per_head_minus_band_24_31",
    ]
    cross_layer_gate = all(
        contrasts[name]["low"] > 0.0 and contrasts[name]["p_holm"] < 0.05
        for name in cross_layer_names
    )
    cross_head = contrasts["all_per_head_minus_all_head_mean"]
    cross_head_gate = cross_head["low"] > 0.0 and cross_head["p_holm"] < 0.05
    interaction = contrasts["layer_by_head_interaction"]
    interaction_gate = (
        interaction["low"] > 0.0 and interaction["p_holm"] < 0.05
    )
    control_auc = metrics["control_position_length"]["within_bin_auroc"]
    full_auc = metrics["all_00_31__per_head"]["within_bin_auroc"]
    deviation_auc = metrics["all_00_31__head_deviation"]["within_bin_auroc"]
    excess = full_auc - control_auc
    deviation_retention = (
        (deviation_auc - control_auc) / excess if excess > 0 else None
    )
    decisions = {
        "cross_layer_gate": bool(cross_layer_gate),
        "cross_head_gate": bool(cross_head_gate),
        "positive_interaction_gate": bool(interaction_gate),
        "full_mechanism_gate": bool(
            cross_layer_gate and cross_head_gate and interaction_gate
        ),
        "head_deviation_excess_retention": deviation_retention,
        "head_deviation_retains_95pct": bool(
            deviation_retention is not None and deviation_retention >= 0.95
        ),
    }

    confounds = {}
    for name in (
        selected_name,
        "all_00_31__head_mean",
        "all_00_31__head_deviation",
        "all_00_31__per_head",
    ):
        confounds[name] = {
            "spearman_position": _spearman(score_map[name], positions),
            "within_bin_spearman_position": _within_bin_spearman(
                score_map[name], positions, labels, positions, args.bin_width
            ),
            "spearman_caption_length_proxy": _spearman(
                score_map[name], caption_proxy
            ),
            "within_bin_spearman_caption_length_proxy": _within_bin_spearman(
                score_map[name],
                caption_proxy,
                labels,
                positions,
                args.bin_width,
            ),
        }

    output = {
        "protocol": "docs/iteration_protocols/allocation_layer_head_factorial.md",
        "n": int(labels.size),
        "images": int(np.unique(image_ids).size),
        "hallucinated": int(labels.sum()),
        "grounded": int((labels == 0).sum()),
        "folds": args.folds,
        "fold_seed": args.fold_seed,
        "model_seeds": list(model_seeds),
        "capacity_dim": 64,
        "balanced": {
            "repeats": args.balanced_repeats,
            "seed": args.balanced_seed,
            "selected_count_per_repeat": balanced_count,
            "valid_bins": balanced_bins,
        },
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "holm_family_size": len(contrast_weights),
        },
        "provenance": {
            "compact_cache": args.compact_cache,
            "compact_cache_sha256": sha256_file(args.compact_cache),
            "grid": grid_provenance,
            "nested_paths": nested["paths"],
        },
        "nested_selection": {
            "per_head": selections,
            "head_mean_same_layer": mean_selections,
        },
        "metrics": metrics,
        "controls": control_metadata,
        "contrasts": contrasts,
        "confounds": confounds,
        "decisions": decisions,
        "elapsed_seconds": time.time() - started,
    }
    output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    output_markdown = Path(args.output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2) + "\n")
    np.savez_compressed(
        output_npz,
        object_id=object_ids,
        image_id=image_ids,
        gen_pos=positions,
        label=labels,
        caption_length_proxy=caption_proxy,
        selected_single_per_head_seed_scores=selected_per_head_seed,
        selected_single_head_mean_seed_scores=selected_head_mean_seed,
        **{f"score__{name}": values for name, values in score_map.items()},
    )
    output_markdown.write_text(
        _markdown_table(metrics, selected_name, decisions) + "\n"
    )
    print(json.dumps(decisions, indent=2), flush=True)
    print(
        f"wrote {output_json}, {output_npz}, and {output_markdown}",
        flush=True,
    )


if __name__ == "__main__":
    main()
