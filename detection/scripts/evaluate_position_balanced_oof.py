#!/usr/bin/env python3
"""Evaluate frozen OOF detector scores on explicitly position-balanced tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.baselines.metrics import roc_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position_free_oof",
        default="experiments/position_free_detector/iteration1_oof.npz",
    )
    parser.add_argument(
        "--vve_oof",
        default=(
            "experiments/visual_value_evidence/"
            "iteration2_size_shared_seed_oof.npz"
        ),
    )
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--min_bin_count", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--output",
        default=(
            "experiments/visual_value_evidence/"
            "position_balanced_oof_metrics.json"
        ),
    )
    return parser.parse_args()


def position_balanced_indices(
    labels: np.ndarray,
    positions: np.ndarray,
    bin_width: int,
    min_bin_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    bins = positions // bin_width
    selected = []
    bin_summary = []
    for bin_id in np.unique(bins):
        in_bin = np.where(bins == bin_id)[0]
        by_class = [in_bin[labels[in_bin] == label] for label in (0, 1)]
        if in_bin.size < min_bin_count or any(indices.size == 0 for indices in by_class):
            continue
        per_class = min(indices.size for indices in by_class)
        selected.extend(
            rng.choice(indices, size=per_class, replace=False) for indices in by_class
        )
        bin_summary.append(
            {
                "bin_start": int(bin_id * bin_width),
                "bin_end": int((bin_id + 1) * bin_width - 1),
                "original_grounded": int(by_class[0].size),
                "original_hallucinated": int(by_class[1].size),
                "selected_per_class": int(per_class),
            }
        )
    if not selected:
        raise ValueError("No valid position bins for balanced evaluation")
    return np.concatenate(selected), bin_summary


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    low, median, high = np.percentile(array, [2.5, 50.0, 97.5])
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p2_5": float(low),
        "median": float(median),
        "p97_5": float(high),
    }


def _load_aligned(path: str, reference: dict[str, np.ndarray]) -> np.lib.npyio.NpzFile:
    data = np.load(path)
    for key in ("object_id", "label", "gen_pos"):
        if not np.array_equal(data[key], reference[key]):
            raise ValueError(f"{path} does not align on {key}")
    return data


def main() -> None:
    args = parse_args()
    position_free = np.load(args.position_free_oof)
    reference = {
        key: position_free[key] for key in ("object_id", "label", "gen_pos")
    }
    vve = _load_aligned(args.vve_oof, reference)
    labels = reference["label"].astype(np.int32)
    positions = reference["gen_pos"].astype(np.int32)
    scores = {
        "raw_l31_per_head": position_free["score__probe_raw_l31_all"],
        "no_rope_l31_per_head": position_free["score__probe_no_rope_l31_all"],
        "attention_confidence": vve["score__probe_attention_confidence"],
        "all_layer_allocation_mass": vve["score__probe_allocation_mass"],
        "vve_core": vve["score__probe_vve_no_rope_gain"],
        "vve_full": vve["score__probe_vve_no_rope"],
    }
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("OOF scores must all be finite")

    rng = np.random.default_rng(args.seed)
    results = {
        name: {"auroc": [], "accuracy_at_zero": []} for name in scores
    }
    per_bin_results: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {} for name in scores
    }
    bin_summary = None
    selected_count = None
    for _ in range(args.repeats):
        indices, current_summary = position_balanced_indices(
            labels,
            positions,
            args.bin_width,
            args.min_bin_count,
            rng,
        )
        if bin_summary is None:
            bin_summary = current_summary
            selected_count = int(indices.size)
        for name, values in scores.items():
            auc = roc_auc(labels[indices], values[indices])
            if auc is None:
                raise ValueError(f"AUROC is undefined for {name}")
            prediction = values[indices] >= 0.0
            results[name]["auroc"].append(auc)
            results[name]["accuracy_at_zero"].append(
                float(np.mean(prediction == labels[indices]))
            )
            for item in current_summary:
                bin_key = f"{item['bin_start']}-{item['bin_end']}"
                in_bin = (
                    (positions[indices] >= item["bin_start"])
                    & (positions[indices] <= item["bin_end"])
                )
                bin_indices = indices[in_bin]
                bin_auc = roc_auc(labels[bin_indices], values[bin_indices])
                if bin_auc is None:
                    raise ValueError(f"AUROC is undefined for {name} in {bin_key}")
                bin_metrics = per_bin_results[name].setdefault(
                    bin_key, {"auroc": [], "accuracy_at_zero": []}
                )
                bin_metrics["auroc"].append(bin_auc)
                bin_prediction = values[bin_indices] >= 0.0
                bin_metrics["accuracy_at_zero"].append(
                    float(np.mean(bin_prediction == labels[bin_indices]))
                )

    output = {
        "position_free_oof": args.position_free_oof,
        "vve_oof": args.vve_oof,
        "bin_width": args.bin_width,
        "min_bin_count": args.min_bin_count,
        "repeats": args.repeats,
        "seed": args.seed,
        "selected_count_per_repeat": selected_count,
        "valid_bins": bin_summary,
        "metrics": {
            name: {
                metric: _summary(values) for metric, values in metrics.items()
            }
            for name, metrics in results.items()
        },
        "per_bin_metrics": {
            name: {
                bin_key: {
                    metric: _summary(values)
                    for metric, values in bin_metrics.items()
                }
                for bin_key, bin_metrics in method_bins.items()
            }
            for name, method_bins in per_bin_results.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    for name, metrics in output["metrics"].items():
        print(
            f"{name:28s} AUROC={metrics['auroc']['mean']:.4f} "
            f"accuracy@0={metrics['accuracy_at_zero']['mean']:.4f}"
        )
    print(f"balanced mentions={selected_count}, repeats={args.repeats}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
