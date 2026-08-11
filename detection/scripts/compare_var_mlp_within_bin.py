#!/usr/bin/env python3
"""Minimal fixed-split comparison of VAR-MLP and ours.

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

from detection.baselines.metrics import roc_auc, within_bin_auc  # noqa: E402
from detection.scripts.analyze_allocation_probe import (  # noqa: E402
    _fit_model,
    _predict_standardized,
)
from detection.scripts.analyze_layer_head_factorial import (  # noqa: E402
    sha256_file,
)
from detection.scripts.evaluate_layer_head_factorial import (  # noqa: E402
    best_balanced_threshold,
)
from detection.scripts.evaluate_visual_value_evidence import (  # noqa: E402
    _load_npz_parts,
)


REPRESENTATIONS = ("var_mlp", "ours")
MODEL_SEED = 17


def fixed_image_split(
    image_ids: np.ndarray,
    train_images: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one deterministic, label-blind image-level train/test split."""

    image_ids = np.asarray(image_ids, dtype=np.int64)
    unique_ids = np.unique(image_ids)
    if unique_ids.size != image_ids.size:
        raise ValueError("Caption manifest image IDs must be unique")
    if train_images <= 0 or train_images >= unique_ids.size:
        raise ValueError(
            f"train_images must be in [1, {unique_ids.size - 1}]"
        )
    permutation = np.random.default_rng(seed).permutation(unique_ids)
    return permutation[:train_images], permutation[train_images:]


def build_representations(
    raw_mass: np.ndarray,
    no_rope_mass: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct only the two native representations in the protocol."""

    expected = (32, 2, 32)
    if raw_mass.ndim != 4 or raw_mass.shape[1:] != expected:
        raise ValueError(f"Unexpected raw mass shape: {raw_mass.shape}")
    if no_rope_mass.shape != raw_mass.shape:
        raise ValueError(
            f"Raw/No-RoPE shapes differ: {raw_mass.shape} versus "
            f"{no_rope_mass.shape}"
        )

    var_mlp = raw_mass[:, 5:19, 1, :]
    ours = no_rope_mass[:, :, 1, :]
    return {
        "var_mlp": np.ascontiguousarray(
            var_mlp.reshape(var_mlp.shape[0], -1), dtype=np.float32
        ),
        "ours": np.ascontiguousarray(
            ours.reshape(ours.shape[0], -1), dtype=np.float32
        ),
    }


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
    parser.add_argument(
        "--generation_manifest",
        default="experiments/coco_llava_7b/generation.json",
    )
    parser.add_argument("--train_images", type=int, default=4000)
    parser.add_argument("--split_seed", type=int, default=17)
    parser.add_argument("--model_seed", type=int, default=MODEL_SEED)
    parser.add_argument("--bin_width", type=int, default=10)
    parser.add_argument("--min_bin_count", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output_dir", default="experiments/var_mlp_matched"
    )
    return parser.parse_args()


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
            raise ValueError(f"Unexpected feature names in {path}")
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
    no_rope_mass = np.asarray(data["no_rope"])[order, ..., 0].astype(
        np.float32
    )
    del data

    compact = np.load(compact_cache, allow_pickle=False)
    compact_order = np.argsort(compact["object_id"], kind="mergesort")
    for key in (
        "object_id",
        "image_id",
        "token_pos",
        "gen_pos",
        "label",
        "word",
    ):
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


def _balanced_accuracy(
    labels: np.ndarray,
    decisions: np.ndarray,
) -> float:
    return float(
        0.5
        * (
            np.mean(decisions[labels == 1])
            + np.mean(~decisions[labels == 0])
        )
    )


def _fit_once(
    features: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    device = torch.device(args.device)
    model, mean, std = _fit_model(
        features,
        labels,
        positions,
        train_indices,
        args,
        args.model_seed,
    )
    train_standardized = (
        features[train_indices].astype(np.float32) - mean
    ) / std
    test_standardized = (
        features[test_indices].astype(np.float32) - mean
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
    threshold = best_balanced_threshold(labels[train_indices], train_scores)
    decisions = test_scores >= threshold

    if not (
        np.isfinite(train_scores).all()
        and np.isfinite(test_scores).all()
        and np.isfinite(threshold)
    ):
        raise ValueError("Non-finite score or threshold")

    test_labels = labels[test_indices]
    test_positions = positions[test_indices]
    within = within_bin_auc(
        test_labels,
        test_scores,
        test_positions,
        bin_width=args.bin_width,
        min_n=args.min_bin_count,
    )
    if within["auc"] is None:
        raise ValueError("Within-bin AUROC is undefined")
    metrics = {
        "input_dim": int(features.shape[1]),
        "overall_auroc": roc_auc(test_labels, test_scores),
        "within_bin_auroc": float(within["auc"]),
        "accuracy_train_threshold": float(np.mean(decisions == test_labels)),
        "balanced_accuracy_train_threshold": _balanced_accuracy(
            test_labels, decisions
        ),
        "train_selected_threshold": float(threshold),
        "within_bins": within["bins"],
    }
    del model, train_standardized, test_standardized
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return train_scores, test_scores, float(threshold), metrics


def _markdown(output: dict) -> str:
    lines = [
        "# Minimal fixed-split VAR-MLP versus ours",
        "",
        "| method | dim | overall AUROC | with-bin AUROC | accuracy | balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in REPRESENTATIONS:
        value = output["metrics"][name]
        lines.append(
            f"| {name} | {value['input_dim']} | "
            f"{value['overall_auroc']:.4f} | "
            f"{value['within_bin_auroc']:.4f} | "
            f"{value['accuracy_train_threshold']:.4f} | "
            f"{value['balanced_accuracy_train_threshold']:.4f} |"
        )
    contrast = output["contrast_ours_minus_var_mlp"]
    lines.extend(
        [
            "",
            "Ours minus VAR-MLP: "
            f"overall AUROC {contrast['overall_auroc']:+.4f}; "
            f"with-bin AUROC {contrast['within_bin_auroc']:+.4f}.",
            "",
            f"Integrity gate: **{output['integrity_gate']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    started = time.time()
    if (
        args.train_images != 4000
        or args.split_seed != 17
        or args.model_seed != MODEL_SEED
        or args.bin_width != 10
        or args.min_bin_count != 30
        or args.epochs != 10
    ):
        raise ValueError("Formal run must use the registered defaults")

    reference, representations, provenance = _align_features(
        args.features_glob,
        args.compact_cache,
    )
    labels = reference["label"].astype(np.int32)
    positions = reference["gen_pos"].astype(np.int32)
    image_ids = reference["image_id"].astype(np.int64)
    token_positions = reference["token_pos"].astype(np.int64)
    if labels.size != 16426 or int(labels.sum()) != 4009:
        raise ValueError("Frozen mention population changed")
    if np.unique(image_ids).size != 4977:
        raise ValueError("Frozen eligible-image count changed")

    generation_manifest = Path(args.generation_manifest)
    generation_records = json.loads(generation_manifest.read_text())
    manifest_image_ids = np.asarray(
        [int(record["image_id"]) for record in generation_records],
        dtype=np.int64,
    )
    if manifest_image_ids.size != 5000:
        raise ValueError("Generation manifest must contain 5,000 images")
    train_image_ids, test_image_ids = fixed_image_split(
        manifest_image_ids,
        args.train_images,
        args.split_seed,
    )
    train_mask = np.isin(image_ids, train_image_ids)
    test_mask = np.isin(image_ids, test_image_ids)
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise ValueError("Every mention must map to exactly one split")
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    if (
        train_indices.size != 13132
        or test_indices.size != 3294
        or np.unique(image_ids[train_indices]).size != 3985
        or np.unique(image_ids[test_indices]).size != 992
    ):
        raise ValueError("Frozen split counts changed")

    metrics = {}
    test_score_arrays = {}
    thresholds = {}
    all_scores_finite = True
    for name in REPRESENTATIONS:
        print(f"fitting {name}", flush=True)
        train_scores, test_scores, threshold, value = _fit_once(
            representations[name],
            labels,
            positions,
            train_indices,
            test_indices,
            args,
        )
        metrics[name] = value
        test_score_arrays[f"score__{name}"] = test_scores
        thresholds[name] = threshold
        all_scores_finite = bool(
            all_scores_finite
            and np.isfinite(train_scores).all()
            and np.isfinite(test_scores).all()
        )

    split_gate = bool(
        manifest_image_ids.size == 5000
        and train_image_ids.size == 4000
        and test_image_ids.size == 1000
        and np.intersect1d(train_image_ids, test_image_ids).size == 0
        and train_indices.size + test_indices.size == labels.size
    )
    feature_identity_gate = bool(
        provenance["max_no_rope_cache_error"] <= 1e-6
    )
    integrity_gate = bool(
        split_gate and feature_identity_gate and all_scores_finite
    )
    contrast = {
        "overall_auroc": float(
            metrics["ours"]["overall_auroc"]
            - metrics["var_mlp"]["overall_auroc"]
        ),
        "within_bin_auroc": float(
            metrics["ours"]["within_bin_auroc"]
            - metrics["var_mlp"]["within_bin_auroc"]
        ),
    }

    output = {
        "protocol": (
            "docs/iteration_protocols/var_mlp_matched_within_bin.md"
        ),
        "caption_manifest_images": int(manifest_image_ids.size),
        "eligible_images": int(np.unique(image_ids).size),
        "eligible_mentions": int(labels.size),
        "split": {
            "seed": args.split_seed,
            "train_manifest_images": int(train_image_ids.size),
            "test_manifest_images": int(test_image_ids.size),
            "train_eligible_images": int(
                np.unique(image_ids[train_indices]).size
            ),
            "test_eligible_images": int(
                np.unique(image_ids[test_indices]).size
            ),
            "train_mentions": int(train_indices.size),
            "test_mentions": int(test_indices.size),
            "train_hallucinated": int(labels[train_indices].sum()),
            "test_hallucinated": int(labels[test_indices].sum()),
        },
        "training": {
            "model_seed": args.model_seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "conditional_bin_width": args.bin_width,
        },
        "provenance": {
            **provenance,
            "generation_manifest": str(generation_manifest),
            "generation_manifest_sha256": sha256_file(generation_manifest),
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "contrast_ours_minus_var_mlp": contrast,
        "gates": {
            "split_gate": split_gate,
            "feature_identity_gate": feature_identity_gate,
            "finite_scores_gate": all_scores_finite,
        },
        "integrity_gate": integrity_gate,
        "elapsed_seconds": time.time() - started,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "summary.json"
    output_npz = output_dir / "heldout_scores.npz"
    output_markdown = output_dir / "summary.md"
    split_manifest = output_dir / "split_manifest.json"
    output_json.write_text(json.dumps(output, indent=2) + "\n")
    output_markdown.write_text(_markdown(output) + "\n")
    split_manifest.write_text(
        json.dumps(
            {
                "source_manifest": str(generation_manifest),
                "algorithm": "sort IDs then numpy default_rng permutation",
                "seed": args.split_seed,
                "train_image_ids": [int(value) for value in train_image_ids],
                "test_image_ids": [int(value) for value in test_image_ids],
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(
        output_npz,
        object_id=reference["object_id"][test_indices],
        image_id=image_ids[test_indices],
        token_pos=token_positions[test_indices],
        gen_pos=positions[test_indices],
        label=labels[test_indices],
        word=reference["word"][test_indices],
        **test_score_arrays,
    )
    print(_markdown(output), flush=True)
    print(
        f"wrote {output_json}, {output_npz}, {output_markdown}, and "
        f"{split_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
