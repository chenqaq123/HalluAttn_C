#!/usr/bin/env python3
"""Evaluate LH-Shape linear prefilters for TDEV-region CHAIR cascades.

This cache-only audit asks whether a cheap internal TDEV-lite score can reduce
expensive OWLv2 calls while preserving the object-mention removals that full
TDEV-region would make. The cascade is conservative: it keeps the full TDEV risk
threshold fixed, but only applies it to mentions selected by a prefilter.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_lh_shape import load_cache, load_gen_pos  # noqa: E402
from evaluate_lh_shape_linear import (  # noqa: E402
    image_group_linear_scores,
    layer_indices_from_spec,
    read_baseline_scores,
)


def parse_float_list(spec: str) -> list[float]:
    values = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("float list cannot be empty")
    return values


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = values.size
    k = max(1, min(int(np.ceil(n * fraction)), n))
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def two_stage_score(target: np.ndarray, margin: np.ndarray, low: float, high: float, margin_threshold: float) -> np.ndarray:
    high_branch = target - high
    medium_branch = np.minimum(target - low, margin - margin_threshold)
    return np.maximum(high_branch, medium_branch)


def tdev_risk(rows: list[dict[str, str]], args: argparse.Namespace) -> np.ndarray:
    target = np.asarray([float(row["target_score"]) for row in rows], dtype=np.float64)
    neighbor = np.asarray([float(row["best_neighbor_score"]) for row in rows], dtype=np.float64)
    margin = np.asarray([float(row["tdev_margin"]) for row in rows], dtype=np.float64)
    dominance = np.maximum(neighbor - target, 0.0)
    if args.tdev_score == "target_absence_plus_neighbor_dominance":
        return -target + args.neighbor_dominance_alpha * dominance
    if args.tdev_score == "hybrid_positive_branch_absence":
        return -two_stage_score(target, margin, args.hybrid_low, args.hybrid_high, args.hybrid_margin)
    if args.tdev_score == "hybrid_mcc_positive_branch_absence":
        return -two_stage_score(target, margin, args.hybrid_low, args.hybrid_high, args.hybrid_mcc_margin)
    if args.tdev_score == "two_stage_absence":
        return -two_stage_score(target, margin, args.two_stage_low, args.two_stage_high, args.two_stage_margin)
    raise ValueError(f"unknown tdev_score {args.tdev_score!r}")


def summarize_prefilter(
    labels: np.ndarray,
    prefilter_values: np.ndarray,
    tdev_values: np.ndarray,
    prefilter_name: str,
    call_rate: float,
    delete_rate: float,
) -> dict:
    selected = top_fraction_mask(prefilter_values, call_rate)
    full_delete = top_fraction_mask(tdev_values, delete_rate)
    cascade_delete_mask = selected & full_delete

    total = int(labels.size)
    hallucinated = int(labels.sum())
    grounded = total - hallucinated
    selected_n = int(selected.sum())
    selected_hallu = int(labels[selected].sum())
    full_deleted = int(full_delete.sum())
    full_deleted_hallu = int(labels[full_delete].sum())
    full_deleted_grounded = full_deleted - full_deleted_hallu
    cascade_deleted = int(cascade_delete_mask.sum())
    cascade_deleted_hallu = int(labels[cascade_delete_mask].sum())
    cascade_deleted_grounded = cascade_deleted - cascade_deleted_hallu

    return {
        "prefilter": prefilter_name,
        "call_rate": call_rate,
        "delete_rate": delete_rate,
        "detector_calls": selected_n,
        "detector_call_savings": 1.0 - selected_n / total,
        "candidate_precision": selected_hallu / selected_n if selected_n else 0.0,
        "candidate_hallucination_coverage": selected_hallu / hallucinated if hallucinated else 0.0,
        "full_tdev_deleted": full_deleted,
        "full_tdev_deleted_hallucinated": full_deleted_hallu,
        "full_tdev_deleted_grounded": full_deleted_grounded,
        "cascade_deleted": cascade_deleted,
        "cascade_deleted_hallucinated": cascade_deleted_hallu,
        "cascade_deleted_grounded": cascade_deleted_grounded,
        "retained_full_deletions": cascade_deleted / full_deleted if full_deleted else 0.0,
        "retained_full_hallucination_reduction": cascade_deleted_hallu / full_deleted_hallu if full_deleted_hallu else 0.0,
        "retained_full_grounded_loss": cascade_deleted_grounded / full_deleted_grounded if full_deleted_grounded else 0.0,
        "hallucination_reduction": cascade_deleted_hallu / hallucinated if hallucinated else 0.0,
        "grounded_loss": cascade_deleted_grounded / grounded if grounded else 0.0,
        "cascade_precision": cascade_deleted_hallu / cascade_deleted if cascade_deleted else 0.0,
    }


def build_lh_scores(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    feats, labels, token_pos, image_ids, layer_indices, _feature_names = load_cache(args.cache, args.cache_glob)
    gen_pos = load_gen_pos(args.generation_json, image_ids, token_pos)
    rows, csv_labels, csv_image_ids, csv_gen_pos = read_baseline_scores(Path(args.baseline_scores_csv))
    if not (np.array_equal(labels, csv_labels) and np.array_equal(image_ids, csv_image_ids) and np.array_equal(gen_pos, csv_gen_pos)):
        raise ValueError("Per-head cache rows do not align with baseline score rows")
    layer_indexes = layer_indices_from_spec(layer_indices, args.layer_set)
    X = feats[:, layer_indexes, :, :].reshape(feats.shape[0], -1).astype(np.float64)
    name = "lh_shape_linear_layers_" + "_".join(str(int(layer_indices[idx])) for idx in layer_indexes)
    scores, folds = image_group_linear_scores(
        X,
        labels,
        image_ids,
        folds=args.folds,
        seed=args.seed,
        l2=args.l2,
        lr=args.lr,
        epochs=args.epochs,
        name=name,
    )
    return scores, {"name": name, "folds": folds, "rows": rows, "labels": labels, "image_ids": image_ids, "gen_pos": gen_pos}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LH-Shape prefilters for TDEV-region cascades")
    p.add_argument("--cache", default="experiments/coco_llava_7b_rows/per_head_row_cache.npz")
    p.add_argument("--cache_glob", default=None)
    p.add_argument("--generation_json", default="experiments/coco_llava_7b/generation.json")
    p.add_argument("--baseline_scores_csv", default="detection/baselines/results/coco_llava_7b_baselines/baseline_scores.csv")
    p.add_argument("--owlv2_scores_csv", default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv")
    p.add_argument("--output_dir", default="detection/baselines/results/lh_shape_tdev_cascade")
    p.add_argument("--layer_set", default="22,31")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--prefilter_cols", default="ic_hallu_score,entropy_hallu_score,pas_layer0_hallu_score,sinkdetect_best_within_bin_hallu_score")
    p.add_argument("--call_rates", default="0.10,0.25,0.50,0.75")
    p.add_argument("--delete_rates", default="0.05,0.10")
    p.add_argument(
        "--tdev_score",
        choices=[
            "target_absence_plus_neighbor_dominance",
            "hybrid_positive_branch_absence",
            "hybrid_mcc_positive_branch_absence",
            "two_stage_absence",
        ],
        default="hybrid_positive_branch_absence",
    )
    p.add_argument("--two_stage_low", type=float, default=0.10)
    p.add_argument("--two_stage_high", type=float, default=0.16)
    p.add_argument("--two_stage_margin", type=float, default=-0.15)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    args = p.parse_args()

    lh_scores, meta = build_lh_scores(args)
    labels = meta["labels"]
    baseline_rows = meta["rows"]
    owlv2_rows = read_csv_rows(Path(args.owlv2_scores_csv))
    owlv2_labels = np.asarray([int(row["label"]) for row in owlv2_rows], dtype=np.int32)
    owlv2_image_ids = np.asarray([int(row["image_id"]) for row in owlv2_rows], dtype=np.int64)
    owlv2_gen_pos = np.asarray([int(row["gen_pos"]) for row in owlv2_rows], dtype=np.int32)
    if not (
        np.array_equal(labels, owlv2_labels)
        and np.array_equal(meta["image_ids"], owlv2_image_ids)
        and np.array_equal(meta["gen_pos"], owlv2_gen_pos)
    ):
        raise ValueError("LH-Shape rows do not align with OWLv2 score rows")

    prefilters: dict[str, np.ndarray] = {meta["name"]: lh_scores, "position_only_gen_pos": meta["gen_pos"].astype(np.float64)}
    for col in [item.strip() for item in args.prefilter_cols.split(",") if item.strip()]:
        prefilters[col] = np.asarray([float(row[col]) for row in baseline_rows], dtype=np.float64)

    tdev_values = tdev_risk(owlv2_rows, args)
    call_rates = parse_float_list(args.call_rates)
    delete_rates = parse_float_list(args.delete_rates)
    summary_rows = []
    for name, values in prefilters.items():
        for call_rate in call_rates:
            for delete_rate in delete_rates:
                summary_rows.append(summarize_prefilter(labels, values, tdev_values, name, call_rate, delete_rate))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "lh_shape_tdev_cascade_metrics.csv", summary_rows)
    payload = {
        "cache": args.cache_glob or args.cache,
        "generation_json": args.generation_json,
        "baseline_scores_csv": args.baseline_scores_csv,
        "owlv2_scores_csv": args.owlv2_scores_csv,
        "num_mentions": int(labels.size),
        "hallucinated_mentions": int(labels.sum()),
        "grounded_mentions": int(labels.size - labels.sum()),
        "lh_prefilter": meta["name"],
        "tdev_score": args.tdev_score,
        "call_rates": call_rates,
        "delete_rates": delete_rates,
        "metrics": summary_rows,
        "folds_detail": meta["folds"],
        "caveat": "Cascade metrics are cache-only CHAIR triage simulations; they estimate detector-call reduction, not fluent caption rewriting or POPE transfer.",
    }
    with (output_dir / "lh_shape_tdev_cascade_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    best = max(
        [row for row in summary_rows if row["delete_rate"] == min(delete_rates)],
        key=lambda row: (row["call_rate"] < 1.0, row["retained_full_hallucination_reduction"], -row["grounded_loss"]),
    )
    print(f"Wrote {output_dir / 'lh_shape_tdev_cascade_metrics.json'}")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
