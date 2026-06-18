#!/usr/bin/env python3
"""Evaluate lightweight LH-Shape scores from cached per-head attention features.

LH-Shape is the TDEV-lite direction: keep late-layer per-head attention-shape
features instead of averaging heads first. This script is cache-only and emits
paper-style controlled AUROC metrics for:

1. training-free late-layer mean-head shape features;
2. image-fold split-selected top-k head/feature averages, where each fold uses
   only train images to choose heads/features and orientation, then scores held-
   out images.

The split-selected scores are supervised calibration baselines, not final
training-free methods. They are useful for testing whether a simple, interpretable
head selection can retain the late-head signal without fitting a full logistic
probe.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def roc_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = labels.astype(np.int32)
    values = values.astype(np.float64)
    valid = np.isfinite(values)
    labels = labels[valid]
    values = values[valid]
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values)
    sv = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sv[j] == sv[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_bin_auroc(values, labels, gen_pos, bin_width=10, min_n=10) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    gen_pos = np.asarray(gen_pos, dtype=np.int32)
    aucs, weights = [], []
    lo, hi = int(gen_pos.min()), int(gen_pos.max())
    for start in range(lo, hi + 1, bin_width):
        mask = (gen_pos >= start) & (gen_pos < start + bin_width)
        if mask.sum() < min_n:
            continue
        valid = np.isfinite(values[mask])
        if valid.sum() < min_n:
            continue
        auc = roc_auc(labels[mask][valid], values[mask][valid])
        if auc is not None:
            aucs.append(auc)
            weights.append(int(valid.sum()))
    return float(np.average(aucs, weights=weights)) if aucs else None


def matched_pair_auc(labels, values, gen_pos, delta=5) -> tuple[float | None, int]:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    gen_pos = np.asarray(gen_pos, dtype=np.int32)
    valid = np.isfinite(values)
    pos_idx = np.where(valid & (labels == 1))[0]
    neg_idx = np.where(valid & (labels == 0))[0]
    wins = 0.0
    pairs = 0
    for idx in pos_idx:
        candidates = neg_idx[np.abs(gen_pos[neg_idx] - gen_pos[idx]) <= delta]
        if candidates.size == 0:
            continue
        diff = values[idx] - values[candidates]
        wins += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
        pairs += int(candidates.size)
    return (float(wins / pairs) if pairs else None), pairs


def residualize_by_position(values, gen_pos, bin_width=10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    gen_pos = np.asarray(gen_pos, dtype=np.int32)
    residual = values.copy()
    lo, hi = int(gen_pos.min()), int(gen_pos.max())
    for start in range(lo, hi + 1, bin_width):
        mask = (gen_pos >= start) & (gen_pos < start + bin_width) & np.isfinite(values)
        if mask.sum() > 0:
            residual[mask] = values[mask] - values[mask].mean()
    return residual


def metric_bundle(name: str, values, labels, gen_pos, bin_width=10) -> dict:
    matched, pairs = matched_pair_auc(labels, values, gen_pos)
    residual = residualize_by_position(values, gen_pos, bin_width=bin_width)
    return {
        "score": name,
        "overall_auroc": roc_auc(labels, values),
        "within_bin_auroc": within_bin_auroc(values, labels, gen_pos, bin_width=bin_width),
        "matched_pair_auroc": matched,
        "matched_pair_count": int(pairs),
        "residual_auroc": roc_auc(labels, residual),
    }


def load_cache(cache: str, cache_glob: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if cache_glob:
        files = sorted(glob.glob(cache_glob))
        if not files:
            raise FileNotFoundError(cache_glob)
        parts = [np.load(file, allow_pickle=True) for file in files]
        feats = np.concatenate([part["feats"] for part in parts], axis=0)
        labels = np.concatenate([part["labels"] for part in parts], axis=0).astype(np.int32)
        token_pos = np.concatenate([part["token_pos"] for part in parts], axis=0).astype(np.int32)
        image_ids = np.concatenate([part["image_ids"] for part in parts], axis=0).astype(np.int64)
        layer_indices = parts[0]["layer_indices"].astype(np.int32)
        feature_names = [str(x) for x in parts[0]["feature_names"]]
    else:
        z = np.load(cache, allow_pickle=True)
        feats = z["feats"]
        labels = z["labels"].astype(np.int32)
        token_pos = z["token_pos"].astype(np.int32)
        image_ids = z["image_ids"].astype(np.int64)
        layer_indices = z["layer_indices"].astype(np.int32)
        feature_names = [str(x) for x in z["feature_names"]]
    return feats, labels, token_pos, image_ids, layer_indices, feature_names


def load_gen_pos(generation_json: str, image_ids: np.ndarray, token_pos: np.ndarray) -> np.ndarray:
    gen = json.loads(Path(generation_json).read_text())
    prompt_end = {int(row["image_id"]): int(row["prompt_end_idx"]) for row in gen}
    return np.asarray([int(pos) - prompt_end[int(image_id)] for pos, image_id in zip(token_pos, image_ids)], dtype=np.int32)


def feature_metadata(layer_indices: np.ndarray, num_heads: int, feature_names: list[str]) -> list[dict]:
    meta = []
    for li, layer in enumerate(layer_indices):
        for head in range(num_heads):
            for fi, feature in enumerate(feature_names):
                meta.append({"layer_index": li, "layer": int(layer), "head": head, "feature_index": fi, "feature": feature})
    return meta


def image_folds(image_ids: np.ndarray, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    unique_images = np.asarray(sorted(set(image_ids.tolist())))
    rng.shuffle(unique_images)
    folds = np.array_split(unique_images, k)
    output = []
    for fold_images in folds:
        test = np.isin(image_ids, fold_images)
        output.append((~test, test))
    return output


def allowed_indices(meta: list[dict], layers: set[int] | None, features: set[str] | None) -> list[int]:
    idxs = []
    for idx, item in enumerate(meta):
        if layers is not None and item["layer"] not in layers:
            continue
        if features is not None and item["feature"] not in features:
            continue
        idxs.append(idx)
    return idxs


def orient_score(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_pos: np.ndarray,
    bin_width: int,
    selection_metric: str,
) -> tuple[float, int]:
    if selection_metric == "within_bin":
        auc = within_bin_auroc(train_values, train_labels, train_pos, bin_width=bin_width)
    elif selection_metric == "residual":
        auc = roc_auc(train_labels, residualize_by_position(train_values, train_pos, bin_width=bin_width))
    elif selection_metric == "overall":
        auc = roc_auc(train_labels, train_values)
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")
    if auc is None:
        return 0.5, 1
    if auc < 0.5:
        return 1.0 - auc, -1
    return auc, 1


def split_selected_scores(
    X: np.ndarray,
    labels: np.ndarray,
    gen_pos: np.ndarray,
    image_ids: np.ndarray,
    meta: list[dict],
    candidate_indices: list[int],
    top_k: int,
    folds: int,
    seed: int,
    bin_width: int,
    selection_metric: str,
) -> tuple[np.ndarray, list[dict]]:
    scores = np.full(labels.shape[0], np.nan, dtype=np.float64)
    selections = []
    for fold_id, (train, test) in enumerate(image_folds(image_ids, folds, seed), start=1):
        ranked = []
        for idx in candidate_indices:
            train_values = X[train, idx]
            quality, direction = orient_score(train_values, labels[train], gen_pos[train], bin_width, selection_metric)
            ranked.append((quality, direction, idx))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = ranked[:top_k]
        fold_parts = []
        fold_selected = []
        for quality, direction, idx in selected:
            mu = X[train, idx].mean()
            sd = X[train, idx].std() + 1e-8
            fold_parts.append(direction * ((X[test, idx] - mu) / sd))
            m = dict(meta[idx])
            m.update({"train_selection_score_oriented": float(quality), "selection_metric": selection_metric, "direction": int(direction)})
            fold_selected.append(m)
        scores[test] = np.stack(fold_parts, axis=1).mean(axis=1)
        selections.append({"fold": fold_id, "top_k": top_k, "selected": fold_selected})
    return scores, selections


def summarize_selections(selections: list[dict]) -> list[dict]:
    counter = Counter()
    quality = defaultdict(list)
    for fold in selections:
        for item in fold["selected"]:
            key = (item["layer"], item["head"], item["feature"], item["direction"])
            counter[key] += 1
            quality[key].append(float(item["train_selection_score_oriented"]))
    rows = []
    for (layer, head, feature, direction), count in counter.most_common():
        rows.append({
            "layer": layer,
            "head": head,
            "feature": feature,
            "direction": direction,
            "selected_count": count,
            "mean_train_selection_score_oriented": float(np.mean(quality[(layer, head, feature, direction)])),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_int_set(spec: str) -> set[int]:
    return {int(item.strip()) for item in spec.split(",") if item.strip()}


def parse_feature_set(spec: str) -> set[str]:
    return {item.strip() for item in spec.split(",") if item.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LH-Shape split-selected per-head scores")
    p.add_argument("--cache", default="experiments/coco_llava_7b_rows/per_head_row_cache.npz")
    p.add_argument("--cache_glob", default=None)
    p.add_argument("--generation_json", default="experiments/coco_llava_7b/generation.json")
    p.add_argument("--output_dir", default="detection/baselines/results/lh_shape")
    p.add_argument("--bin_width", type=int, default=10)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_ks", default="1,3,5")
    p.add_argument("--candidate_layer_sets", default="31;22,31")
    p.add_argument("--candidate_features", default="entropy,neg_top1_mass,neg_top5_mass,neg_max_over_mea")
    p.add_argument("--selection_metric", choices=["residual", "within_bin", "overall"], default="residual")
    args = p.parse_args()

    feats, labels, token_pos, image_ids, layer_indices, feature_names = load_cache(args.cache, args.cache_glob)
    gen_pos = load_gen_pos(args.generation_json, image_ids, token_pos)
    n, num_layers, num_heads, num_features = feats.shape
    X = feats.reshape(n, num_layers * num_heads * num_features).astype(np.float64)
    meta = feature_metadata(layer_indices, num_heads, feature_names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    selection_payload = {}

    # Training-free mean-over-head baselines for late layers.
    for layer in [22, 31]:
        if layer not in set(map(int, layer_indices)):
            continue
        li = int(np.where(layer_indices == layer)[0][0])
        for fi, feature in enumerate(feature_names):
            values = feats[:, li, :, fi].mean(axis=1)
            metric_rows.append(metric_bundle(f"training_free_layer{layer}_mean_heads_{feature}", values, labels, gen_pos, args.bin_width))

    top_ks = [int(item) for item in args.top_ks.split(",") if item.strip()]
    feature_filter = parse_feature_set(args.candidate_features)
    for layer_spec in [item.strip() for item in args.candidate_layer_sets.split(";") if item.strip()]:
        layers = parse_int_set(layer_spec)
        candidates = allowed_indices(meta, layers, feature_filter)
        label = "layers_" + "_".join(str(x) for x in sorted(layers))
        for top_k in top_ks:
            print(f"Evaluating {label} top{top_k} with {len(candidates)} candidates ({args.selection_metric} selection)", flush=True)
            scores, selections = split_selected_scores(
                X,
                labels,
                gen_pos,
                image_ids,
                meta,
                candidates,
                top_k=top_k,
                folds=args.folds,
                seed=args.seed,
                bin_width=args.bin_width,
                selection_metric=args.selection_metric,
            )
            name = f"split_selected_{label}_top{top_k}"
            metric_rows.append(metric_bundle(name, scores, labels, gen_pos, args.bin_width))
            selection_payload[name] = {
                "folds": selections,
                "summary": summarize_selections(selections),
            }

    write_csv(output_dir / "lh_shape_metrics.csv", metric_rows)
    payload = {
        "cache": args.cache_glob or args.cache,
        "generation_json": args.generation_json,
        "num_mentions": int(n),
        "num_images": int(len(set(image_ids.tolist()))),
        "hallucinated_mentions": int(labels.sum()),
        "grounded_mentions": int(n - labels.sum()),
        "layer_indices": [int(x) for x in layer_indices],
        "num_heads": int(num_heads),
        "feature_names": feature_names,
        "bin_width": int(args.bin_width),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "top_ks": top_ks,
        "candidate_layer_sets": args.candidate_layer_sets,
        "candidate_features": sorted(feature_filter),
        "selection_metric": args.selection_metric,
        "metrics": metric_rows,
        "selections": selection_payload,
        "caveat": "split-selected variants use train-fold labels to choose heads/features and orientation; residual selection is position-controlled but still supervised calibration, not a training-free final method",
    }
    with (output_dir / "lh_shape_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'lh_shape_metrics.json'}")
    best = max(metric_rows, key=lambda row: row["within_bin_auroc"] or -1)
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
