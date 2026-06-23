#!/usr/bin/env python3
"""Hidden-state target-vs-neighbor contrast probe for POPE.

This is a stronger internal supervised diagnostic than the attention-shape probe:
for the target prompt and semantic-neighbor prompt, extract selected-layer object
question-token hidden states, visual-token mean hidden states, and their
contrasts. No external detector is used. The readout is image-grouped
out-of-fold logistic regression for target absence, so it is a supervised
ceiling/foil rather than a training-free method.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import load_model_and_processor  # noqa: E402
from src.data import iter_pope_records, load_pope_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hidden-state contrast probe TDEV on POPE")
    p.add_argument("--model_path", default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--audit_csv", default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--top_neighbors", type=int, default=1)
    p.add_argument("--layers", default="22,31")
    p.add_argument("--target_token_policy", choices=["first", "last", "mean"], default="mean")
    p.add_argument("--feature_blocks", default="obj,vis,diff", help="Comma list from obj,vis,diff,obj_vis")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--l2", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def split_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_layers(spec: str) -> list[int]:
    layers = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not layers:
        raise ValueError("--layers cannot be empty")
    return layers


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_audit_rows(path: Path, splits: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    split_set = set(splits)
    rows = {}
    for row in read_csv(path):
        if row["split"] in split_set:
            rows[(row["split"], str(row["question_id"]))] = row
    if not rows:
        raise ValueError(f"No audit rows found in {path} for splits={splits}")
    return rows


def read_neighbors(path: Path, top_k: int) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {obj: [item["object"] for item in items[:top_k]] for obj, items in raw.items()}


def question_for_object(obj: str) -> str:
    article = "an" if obj[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    if obj.endswith("s") and obj not in {"scissors", "skis"}:
        return f"Are there any {obj} in the image?"
    return f"Is there {article} {obj} in the image?"


def pope_prompt(question: str) -> str:
    return f"<image>\nUSER: {question} Please just answer yes or no.\nASSISTANT:"


def candidate_tokenizations(tokenizer, target: str) -> list[list[int]]:
    candidates = []
    for text in (target, " " + target, target.replace(" ", "_"), " " + target.replace(" ", "_")):
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if ids and ids not in candidates:
            candidates.append(ids)
    return candidates


def find_subsequence(sequence: list[int], pattern: list[int], end_before: int | None = None) -> tuple[int, int] | None:
    if end_before is not None:
        sequence = sequence[:end_before]
    n = len(pattern)
    if n == 0 or len(sequence) < n:
        return None
    matches = []
    for start in range(0, len(sequence) - n + 1):
        if sequence[start:start + n] == pattern:
            matches.append((start, start + n))
    return matches[-1] if matches else None


def assistant_start(input_ids: list[int], tokenizer) -> int:
    best = None
    for text in ("ASSISTANT:", " ASSISTANT:", "\nASSISTANT:"):
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if not ids:
            continue
        match = find_subsequence(input_ids, ids)
        if match is not None and (best is None or match[0] > best[0]):
            best = match
    return best[0] if best else len(input_ids)


def find_target_span(input_ids: list[int], tokenizer, target: str) -> tuple[int, int, list[int]]:
    end_before = assistant_start(input_ids, tokenizer)
    for pattern in candidate_tokenizations(tokenizer, target):
        match = find_subsequence(input_ids, pattern, end_before=end_before)
        if match is not None:
            return match[0], match[1], pattern
    decoded = tokenizer.decode(input_ids[:end_before], skip_special_tokens=False)
    raise ValueError(f"Could not locate target {target!r}. Prompt prefix: {decoded!r}")


def token_positions(start: int, end: int, policy: str) -> list[int]:
    if policy == "first":
        return [start]
    if policy == "last":
        return [end - 1]
    return list(range(start, end))


def normalize_vec(vec: torch.Tensor) -> torch.Tensor:
    return F.normalize(vec.float(), dim=-1).cpu()


def prompt_hidden_features(model, processor, tokenizer, image, obj: str, question: str, layers: list[int], policy: str, device: torch.device, image_token_id: int) -> tuple[dict[str, np.ndarray], dict]:
    prompt = pope_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    input_ids_tensor = inputs["input_ids"][0]
    input_ids = [int(x) for x in input_ids_tensor.detach().cpu().tolist()]
    start, end, pattern = find_target_span(input_ids, tokenizer, obj)
    positions = torch.tensor(token_positions(start, end, policy), dtype=torch.long, device=device)
    vis_start, vis_end = find_vis_bounds(input_ids_tensor, image_token_id)
    with torch.inference_mode():
        outputs = model.forward(**inputs, output_hidden_states=True, output_attentions=False)
    obj_parts = []
    vis_parts = []
    obj_vis_parts = []
    for layer in layers:
        hs = outputs.hidden_states[layer][0].float()
        obj_vec = hs[positions].mean(dim=0)
        vis_vec = hs[vis_start:vis_end].mean(dim=0)
        obj_n = normalize_vec(obj_vec)
        vis_n = normalize_vec(vis_vec)
        obj_parts.append(obj_n.numpy())
        vis_parts.append(vis_n.numpy())
        obj_vis_parts.append((obj_n - vis_n).numpy())
    info = {"token_start": int(start), "token_end": int(end), "token_ids": pattern, "vis_tokens": int(vis_end - vis_start)}
    del inputs, outputs
    return {
        "obj": np.concatenate(obj_parts).astype(np.float32),
        "vis": np.concatenate(vis_parts).astype(np.float32),
        "obj_vis": np.concatenate(obj_vis_parts).astype(np.float32),
    }, info


def iter_joined_records(args: argparse.Namespace, splits: list[str], audit_rows: dict[tuple[str, str], dict[str, str]]) -> list[dict]:
    joined = []
    for split in splits:
        for record in iter_pope_records(args.pope_dir, args.coco_path, split, limit=args.limit):
            key = (split, str(record["question_id"]))
            audit = audit_rows.get(key)
            if audit is not None:
                joined.append({**record, **audit, "split": split, "question_id": str(record["question_id"])})
    if args.num_shards > 1:
        joined = joined[args.shard_idx::args.num_shards]
    return joined


def image_folds(image_ids: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(seed)
    unique_images = np.asarray(sorted(set(int(x) for x in image_ids.tolist())))
    rng.shuffle(unique_images)
    fold_images = np.array_split(unique_images, folds)
    output = []
    for items in fold_images:
        test = np.isin(image_ids, items)
        output.append((~test, test))
    return output


def train_logreg(X: np.ndarray, y: np.ndarray, l2: float, lr: float, epochs: int) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    y = y.astype(np.float64)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        w -= lr * ((X.T @ g) / n + l2 * w / n)
        b -= lr * float(g.mean())
    return w, b


def binary_metrics(labels: np.ndarray, pred_absent: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int32)
    pred_absent = np.asarray(pred_absent, dtype=bool)
    tp = int(((labels == 1) & pred_absent).sum())
    fp = int(((labels == 0) & pred_absent).sum())
    tn = int(((labels == 0) & ~pred_absent).sum())
    fn = int(((labels == 1) & ~pred_absent).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "samples": int(tp + fp + tn + fn),
        "absent_tpr": tp / (tp + fn) if tp + fn else 0.0,
        "present_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0,
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite scores")
    unique = np.unique(finite)
    candidates = [float(unique[0] - 1e-6), float(unique[-1] + 1e-6)]
    candidates.extend(float((a + b) / 2.0) for a, b in zip(unique, unique[1:]))
    best_threshold = candidates[0]
    best_metrics = binary_metrics(labels, scores > best_threshold)
    for threshold in candidates[1:]:
        vals = binary_metrics(labels, scores > threshold)
        if vals["mcc"] > best_metrics["mcc"]:
            best_threshold = threshold
            best_metrics = vals
    return best_threshold, best_metrics


def oof_probe_scores(X: np.ndarray, labels: np.ndarray, image_ids: np.ndarray, folds: int, seed: int, l2: float, lr: float, epochs: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    pred_absent = np.zeros(labels.shape[0], dtype=bool)
    details = []
    for fold_id, (train, test) in enumerate(image_folds(image_ids, folds, seed), start=1):
        if np.unique(labels[train]).size < 2:
            raise ValueError(f"Fold {fold_id} training rows do not contain both classes")
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0) + 1e-8
        X_train = (X[train] - mu) / sd
        X_test = (X[test] - mu) / sd
        w, b = train_logreg(X_train, labels[train], l2=l2, lr=lr, epochs=epochs)
        train_scores = X_train @ w + b
        threshold, train_metrics = choose_threshold(labels[train], train_scores)
        test_scores = X_test @ w + b
        scores[test] = test_scores
        pred_absent[test] = test_scores > threshold
        details.append({
            "fold": fold_id,
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "threshold": float(threshold),
            "train_mcc": float(train_metrics["mcc"]),
            "train_absent_tpr": float(train_metrics["absent_tpr"]),
            "train_present_fpr": float(train_metrics["present_fpr"]),
            "weight_l2_norm": float(np.linalg.norm(w)),
            "bias": float(b),
        })
    return scores, pred_absent, details


def roc_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    labels = labels[valid]
    values = values[valid]
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def subset_summary(rows: list[dict], labels: np.ndarray, scores: np.ndarray, pred_absent: np.ndarray) -> list[dict]:
    splits = sorted({row["split"] for row in rows})
    subsets = ["all", "positive", "negative", "negative_related_present", "negative_absent_plain", "negative_target_present_coco_label"]
    out = []
    split_arr = np.asarray([row["split"] for row in rows])
    neg_arr = np.asarray([row.get("negative_type", "") for row in rows])
    for split in splits + ["macro"]:
        split_mask = np.ones(len(rows), dtype=bool) if split == "macro" else split_arr == split
        for subset in subsets:
            if subset == "all":
                mask = split_mask
            elif subset == "positive":
                mask = split_mask & (labels == 0)
            elif subset == "negative":
                mask = split_mask & (labels == 1)
            else:
                mask = split_mask & (neg_arr == subset)
            if not mask.any():
                continue
            out.append({
                "split": split,
                "subset": subset,
                "auroc_absent": roc_auc(labels[mask], scores[mask]),
                **binary_metrics(labels[mask], pred_absent[mask]),
            })
    return out


def build_feature(target: dict[str, np.ndarray], neighbor: dict[str, np.ndarray], blocks: list[str]) -> np.ndarray:
    parts = []
    if "obj" in blocks:
        parts.extend([target["obj"], neighbor["obj"]])
    if "vis" in blocks:
        parts.extend([target["vis"], neighbor["vis"]])
    if "diff" in blocks:
        parts.extend([target["obj"] - neighbor["obj"], target["vis"] - neighbor["vis"]])
    if "obj_vis" in blocks:
        parts.extend([target["obj_vis"], neighbor["obj_vis"], target["obj_vis"] - neighbor["obj_vis"]])
    if not parts:
        raise ValueError("No feature blocks selected")
    return np.concatenate(parts).astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_list(args.splits)
    layers = parse_layers(args.layers)
    blocks = split_list(args.feature_blocks)
    allowed = {"obj", "vis", "diff", "obj_vis"}
    unknown = sorted(set(blocks) - allowed)
    if unknown:
        raise ValueError(f"Unknown feature blocks: {unknown}")
    audit_rows = read_audit_rows(Path(args.audit_csv), splits)
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    records = iter_joined_records(args, splits, audit_rows)
    logger.info("Extracting hidden contrast features for %d rows layers=%s top_neighbors=%d blocks=%s", len(records), layers, args.top_neighbors, blocks)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager")
    model.eval()
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    feature_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []
    labels_absent: list[int] = []
    failures: list[dict] = []
    for step, record in enumerate(tqdm(records, desc="Hidden contrast features"), start=1):
        target = str(record["target"])
        neighbor_list = neighbors.get(target, [])
        if not neighbor_list:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": "no semantic neighbors"})
            continue
        try:
            image = load_pope_image(record)
            try:
                target_feats, target_info = prompt_hidden_features(
                    model, processor, tokenizer, image, target, str(record["question"]), layers, args.target_token_policy, device, image_token_id
                )
                neighbor_feats_all = []
                neighbor_names = []
                neighbor_token_starts = []
                for neighbor in neighbor_list:
                    n_feats, n_info = prompt_hidden_features(
                        model, processor, tokenizer, image, neighbor, question_for_object(neighbor), layers, args.target_token_policy, device, image_token_id
                    )
                    neighbor_feats_all.append(n_feats)
                    neighbor_names.append(neighbor)
                    neighbor_token_starts.append(str(n_info["token_start"]))
                neighbor_feats = {key: np.mean(np.stack([x[key] for x in neighbor_feats_all], axis=0), axis=0).astype(np.float32) for key in neighbor_feats_all[0]}
            finally:
                image.close()
            feature_rows.append(build_feature(target_feats, neighbor_feats, blocks))
            labels_absent.append(1 if str(record["label"]).lower() == "no" else 0)
            meta_rows.append({
                "split": str(record["split"]),
                "question_id": str(record["question_id"]),
                "image_id": str(record.get("image_id", "")),
                "label": str(record["label"]).lower(),
                "target": target,
                "question": str(record["question"]),
                "negative_type": str(record.get("negative_type", "")),
                "best_neighbor": neighbor_names[0],
                "neighbors": "|".join(neighbor_names),
                "target_token_start": target_info["token_start"],
                "target_token_end": target_info["token_end"],
                "neighbor_token_starts": "|".join(neighbor_token_starts),
            })
        except Exception as exc:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": repr(exc)})
        finally:
            if step % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if not feature_rows:
        raise ValueError("No rows were scored")
    X = np.stack(feature_rows, axis=0).astype(np.float64)
    labels = np.asarray(labels_absent, dtype=np.int32)
    image_ids = np.asarray([int(row["image_id"]) for row in meta_rows], dtype=np.int64)
    scores_absent, pred_absent, fold_details = oof_probe_scores(X, labels, image_ids, args.folds, args.seed, args.l2, args.lr, args.epochs)

    pred_rows = []
    for row, absent_score, absent_pred in zip(meta_rows, scores_absent, pred_absent):
        support_score = -float(absent_score)
        pred_rows.append({
            **row,
            "absence_score": float(absent_score),
            "support_score": support_score,
            "tdev_margin": support_score,
            "prediction": "no" if bool(absent_pred) else "yes",
        })
    metric_rows = subset_summary(meta_rows, labels, scores_absent, pred_absent)

    pred_path = output_dir / "hidden_contrast_probe_predictions.csv"
    metrics_path = output_dir / "hidden_contrast_probe_metrics.csv"
    write_csv(pred_path, pred_rows)
    write_csv(metrics_path, metric_rows)
    payload = {
        "audit_csv": args.audit_csv,
        "backend": "internal_hidden_state_contrast_probe",
        "external_detector": False,
        "supervised": True,
        "caveat": "Image-grouped out-of-fold supervised probe over hidden states; use as diagnostic ceiling/foil, not as the training-free headline method.",
        "model_path": args.model_path,
        "neighbors_json": args.neighbors_json,
        "predictions_csv": str(pred_path),
        "metrics_csv": str(metrics_path),
        "rows_requested": len(records),
        "rows_scored": len(pred_rows),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "splits": splits,
        "layers": layers,
        "top_neighbors": args.top_neighbors,
        "target_token_policy": args.target_token_policy,
        "feature_blocks": blocks,
        "feature_dim": int(X.shape[1]),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "l2": float(args.l2),
        "lr": float(args.lr),
        "epochs": int(args.epochs),
        "fold_details": fold_details,
        "metrics": metric_rows,
        "score_direction": "larger support_score means stronger target-present/yes evidence; larger absence_score means target-absent/no evidence",
    }
    (output_dir / "hidden_contrast_probe_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", pred_path)
    logger.info("Wrote %s", output_dir / "hidden_contrast_probe_metrics.json")


if __name__ == "__main__":
    main()
