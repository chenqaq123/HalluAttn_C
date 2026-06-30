#!/usr/bin/env python3
"""Per-layer hidden-margin features + answer logits for POPE (v2).

Replaces evaluate_hidden_margin_tdev_pope.py with richer output:
  - Hidden cosine margins computed *per layer* (not averaged across layers).
  - Answer yes/no logits extracted from the same forward pass (no extra call).
  - Default layers: 16, 22, 27, 31 (configurable via --layers).

Output columns per row (prefix l{L}_ for each layer L):
  l{L}_target_align, l{L}_neighbor_align, l{L}_align_margin,
  l{L}_cross_tov_nv, l{L}_cross_nov_tv, l{L}_cross_margin,
  l{L}_obj_separation, l{L}_vis_separation
Answer columns:
  target_yes_margin, target_yes_logit, target_no_logit,
  neighbor_yes_margin, neighbor_yes_logit, neighbor_no_logit,
  answer_support_score (= target_yes_margin),
  answer_absence_score (= -target_yes_margin),
  answer_contrast_margin (= target - neighbor yes margin),
  answer_neighbor_dominance (= neighbor - target yes margin)
Aggregate columns (backward-compatible with v1 cross_split_verifier):
  hidden_align_margin, hidden_cross_margin,
  hidden_obj_separation, hidden_vis_separation, hidden_margin_score

The worst neighbor is selected by the minimum mean combined score across layers.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
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
    p = argparse.ArgumentParser(description="Per-layer hidden-margin + answer features for POPE (v2)")
    p.add_argument("--model_path", default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--audit_csv", default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--top_neighbors", type=int, default=1)
    p.add_argument("--layers", default="16,22,27,31")
    p.add_argument("--target_token_policy", choices=["first", "last", "mean"], default="mean")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def split_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_layers(spec: str) -> list[int]:
    layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
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


def first_token_candidates(tokenizer, word: str) -> list[int]:
    ids = []
    for text in (word, word.capitalize(), word.upper()):
        toks = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(toks[0])
    if not ids:
        raise ValueError(f"Could not tokenize {word!r} as single-token variants")
    return ids


def logsumexp_ids(logits: torch.Tensor, ids: list[int]) -> float:
    vals = logits[torch.tensor(ids, dtype=torch.long, device=logits.device)].float()
    return float(torch.logsumexp(vals, dim=0).item())


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item())


def norm_vec(v: torch.Tensor) -> torch.Tensor:
    return F.normalize(v.float(), dim=-1)


def prompt_features(
    model, processor, tokenizer,
    image, obj: str, question: str,
    layers: list[int], policy: str,
    device: torch.device, image_token_id: int,
    yes_ids: list[int], no_ids: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, float], dict]:
    """Run one forward pass; return per-layer hidden vecs + answer logits."""
    prompt = pope_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    input_ids_tensor = inputs["input_ids"][0]
    input_ids = [int(x) for x in input_ids_tensor.detach().cpu().tolist()]
    start, end, pattern = find_target_span(input_ids, tokenizer, obj)
    positions = torch.tensor(token_positions(start, end, policy), dtype=torch.long, device=device)
    vis_start, vis_end = find_vis_bounds(input_ids_tensor, image_token_id)

    with torch.inference_mode():
        outputs = model.forward(**inputs, output_hidden_states=True, output_attentions=False)

    # Per-layer hidden vectors
    layer_vecs: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        hs = outputs.hidden_states[layer][0].float()
        obj_vec = norm_vec(hs[positions].mean(dim=0)).cpu()
        vis_vec = norm_vec(hs[vis_start:vis_end].mean(dim=0)).cpu()
        layer_vecs[layer] = {"obj": obj_vec, "vis": vis_vec}

    # Answer logits from same forward pass (free)
    next_logits = outputs.logits[0, -1].float()
    yes_logit = logsumexp_ids(next_logits, yes_ids)
    no_logit = logsumexp_ids(next_logits, no_ids)
    answer = {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_margin": yes_logit - no_logit,
    }

    info = {
        "token_start": int(start), "token_end": int(end),
        "token_ids": pattern, "vis_tokens": int(vis_end - vis_start),
    }
    del inputs, outputs, next_logits
    return layer_vecs, answer, info


def compute_neighbor_features(
    target_vecs: dict[int, dict[str, torch.Tensor]],
    neighbor_vecs: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
) -> dict[str, float]:
    """Compute per-layer cosine margins between target and one neighbor."""
    features: dict[str, float] = {}
    combined_scores = []
    for layer in layers:
        tv = target_vecs[layer]
        nv = neighbor_vecs[layer]
        t_align = cos(tv["obj"], tv["vis"])
        n_align = cos(nv["obj"], nv["vis"])
        cross_tov_nv = cos(tv["obj"], nv["vis"])
        cross_nov_tv = cos(nv["obj"], tv["vis"])
        obj_sep = 1.0 - cos(tv["obj"], nv["obj"])
        vis_sep = 1.0 - cos(tv["vis"], nv["vis"])
        align_margin = t_align - n_align
        cross_margin = t_align - max(cross_tov_nv, cross_nov_tv)
        combined = align_margin + cross_margin + 0.25 * obj_sep + 0.25 * vis_sep
        combined_scores.append(combined)
        pfx = f"l{layer}_"
        features[f"{pfx}target_align"] = t_align
        features[f"{pfx}neighbor_align"] = n_align
        features[f"{pfx}align_margin"] = align_margin
        features[f"{pfx}cross_tov_nv"] = cross_tov_nv
        features[f"{pfx}cross_nov_tv"] = cross_nov_tv
        features[f"{pfx}cross_margin"] = cross_margin
        features[f"{pfx}obj_separation"] = obj_sep
        features[f"{pfx}vis_separation"] = vis_sep
    # Aggregate (backward-compatible with v1)
    n = len(layers)
    features["hidden_align_margin"] = sum(features[f"l{l}_align_margin"] for l in layers) / n
    features["hidden_cross_margin"] = sum(features[f"l{l}_cross_margin"] for l in layers) / n
    features["hidden_obj_separation"] = sum(features[f"l{l}_obj_separation"] for l in layers) / n
    features["hidden_vis_separation"] = sum(features[f"l{l}_vis_separation"] for l in layers) / n
    features["hidden_margin_score"] = sum(combined_scores) / n
    return features


def iter_joined_records(
    args: argparse.Namespace, splits: list[str],
    audit_rows: dict[tuple[str, str], dict[str, str]],
) -> list[dict]:
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


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_list(args.splits)
    layers = parse_layers(args.layers)

    audit_rows = read_audit_rows(Path(args.audit_csv), splits)
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    records = iter_joined_records(args, splits, audit_rows)
    logger.info(
        "Scoring %d rows, layers=%s, top_neighbors=%d", len(records), layers, args.top_neighbors
    )

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(
        args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager"
    )
    model.eval()
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)
    yes_ids = first_token_candidates(tokenizer, "yes")
    no_ids = first_token_candidates(tokenizer, "no")
    logger.info("yes_ids=%s no_ids=%s", yes_ids, no_ids)

    scored: list[dict] = []
    failures: list[dict] = []

    for step, record in enumerate(tqdm(records, desc="Hidden-margin v2"), start=1):
        target = str(record["target"])
        neighbor_list = neighbors.get(target, [])
        if not neighbor_list:
            failures.append({"split": record.get("split", ""), "question_id": record.get("question_id", ""), "target": target, "error": "no semantic neighbors"})
            continue
        try:
            image = load_pope_image(record)
            try:
                target_vecs, target_answer, target_info = prompt_features(
                    model, processor, tokenizer, image, target, str(record["question"]),
                    layers, args.target_token_policy, device, image_token_id, yes_ids, no_ids,
                )
                candidates = []
                for neighbor in neighbor_list:
                    n_vecs, n_answer, n_info = prompt_features(
                        model, processor, tokenizer, image, neighbor, question_for_object(neighbor),
                        layers, args.target_token_policy, device, image_token_id, yes_ids, no_ids,
                    )
                    feats = compute_neighbor_features(target_vecs, n_vecs, layers)
                    candidates.append({
                        "neighbor": neighbor,
                        "n_info": n_info,
                        "n_answer": n_answer,
                        "feats": feats,
                    })
            finally:
                image.close()

            # Select worst neighbor (minimum combined margin across layers)
            best = min(candidates, key=lambda c: c["feats"]["hidden_margin_score"])
            feats = best["feats"]
            n_ans = best["n_answer"]

            # Answer features
            target_yes_margin = float(target_answer["yes_margin"])
            neighbor_yes_margin = float(n_ans["yes_margin"])

            row: dict = {
                "split": str(record["split"]),
                "question_id": str(record["question_id"]),
                "image_id": str(record.get("image_id", "")),
                "label": str(record["label"]).lower(),
                "target": target,
                "question": str(record["question"]),
                "negative_type": str(record.get("negative_type", "")),
                "best_neighbor": best["neighbor"],
                "neighbors": "|".join(neighbor_list),
                "target_token_start": target_info["token_start"],
                "target_token_end": target_info["token_end"],
                "neighbor_token_start": best["n_info"]["token_start"],
                "neighbor_token_end": best["n_info"]["token_end"],
                # Answer features
                "target_yes_margin": target_yes_margin,
                "target_yes_logit": float(target_answer["yes_logit"]),
                "target_no_logit": float(target_answer["no_logit"]),
                "neighbor_yes_margin": neighbor_yes_margin,
                "neighbor_yes_logit": float(n_ans["yes_logit"]),
                "neighbor_no_logit": float(n_ans["no_logit"]),
                "answer_support_score": target_yes_margin,
                "answer_absence_score": -target_yes_margin,
                "answer_contrast_margin": target_yes_margin - neighbor_yes_margin,
                "answer_neighbor_dominance": neighbor_yes_margin - target_yes_margin,
            }
            # Per-layer hidden features
            row.update(feats)
            # Vanilla-compatible prediction (target yes margin > 0)
            row["prediction"] = "yes" if target_yes_margin > 0.0 else "no"
            row["tdev_margin"] = float(feats["hidden_margin_score"])

            scored.append(row)
        except Exception as exc:
            failures.append({"split": record.get("split", ""), "question_id": record.get("question_id", ""), "target": target, "error": repr(exc)})
        finally:
            if step % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    pred_path = output_dir / "hidden_margin_v2_predictions.csv"
    write_csv(pred_path, scored)
    payload = {
        "backend": "internal_hidden_margin_v2_per_layer_plus_answer",
        "external_detector": False,
        "supervised": False,
        "model_path": args.model_path,
        "layers": layers,
        "top_neighbors": args.top_neighbors,
        "target_token_policy": args.target_token_policy,
        "splits": splits,
        "rows_requested": len(records),
        "rows_scored": len(scored),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "predictions_csv": str(pred_path),
        "per_layer_features": [f"l{l}_{f}" for l in layers for f in ["target_align", "neighbor_align", "align_margin", "cross_tov_nv", "cross_nov_tv", "cross_margin", "obj_separation", "vis_separation"]],
        "answer_features": ["target_yes_margin", "neighbor_yes_margin", "answer_support_score", "answer_absence_score", "answer_contrast_margin", "answer_neighbor_dominance"],
        "aggregate_features": ["hidden_align_margin", "hidden_cross_margin", "hidden_obj_separation", "hidden_vis_separation", "hidden_margin_score"],
        "note": "Per-layer hidden margins + answer logits in same forward pass. Worst neighbor selected by min mean combined score.",
    }
    (output_dir / "hidden_margin_v2_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Wrote %d rows → %s", len(scored), pred_path)
    logger.info("Failures: %d", len(failures))


if __name__ == "__main__":
    main()
