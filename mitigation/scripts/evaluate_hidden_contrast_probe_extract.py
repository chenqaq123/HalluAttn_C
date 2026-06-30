#!/usr/bin/env python3
"""Stage 1 of supervised hidden-contrast probe for full POPE.

Extracts per-layer hidden features (object tokens + visual mean) for the target
prompt and semantic-neighbor prompt.  Saves a feature CSV per shard; does NOT
run logistic regression (that is done by merge_evaluate_hidden_contrast_probe.py
after all shards are merged).

This script is shardable and can be run in parallel across GPUs.

Usage (one shard):
  python evaluate_hidden_contrast_probe_extract.py \\
    --coco_path /data/coco \\
    --pope_dir /data/pope \\
    --output_dir results/hidden_contrast_probe_full/shard0 \\
    --layers 22,31 \\
    --shard_idx 0 --num_shards 5 --device 0
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
    p = argparse.ArgumentParser()
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
    p.add_argument("--feature_blocks", default="obj,vis,diff")
    p.add_argument("--target_token_policy", choices=["first", "last", "mean"], default="mean")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def split_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def parse_layers(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_audit_rows(path: Path, splits: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    split_set = set(splits)
    rows = {}
    for row in read_csv(path):
        if row["split"] in split_set:
            rows[(row["split"], str(row["question_id"]))] = row
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


def find_subsequence(seq: list[int], pat: list[int], end_before: int | None = None) -> tuple[int, int] | None:
    if end_before is not None:
        seq = seq[:end_before]
    n = len(pat)
    if n == 0 or len(seq) < n:
        return None
    matches = [(i, i + n) for i in range(len(seq) - n + 1) if seq[i:i + n] == pat]
    return matches[-1] if matches else None


def assistant_start(input_ids: list[int], tokenizer) -> int:
    best = None
    for text in ("ASSISTANT:", " ASSISTANT:", "\nASSISTANT:"):
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        m = find_subsequence(input_ids, ids)
        if m and (best is None or m[0] > best[0]):
            best = m
    return best[0] if best else len(input_ids)


def find_target_span(input_ids: list[int], tokenizer, target: str) -> tuple[int, int, list[int]]:
    end_before = assistant_start(input_ids, tokenizer)
    for pat in candidate_tokenizations(tokenizer, target):
        m = find_subsequence(input_ids, pat, end_before=end_before)
        if m:
            return m[0], m[1], pat
    raise ValueError(f"Cannot locate {target!r}")


def token_positions(start: int, end: int, policy: str) -> list[int]:
    if policy == "first":
        return [start]
    if policy == "last":
        return [end - 1]
    return list(range(start, end))


def extract_hidden(model, processor, tokenizer, image, obj: str, question: str,
                   layers: list[int], policy: str, device: torch.device,
                   image_token_id: int) -> tuple[dict[str, np.ndarray], dict]:
    prompt = pope_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    ids_tensor = inputs["input_ids"][0]
    ids = ids_tensor.detach().cpu().tolist()
    start, end, pat = find_target_span(ids, tokenizer, obj)
    positions = torch.tensor(token_positions(start, end, policy), dtype=torch.long, device=device)
    vis_start, vis_end = find_vis_bounds(ids_tensor, image_token_id)
    with torch.inference_mode():
        out = model.forward(**inputs, output_hidden_states=True, output_attentions=False)
    obj_parts, vis_parts, obj_vis_parts = [], [], []
    for layer in layers:
        hs = out.hidden_states[layer][0].float()
        ov = F.normalize(hs[positions].mean(0), dim=-1).cpu().numpy().astype(np.float32)
        vv = F.normalize(hs[vis_start:vis_end].mean(0), dim=-1).cpu().numpy().astype(np.float32)
        obj_parts.append(ov)
        vis_parts.append(vv)
        obj_vis_parts.append((ov - vv))
    del inputs, out
    return {
        "obj": np.concatenate(obj_parts),
        "vis": np.concatenate(vis_parts),
        "obj_vis": np.concatenate(obj_vis_parts),
    }, {"token_start": start, "token_end": end, "token_ids": pat}


def build_feature(target: dict[str, np.ndarray], neighbor: dict[str, np.ndarray], blocks: list[str]) -> np.ndarray:
    parts = []
    if "obj" in blocks:
        parts += [target["obj"], neighbor["obj"]]
    if "vis" in blocks:
        parts += [target["vis"], neighbor["vis"]]
    if "diff" in blocks:
        parts += [target["obj"] - neighbor["obj"], target["vis"] - neighbor["vis"]]
    if "obj_vis" in blocks:
        parts += [target["obj_vis"], neighbor["obj_vis"], target["obj_vis"] - neighbor["obj_vis"]]
    return np.concatenate(parts).astype(np.float32)


def iter_joined_records(args, splits, audit_rows):
    joined = []
    for split in splits:
        for rec in iter_pope_records(args.pope_dir, args.coco_path, split, limit=args.limit):
            key = (split, str(rec["question_id"]))
            audit = audit_rows.get(key)
            if audit:
                joined.append({**rec, **audit, "split": split, "question_id": str(rec["question_id"])})
    if args.num_shards > 1:
        joined = joined[args.shard_idx::args.num_shards]
    return joined


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = split_list(args.splits)
    layers = parse_layers(args.layers)
    blocks = split_list(args.feature_blocks)

    audit_rows = read_audit_rows(Path(args.audit_csv), splits)
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    records = iter_joined_records(args, splits, audit_rows)
    logger.info("Extracting features: %d rows, layers=%s, blocks=%s", len(records), layers, blocks)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(
        args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager"
    )
    model.eval()
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", getattr(processor, "image_token_index", 32000))

    feature_rows: list[np.ndarray] = []
    meta_rows: list[dict] = []
    labels: list[int] = []
    failures: list[dict] = []

    for step, rec in enumerate(tqdm(records, desc="Feature extraction"), start=1):
        target = str(rec["target"])
        neighbor_list = neighbors.get(target, [])
        if not neighbor_list:
            failures.append({"split": rec.get("split", ""), "question_id": rec.get("question_id", ""), "target": target, "error": "no neighbors"})
            continue
        try:
            image = load_pope_image(rec)
            try:
                t_feats, t_info = extract_hidden(
                    model, processor, tokenizer, image, target, str(rec["question"]),
                    layers, args.target_token_policy, device, image_token_id
                )
                n_feats_all = []
                for neighbor in neighbor_list:
                    nf, _ = extract_hidden(
                        model, processor, tokenizer, image, neighbor, question_for_object(neighbor),
                        layers, args.target_token_policy, device, image_token_id
                    )
                    n_feats_all.append(nf)
                n_feats = {k: np.mean(np.stack([x[k] for x in n_feats_all]), axis=0).astype(np.float32) for k in n_feats_all[0]}
            finally:
                image.close()
            feature_rows.append(build_feature(t_feats, n_feats, blocks))
            labels.append(1 if str(rec["label"]).lower() == "no" else 0)
            meta_rows.append({
                "split": str(rec["split"]),
                "question_id": str(rec["question_id"]),
                "image_id": str(rec.get("image_id", "")),
                "label": str(rec["label"]).lower(),
                "target": target,
                "question": str(rec["question"]),
                "negative_type": str(rec.get("negative_type", "")),
                "best_neighbor": neighbor_list[0],
            })
        except Exception as exc:
            failures.append({"split": rec.get("split", ""), "question_id": rec.get("question_id", ""), "target": target, "error": repr(exc)})
        finally:
            if step % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if not feature_rows:
        raise ValueError("No rows scored")

    X = np.stack(feature_rows, axis=0)
    np.save(out_dir / "features.npy", X)
    np.save(out_dir / "labels.npy", np.array(labels, dtype=np.int32))

    with (out_dir / "meta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(meta_rows)

    (out_dir / "extract_metrics.json").write_text(json.dumps({
        "rows_requested": len(records),
        "rows_scored": len(feature_rows),
        "failures": len(failures),
        "failure_examples": failures[:5],
        "feature_dim": int(X.shape[1]),
        "layers": layers,
        "blocks": blocks,
        "shard_idx": args.shard_idx,
        "num_shards": args.num_shards,
    }, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %d feature rows (dim=%d) → %s", len(feature_rows), X.shape[1], out_dir)


if __name__ == "__main__":
    main()
