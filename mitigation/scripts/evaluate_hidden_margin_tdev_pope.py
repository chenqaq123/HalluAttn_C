#!/usr/bin/env python3
"""Training-free hidden-state target-vs-neighbor margins for POPE.

This is a deployable D-lite variant: no external detector and no supervised
readout. For each target and semantic-neighbor prompt, it extracts selected-layer
object-token hidden states and visual-token mean hidden states, then computes
low-dimensional alignment/separation margins.
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
    p = argparse.ArgumentParser(description="Run training-free hidden-margin TDEV on POPE")
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


def mean_norm(vecs: list[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.stack(vecs, dim=0).mean(dim=0).float(), dim=0)


def prompt_vectors(model, processor, tokenizer, image, obj: str, question: str, layers: list[int], policy: str, device: torch.device, image_token_id: int) -> tuple[dict[str, torch.Tensor], dict]:
    prompt = pope_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    input_ids_tensor = inputs["input_ids"][0]
    input_ids = [int(x) for x in input_ids_tensor.detach().cpu().tolist()]
    start, end, pattern = find_target_span(input_ids, tokenizer, obj)
    positions = torch.tensor(token_positions(start, end, policy), dtype=torch.long, device=device)
    vis_start, vis_end = find_vis_bounds(input_ids_tensor, image_token_id)
    with torch.inference_mode():
        outputs = model.forward(**inputs, output_hidden_states=True, output_attentions=False)
    obj_vecs = []
    vis_vecs = []
    for layer in layers:
        hs = outputs.hidden_states[layer][0].float()
        obj_vecs.append(hs[positions].mean(dim=0))
        vis_vecs.append(hs[vis_start:vis_end].mean(dim=0))
    vectors = {"obj": mean_norm(obj_vecs).cpu(), "vis": mean_norm(vis_vecs).cpu()}
    info = {"token_start": int(start), "token_end": int(end), "token_ids": pattern, "vis_tokens": int(vis_end - vis_start)}
    del inputs, outputs
    return vectors, info


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.float(), b.float()).item())


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
    logger.info("Scoring %d rows with hidden-margin D-lite layers=%s top_neighbors=%d", len(records), layers, args.top_neighbors)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager")
    model.eval()
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    scored: list[dict] = []
    failures: list[dict] = []
    for step, record in enumerate(tqdm(records, desc="Hidden-margin TDEV"), start=1):
        target = str(record["target"])
        neighbor_list = neighbors.get(target, [])
        if not neighbor_list:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": "no semantic neighbors"})
            continue
        try:
            image = load_pope_image(record)
            try:
                target_vecs, target_info = prompt_vectors(
                    model, processor, tokenizer, image, target, str(record["question"]), layers, args.target_token_policy, device, image_token_id
                )
                target_align = cos(target_vecs["obj"], target_vecs["vis"])
                candidates = []
                for neighbor in neighbor_list:
                    n_vecs, n_info = prompt_vectors(
                        model, processor, tokenizer, image, neighbor, question_for_object(neighbor), layers, args.target_token_policy, device, image_token_id
                    )
                    neighbor_align = cos(n_vecs["obj"], n_vecs["vis"])
                    cross_target_obj_neighbor_vis = cos(target_vecs["obj"], n_vecs["vis"])
                    cross_neighbor_obj_target_vis = cos(n_vecs["obj"], target_vecs["vis"])
                    obj_sep = 1.0 - cos(target_vecs["obj"], n_vecs["obj"])
                    vis_sep = 1.0 - cos(target_vecs["vis"], n_vecs["vis"])
                    margin = target_align - neighbor_align
                    cross_margin = target_align - max(cross_target_obj_neighbor_vis, cross_neighbor_obj_target_vis)
                    combined = margin + cross_margin + 0.25 * obj_sep + 0.25 * vis_sep
                    candidates.append({
                        "neighbor": neighbor,
                        "neighbor_info": n_info,
                        "target_align": target_align,
                        "neighbor_align": neighbor_align,
                        "hidden_align_margin": margin,
                        "hidden_cross_margin": cross_margin,
                        "hidden_obj_separation": obj_sep,
                        "hidden_vis_separation": vis_sep,
                        "hidden_margin_score": combined,
                        "cross_target_obj_neighbor_vis": cross_target_obj_neighbor_vis,
                        "cross_neighbor_obj_target_vis": cross_neighbor_obj_target_vis,
                    })
            finally:
                image.close()
            best = min(candidates, key=lambda item: item["hidden_margin_score"])
            score = float(best["hidden_margin_score"])
            scored.append({
                "split": str(record["split"]),
                "question_id": str(record["question_id"]),
                "image_id": str(record.get("image_id", "")),
                "label": str(record["label"]).lower(),
                "target": target,
                "question": str(record["question"]),
                "negative_type": str(record.get("negative_type", "")),
                "prediction": "yes" if score > 0.0 else "no",
                "tdev_margin": score,
                "hidden_margin_score": score,
                "hidden_align_margin": best["hidden_align_margin"],
                "hidden_cross_margin": best["hidden_cross_margin"],
                "hidden_obj_separation": best["hidden_obj_separation"],
                "hidden_vis_separation": best["hidden_vis_separation"],
                "target_align": best["target_align"],
                "neighbor_align": best["neighbor_align"],
                "cross_target_obj_neighbor_vis": best["cross_target_obj_neighbor_vis"],
                "cross_neighbor_obj_target_vis": best["cross_neighbor_obj_target_vis"],
                "best_neighbor": best["neighbor"],
                "neighbors": "|".join(neighbor_list),
                "target_token_start": target_info["token_start"],
                "target_token_end": target_info["token_end"],
                "neighbor_token_start": best["neighbor_info"]["token_start"],
                "neighbor_token_end": best["neighbor_info"]["token_end"],
            })
        except Exception as exc:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": repr(exc)})
        finally:
            if step % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    pred_path = output_dir / "hidden_margin_tdev_predictions.csv"
    write_csv(pred_path, scored)
    payload = {
        "audit_csv": args.audit_csv,
        "backend": "internal_hidden_margin_training_free",
        "external_detector": False,
        "supervised": False,
        "model_path": args.model_path,
        "neighbors_json": args.neighbors_json,
        "predictions_csv": str(pred_path),
        "rows_requested": len(records),
        "rows_scored": len(scored),
        "failures": len(failures),
        "failure_examples": failures[:10],
        "splits": splits,
        "layers": layers,
        "top_neighbors": args.top_neighbors,
        "target_token_policy": args.target_token_policy,
        "score_direction": "larger hidden_margin_score means stronger target-present/yes evidence relative to semantic neighbor",
        "formula": "hidden_align_margin + hidden_cross_margin + 0.25*hidden_obj_separation + 0.25*hidden_vis_separation; worst neighbor selected by minimum score",
    }
    (output_dir / "hidden_margin_tdev_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", pred_path)
    logger.info("Wrote %s", output_dir / "hidden_margin_tdev_metrics.json")


if __name__ == "__main__":
    main()
