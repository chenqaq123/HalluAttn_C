#!/usr/bin/env python3
"""Attention-region target-vs-neighbor evidence for POPE.

Route B from the proposal: compare where target and semantic-neighbor prompts
attend inside the image. This uses only LLaVA internals, no external detector.

For each POPE row, the script runs a target prompt and one or more neighbor
prompts. It extracts visual-token attention rows at object tokens, averages over
selected layers/heads, and reports overlap/discriminability scores.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import load_model_and_processor  # noqa: E402
from src.data import iter_pope_records, load_pope_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run attention-region TDEV on POPE")
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
    p.add_argument("--score", choices=["low_overlap", "target_concentration", "combined"], default="combined")
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


def attention_region_row(model, processor, tokenizer, image, obj: str, question: str, layers: list[int], policy: str, device: torch.device, image_token_id: int) -> tuple[torch.Tensor, dict]:
    prompt = pope_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    input_ids_tensor = inputs["input_ids"][0]
    input_ids = [int(x) for x in input_ids_tensor.detach().cpu().tolist()]
    start, end, pattern = find_target_span(input_ids, tokenizer, obj)
    positions = torch.tensor(token_positions(start, end, policy), dtype=torch.long, device=device)
    vis_start, vis_end = find_vis_bounds(input_ids_tensor, image_token_id)
    with torch.inference_mode():
        outputs = model.forward(**inputs, output_attentions=True, output_hidden_states=False)
    rows = []
    for layer in layers:
        attn = outputs.attentions[layer][0]
        row = attn[:, positions, vis_start:vis_end].float().mean(dim=(0, 1))
        rows.append(row)
    merged = torch.stack(rows, dim=0).mean(dim=0)
    merged = merged / merged.sum().clamp_min(_EPS)
    info = {"token_start": start, "token_end": end, "token_ids": pattern}
    del inputs, outputs
    return merged.detach(), info


def row_features(row: torch.Tensor) -> dict[str, float]:
    row = row.float()
    sorted_row, _ = row.sort(descending=True)
    ent = -(row * row.clamp_min(_EPS).log()).sum() / torch.log(torch.tensor(float(row.numel()), device=row.device))
    return {
        "entropy": float(ent.item()),
        "top1_mass": float(sorted_row[0].item()),
        "top5_mass": float(sorted_row[:5].sum().item()),
        "max_over_mean": float((sorted_row[0] / row.mean().clamp_min(_EPS)).item()),
    }


def region_metrics(target_row: torch.Tensor, neighbor_row: torch.Tensor) -> dict[str, float]:
    target = target_row.float().clamp_min(_EPS)
    neighbor = neighbor_row.float().clamp_min(_EPS)
    target = target / target.sum().clamp_min(_EPS)
    neighbor = neighbor / neighbor.sum().clamp_min(_EPS)
    overlap = torch.minimum(target, neighbor).sum()
    dot = torch.dot(target, neighbor)
    js_m = 0.5 * (target + neighbor)
    js = 0.5 * (target * (target / js_m).log()).sum() + 0.5 * (neighbor * (neighbor / js_m).log()).sum()
    t_feat = row_features(target)
    n_feat = row_features(neighbor)
    return {
        "region_overlap": float(overlap.item()),
        "region_dot": float(dot.item()),
        "region_jsd": float(js.item()),
        "target_top5_mass": t_feat["top5_mass"],
        "neighbor_top5_mass": n_feat["top5_mass"],
        "target_entropy": t_feat["entropy"],
        "neighbor_entropy": n_feat["entropy"],
        "target_concentration_margin": t_feat["top5_mass"] - n_feat["top5_mass"],
    }


def score_from_metrics(vals: dict[str, float], mode: str) -> float:
    low_overlap = 1.0 - vals["region_overlap"]
    concentration = vals["target_concentration_margin"]
    if mode == "low_overlap":
        return low_overlap
    if mode == "target_concentration":
        return concentration
    return low_overlap + concentration


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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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
    logger.info("Scoring %d rows with attention-region TDEV layers=%s top_neighbors=%d", len(records), layers, args.top_neighbors)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager")
    model.eval()
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    scored: list[dict] = []
    failures: list[dict] = []
    for step, record in enumerate(tqdm(records, desc="Attention-region TDEV"), start=1):
        target = str(record["target"])
        try:
            image = load_pope_image(record)
            try:
                target_row, target_info = attention_region_row(
                    model, processor, tokenizer, image, target, str(record["question"]), layers, args.target_token_policy, device, image_token_id
                )
                candidates = []
                for neighbor in neighbors.get(target, []):
                    neighbor_question = question_for_object(neighbor)
                    neighbor_row, neighbor_info = attention_region_row(
                        model, processor, tokenizer, image, neighbor, neighbor_question, layers, args.target_token_policy, device, image_token_id
                    )
                    vals = region_metrics(target_row.to(device), neighbor_row.to(device))
                    score = score_from_metrics(vals, args.score)
                    candidates.append((score, neighbor, vals, neighbor_info))
                if candidates:
                    best_score, best_neighbor, vals, neighbor_info = max(candidates, key=lambda item: item[0])
                else:
                    best_score, best_neighbor, vals, neighbor_info = 0.0, "", {}, {"token_start": -1, "token_end": -1, "token_ids": []}
            finally:
                image.close()
            scored.append({
                "split": str(record["split"]),
                "question_id": str(record["question_id"]),
                "image_id": str(record.get("image_id", "")),
                "label": str(record["label"]).lower(),
                "target": target,
                "question": str(record["question"]),
                "negative_type": str(record.get("negative_type", "")),
                "prediction": "yes" if best_score > 0.0 else "no",
                "tdev_margin": best_score,
                "attention_region_score": best_score,
                "best_neighbor": best_neighbor,
                "neighbors": "|".join(neighbors.get(target, [])),
                "region_overlap": vals.get("region_overlap", 0.0),
                "region_dot": vals.get("region_dot", 0.0),
                "region_jsd": vals.get("region_jsd", 0.0),
                "target_top5_mass": vals.get("target_top5_mass", 0.0),
                "neighbor_top5_mass": vals.get("neighbor_top5_mass", 0.0),
                "target_entropy": vals.get("target_entropy", 0.0),
                "neighbor_entropy": vals.get("neighbor_entropy", 0.0),
                "target_concentration_margin": vals.get("target_concentration_margin", 0.0),
                "target_token_start": target_info["token_start"],
                "target_token_end": target_info["token_end"],
                "neighbor_token_start": neighbor_info["token_start"],
                "neighbor_token_end": neighbor_info["token_end"],
            })
        except Exception as exc:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": repr(exc)})
        finally:
            if step % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    pred_path = output_dir / "attention_region_tdev_predictions.csv"
    write_csv(pred_path, scored)
    payload = {
        "backend": "internal_attention_region",
        "external_detector": False,
        "model_path": args.model_path,
        "audit_csv": args.audit_csv,
        "neighbors_json": args.neighbors_json,
        "splits": splits,
        "top_neighbors": args.top_neighbors,
        "layers": layers,
        "target_token_policy": args.target_token_policy,
        "score": args.score,
        "rows_requested": len(records),
        "rows_scored": len(scored),
        "failures": len(failures),
        "failure_examples": failures[:20],
        "predictions_csv": str(pred_path),
        "score_direction": "larger attention_region_score means stronger target-vs-neighbor region discriminability",
    }
    (output_dir / "attention_region_tdev_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        write_csv(output_dir / "attention_region_tdev_failures.csv", failures)
    logger.info("Wrote %s", pred_path)
    logger.info("Wrote %s", output_dir / "attention_region_tdev_metrics.json")


if __name__ == "__main__":
    main()
