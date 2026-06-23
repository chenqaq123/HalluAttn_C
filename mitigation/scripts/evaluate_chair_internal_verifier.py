#!/usr/bin/env python3
"""Internal target-vs-neighbor verifier features for CHAIR object mentions.

This scores generated caption object claims from an existing CHAIR object cache.
No external detector is used. For each object mention, it prompts LLaVA with the
mentioned object and its top semantic neighbor, then caches hidden-margin and
answer-confidence features aligned with the POPE internal verifier.
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

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.sink_utils import find_vis_bounds  # noqa: E402
from sinkdetect.utils import load_model_and_processor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score CHAIR object mentions with an internal target-vs-neighbor verifier")
    p.add_argument("--model_path", default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--object_cache", default="detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_neighbors", type=int, default=1)
    p.add_argument("--layers", default="22,31")
    p.add_argument("--target_token_policy", choices=["first", "last", "mean"], default="mean")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def parse_layers(spec: str) -> list[int]:
    layers = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not layers:
        raise ValueError("--layers cannot be empty")
    return layers


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_neighbors(path: Path, top_k: int) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {obj: [item["object"] for item in items[:top_k]] for obj, items in raw.items()}


def normalize_obj(obj: str) -> str:
    return str(obj).strip().lower().replace("_", " ")


def question_for_object(obj: str) -> str:
    article = "an" if obj[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    if obj.endswith("s") and obj not in {"scissors", "skis"}:
        return f"Are there any {obj} in the image?"
    return f"Is there {article} {obj} in the image?"


def pope_prompt(question: str) -> str:
    return f"<image>\nUSER: {question} Please just answer yes or no.\nASSISTANT:"


def image_path(coco_path: Path, image_id: int) -> Path:
    return coco_path / "val2014" / f"COCO_val2014_{int(image_id):012d}.jpg"


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


def find_target_span(input_ids: list[int], tokenizer, target: str) -> tuple[int, int]:
    end_before = assistant_start(input_ids, tokenizer)
    for pattern in candidate_tokenizations(tokenizer, target):
        match = find_subsequence(input_ids, pattern, end_before=end_before)
        if match is not None:
            return match[0], match[1]
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


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a.float(), b.float()).item())


def first_token_candidates(tokenizer, word: str) -> list[int]:
    ids = []
    for text in (word, word.capitalize(), word.upper()):
        toks = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
        if len(toks) == 1 and toks[0] not in ids:
            ids.append(toks[0])
    if not ids:
        raise ValueError(f"Could not tokenize candidate word {word!r} as single-token variants")
    return ids


def logsumexp_ids(logits: torch.Tensor, ids: list[int]) -> float:
    vals = logits[torch.tensor(ids, dtype=torch.long, device=logits.device)].float()
    return float(torch.logsumexp(vals, dim=0).item())


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def prompt_features(model, processor, tokenizer, image, obj: str, layers: list[int], policy: str, device: torch.device, image_token_id: int, yes_ids: list[int], no_ids: list[int]) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    prompt = pope_prompt(question_for_object(obj))
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    input_ids_tensor = inputs["input_ids"][0]
    input_ids = [int(x) for x in input_ids_tensor.detach().cpu().tolist()]
    start, end = find_target_span(input_ids, tokenizer, obj)
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
    next_logits = outputs.logits[0, -1].float()
    yes_logit = logsumexp_ids(next_logits, yes_ids)
    no_logit = logsumexp_ids(next_logits, no_ids)
    yes_margin = yes_logit - no_logit
    vectors = {"obj": mean_norm(obj_vecs).cpu(), "vis": mean_norm(vis_vecs).cpu()}
    info = {
        "token_start": int(start),
        "token_end": int(end),
        "vis_tokens": int(vis_end - vis_start),
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_margin": yes_margin,
        "yes_prob": sigmoid(yes_margin),
    }
    del inputs, outputs, next_logits
    return vectors, info


def iter_rows(args: argparse.Namespace) -> list[dict]:
    rows = read_jsonl(Path(args.object_cache))
    if args.limit > 0:
        rows = rows[:args.limit]
    if args.num_shards > 1:
        rows = rows[args.shard_idx::args.num_shards]
    return rows


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
    layers = parse_layers(args.layers)
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    rows = iter_rows(args)
    logger.info("Scoring %d CHAIR object mentions layers=%s top_neighbors=%d", len(rows), layers, args.top_neighbors)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager")
    model.eval()
    tokenizer = processor.tokenizer
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)
    yes_ids = first_token_candidates(tokenizer, "yes")
    no_ids = first_token_candidates(tokenizer, "no")
    coco_path = Path(args.coco_path)

    scored: list[dict] = []
    failures: list[dict] = []
    image_cache: dict[int, Image.Image] = {}
    for step, row in enumerate(tqdm(rows, desc="CHAIR internal verifier"), start=1):
        target = normalize_obj(row["word"])
        neighbor_list = [normalize_obj(x) for x in neighbors.get(target, [])]
        if not neighbor_list:
            failures.append({"object_id": row.get("object_id"), "image_id": row.get("image_id"), "word": target, "error": "no semantic neighbors"})
            continue
        try:
            iid = int(row["image_id"])
            image = image_cache.get(iid)
            if image is None:
                image = Image.open(image_path(coco_path, iid)).convert("RGB")
                image_cache[iid] = image
            target_vecs, target_info = prompt_features(model, processor, tokenizer, image, target, layers, args.target_token_policy, device, image_token_id, yes_ids, no_ids)
            target_align = cos(target_vecs["obj"], target_vecs["vis"])
            candidates = []
            for neighbor in neighbor_list:
                n_vecs, n_info = prompt_features(model, processor, tokenizer, image, neighbor, layers, args.target_token_policy, device, image_token_id, yes_ids, no_ids)
                neighbor_align = cos(n_vecs["obj"], n_vecs["vis"])
                cross_target_obj_neighbor_vis = cos(target_vecs["obj"], n_vecs["vis"])
                cross_neighbor_obj_target_vis = cos(n_vecs["obj"], target_vecs["vis"])
                obj_sep = 1.0 - cos(target_vecs["obj"], n_vecs["obj"])
                vis_sep = 1.0 - cos(target_vecs["vis"], n_vecs["vis"])
                hidden_align_margin = target_align - neighbor_align
                hidden_cross_margin = target_align - max(cross_target_obj_neighbor_vis, cross_neighbor_obj_target_vis)
                hidden_margin_score = hidden_align_margin + hidden_cross_margin + 0.25 * obj_sep + 0.25 * vis_sep
                answer_contrast_margin = float(target_info["yes_margin"]) - float(n_info["yes_margin"])
                candidates.append({
                    "neighbor": neighbor,
                    "neighbor_info": n_info,
                    "neighbor_align": neighbor_align,
                    "hidden_align_margin": hidden_align_margin,
                    "hidden_cross_margin": hidden_cross_margin,
                    "hidden_obj_separation": obj_sep,
                    "hidden_vis_separation": vis_sep,
                    "hidden_margin_score": hidden_margin_score,
                    "cross_target_obj_neighbor_vis": cross_target_obj_neighbor_vis,
                    "cross_neighbor_obj_target_vis": cross_neighbor_obj_target_vis,
                    "answer_contrast_margin": answer_contrast_margin,
                })
            best = min(candidates, key=lambda item: item["hidden_margin_score"])
            answer_support_score = float(target_info["yes_margin"])
            answer_neighbor_dominance = float(best["neighbor_info"]["yes_margin"]) - answer_support_score
            scored.append({
                "object_id": str(row["object_id"]),
                "image_id": str(row["image_id"]),
                "word": target,
                "label": int(row["label"]),
                "token_pos": int(row.get("token_pos", -1)),
                "gen_pos": int(row.get("gen_pos", -1)),
                "caption": str(row.get("caption", "")),
                "hidden_margin_score": best["hidden_margin_score"],
                "hidden_align_margin": best["hidden_align_margin"],
                "hidden_cross_margin": best["hidden_cross_margin"],
                "hidden_obj_separation": best["hidden_obj_separation"],
                "hidden_vis_separation": best["hidden_vis_separation"],
                "target_align": target_align,
                "neighbor_align": best["neighbor_align"],
                "cross_target_obj_neighbor_vis": best["cross_target_obj_neighbor_vis"],
                "cross_neighbor_obj_target_vis": best["cross_neighbor_obj_target_vis"],
                "answer_support_score": answer_support_score,
                "answer_absence_score": -answer_support_score,
                "answer_contrast_margin": best["answer_contrast_margin"],
                "answer_neighbor_dominance": answer_neighbor_dominance,
                "target_yes_logit": target_info["yes_logit"],
                "target_no_logit": target_info["no_logit"],
                "target_yes_prob": target_info["yes_prob"],
                "neighbor_yes_logit": best["neighbor_info"]["yes_logit"],
                "neighbor_no_logit": best["neighbor_info"]["no_logit"],
                "neighbor_yes_prob": best["neighbor_info"]["yes_prob"],
                "best_neighbor": best["neighbor"],
                "neighbors": "|".join(neighbor_list),
                "target_token_start": target_info["token_start"],
                "target_token_end": target_info["token_end"],
                "neighbor_token_start": best["neighbor_info"]["token_start"],
                "neighbor_token_end": best["neighbor_info"]["token_end"],
            })
        except Exception as exc:
            failures.append({"object_id": row.get("object_id"), "image_id": row.get("image_id"), "word": target, "error": repr(exc)})
        finally:
            if step % 16 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                if len(image_cache) > 64:
                    for img in image_cache.values():
                        img.close()
                    image_cache.clear()
    for img in image_cache.values():
        img.close()

    pred_path = output_dir / "chair_internal_verifier_scores.csv"
    write_csv(pred_path, scored)
    payload = {
        "object_cache": args.object_cache,
        "backend": "internal_hidden_answer_target_vs_neighbor",
        "external_detector": False,
        "supervised": False,
        "model_path": args.model_path,
        "neighbors_json": args.neighbors_json,
        "predictions_csv": str(pred_path),
        "rows_requested": len(rows),
        "rows_scored": len(scored),
        "failures": len(failures),
        "failure_examples": failures[:20],
        "layers": layers,
        "top_neighbors": args.top_neighbors,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "score_direction": "label=1 means CHAIR hallucinated; lower hidden/answer support and larger absence/dominance should indicate hallucination",
    }
    (output_dir / "chair_internal_verifier_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", pred_path)
    logger.info("Wrote %s", output_dir / "chair_internal_verifier_metrics.json")


if __name__ == "__main__":
    main()
