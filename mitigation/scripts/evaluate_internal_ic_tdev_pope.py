#!/usr/bin/env python3
"""Internal IC/logit-lens target-vs-neighbor evidence for POPE.

No external detector is used. Each row runs one LLaVA forward pass, projects
visual-token hidden states through the LM head, and compares image-level visual
logit-lens confidence for the queried target against its semantic neighbors.
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
_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run internal IC/logit-lens TDEV on POPE")
    p.add_argument("--model_path", default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--audit_csv", default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--top_neighbors", type=int, default=10)
    p.add_argument("--image_layer", type=int, default=-1)
    p.add_argument("--phrase_pool", choices=["geom_mean", "mean", "min", "max"], default="geom_mean")
    p.add_argument("--prompt_mode", choices=["pope", "caption"], default="pope")
    p.add_argument("--target_score_threshold", type=float, default=0.0)
    p.add_argument("--margin_threshold", type=float, default=0.0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def split_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


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


def pope_prompt(question: str) -> str:
    return f"<image>\nUSER: {question} Please just answer yes or no.\nASSISTANT:"


def caption_probe_prompt() -> str:
    return "<image>\nUSER: Please describe the image.\nASSISTANT:"


def get_lm_head(model):
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.lm_head
    raise AttributeError("Cannot find language-model lm_head on this model")


def canonical_image_name(record: dict) -> str:
    image_id = str(record.get("image_id", ""))
    if image_id:
        return f"COCO_val2014_{int(image_id):012d}.jpg"
    return str(record.get("image", ""))


def object_forms(obj: str) -> list[str]:
    forms = [obj, " " + obj]
    if " " in obj:
        forms.extend([obj.replace(" ", "_"), " " + obj.replace(" ", "_")])
    out = []
    for form in forms:
        if form not in out:
            out.append(form)
    return out


def phrase_confidence(probs: torch.Tensor, token_ids: list[int], pool: str) -> float:
    vals = probs[:, token_ids].float().clamp_min(_EPS)
    if vals.ndim == 1:
        per_visual = vals
    elif pool == "geom_mean":
        per_visual = vals.log().mean(dim=-1).exp()
    elif pool == "mean":
        per_visual = vals.mean(dim=-1)
    elif pool == "min":
        per_visual = vals.min(dim=-1).values
    elif pool == "max":
        per_visual = vals.max(dim=-1).values
    else:
        raise ValueError(pool)
    return float(per_visual.max().item())


def object_confidence(probs: torch.Tensor, tokenizer, obj: str, pool: str) -> tuple[float, str, list[int]]:
    best_score = float("-inf")
    best_form = ""
    best_ids: list[int] = []
    for form in object_forms(obj):
        token_ids = [int(x) for x in tokenizer.encode(form, add_special_tokens=False)]
        if not token_ids:
            continue
        score = phrase_confidence(probs, token_ids, pool)
        if score > best_score:
            best_score = score
            best_form = form
            best_ids = token_ids
    if best_score == float("-inf"):
        return 0.0, "", []
    return best_score, best_form, best_ids


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


def clear_locals(names: list[str], scope: dict) -> None:
    for name in names:
        if name in scope:
            del scope[name]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_list(args.splits)
    audit_rows = read_audit_rows(Path(args.audit_csv), splits)
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    records = iter_joined_records(args, splits, audit_rows)
    logger.info("Scoring %d POPE rows for splits=%s shard=%d/%d", len(records), splits, args.shard_idx, args.num_shards)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(
        args.model_path,
        device,
        cache_dir=args.cache_dir or None,
        attn_implementation="eager",
    )
    model.eval()
    tokenizer = processor.tokenizer
    lm_head = get_lm_head(model)
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)

    scored: list[dict] = []
    failures: list[dict] = []
    for step, record in enumerate(tqdm(records, desc="Internal IC TDEV"), start=1):
        target = str(record["target"])
        try:
            prompt = pope_prompt(str(record["question"])) if args.prompt_mode == "pope" else caption_probe_prompt()
            image = load_pope_image(record)
            try:
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
            finally:
                image.close()
            vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], image_token_id)
            with torch.no_grad():
                outputs = model.forward(**inputs, output_hidden_states=True, output_attentions=False)
                hidden_states = outputs.hidden_states
                layer_idx = args.image_layer if args.image_layer >= 0 else len(hidden_states) + args.image_layer
                visual_hidden = hidden_states[layer_idx][0, vis_start:vis_end, :]
                visual_logits = lm_head(visual_hidden).float()
                visual_probs = torch.softmax(visual_logits, dim=-1)
                target_score, target_form, target_token_ids = object_confidence(visual_probs, tokenizer, target, args.phrase_pool)
                neighbor_scores = []
                for neighbor in neighbors.get(target, []):
                    score, form, token_ids = object_confidence(visual_probs, tokenizer, neighbor, args.phrase_pool)
                    neighbor_scores.append((neighbor, score, form, token_ids))
                if neighbor_scores:
                    best_neighbor, best_neighbor_score, best_neighbor_form, best_neighbor_token_ids = max(neighbor_scores, key=lambda item: item[1])
                    margin = target_score - best_neighbor_score
                else:
                    best_neighbor, best_neighbor_score, best_neighbor_form, best_neighbor_token_ids = "", 0.0, "", []
                    margin = target_score
            prediction = "yes" if (target_score > args.target_score_threshold and margin > args.margin_threshold) else "no"
            scored.append({
                "split": str(record["split"]),
                "question_id": str(record["question_id"]),
                "image": canonical_image_name(record),
                "image_id": str(record.get("image_id", "")),
                "label": str(record["label"]).lower(),
                "target": target,
                "question": str(record["question"]),
                "negative_type": str(record.get("negative_type", "")),
                "target_score": target_score,
                "best_neighbor": best_neighbor,
                "best_neighbor_score": best_neighbor_score,
                "tdev_margin": margin,
                "ic_neighbor_dominance": best_neighbor_score - target_score,
                "prediction": prediction,
                "neighbors": "|".join(neighbors.get(target, [])),
                "target_token_form": target_form,
                "target_token_ids": "|".join(map(str, target_token_ids)),
                "best_neighbor_token_form": best_neighbor_form,
                "best_neighbor_token_ids": "|".join(map(str, best_neighbor_token_ids)),
            })
        except Exception as exc:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": repr(exc)})
        finally:
            clear_locals(["inputs", "outputs", "hidden_states", "visual_hidden", "visual_logits", "visual_probs"], locals())
            if step % 16 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    predictions_path = output_dir / "internal_ic_tdev_predictions.csv"
    write_csv(predictions_path, scored)
    payload = {
        "backend": "internal_ic_logit_lens",
        "external_detector": False,
        "model_path": args.model_path,
        "audit_csv": args.audit_csv,
        "neighbors_json": args.neighbors_json,
        "splits": splits,
        "top_neighbors": args.top_neighbors,
        "image_layer": args.image_layer,
        "phrase_pool": args.phrase_pool,
        "prompt_mode": args.prompt_mode,
        "rows_requested": len(records),
        "rows_scored": len(scored),
        "failures": len(failures),
        "failure_examples": failures[:20],
        "predictions_csv": str(predictions_path),
        "score_direction": "larger tdev_margin means stronger target support; larger ic_neighbor_dominance means stronger hallucination evidence",
    }
    (output_dir / "internal_ic_tdev_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        write_csv(output_dir / "internal_ic_tdev_failures.csv", failures)
    logger.info("Wrote %s", predictions_path)
    logger.info("Wrote %s", output_dir / "internal_ic_tdev_metrics.json")


if __name__ == "__main__":
    main()
