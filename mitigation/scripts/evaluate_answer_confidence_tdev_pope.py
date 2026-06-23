#!/usr/bin/env python3
"""Answer-confidence target-vs-neighbor evidence for POPE.

No external detector is used. For each target prompt and semantic-neighbor
prompt, the script reads the next-token yes/no logits at the assistant position
and caches low-dimensional confidence/contrast features for verifier tuning.
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.utils import load_model_and_processor  # noqa: E402
from src.data import iter_pope_records, load_pope_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run answer-confidence TDEV feature extraction on POPE")
    p.add_argument("--model_path", default="/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", required=True)
    p.add_argument("--pope_dir", required=True)
    p.add_argument("--audit_csv", default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv")
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", default="random,popular,adversarial")
    p.add_argument("--top_neighbors", type=int, default=1)
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


def question_for_object(obj: str) -> str:
    article = "an" if obj[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    if obj.endswith("s") and obj not in {"scissors", "skis"}:
        return f"Are there any {obj} in the image?"
    return f"Is there {article} {obj} in the image?"


def pope_prompt(question: str) -> str:
    return f"<image>\nUSER: {question} Please just answer yes or no.\nASSISTANT:"


def first_token_candidates(tokenizer, word: str) -> list[int]:
    ids = []
    # Use only variants that are themselves a single lexical token. Leading-space
    # variants often tokenize as a standalone whitespace token followed by the
    # word; including that shared whitespace token would contaminate yes/no margins.
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


def answer_confidence(model, processor, tokenizer, image, question: str, yes_ids: list[int], no_ids: list[int], device: torch.device) -> dict[str, float]:
    prompt = pope_prompt(question)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    with torch.inference_mode():
        outputs = model.forward(**inputs, output_hidden_states=False, output_attentions=False)
    next_logits = outputs.logits[0, -1].float()
    yes_logit = logsumexp_ids(next_logits, yes_ids)
    no_logit = logsumexp_ids(next_logits, no_ids)
    margin = yes_logit - no_logit
    del inputs, outputs, next_logits
    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_margin": margin,
        "yes_prob": sigmoid(margin),
    }


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
    audit_rows = read_audit_rows(Path(args.audit_csv), splits)
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    records = iter_joined_records(args, splits, audit_rows)
    logger.info("Scoring %d rows with answer-confidence TDEV top_neighbors=%d", len(records), args.top_neighbors)

    device = torch.device(f"cuda:{args.device}")
    model, processor = load_model_and_processor(args.model_path, device, cache_dir=args.cache_dir or None, attn_implementation="eager")
    model.eval()
    tokenizer = processor.tokenizer
    yes_ids = first_token_candidates(tokenizer, "yes")
    no_ids = first_token_candidates(tokenizer, "no")
    logger.info("yes token candidates=%s no token candidates=%s", yes_ids, no_ids)

    scored: list[dict] = []
    failures: list[dict] = []
    for step, record in enumerate(tqdm(records, desc="Answer-confidence TDEV"), start=1):
        target = str(record["target"])
        neighbor_list = neighbors.get(target, [])
        if not neighbor_list:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": "no semantic neighbors"})
            continue
        image = None
        try:
            image = load_pope_image(record)
            target_conf = answer_confidence(model, processor, tokenizer, image, str(record["question"]), yes_ids, no_ids, device)
            neighbor_confs = []
            for neighbor in neighbor_list:
                conf = answer_confidence(model, processor, tokenizer, image, question_for_object(neighbor), yes_ids, no_ids, device)
                neighbor_confs.append({"neighbor": neighbor, **conf})
            best_neighbor = max(neighbor_confs, key=lambda item: item["yes_margin"])
            target_yes_margin = target_conf["yes_margin"]
            neighbor_yes_margin = best_neighbor["yes_margin"]
            contrast_margin = target_yes_margin - neighbor_yes_margin
            neighbor_dominance = neighbor_yes_margin - target_yes_margin
            absence_score = -target_yes_margin
            scored.append({
                "split": str(record["split"]),
                "question_id": str(record["question_id"]),
                "image_id": str(record.get("image_id", "")),
                "label": str(record["label"]).lower(),
                "target": target,
                "question": str(record["question"]),
                "negative_type": str(record.get("negative_type", "")),
                "prediction": "yes" if target_yes_margin > 0.0 else "no",
                "answer_support_score": target_yes_margin,
                "answer_absence_score": absence_score,
                "answer_contrast_margin": contrast_margin,
                "answer_neighbor_dominance": neighbor_dominance,
                "target_yes_logit": target_conf["yes_logit"],
                "target_no_logit": target_conf["no_logit"],
                "target_yes_margin": target_yes_margin,
                "target_yes_prob": target_conf["yes_prob"],
                "neighbor_yes_logit": best_neighbor["yes_logit"],
                "neighbor_no_logit": best_neighbor["no_logit"],
                "neighbor_yes_margin": neighbor_yes_margin,
                "neighbor_yes_prob": best_neighbor["yes_prob"],
                "best_neighbor": best_neighbor["neighbor"],
                "neighbors": "|".join(neighbor_list),
                "yes_token_ids": "|".join(str(x) for x in yes_ids),
                "no_token_ids": "|".join(str(x) for x in no_ids),
            })
        except Exception as exc:
            failures.append({"split": str(record.get("split", "")), "question_id": str(record.get("question_id", "")), "target": target, "error": repr(exc)})
        finally:
            if image is not None:
                image.close()
            if step % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    pred_path = output_dir / "answer_confidence_tdev_predictions.csv"
    write_csv(pred_path, scored)
    payload = {
        "audit_csv": args.audit_csv,
        "backend": "internal_answer_confidence_training_free",
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
        "top_neighbors": args.top_neighbors,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "score_direction": "larger answer_support_score/answer_contrast_margin means stronger target-present/yes evidence; larger answer_absence_score/answer_neighbor_dominance means stronger absence/neighbor dominance evidence",
    }
    (output_dir / "answer_confidence_tdev_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote %s", pred_path)
    logger.info("Wrote %s", output_dir / "answer_confidence_tdev_metrics.json")


if __name__ == "__main__":
    main()
