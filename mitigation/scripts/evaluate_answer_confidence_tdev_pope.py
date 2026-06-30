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
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from sinkdetect.utils import load_model_and_processor  # noqa: E402
from src.data import iter_pope_records, load_pope_image  # noqa: E402
from src.tdev_core import (  # noqa: E402
    existence_question,
    first_token_candidates,
    read_neighbor_map,
    score_yes_no_question,
)

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
    neighbors = read_neighbor_map(args.neighbors_json, args.top_neighbors)
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
            target_conf = score_yes_no_question(model, processor, image, str(record["question"]), yes_ids, no_ids, device)
            neighbor_confs = []
            for neighbor in neighbor_list:
                conf = score_yes_no_question(model, processor, image, existence_question(neighbor), yes_ids, no_ids, device)
                neighbor_confs.append({"neighbor": neighbor, **conf.__dict__})
            best_neighbor = max(neighbor_confs, key=lambda item: item["yes_margin"])
            target_yes_margin = target_conf.yes_margin
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
                "target_yes_logit": target_conf.yes_logit,
                "target_no_logit": target_conf.no_logit,
                "target_yes_margin": target_yes_margin,
                "target_yes_prob": target_conf.yes_prob,
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
