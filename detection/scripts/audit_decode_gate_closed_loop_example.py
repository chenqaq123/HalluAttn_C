#!/usr/bin/env python3
"""Audit new object claims introduced by a gated caption smoke run.

The decode gate can suppress denied surface phrases, but a caption may route to
new object claims. This script checks those newly introduced claims with the
same target-vs-neighbor OWLv2 evidence used by TDEV.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
MITIGATION_SCRIPTS = REPO_ROOT / "mitigation" / "scripts"
PAS_SRC = REPO_ROOT.parent / "pas" / "src"
for path in (DETECTION_SRC, MITIGATION_SCRIPTS, PAS_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from owlv2_cache_utils import encode_image_object_scores
from sinkdetect.chair import evaluate_chair, load_chair_evaluator
from sinkdetect.open_vocab_claims import open_vocab_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--examples_json",
        default="detection/baselines/results/tdev_decode_gate_caption_smoke/gated_generation_examples.json",
    )
    p.add_argument("--neighbors_json", default="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json")
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    p.add_argument("--variant_aliases_json", default="detection/config/object_variant_aliases.json")
    p.add_argument("--variant_roots_json", default="detection/config/object_variant_roots.json")
    p.add_argument("--output_dir", default="detection/baselines/results/tdev_decode_gate_closed_loop_example")
    p.add_argument(
        "--owlv2_model_path",
        default="/home/chenguanxu/common_model/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d",
    )
    p.add_argument("--device", default="cuda:5")
    p.add_argument("--top_neighbors", type=int, default=10)
    p.add_argument("--open_vocab_min_token_len", type=int, default=3)
    p.add_argument("--open_vocab_candidate_limit", type=int, default=32)
    p.add_argument("--two_stage_low", type=float, default=0.10)
    p.add_argument("--two_stage_high", type=float, default=0.16)
    p.add_argument("--two_stage_margin", type=float, default=-0.15)
    return p.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_neighbors(path: Path, top_k: int) -> dict[str, list[str]]:
    raw = read_json(path)
    return {obj: [item["object"] for item in items[:top_k]] for obj, items in raw.items()}


def read_variant_aliases(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw = read_json(path)
    return {
        str(obj).strip().lower(): [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
        for obj, aliases in raw.items()
    }


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.lower())
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def variant_hits(caption: str, denied_words: set[str], aliases: dict[str, list[str]]) -> list[dict[str, str]]:
    text = caption.lower()
    hits: list[dict[str, str]] = []
    for word in sorted(denied_words):
        for alias in aliases.get(word, []):
            if phrase_pattern(alias).search(text):
                hits.append({"word": word, "alias": alias})
    return hits


def read_variant_roots(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw = read_json(path)
    return {
        str(obj).strip().lower(): [str(root).strip().lower() for root in roots if str(root).strip()]
        for obj, roots in raw.items()
    }


def root_hits(caption: str, denied_words: set[str], roots_by_word: dict[str, list[str]]) -> list[dict[str, str]]:
    tokens = re.findall(r"[a-z0-9][a-z0-9-]*", caption.lower())
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for word in sorted(denied_words):
        for root in roots_by_word.get(word, []):
            for token in tokens:
                if root in token:
                    key = (word, root, token)
                    if key not in seen:
                        hits.append({"word": word, "root": root, "token": token})
                        seen.add(key)
    return hits


def two_stage_score(target_score: float, margin: float, low: float, high: float, margin_threshold: float) -> float:
    return max(target_score - high, min(target_score - low, margin - margin_threshold))


def two_stage_present(target_score: float, margin: float, low: float, high: float, margin_threshold: float) -> bool:
    return target_score > high or (target_score > low and margin > margin_threshold)


def chair_words(chair_pkl: str, rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    evaluator = load_chair_evaluator(chair_pkl)
    per_sample, _ = evaluate_chair(
        evaluator,
        data=rows,
        json_path=str(output_dir / "closed_loop_chair_input.json"),
    )
    out = []
    for sample in per_sample:
        generated = list(sample.get("mscoco_generated_words", []))
        grounded = set(sample.get("mscoco_gt_words", []))
        out.append(
            {
                "image_id": int(sample["image_id"]),
                "caption": sample.get("caption", ""),
                "generated_words": generated,
                "grounded_words": sorted(grounded),
                "hallucinated_words": [word for word in generated if word not in grounded],
            }
        )
    return out


def score_claims(
    image_path: Path,
    claims: list[str],
    neighbors: dict[str, list[str]],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if not claims:
        return {}
    object_names = set(claims)
    for claim in claims:
        object_names.update(neighbors.get(claim, []))
    object_list = sorted(object_names)
    prompts = [f"a photo of a {obj}" for obj in object_list]
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    processor = Owlv2Processor.from_pretrained(args.owlv2_model_path, local_files_only=True)
    model = Owlv2ForObjectDetection.from_pretrained(args.owlv2_model_path, local_files_only=True).to(device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    scores = encode_image_object_scores(model, processor, [image], prompts, object_list, device)[0]
    image.close()

    out: dict[str, dict[str, Any]] = {}
    for claim in claims:
        neighbor_scores = [(name, scores[name]) for name in neighbors.get(claim, []) if name in scores]
        if neighbor_scores:
            best_neighbor, best_neighbor_score = max(neighbor_scores, key=lambda item: item[1])
            margin = scores[claim] - best_neighbor_score
        else:
            best_neighbor, best_neighbor_score = "", 0.0
            margin = scores[claim]
        ts_score = two_stage_score(
            scores[claim],
            margin,
            args.two_stage_low,
            args.two_stage_high,
            args.two_stage_margin,
        )
        out[claim] = {
            "target_score": scores[claim],
            "best_neighbor": best_neighbor,
            "best_neighbor_score": best_neighbor_score,
            "tdev_margin": margin,
            "two_stage_score": ts_score,
            "two_stage_present": int(
                two_stage_present(
                    scores[claim],
                    margin,
                    args.two_stage_low,
                    args.two_stage_high,
                    args.two_stage_margin,
                )
            ),
            "top_neighbors": neighbors.get(claim, []),
        }
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_json(Path(args.examples_json))
    neighbors = read_neighbors(Path(args.neighbors_json), args.top_neighbors)
    variant_aliases = read_variant_aliases(Path(args.variant_aliases_json))
    variant_roots = read_variant_roots(Path(args.variant_roots_json))

    caption_rows: list[dict[str, Any]] = []
    for example in examples:
        image_id = int(example["image_id"])
        caption_rows.extend(
            [
                {"image_id": image_id, "caption": example["vanilla_caption"], "variant": "vanilla"},
                {"image_id": image_id, "caption": example["gated_caption"], "variant": "gated"},
            ]
        )
    chair = chair_words(args.chair_pkl, caption_rows, output_dir)
    chair_by_key = {
        (int(row["image_id"]), caption_rows[idx]["variant"]): row
        for idx, row in enumerate(chair)
    }

    audited = []
    for example in examples:
        image_id = int(example["image_id"])
        vanilla = chair_by_key[(image_id, "vanilla")]
        gated = chair_by_key[(image_id, "gated")]
        vanilla_words = set(vanilla["generated_words"])
        gated_words = set(gated["generated_words"])
        denied = {item["word"] for item in example.get("denied_items", [])}
        introduced = sorted(gated_words - vanilla_words)
        removed = sorted(vanilla_words - gated_words)
        introduced_hallucinated = sorted(set(gated["hallucinated_words"]) & set(introduced))
        vanilla_variant_hits = variant_hits(example["vanilla_caption"], denied, variant_aliases)
        gated_variant_hits = variant_hits(example["gated_caption"], denied, variant_aliases)
        vanilla_hit_keys = {(hit["word"], hit["alias"]) for hit in vanilla_variant_hits}
        introduced_variant_leaks = [
            hit for hit in gated_variant_hits
            if (hit["word"], hit["alias"]) not in vanilla_hit_keys
        ]
        vanilla_root_hits = root_hits(example["vanilla_caption"], denied, variant_roots)
        gated_root_hits = root_hits(example["gated_caption"], denied, variant_roots)
        vanilla_root_keys = {(hit["word"], hit["root"], hit["token"]) for hit in vanilla_root_hits}
        introduced_root_leaks = [
            hit for hit in gated_root_hits
            if (hit["word"], hit["root"], hit["token"]) not in vanilla_root_keys
        ]
        vanilla_open_vocab = open_vocab_candidates(
            example["vanilla_caption"],
            min_len=args.open_vocab_min_token_len,
            limit=args.open_vocab_candidate_limit,
        )
        gated_open_vocab = open_vocab_candidates(
            example["gated_caption"],
            min_len=args.open_vocab_min_token_len,
            limit=args.open_vocab_candidate_limit,
        )
        vanilla_open_vocab_set = set(vanilla_open_vocab)
        introduced_open_vocab = [claim for claim in gated_open_vocab if claim not in vanilla_open_vocab_set]
        all_claims = sorted(set(introduced) | set(introduced_open_vocab))
        all_claim_scores = score_claims(Path(example["image_path"]), all_claims, neighbors, args)
        claim_scores = {claim: all_claim_scores[claim] for claim in introduced if claim in all_claim_scores}
        open_vocab_claim_scores = {
            claim: all_claim_scores[claim]
            for claim in introduced_open_vocab
            if claim in all_claim_scores
        }
        unsupported_open_vocab_claims = [
            claim for claim, score in open_vocab_claim_scores.items()
            if int(score.get("two_stage_present", 0)) == 0
        ]
        audited.append(
            {
                "image_id": image_id,
                "image_path": example["image_path"],
                "denied_words": sorted(denied),
                "vanilla_words": vanilla["generated_words"],
                "gated_words": gated["generated_words"],
                "removed_words": removed,
                "introduced_words": introduced,
                "introduced_hallucinated_words": introduced_hallucinated,
                "gated_variant_hits": gated_variant_hits,
                "introduced_variant_leaks": introduced_variant_leaks,
                "gated_root_hits": gated_root_hits,
                "introduced_root_leaks": introduced_root_leaks,
                "open_vocab_vanilla_candidates": vanilla_open_vocab,
                "open_vocab_gated_candidates": gated_open_vocab,
                "introduced_open_vocab_candidates": introduced_open_vocab,
                "introduced_open_vocab_claim_scores": open_vocab_claim_scores,
                "unsupported_open_vocab_claims": unsupported_open_vocab_claims,
                "introduced_claim_scores": claim_scores,
                "interpretation": (
                    "A closed-loop gate should verify introduced_words before allowing "
                    "the rewritten caption path to continue."
                ),
            }
        )

    payload = {
        "examples_json": args.examples_json,
        "neighbors_json": args.neighbors_json,
        "variant_aliases_json": args.variant_aliases_json,
        "variant_roots_json": args.variant_roots_json,
        "owlv2_model_path": args.owlv2_model_path,
        "top_neighbors": args.top_neighbors,
        "open_vocab_min_token_len": args.open_vocab_min_token_len,
        "open_vocab_candidate_limit": args.open_vocab_candidate_limit,
        "two_stage_low": args.two_stage_low,
        "two_stage_high": args.two_stage_high,
        "two_stage_margin": args.two_stage_margin,
        "examples": audited,
    }
    with (output_dir / "closed_loop_example_audit.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
