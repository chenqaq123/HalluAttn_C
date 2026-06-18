#!/usr/bin/env python3
"""Physically edit captions by deleting high-risk object mentions.

This is a deterministic text-edit proxy for caption-side mitigation. It uses
TDEV-derived object risk scores to select CHAIR object mentions, removes a
matching object phrase from the caption, and reports deletion-only CHAIR-style
accounting. It does not regenerate captions. With --run_chair, it also reruns the official CHAIR scorer on the original and edited captions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = REPO_ROOT / "detection" / "src"
PAS_SRC = REPO_ROOT.parent / "pas" / "src"
for import_path in (DETECTION_SRC, PAS_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TDEV caption text-edit proxy")
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--chair_source",
        default="../pas/src/pas/evaluate/chair.py",
        help="PAS CHAIR source file used only to parse synonyms_txt.",
    )
    p.add_argument(
        "--output_dir",
        default="detection/baselines/results/tdev_caption_edit",
    )
    p.add_argument("--score", default="hybrid_positive_branch_absence")
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    p.add_argument("--run_chair", action="store_true", help="Rerun official PAS CHAIR on original and edited captions.")
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def two_stage_score(
    target: np.ndarray,
    margin: np.ndarray,
    low: float,
    high: float,
    margin_threshold: float,
) -> np.ndarray:
    high_branch = target - high
    medium_branch = np.minimum(target - low, margin - margin_threshold)
    return np.maximum(high_branch, medium_branch)


def build_scores(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, np.ndarray]:
    target = np.asarray([float(row["target_score"]) for row in rows], dtype=np.float64)
    neighbor = np.asarray([float(row["best_neighbor_score"]) for row in rows], dtype=np.float64)
    margin = np.asarray([float(row["tdev_margin"]) for row in rows], dtype=np.float64)
    dominance = np.maximum(neighbor - target, 0.0)
    return {
        "target_absence": -target,
        "margin_absence": -margin,
        "hybrid_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_margin,
        ),
        "hybrid_mcc_positive_branch_absence": -two_stage_score(
            target,
            margin,
            args.hybrid_low,
            args.hybrid_high,
            args.hybrid_mcc_margin,
        ),
        f"target_absence_plus_neighbor_dominance_{args.neighbor_dominance_alpha:g}": (
            -target + args.neighbor_dominance_alpha * dominance
        ),
    }


def load_synonyms(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"synonyms_txt\s*=\s*'''(.*?)'''", text, flags=re.S)
    if not match:
        raise ValueError(f"Could not find synonyms_txt in {path}")
    synonyms: dict[str, list[str]] = {}
    for raw_line in match.group(1).splitlines():
        items = [item.strip().lower() for item in raw_line.split(",") if item.strip()]
        if not items:
            continue
        canonical = items[0]
        deduped = list(OrderedDict.fromkeys(items + [canonical]))
        synonyms[canonical] = sorted(deduped, key=lambda item: (-len(item), item))
    return synonyms


def pluralize_last_token(phrase: str) -> str:
    parts = phrase.split()
    if not parts:
        return phrase
    last = parts[-1]
    if last.endswith(("s", "x", "ch", "sh")):
        parts[-1] = f"{last}es"
    elif last.endswith("y") and len(last) > 1 and last[-2] not in "aeiou":
        parts[-1] = f"{last[:-1]}ies"
    else:
        parts[-1] = f"{last}s"
    return " ".join(parts)


def phrase_variants(phrase: str) -> list[str]:
    variants = [phrase, pluralize_last_token(phrase)]
    return list(OrderedDict.fromkeys(variants))


def compile_patterns(phrases: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    expanded: list[str] = []
    for phrase in phrases:
        expanded.extend(phrase_variants(phrase))
    patterns = []
    for phrase in sorted(OrderedDict.fromkeys(expanded), key=lambda item: (-len(item), item)):
        escaped = re.escape(phrase).replace(r"\ ", r"\s+")
        patterns.append((phrase, re.compile(r"(?<!\w)(?:(?:a|an|the)\s+)?" + escaped + r"(?!\w)", flags=re.I)))
    return patterns


def top_fraction_mask(values: np.ndarray, frac: float) -> np.ndarray:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"top_frac must be in (0,1): {frac}")
    n_remove = max(1, int(round(values.size * frac)))
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:n_remove]] = True
    return mask


def clean_caption(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\b(a|an|the)(?=[,.;:!?])", "", text, flags=re.I)
    text = re.sub(r"\b(a|an|the)\s+([,.;:!?])", r"\2", text, flags=re.I)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def delete_one_phrase(caption: str, patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[str, str | None]:
    for phrase, pattern in patterns:
        match = pattern.search(caption)
        if not match:
            continue
        edited = caption[: match.start()] + caption[match.end() :]
        return clean_caption(edited), phrase
    return caption, None


def image_hallucination_rate(rows: list[dict[str, str]], keep_mask: np.ndarray) -> float:
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    unique_images = np.unique(image_ids)
    if unique_images.size == 0:
        return 0.0
    hallucinated_images = 0
    for image_id in unique_images:
        mask = (image_ids == image_id) & keep_mask
        hallucinated_images += int(labels[mask].sum() > 0)
    return hallucinated_images / unique_images.size


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def caption_stats(data: list[dict[str, object]]) -> dict[str, float | int]:
    lengths = [len(str(row["caption"]).split()) for row in data]
    return {
        "num_captions": len(data),
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def chair_summary(data: list[dict[str, object]], chair_pkl: str, output_dir: Path, prefix: str) -> dict:
    from sinkdetect.chair import evaluate_chair, load_chair_evaluator

    evaluator = load_chair_evaluator(chair_pkl)
    per_sample, overall = evaluate_chair(
        evaluator,
        data=data,
        json_path=str(output_dir / f"{prefix}_chair_input.json"),
    )
    mention_counts = []
    hallucinated_counts = []
    with (output_dir / f"{prefix}_chair_per_sample.jsonl").open("w", encoding="utf-8") as f:
        for sample in per_sample:
            generated = list(sample.get("mscoco_generated_words", []))
            grounded = set(sample.get("mscoco_gt_words", []))
            hallucinated = sum(word not in grounded for word in generated)
            mention_counts.append(len(generated))
            hallucinated_counts.append(hallucinated)
            f.write(json.dumps({
                "image_id": sample.get("image_id"),
                "caption": sample.get("caption", ""),
                "object_mentions": len(generated),
                "hallucinated_mentions": hallucinated,
            }, ensure_ascii=False) + "\n")
    stats = caption_stats(data)
    stats.update({
        "mean_object_mentions": sum(mention_counts) / len(mention_counts) if mention_counts else 0.0,
        "mean_hallucinated_mentions": sum(hallucinated_counts) / len(hallucinated_counts) if hallucinated_counts else 0.0,
        "total_object_mentions": sum(mention_counts),
        "total_hallucinated_mentions": sum(hallucinated_counts),
    })
    return {"chair": overall, "caption_stats": stats}


def metric_delta(edited: dict, vanilla: dict) -> dict[str, float]:
    delta = {}
    for section in ("chair", "caption_stats"):
        for key, value in edited[section].items():
            base = vanilla[section].get(key)
            if isinstance(value, (int, float)) and isinstance(base, (int, float)):
                delta[f"delta_{section}_{key}"] = value - base
    return delta


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.scores_csv))
    scores = build_scores(rows, args)
    if args.score not in scores:
        raise KeyError(f"Unknown score {args.score!r}; available: {sorted(scores)}")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    values = scores[args.score]
    selected_mask = top_fraction_mask(values, args.top_frac)

    synonyms = load_synonyms(Path(args.chair_source))
    pattern_cache = {word: compile_patterns(phrases) for word, phrases in synonyms.items()}

    captions: OrderedDict[int, str] = OrderedDict()
    for row in rows:
        image_id = int(row["image_id"])
        captions.setdefault(image_id, row["caption"])
    original_captions = dict(captions)

    deleted_mask = np.zeros(len(rows), dtype=bool)
    deletion_rows: list[dict[str, object]] = []
    selected_indices = np.flatnonzero(selected_mask)
    selected_indices = selected_indices[np.argsort(-values[selected_indices], kind="mergesort")]
    for idx in selected_indices:
        row = rows[int(idx)]
        image_id = int(row["image_id"])
        word = row["word"].strip().lower()
        patterns = pattern_cache.get(word, compile_patterns([word]))
        edited, phrase = delete_one_phrase(captions[image_id], patterns)
        deleted = phrase is not None
        if deleted:
            captions[image_id] = edited
            deleted_mask[int(idx)] = True
        deletion_rows.append(
            {
                "object_id": int(row["object_id"]),
                "image_id": image_id,
                "word": row["word"],
                "label": int(row["label"]),
                "score": float(values[int(idx)]),
                "matched_phrase": phrase or "",
                "deleted": int(deleted),
            }
        )

    mentions = int(labels.size)
    hallucinated = int(labels.sum())
    grounded = mentions - hallucinated
    selected = int(selected_mask.sum())
    selected_hallucinated = int(labels[selected_mask].sum())
    selected_grounded = selected - selected_hallucinated
    deleted = int(deleted_mask.sum())
    deleted_hallucinated = int(labels[deleted_mask].sum())
    deleted_grounded = deleted - deleted_hallucinated
    keep_mask = ~deleted_mask
    remaining = int(keep_mask.sum())
    remaining_hallucinated = int(labels[keep_mask].sum())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edited_captions = [
        {"image_id": image_id, "caption": caption}
        for image_id, caption in captions.items()
    ]
    edited_pairs = [
        {
            "image_id": image_id,
            "caption": captions[image_id],
            "original_caption": original_captions[image_id],
        }
        for image_id in captions
    ]
    with (output_dir / "edited_captions.json").open("w", encoding="utf-8") as f:
        json.dump(edited_captions, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with (output_dir / "edited_caption_pairs.json").open("w", encoding="utf-8") as f:
        json.dump(edited_pairs, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_csv(output_dir / "deleted_mentions.csv", deletion_rows)

    metrics = {
        "scores_csv": args.scores_csv,
        "score": args.score,
        "score_direction": "larger score means higher risk and is selected first",
        "top_frac": args.top_frac,
        "mentions": mentions,
        "hallucinated_mentions": hallucinated,
        "grounded_mentions": grounded,
        "selected_mentions": selected,
        "selected_hallucinated": selected_hallucinated,
        "selected_grounded": selected_grounded,
        "deleted_mentions": deleted,
        "deleted_hallucinated": deleted_hallucinated,
        "deleted_grounded": deleted_grounded,
        "delete_success_rate": deleted / selected if selected else 0.0,
        "deletion_precision": deleted_hallucinated / deleted if deleted else 0.0,
        "hallucinated_reduction": deleted_hallucinated / hallucinated if hallucinated else 0.0,
        "grounded_loss": deleted_grounded / grounded if grounded else 0.0,
        "remaining_mentions": remaining,
        "remaining_hallucinated": remaining_hallucinated,
        "remaining_mention_hallucination_rate": remaining_hallucinated / remaining if remaining else 0.0,
        "remaining_image_hallucination_rate": image_hallucination_rate(rows, keep_mask),
        "chair_rerun_status": "not_run_use_--run_chair",
    }

    if args.run_chair:
        vanilla_data = [
            {"image_id": image_id, "caption": original_captions[image_id]}
            for image_id in captions
        ]
        official_chair = {
            "chair_pkl": args.chair_pkl,
            "sample_scope": "images with CHAIR object mentions in the OWLv2 score CSV",
            "vanilla": chair_summary(vanilla_data, args.chair_pkl, output_dir, "vanilla"),
            "edited": chair_summary(edited_captions, args.chair_pkl, output_dir, "edited"),
        }
        official_chair["delta"] = metric_delta(official_chair["edited"], official_chair["vanilla"])
        with (output_dir / "chair_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(official_chair, f, indent=2, sort_keys=True)
            f.write("\n")
        metrics["chair_rerun_status"] = "run"
        metrics["chair_metrics_file"] = str(output_dir / "chair_metrics.json")
    with (output_dir / "caption_edit_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {output_dir / 'caption_edit_metrics.json'}")


if __name__ == "__main__":
    main()
