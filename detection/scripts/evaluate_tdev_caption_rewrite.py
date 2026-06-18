#!/usr/bin/env python3
"""Evaluate a sentence-local neutral rewrite proxy for TDEV caption mitigation.

The existing caption-edit proxy physically deletes high-risk object mentions.
This script tests a slightly more natural deterministic rewrite: select the same
TDEV-risk object mentions, replace the matched object phrase with a neutral
placeholder (``something`` or ``someone``), and rerun CHAIR on the rewritten
captions. It is still a proxy, not a fluent neural rewriter, but it separates the
TDEV decision rule from raw phrase deletion.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
from types import ModuleType

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EDIT_SCRIPT = REPO_ROOT / "detection" / "scripts" / "evaluate_tdev_caption_edit.py"


GENERIC_REPLACEMENTS = {
    "dining table": ("a surface", "some surfaces"),
    "table": ("a surface", "some surfaces"),
    "potted plant": ("a decorative item", "some decorative items"),
    "tv": ("a device", "some devices"),
    "television": ("a device", "some devices"),
    "remote": ("a device", "some devices"),
    "cell phone": ("a device", "some devices"),
    "laptop": ("a device", "some devices"),
    "microwave": ("an appliance", "some appliances"),
    "toaster": ("an appliance", "some appliances"),
    "oven": ("an appliance", "some appliances"),
    "refrigerator": ("an appliance", "some appliances"),
    "car": ("a vehicle", "some vehicles"),
    "airplane": ("a vehicle", "some vehicles"),
    "bus": ("a vehicle", "some vehicles"),
    "truck": ("a vehicle", "some vehicles"),
    "bicycle": ("a vehicle", "some vehicles"),
    "skis": ("some equipment", "some equipment"),
    "sports ball": ("an item", "some items"),
    "baseball bat": ("some equipment", "some equipment"),
    "baseball glove": ("some equipment", "some equipment"),
}

PLURAL_HINTS = {
    "people",
    "men",
    "women",
    "children",
    "boys",
    "girls",
    "skis",
    "scissors",
}

PERSON_WORDS = {
    "person",
    "people",
    "man",
    "men",
    "woman",
    "women",
    "boy",
    "boys",
    "girl",
    "girls",
    "child",
    "children",
}

LEFT_MODIFIER_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "with",
    "without",
    "near",
    "by",
    "to",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "has",
    "have",
    "had",
    "having",
    "include",
    "includes",
    "including",
    "contain",
    "contains",
    "containing",
    "feature",
    "features",
    "featuring",
    "around",
    "inside",
    "outside",
}


def load_edit_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tdev_caption_edit", EDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {EDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TDEV caption neutral-rewrite proxy")
    p.add_argument(
        "--scores_csv",
        default="detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv",
    )
    p.add_argument(
        "--chair_source",
        default="../pas/src/pas/evaluate/chair.py",
        help="PAS CHAIR source file used only to parse synonyms_txt.",
    )
    p.add_argument("--output_dir", default="detection/baselines/results/tdev_caption_rewrite_neutral")
    p.add_argument("--score", default="hybrid_positive_branch_absence")
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--hybrid_low", type=float, default=0.04)
    p.add_argument("--hybrid_high", type=float, default=0.12)
    p.add_argument("--hybrid_margin", type=float, default=-0.20)
    p.add_argument("--hybrid_mcc_margin", type=float, default=-0.30)
    p.add_argument("--neighbor_dominance_alpha", type=float, default=0.25)
    p.add_argument("--rewrite_policy", choices=("neutral_placeholder", "generic_noun"), default="neutral_placeholder")
    p.add_argument("--object_placeholder", default="something")
    p.add_argument("--person_placeholder", default="someone")
    p.add_argument("--run_chair", action="store_true", help="Rerun official PAS CHAIR on original and rewritten captions.")
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _is_plural_phrase(text: str) -> bool:
    tokens = [token.strip(".,;:!?()[]{}\"'").lower() for token in text.split()]
    tokens = [token for token in tokens if token]
    if not tokens:
        return False
    if any(token in PLURAL_HINTS for token in tokens):
        return True
    last = tokens[-1]
    return last.endswith("s") and not last.endswith("ss")


def choose_placeholder(target_word: str, matched_phrase: str, args: argparse.Namespace) -> str:
    tokens = set(target_word.lower().split()) | set(matched_phrase.lower().split())
    if tokens & PERSON_WORDS:
        return args.person_placeholder
    if args.rewrite_policy == "generic_noun":
        key = target_word.lower().strip()
        singular, plural = GENERIC_REPLACEMENTS.get(key, ("an item", "some items"))
        return plural if _is_plural_phrase(matched_phrase) else singular
    return args.object_placeholder


def preserve_case(replacement: str, matched_text: str) -> str:
    stripped = matched_text.lstrip()
    if stripped[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def drop_leading_determiner(replacement: str) -> str:
    lowered = replacement.lower()
    for prefix in ("an ", "a ", "some "):
        if lowered.startswith(prefix):
            return replacement[len(prefix):]
    return replacement


def starts_with_determiner(text: str) -> bool:
    lowered = text.lstrip().lower()
    return lowered.startswith(("a ", "an ", "the ", "some "))


def has_left_modifier(caption: str, start: int) -> bool:
    prefix = caption[:start].rstrip()
    if not prefix:
        return False
    if prefix[-1:] in ".!?;:(,":
        return False
    token = prefix.split()[-1].strip(".,;:!?\"'()[]{}")
    if not token:
        return False
    return token[-1:].isalnum() and token.lower() not in LEFT_MODIFIER_STOP_WORDS


def rewrite_one_phrase(caption: str, patterns: list[tuple[str, object]], target_word: str, args: argparse.Namespace, edit) -> tuple[str, str | None, str | None]:
    for phrase, pattern in patterns:
        match = pattern.search(caption)
        if not match:
            continue
        replacement = choose_placeholder(target_word, phrase, args)
        if (
            args.rewrite_policy == "generic_noun"
            and not starts_with_determiner(match.group(0))
            and has_left_modifier(caption, match.start())
        ):
            replacement = drop_leading_determiner(replacement)
        replacement = preserve_case(replacement, match.group(0))
        rewritten = caption[: match.start()] + replacement + caption[match.end() :]
        return edit.clean_caption(rewritten), phrase, replacement
    return caption, None, None


def main() -> None:
    args = parse_args()
    edit = load_edit_module()

    rows = edit.read_rows(Path(args.scores_csv))
    scores = edit.build_scores(rows, args)
    if args.score not in scores:
        raise KeyError(f"Unknown score {args.score!r}; available: {sorted(scores)}")

    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    values = scores[args.score]
    selected_mask = edit.top_fraction_mask(values, args.top_frac)

    synonyms = edit.load_synonyms(Path(args.chair_source))
    pattern_cache = {word: edit.compile_patterns(phrases) for word, phrases in synonyms.items()}

    captions: OrderedDict[int, str] = OrderedDict()
    for row in rows:
        image_id = int(row["image_id"])
        captions.setdefault(image_id, row["caption"])
    original_captions = dict(captions)

    rewritten_mask = np.zeros(len(rows), dtype=bool)
    rewrite_rows: list[dict[str, object]] = []
    selected_indices = np.flatnonzero(selected_mask)
    selected_indices = selected_indices[np.argsort(-values[selected_indices], kind="mergesort")]
    for idx in selected_indices:
        row = rows[int(idx)]
        image_id = int(row["image_id"])
        word = row["word"].strip().lower()
        patterns = pattern_cache.get(word, edit.compile_patterns([word]))
        rewritten, phrase, replacement = rewrite_one_phrase(captions[image_id], patterns, word, args, edit)
        changed = phrase is not None and rewritten != captions[image_id]
        if changed:
            captions[image_id] = rewritten
            rewritten_mask[int(idx)] = True
        rewrite_rows.append(
            {
                "object_id": int(row["object_id"]),
                "image_id": image_id,
                "word": row["word"],
                "label": int(row["label"]),
                "score": float(values[int(idx)]),
                "matched_phrase": phrase or "",
                "replacement": replacement or "",
                "rewritten": int(changed),
            }
        )

    mentions = int(labels.size)
    hallucinated = int(labels.sum())
    grounded = mentions - hallucinated
    selected = int(selected_mask.sum())
    selected_hallucinated = int(labels[selected_mask].sum())
    selected_grounded = selected - selected_hallucinated
    rewritten = int(rewritten_mask.sum())
    rewritten_hallucinated = int(labels[rewritten_mask].sum())
    rewritten_grounded = rewritten - rewritten_hallucinated
    keep_mask = ~rewritten_mask
    remaining = int(keep_mask.sum())
    remaining_hallucinated = int(labels[keep_mask].sum())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rewritten_captions = [
        {"image_id": image_id, "caption": caption}
        for image_id, caption in captions.items()
    ]
    rewritten_pairs = [
        {
            "image_id": image_id,
            "caption": captions[image_id],
            "original_caption": original_captions[image_id],
        }
        for image_id in captions
    ]
    (output_dir / "rewritten_captions.json").write_text(json.dumps(rewritten_captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "rewritten_caption_pairs.json").write_text(json.dumps(rewritten_pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "rewritten_mentions.csv", rewrite_rows)

    metrics = {
        "scores_csv": args.scores_csv,
        "score": args.score,
        "score_direction": "larger score means higher risk and is selected first",
        "top_frac": args.top_frac,
        "rewrite_policy": args.rewrite_policy,
        "object_placeholder": args.object_placeholder,
        "person_placeholder": args.person_placeholder,
        "mentions": mentions,
        "hallucinated_mentions": hallucinated,
        "grounded_mentions": grounded,
        "selected_mentions": selected,
        "selected_hallucinated": selected_hallucinated,
        "selected_grounded": selected_grounded,
        "rewritten_mentions": rewritten,
        "rewritten_hallucinated": rewritten_hallucinated,
        "rewritten_grounded": rewritten_grounded,
        "rewrite_success_rate": rewritten / selected if selected else 0.0,
        "rewrite_precision": rewritten_hallucinated / rewritten if rewritten else 0.0,
        "hallucinated_reduction_proxy": rewritten_hallucinated / hallucinated if hallucinated else 0.0,
        "grounded_rewrite_rate": rewritten_grounded / grounded if grounded else 0.0,
        "remaining_mentions_proxy": remaining,
        "remaining_hallucinated_proxy": remaining_hallucinated,
        "remaining_mention_hallucination_rate_proxy": remaining_hallucinated / remaining if remaining else 0.0,
        "remaining_image_hallucination_rate_proxy": edit.image_hallucination_rate(rows, keep_mask),
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
            "vanilla": edit.chair_summary(vanilla_data, args.chair_pkl, output_dir, "vanilla"),
            "rewritten": edit.chair_summary(rewritten_captions, args.chair_pkl, output_dir, "rewritten"),
        }
        official_chair["delta"] = edit.metric_delta(official_chair["rewritten"], official_chair["vanilla"])
        (output_dir / "chair_metrics.json").write_text(json.dumps(official_chair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metrics["chair_rerun_status"] = "run"
        metrics["chair_metrics_file"] = str(output_dir / "chair_metrics.json")

    (output_dir / "caption_rewrite_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'caption_rewrite_metrics.json'}")


if __name__ == "__main__":
    main()
