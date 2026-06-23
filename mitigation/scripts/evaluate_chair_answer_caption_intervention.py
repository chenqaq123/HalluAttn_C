#!/usr/bin/env python3
"""Caption intervention from internal CHAIR answer-absence scores.

This is a deterministic post-hoc caption intervention: select high-risk CHAIR
object mentions according to an internal score (default answer_absence_score),
remove or generic-rewrite the matched object phrase, then optionally rerun the
official CHAIR evaluator. No external detector is used.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_SRC = PROJECT_ROOT / "detection" / "src"
PAS_SRC = PROJECT_ROOT.parent / "pas" / "src"
for import_path in (DETECTION_SRC, PAS_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

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
    "motorcycle": ("a vehicle", "some vehicles"),
    "boat": ("a vehicle", "some vehicles"),
    "train": ("a vehicle", "some vehicles"),
    "skis": ("some equipment", "some equipment"),
    "sports ball": ("an item", "some items"),
    "baseball bat": ("some equipment", "some equipment"),
    "baseball glove": ("some equipment", "some equipment"),
}
PERSON_WORDS = {"person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children"}
PLURAL_HINTS = {"people", "men", "women", "children", "boys", "girls", "skis", "scissors"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate internal-score CHAIR caption intervention")
    p.add_argument("--scores_csv", default="mitigation/results/semantic_neighbor_audit/chair_internal_verifier_full/chair_internal_verifier_scores.csv")
    p.add_argument("--score_field", default="answer_absence_score")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--top_frac", type=float, default=0.05)
    p.add_argument("--mode", choices=("delete", "generic_noun", "neutral_placeholder"), default="delete")
    p.add_argument("--chair_source", default="../pas/src/pas/evaluate/chair.py")
    p.add_argument("--chair_pkl", default="../pas/data/chair_coco.pkl")
    p.add_argument("--run_chair", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


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
        synonyms[canonical] = sorted(list(OrderedDict.fromkeys(items + [canonical])), key=lambda s: (-len(s), s))
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
    return list(OrderedDict.fromkeys([phrase, pluralize_last_token(phrase)]))


def compile_patterns(phrases: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    expanded: list[str] = []
    for phrase in phrases:
        expanded.extend(phrase_variants(phrase))
    out = []
    for phrase in sorted(OrderedDict.fromkeys(expanded), key=lambda item: (-len(item), item)):
        escaped = re.escape(phrase).replace(r"\ ", r"\s+")
        out.append((phrase, re.compile(r"(?<!\w)(?:(?:a|an|the)\s+)?" + escaped + r"(?!\w)", flags=re.I)))
    return out


def clean_caption(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\b(a|an|the)\s+([,.;:!?])", r"\2", text, flags=re.I)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def is_plural(text: str) -> bool:
    toks = [t.strip(".,;:!?()[]{}\"'").lower() for t in text.split() if t.strip(".,;:!?()[]{}\"'")]
    if not toks:
        return False
    if any(t in PLURAL_HINTS for t in toks):
        return True
    return toks[-1].endswith("s") and not toks[-1].endswith("ss")


def replacement_for(word: str, matched_text: str, mode: str) -> str:
    toks = set(word.lower().split()) | set(matched_text.lower().split())
    if mode == "neutral_placeholder":
        return "someone" if toks & PERSON_WORDS else "something"
    if toks & PERSON_WORDS:
        repl = ("a person", "people")
    else:
        repl = GENERIC_REPLACEMENTS.get(word.lower(), ("an item", "some items"))
    return repl[1] if is_plural(matched_text) else repl[0]


def preserve_case(replacement: str, matched_text: str) -> str:
    stripped = matched_text.lstrip()
    if stripped[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def edit_one(caption: str, patterns: list[tuple[str, re.Pattern[str]]], word: str, mode: str) -> tuple[str, str | None, str | None]:
    for phrase, pattern in patterns:
        match = pattern.search(caption)
        if not match:
            continue
        matched = caption[match.start():match.end()]
        if mode == "delete":
            edited = caption[:match.start()] + caption[match.end():]
            replacement = ""
        else:
            replacement = preserve_case(replacement_for(word, matched, mode), matched)
            edited = caption[:match.start()] + replacement + caption[match.end():]
        return clean_caption(edited), phrase, replacement
    return caption, None, None


def top_fraction_mask(values: np.ndarray, frac: float) -> np.ndarray:
    if not 0.0 < frac < 1.0:
        raise ValueError(f"top_frac must be in (0,1): {frac}")
    n = max(1, int(round(values.size * frac)))
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:n]] = True
    return mask


def caption_stats(data: list[dict[str, object]]) -> dict[str, float | int]:
    lengths = [len(str(row["caption"]).split()) for row in data]
    return {"num_captions": len(data), "mean_words": sum(lengths) / len(lengths) if lengths else 0.0}


def chair_summary(data: list[dict[str, object]], chair_pkl: str, output_dir: Path, prefix: str) -> dict:
    from sinkdetect.chair import evaluate_chair, load_chair_evaluator
    evaluator = load_chair_evaluator(chair_pkl)
    per_sample, overall = evaluate_chair(evaluator, data=data, json_path=str(output_dir / f"{prefix}_chair_input.json"))
    mention_counts, hall_counts = [], []
    with (output_dir / f"{prefix}_chair_per_sample.jsonl").open("w", encoding="utf-8") as f:
        for sample in per_sample:
            generated = list(sample.get("mscoco_generated_words", []))
            grounded = set(sample.get("mscoco_gt_words", []))
            hallucinated = sum(word not in grounded for word in generated)
            mention_counts.append(len(generated))
            hall_counts.append(hallucinated)
            f.write(json.dumps({
                "image_id": sample.get("image_id"),
                "caption": sample.get("caption", ""),
                "object_mentions": len(generated),
                "hallucinated_mentions": hallucinated,
            }, ensure_ascii=False) + "\n")
    stats = caption_stats(data)
    stats.update({
        "mean_object_mentions": sum(mention_counts) / len(mention_counts) if mention_counts else 0.0,
        "mean_hallucinated_mentions": sum(hall_counts) / len(hall_counts) if hall_counts else 0.0,
        "total_object_mentions": sum(mention_counts),
        "total_hallucinated_mentions": sum(hall_counts),
    })
    return {"chair": overall, "caption_stats": stats}


def metric_delta(new: dict, base: dict) -> dict[str, float]:
    out = {}
    for section in ("chair", "caption_stats"):
        for key, value in new[section].items():
            b = base[section].get(key)
            if isinstance(value, (int, float)) and isinstance(b, (int, float)):
                out[f"delta_{section}_{key}"] = value - b
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.scores_csv))
    if args.score_field not in rows[0]:
        raise KeyError(f"{args.score_field} not found in {args.scores_csv}")
    values = np.asarray([float(r[args.score_field]) for r in rows], dtype=np.float64)
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    selected_mask = top_fraction_mask(values, args.top_frac)
    synonyms = load_synonyms(Path(args.chair_source))
    pattern_cache = {word: compile_patterns(phrases) for word, phrases in synonyms.items()}

    captions: OrderedDict[int, str] = OrderedDict()
    for row in rows:
        captions.setdefault(int(row["image_id"]), row["caption"])
    original = dict(captions)
    edited_mask = np.zeros(len(rows), dtype=bool)
    edit_rows = []
    selected_indices = np.flatnonzero(selected_mask)
    selected_indices = selected_indices[np.argsort(-values[selected_indices], kind="mergesort")]
    for idx in selected_indices:
        row = rows[int(idx)]
        image_id = int(row["image_id"])
        word = row["word"].strip().lower()
        patterns = pattern_cache.get(word, compile_patterns([word]))
        edited, phrase, replacement = edit_one(captions[image_id], patterns, word, args.mode)
        changed = phrase is not None and edited != captions[image_id]
        if changed:
            captions[image_id] = edited
            edited_mask[int(idx)] = True
        edit_rows.append({
            "object_id": int(row["object_id"]),
            "image_id": image_id,
            "word": word,
            "label": int(row["label"]),
            "score": float(values[int(idx)]),
            "matched_phrase": phrase or "",
            "replacement": replacement or "",
            "changed": int(changed),
        })

    intervention_data = [{"image_id": iid, "caption": cap} for iid, cap in captions.items()]
    vanilla_data = [{"image_id": iid, "caption": original[iid]} for iid in captions]
    with (output_dir / "intervention_captions.json").open("w", encoding="utf-8") as f:
        json.dump(intervention_data, f, indent=2, ensure_ascii=False); f.write("\n")
    with (output_dir / "caption_pairs.json").open("w", encoding="utf-8") as f:
        json.dump([{"image_id": iid, "original_caption": original[iid], "caption": captions[iid]} for iid in captions], f, indent=2, ensure_ascii=False); f.write("\n")
    write_csv(output_dir / "intervened_mentions.csv", edit_rows)

    changed = int(edited_mask.sum())
    changed_hall = int(labels[edited_mask].sum())
    metrics = {
        "scores_csv": args.scores_csv,
        "score_field": args.score_field,
        "mode": args.mode,
        "top_frac": args.top_frac,
        "mentions": int(labels.size),
        "hallucinated_mentions": int(labels.sum()),
        "selected_mentions": int(selected_mask.sum()),
        "selected_hallucinated": int(labels[selected_mask].sum()),
        "selected_precision": float(labels[selected_mask].mean()) if selected_mask.any() else 0.0,
        "changed_mentions": changed,
        "changed_hallucinated": changed_hall,
        "changed_grounded": changed - changed_hall,
        "change_success_rate": changed / int(selected_mask.sum()) if selected_mask.any() else 0.0,
        "change_precision": changed_hall / changed if changed else 0.0,
        "hallucinated_claim_change_rate": changed_hall / int(labels.sum()) if labels.sum() else 0.0,
        "grounded_claim_change_rate": (changed - changed_hall) / int((labels == 0).sum()) if (labels == 0).sum() else 0.0,
        "chair_rerun_status": "not_run_use_--run_chair",
    }
    if args.run_chair:
        chair = {
            "chair_pkl": args.chair_pkl,
            "sample_scope": "images with CHAIR object mentions in internal score CSV",
            "vanilla": chair_summary(vanilla_data, args.chair_pkl, output_dir, "vanilla"),
            "intervention": chair_summary(intervention_data, args.chair_pkl, output_dir, "intervention"),
        }
        chair["delta"] = metric_delta(chair["intervention"], chair["vanilla"])
        with (output_dir / "chair_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(chair, f, indent=2, sort_keys=True); f.write("\n")
        metrics["chair_rerun_status"] = "run"
        metrics["chair_metrics_file"] = str(output_dir / "chair_metrics.json")
    with (output_dir / "caption_intervention_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
