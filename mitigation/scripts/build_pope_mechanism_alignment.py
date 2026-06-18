#!/usr/bin/env python3
"""Build mechanism-alignment examples for the looking-is-not-grounding claim.

The output table is not a new benchmark. It selects already-audited POPE
semantic-neighbor failures where vanilla answers yes to an absent target while
related objects are present, then attaches TDEV target/neighbor evidence,
attention-intervention outputs, and LH-Shape scores. The purpose is to keep the
paper story tied to the original mechanism: plausible visual/associated evidence
is not target-object verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build POPE mechanism-alignment examples")
    p.add_argument("--audit_csv", default="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv")
    p.add_argument("--tdev_predictions_csv", default="mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_predictions.csv")
    p.add_argument("--lh_predictions_csv", default="mitigation/results/pope_lh_shape_transfer_full_imagecv/pope_lh_shape_transfer_predictions.csv")
    p.add_argument("--attention_root", default="mitigation/results/coco_llava_7b_attention_only/pope")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lh_score", default="lh_shape_pope_layers_22_31")
    p.add_argument("--examples_per_split", type=int, default=6)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def answer_from_text(text: str) -> str:
    first = re.sub(r"[^a-z]", "", text.strip().lower().split()[0] if text.strip() else "")
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return "invalid"


def read_jsonl_predictions(path: Path) -> dict[str, str]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            out[str(item["question_id"])] = answer_from_text(item.get("text", ""))
    return out


def load_attention_predictions(root: Path) -> dict[tuple[str, str, str], str]:
    methods = ["vanilla", "pai", "clearsight", "visattnsink"]
    splits = ["random", "popular", "adversarial"]
    output = {}
    for split in splits:
        for method in methods:
            path = root / split / method / "predictions.jsonl"
            if not path.exists():
                continue
            preds = read_jsonl_predictions(path)
            for qid, pred in preds.items():
                output[(split, qid, method)] = pred
    return output


def fmt(x: float) -> str:
    return f"{x:.3f}"


def markdown_table(rows: list[dict[str, str]]) -> str:
    cols = [
        "split", "question_id", "target", "related_present", "best_neighbor",
        "target_score", "best_neighbor_score", "tdev_margin", "vanilla",
        "pai", "clearsight", "visattnsink", "tdev", "lh_absent_score",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(c, "") for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = read_csv(Path(args.audit_csv))
    audit_by_key = {(r["split"], r["question_id"]): r for r in audit_rows}
    tdev_rows = read_csv(Path(args.tdev_predictions_csv))
    lh_rows = [r for r in read_csv(Path(args.lh_predictions_csv)) if r["score"] == args.lh_score]
    lh_by_key = {(r["split"], r["question_id"]): r for r in lh_rows}
    attention = load_attention_predictions(Path(args.attention_root))

    summary = Counter()
    by_split = defaultdict(list)
    all_related = []

    for row in tdev_rows:
        if row["negative_type"] != "negative_related_present":
            continue
        key = (row["split"], row["question_id"])
        audit = audit_by_key.get(key, {})
        lh = lh_by_key.get(key, {})
        target_score = float(row["target_score"])
        neighbor_score = float(row["best_neighbor_score"])
        margin = float(row["tdev_margin"])
        vanilla = row["base_prediction"]
        tdev = row["prediction"]
        item = {
            "split": row["split"],
            "question_id": row["question_id"],
            "image": row["image"],
            "target": row["target"],
            "question": audit.get("question", ""),
            "present_objects": audit.get("present_objects", ""),
            "related_present": audit.get("related_present", ""),
            "top_cooccurrence_neighbors": audit.get("top_cooccurrence_neighbors", row.get("neighbors", "")),
            "best_neighbor": row["best_neighbor"],
            "target_score": fmt(target_score),
            "best_neighbor_score": fmt(neighbor_score),
            "tdev_margin": fmt(margin),
            "vanilla": vanilla,
            "pai": attention.get((row["split"], row["question_id"], "pai"), "missing"),
            "clearsight": attention.get((row["split"], row["question_id"], "clearsight"), "missing"),
            "visattnsink": attention.get((row["split"], row["question_id"], "visattnsink"), "missing"),
            "tdev": tdev,
            "lh_absent_score": fmt(float(lh["absent_score"])) if lh else "missing",
            "lh_pred_absent": lh.get("pred_absent", "missing"),
            "neighbor_dominates": int(neighbor_score > target_score),
            "tdev_corrects_vanilla_fp": int(vanilla == "yes" and tdev == "no"),
            "attention_yes_count": sum(attention.get((row["split"], row["question_id"], m), "missing") == "yes" for m in ["pai", "clearsight", "visattnsink"]),
        }
        all_related.append(item)
        summary["related_present_negatives"] += 1
        summary[f"{row['split']}_related_present_negatives"] += 1
        if vanilla == "yes":
            summary["vanilla_related_fp"] += 1
            summary[f"{row['split']}_vanilla_related_fp"] += 1
        if vanilla == "yes" and tdev == "no":
            summary["tdev_corrected_vanilla_related_fp"] += 1
        if neighbor_score > target_score:
            summary["neighbor_score_gt_target_score"] += 1
        if vanilla == "yes" and neighbor_score > target_score:
            summary["vanilla_fp_with_neighbor_dominance"] += 1
        if vanilla == "yes" and tdev == "no" and neighbor_score > target_score:
            by_split[row["split"]].append(item)

    selected = []
    for split in ["random", "popular", "adversarial"]:
        candidates = sorted(
            by_split[split],
            key=lambda r: (
                int(r["attention_yes_count"]),
                float(r["best_neighbor_score"]),
                -float(r["target_score"]),
                -float(r["tdev_margin"]),
            ),
            reverse=True,
        )
        selected.extend(candidates[: args.examples_per_split])

    csv_path = output_dir / "pope_mechanism_alignment_examples.csv"
    fieldnames = [
        "split", "question_id", "image", "target", "question", "present_objects",
        "related_present", "top_cooccurrence_neighbors", "best_neighbor",
        "target_score", "best_neighbor_score", "tdev_margin", "vanilla", "pai",
        "clearsight", "visattnsink", "tdev", "lh_absent_score", "lh_pred_absent",
        "neighbor_dominates", "tdev_corrects_vanilla_fp", "attention_yes_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    denominator = max(summary["related_present_negatives"], 1)
    vanilla_fp = summary["vanilla_related_fp"]
    payload = {
        "inputs": {
            "audit_csv": args.audit_csv,
            "tdev_predictions_csv": args.tdev_predictions_csv,
            "lh_predictions_csv": args.lh_predictions_csv,
            "attention_root": args.attention_root,
            "lh_score": args.lh_score,
        },
        "summary_counts": dict(summary),
        "summary_rates": {
            "vanilla_related_fpr": vanilla_fp / denominator,
            "tdev_correction_rate_among_vanilla_related_fp": summary["tdev_corrected_vanilla_related_fp"] / max(vanilla_fp, 1),
            "neighbor_dominance_rate_all_related": summary["neighbor_score_gt_target_score"] / denominator,
            "neighbor_dominance_rate_vanilla_related_fp": summary["vanilla_fp_with_neighbor_dominance"] / max(vanilla_fp, 1),
        },
        "selection_rule": "negative_related_present rows where vanilla/base predicts yes, TDEV predicts no, and best_neighbor_score > target_score; ranked by attention-intervention yes count and neighbor evidence.",
        "examples_csv": str(csv_path),
        "examples": selected,
    }
    json_path = output_dir / "pope_mechanism_alignment_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    md_path = output_dir / "pope_mechanism_alignment_examples.md"
    md_path.write_text(
        "# POPE Mechanism Alignment Examples\n\n"
        "These are related-present negative POPE rows where vanilla answers `yes`, "
        "TDEV answers `no`, and the strongest semantic-neighbor evidence exceeds "
        "target evidence. They illustrate associated-evidence grounding: the image "
        "contains plausible related objects, but not the queried target.\n\n"
        f"Vanilla related-present FPR in this table source: {payload['summary_rates']['vanilla_related_fpr']:.3f}. "
        f"TDEV corrects {payload['summary_rates']['tdev_correction_rate_among_vanilla_related_fp']:.3f} of vanilla related-present false positives. "
        f"Among vanilla related-present false positives, neighbor evidence exceeds target evidence in {payload['summary_rates']['neighbor_dominance_rate_vanilla_related_fp']:.3f}.\n\n"
        + markdown_table(selected),
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(payload["summary_rates"], indent=2, sort_keys=True))
    print(markdown_table(selected[: min(8, len(selected))]))


if __name__ == "__main__":
    main()
