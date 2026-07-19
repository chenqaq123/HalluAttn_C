#!/usr/bin/env python3
"""Audit target-grounding proposals with target-free regional re-identification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mitigation.src.data import iter_pope_records, load_pope_image  # noqa: E402
from mitigation.src.xev import (  # noqa: E402
    inventory_lexically_supports_target,
    lexical_equivalent,
    make_counterfactual_views,
    parse_candidate_list,
    parse_grounding_boxes,
    robust_local_identity_proposal,
    select_pairwise_candidates,
)
from run_qwen3vl_xev_pope import (  # noqa: E402
    DEFAULT_COCO,
    DEFAULT_MODEL,
    DEFAULT_POPE,
    Qwen3VLEngine,
    RelationFilter,
    build_coco_knowledge,
    open_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Iteration-11 JSONL files")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--selection", choices=("actions", "conflicts"), default="actions")
    parser.add_argument(
        "--box_inputs",
        nargs="+",
        help="Saved grounded-cycle JSONL files whose pixel boxes are reused",
    )
    parser.add_argument(
        "--view_mode",
        choices=("clear_canvas", "crop"),
        default="clear_canvas",
    )
    parser.add_argument(
        "--conflict_relations",
        choices=("CN", "C"),
        default="CN",
        help="Relations eligible for pairwise identity conflict",
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--pope_dir", default=DEFAULT_POPE)
    parser.add_argument("--coco_root", default=DEFAULT_COCO)
    parser.add_argument("--pope_split", default="adversarial")
    parser.add_argument(
        "--knowledge_cache",
        default=str(PROJECT_ROOT / "experiments/qwen3vl_xev/coco_knowledge.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--attention_layers", default="18:30")
    parser.add_argument("--attention_map", default="no_rope")
    parser.add_argument("--max_grounding_boxes", type=int, default=3)
    parser.add_argument("--grounding_max_new_tokens", type=int, default=96)
    parser.add_argument("--max_open_candidates", type=int, default=6)
    parser.add_argument("--candidate_max_new_tokens", type=int, default=32)
    parser.add_argument("--closed_k", type=int, default=8)
    parser.add_argument("--pairwise_top_k", type=int, default=3)
    parser.add_argument("--margin_batch_size", type=int, default=4)
    parser.add_argument("--relation_max_new_tokens", type=int, default=32)
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path_text in paths:
        with Path(path_text).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    by_id = {str(row["question_id"]): row for row in rows}
    return sorted(by_id.values(), key=lambda row: int(row["question_id"]))


def has_local_conflict(row: dict) -> bool:
    return bool(
        row.get("global_support_verification", {}).get(
            "local_identity_proposals", []
        )
    )


def is_selected(row: dict, selection: str) -> bool:
    if selection == "actions":
        return row.get("global_support_verification", {}).get("certificate") is not None
    return has_local_conflict(row)


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        return {str(json.loads(line)["question_id"]) for line in handle if line.strip()}


def grounding_prompt(target: str) -> str:
    return (
        "Locate every instance that belongs to the following category: "
        f'"{target}". Report bbox coordinates in JSON format.'
    )


def crop_candidates(
    engine: Qwen3VLEngine,
    image,
    limit: int,
    max_new_tokens: int,
) -> tuple[list[str], str]:
    prompt = (
        "Without knowing any question target, list up to "
        f"{limit} concrete object identities visibly supported in this image crop.\n\n"
        "Use singular noun phrases at the most specific recognizable level. Do "
        "not list attributes, parts, scene types, broader classes, or explanations. "
        "Return a comma-separated list only."
    )
    raw = engine.generate(prompt, image=image, max_new_tokens=max_new_tokens)
    return parse_candidate_list(raw, limit=limit), raw


def saved_boxes(row: dict, image_size: tuple[int, int], limit: int) -> list[tuple[int, int, int, int]]:
    width, height = image_size
    boxes = []
    for values in row.get("grounding_boxes", []):
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"Invalid saved grounding box: {values!r}")
        box = tuple(int(value) for value in values)
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Saved grounding box outside image: {box!r}")
        boxes.append(box)
        if len(boxes) >= limit:
            break
    return boxes


def process_row(
    source: dict,
    record: dict,
    engine: Qwen3VLEngine,
    relation_filter: RelationFilter,
    knowledge: dict,
    args: argparse.Namespace,
    box_source: dict | None = None,
) -> dict:
    started = time.perf_counter()
    target = source["target"]
    image = load_pope_image(record)
    if box_source is None:
        raw = engine.generate(
            grounding_prompt(target),
            image=image,
            max_new_tokens=args.grounding_max_new_tokens,
        )
        boxes = parse_grounding_boxes(
            raw,
            image.size,
            limit=args.max_grounding_boxes,
        )
        box_source_name = "generated"
    else:
        raw = box_source.get("grounding_raw", "")
        boxes = saved_boxes(box_source, image.size, args.max_grounding_boxes)
        box_source_name = "saved"
    closed_candidates = [
        row["name"]
        for row in knowledge["neighbors"].get(target, [])[: args.closed_k]
        if not lexical_equivalent(row["name"], target)
    ]

    region_rows = []
    for index, bbox in enumerate(boxes):
        drop = None
        if args.view_mode == "crop":
            keep = image.crop(bbox).convert("RGB")
            generated_candidates, candidate_raw = crop_candidates(
                engine,
                keep,
                limit=args.max_open_candidates,
                max_new_tokens=args.candidate_max_new_tokens,
            )
        else:
            keep, drop = make_counterfactual_views(image, bbox)
            generated_candidates, candidate_raw = open_candidates(
                engine,
                keep,
                limit=args.max_open_candidates,
                max_new_tokens=args.candidate_max_new_tokens,
            )
        candidate_sources = {
            "closed": closed_candidates,
            "open": generated_candidates,
        }
        all_candidates = list(
            dict.fromkeys(
                candidate
                for candidates in candidate_sources.values()
                for candidate in candidates
                if not lexical_equivalent(candidate, target)
            )
        )
        relations, relation_metadata = relation_filter.classify(target, all_candidates)
        hypotheses = [target, *all_candidates]
        keep_margins = engine.margins(
            [keep] * len(hypotheses),
            hypotheses,
            batch_size=args.margin_batch_size,
            questions=[source["question"], *([None] * len(all_candidates))],
        )
        keep_margin_map = dict(zip(hypotheses, keep_margins))
        allowed_relations = ("C",) if args.conflict_relations == "C" else ("C", "N")
        pairwise_sources = {
            source_name: [
                candidate
                for candidate in candidates
                if relations.get(candidate, "N") in allowed_relations
            ]
            for source_name, candidates in candidate_sources.items()
        }
        pairwise_candidates = select_pairwise_candidates(
            target=target,
            candidate_sources=pairwise_sources,
            relations=relations,
            keep_margins=keep_margin_map,
            limit=args.pairwise_top_k,
            block_on_open_target=False,
        )
        comparisons = engine.pairwise_comparisons(
            keep,
            target,
            pairwise_candidates,
            batch_size=args.margin_batch_size,
        )
        conflict = robust_local_identity_proposal(
            comparisons,
            required_sources=(),
            require_directional_switch=True,
            allowed_relations=allowed_relations,
        )
        target_named = inventory_lexically_supports_target(
            target, generated_candidates
        )
        target_margin = float(keep_margin_map[target])
        cycle_support = target_named and target_margin > 0 and conflict is None
        region_rows.append(
            {
                "index": index,
                "bbox": list(bbox),
                "view_mode": args.view_mode,
                "view_size": list(keep.size),
                "target_free_candidates": generated_candidates,
                "target_free_raw": candidate_raw,
                "target_named": target_named,
                "target_margin": target_margin,
                "relations": relations,
                "relation_filter": relation_metadata,
                "candidate_keep_margins": {
                    candidate: float(keep_margin_map[candidate])
                    for candidate in all_candidates
                },
                "pairwise_comparisons": comparisons,
                "directional_conflict": conflict,
                "cycle_support": cycle_support,
            }
        )
        keep.close()
        if drop is not None:
            drop.close()

    image.close()
    cycle_support = any(region["cycle_support"] for region in region_rows)
    local_conflict = has_local_conflict(source)
    return {
        "schema_version": 1,
        "question_id": str(source["question_id"]),
        "image_id": source["image_id"],
        "target": target,
        "label": source["label"],
        "baseline_margin": source["baseline_margin"],
        "local_conflict": local_conflict,
        "grounding_prompt": grounding_prompt(target),
        "grounding_raw": raw,
        "grounding_boxes": [list(box) for box in boxes],
        "box_source": box_source_name,
        "regions": region_rows,
        "cycle_support_found": cycle_support,
        "action": bool(source["baseline_margin"] >= 0 and local_conflict and not cycle_support),
        "runtime_seconds": time.perf_counter() - started,
        "run": {
            "selection": args.selection,
            "max_grounding_boxes": args.max_grounding_boxes,
            "max_open_candidates": args.max_open_candidates,
            "pairwise_top_k": args.pairwise_top_k,
            "view_mode": args.view_mode,
            "conflict_relations": args.conflict_relations,
            "model_path": args.model_path,
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_grounding_boxes <= 0:
        raise ValueError("--max_grounding_boxes must be positive")
    source_rows = [row for row in read_rows(args.inputs) if is_selected(row, args.selection)]
    box_rows = read_rows(args.box_inputs) if args.box_inputs else []
    box_by_id = {str(row["question_id"]): row for row in box_rows}
    if args.box_inputs:
        missing_boxes = [
            str(row["question_id"])
            for row in source_rows
            if str(row["question_id"]) not in box_by_id
        ]
        if missing_boxes:
            raise ValueError(f"Missing saved grounding rows: {missing_boxes}")
    records = {
        str(row["question_id"]): row
        for row in iter_pope_records(
            args.pope_dir,
            args.coco_root,
            args.pope_split,
            limit=0,
        )
    }
    missing = [row["question_id"] for row in source_rows if str(row["question_id"]) not in records]
    if missing:
        raise ValueError(f"Missing POPE records: {missing}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(output_path)
    knowledge = build_coco_knowledge(
        Path(args.coco_root),
        Path(args.knowledge_cache),
    )
    engine = Qwen3VLEngine(args)
    relation_filter = RelationFilter(
        engine,
        max_new_tokens=args.relation_max_new_tokens,
    )

    with output_path.open("a", encoding="utf-8") as output:
        iterator = tqdm(source_rows, desc=f"grounded-cycle:{args.selection}")
        for source in iterator:
            question_id = str(source["question_id"])
            if question_id in completed:
                continue
            payload = process_row(
                source,
                records[question_id],
                engine,
                relation_filter,
                knowledge,
                args,
                box_source=box_by_id.get(question_id),
            )
            output.write(json.dumps(payload) + "\n")
            output.flush()
            iterator.set_postfix(seconds=f"{payload['runtime_seconds']:.1f}")


if __name__ == "__main__":
    main()
