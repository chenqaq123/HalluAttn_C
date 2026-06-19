#!/usr/bin/env python3
"""Convert AIR official answers.jsonl to this repo's POPE prediction schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AIR answers.jsonl for semantic-neighbor evaluation")
    parser.add_argument("--answers_file", required=True, help="AIR official answers.jsonl")
    parser.add_argument("--questions_file", required=True, help="Question JSONL exported by export_air_pope_subset.py")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--method", default="air")
    parser.add_argument("--strict", action="store_true", help="Fail on unknown, duplicate, or missing question ids")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sortable_question_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def main() -> None:
    args = parse_args()
    answers = read_jsonl(Path(args.answers_file))
    questions = {str(row["question_id"]): row for row in read_jsonl(Path(args.questions_file))}
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    answer_ids = [str(answer["question_id"]) for answer in answers]
    duplicate_answer_ids = sorted(
        [qid for qid, count in Counter(answer_ids).items() if count > 1],
        key=sortable_question_id,
    )
    missing_answer_ids = sorted(set(questions) - set(answer_ids), key=sortable_question_id)

    unknown = []
    converted = []
    for answer in answers:
        question_id = str(answer["question_id"])
        question = questions.get(question_id)
        if question is None:
            unknown.append(question_id)
            if args.strict:
                continue
            question = {}
        converted.append(
            {
                "question_id": question_id,
                "image": question.get("image", ""),
                "prompt": answer.get("prompt", question.get("text", "")),
                "text": answer.get("text", ""),
                "label": question.get("label", ""),
                "split": question.get("split", ""),
                "target": question.get("target", ""),
                "negative_type": question.get("negative_type", ""),
                "method": args.method,
                "answer_id": answer.get("answer_id", ""),
                "model_id": answer.get("model_id", ""),
            }
        )

    if args.strict:
        errors = []
        if unknown:
            errors.append(f"{len(unknown)} AIR answers missing question metadata, first={unknown[:5]}")
        if duplicate_answer_ids:
            errors.append(f"{len(duplicate_answer_ids)} duplicate AIR answer ids, first={duplicate_answer_ids[:5]}")
        if missing_answer_ids:
            errors.append(f"{len(missing_answer_ids)} questions missing AIR answers, first={missing_answer_ids[:5]}")
        if errors:
            raise ValueError("; ".join(errors))

    with output_file.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Read {len(answers)} AIR answers")
    print(f"Wrote {len(converted)} predictions to {output_file}")
    if unknown:
        print(f"Warning: {len(unknown)} answers missing question metadata, first={unknown[:5]}")
    if duplicate_answer_ids:
        print(f"Warning: {len(duplicate_answer_ids)} duplicate AIR answer ids, first={duplicate_answer_ids[:5]}")
    if missing_answer_ids:
        print(f"Warning: {len(missing_answer_ids)} questions missing AIR answers, first={missing_answer_ids[:5]}")


if __name__ == "__main__":
    main()
