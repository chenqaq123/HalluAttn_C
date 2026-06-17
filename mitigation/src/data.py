"""Dataset readers shared by mitigation POPE and CHAIR runs."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator

from PIL import Image


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pope_jsonl_path(question_dir: Path, split: str) -> Path:
    candidates = [
        question_dir / "coco" / f"coco_pope_{split}.jsonl",
        question_dir / f"coco_pope_{split}.jsonl",
        question_dir / f"coco_pope_{split}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find COCO POPE {split!r} questions under {question_dir}")


def iter_pope_records(
    question_dir: str | Path,
    coco_path: str | Path,
    split: str,
    shard_idx: int = 0,
    num_shards: int = 1,
    limit: int = 0,
) -> Iterator[dict]:
    """Yield normalized COCO POPE records from JSONL or parquet layouts."""
    question_dir = Path(question_dir)
    coco_path = Path(coco_path)
    parquet_path = question_dir / "Full" / f"{split}-00000-of-00001.parquet"

    if parquet_path.exists():
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyarrow is required for parquet-format POPE data") from exc
        rows = pq.read_table(
            parquet_path,
            columns=["question_id", "question", "answer", "image_source", "image"],
        ).to_pylist()
        normalized = [
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "label": str(row["answer"]).lower(),
                "image": row.get("image_source", ""),
                "image_bytes": row["image"]["bytes"],
            }
            for row in rows
        ]
    else:
        rows = _read_jsonl(_pope_jsonl_path(question_dir, split))
        normalized = [
            {
                "question_id": row.get("question_id", row.get("qid")),
                "question": row.get("text", row.get("question", row.get("prompt"))),
                "label": str(row.get("label", row.get("answer"))).lower(),
                "image": row["image"],
                "image_path": str(coco_path / "val2014" / row["image"]),
            }
            for row in rows
        ]

    if limit > 0:
        normalized = normalized[:limit]
    selected = normalized[shard_idx::num_shards]
    yield from selected


def load_pope_image(record: dict) -> Image.Image:
    if "image_bytes" in record:
        return Image.open(io.BytesIO(record["image_bytes"])).convert("RGB")
    return Image.open(record["image_path"]).convert("RGB")


def load_chair_manifest(path: str | Path, shard_idx: int = 0, num_shards: int = 1, limit: int = 0) -> list[dict]:
    """Load the fixed COCO image set used by the existing detection experiment."""
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if limit > 0:
        rows = rows[:limit]
    rows = rows[shard_idx::num_shards]
    return [{"image_id": int(row["image_id"])} for row in rows]
