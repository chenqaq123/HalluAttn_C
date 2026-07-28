#!/usr/bin/env python3
"""Audit and, when safe, normalize registered LLaVA-NeXT feature shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.scripts.extract_llava_next_image_features import FEATURE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object_cache",
        default=(
            "detection/baselines/results/"
            "coco_llava_7b_baselines/object_cache.jsonl"
        ),
    )
    parser.add_argument(
        "--exclude_manifest",
        default=(
            "/tmp/sinkdetect_datasets/"
            "open_caption_manifests/chair.jsonl"
        ),
    )
    parser.add_argument(
        "--features_dir",
        default="experiments/registered_caption/llava_next",
    )
    parser.add_argument("--source_images", type=int, default=3000)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--normalize_duplicates", action="store_true")
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalize_rows(
    path: Path,
    rows: list[dict[str, Any]],
    expected_ids: list[int],
) -> tuple[list[dict[str, Any]], Path | None]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(row)
    duplicate_ids = sorted(
        image_id for image_id, records in grouped.items()
        if len(records) > 1
    )
    if not duplicate_ids:
        return rows, None

    divergent = [
        image_id
        for image_id in duplicate_ids
        if len(
            {
                _canonical_record(record)
                for record in grouped[image_id]
            }
        )
        != 1
    ]
    if divergent:
        raise RuntimeError(
            "Duplicate JSONL records are not identical for image IDs "
            f"{divergent[:10]}"
        )
    expected_set = set(expected_ids)
    actual_set = set(grouped)
    if actual_set != expected_set:
        raise RuntimeError(
            "Refusing normalization before the shard is complete: "
            f"missing={sorted(expected_set - actual_set)[:10]}, "
            f"unexpected={sorted(actual_set - expected_set)[:10]}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(
        path.name + f".duplicate_checkpoint_{timestamp}"
    )
    if archive.exists():
        raise FileExistsError(archive)
    shutil.copy2(path, archive)
    normalized = [grouped[image_id][0] for image_id in expected_ids]
    _atomic_jsonl(path, normalized)
    return normalized, archive


def audit_shard(
    jsonl_path: Path,
    npz_path: Path,
    expected_ids: list[int],
    excluded_ids: set[int],
    *,
    normalize_duplicates: bool = False,
) -> dict[str, Any]:
    if not jsonl_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            f"Both shard files are required: {jsonl_path}, {npz_path}"
        )
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    original_rows = len(rows)
    ids = [int(row["image_id"]) for row in rows]
    counts = Counter(ids)
    duplicate_ids = sorted(
        image_id for image_id, count in counts.items() if count > 1
    )
    archive = None
    if duplicate_ids:
        if not normalize_duplicates:
            raise RuntimeError(
                f"Duplicate JSONL image IDs {duplicate_ids[:10]}"
            )
        rows, archive = _normalize_rows(
            jsonl_path, rows, expected_ids
        )
        ids = [int(row["image_id"]) for row in rows]

    expected_set = set(expected_ids)
    actual_set = set(ids)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected or len(ids) != len(expected_ids):
        raise RuntimeError(
            "JSONL image set does not match the registered shard: "
            f"rows={len(ids)}, expected={len(expected_ids)}, "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    if ids != expected_ids:
        raise RuntimeError("JSONL image order differs from registered order")
    leaked_json = sorted(actual_set & excluded_ids)
    if leaked_json:
        raise RuntimeError(
            f"Excluded evaluation images occur in JSONL: {leaked_json[:10]}"
        )

    with np.load(npz_path, allow_pickle=False) as cached:
        missing_keys = [
            key for key in FEATURE_KEYS if key not in cached.files
        ]
        if missing_keys:
            raise RuntimeError(f"NPZ lacks feature keys {missing_keys}")
        arrays = {
            key: np.asarray(cached[key])
            for key in FEATURE_KEYS
        }
        feature_names = (
            np.asarray(cached["feature_names"]).tolist()
            if "feature_names" in cached.files
            else None
        )
    lengths = {
        key: int(value.shape[0])
        for key, value in arrays.items()
    }
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"NPZ field row counts differ: {lengths}")
    mention_count = next(iter(lengths.values()))
    if arrays["no_rope"].ndim != 5:
        raise RuntimeError(
            f"no_rope must be rank 5, got {arrays['no_rope'].shape}"
        )
    for key in ("no_rope", "first_logp"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"NPZ field {key} contains non-finite values")

    object_ids = arrays["object_id"].astype(np.int64, copy=False)
    if np.unique(object_ids).size != mention_count:
        raise RuntimeError("NPZ object IDs are not unique")
    feature_image_ids = arrays["image_id"].astype(
        np.int64, copy=False
    )
    feature_image_set = set(map(int, feature_image_ids))
    extra_feature_images = sorted(feature_image_set - actual_set)
    leaked_features = sorted(feature_image_set & excluded_ids)
    if extra_feature_images or leaked_features:
        raise RuntimeError(
            "NPZ contains invalid image IDs: "
            f"unexpected={extra_feature_images[:10]}, "
            f"excluded={leaked_features[:10]}"
        )

    logged_counts = {
        int(row["image_id"]): len(row.get("mentions", []))
        for row in rows
    }
    feature_counts = Counter(map(int, feature_image_ids))
    mismatched_counts = [
        image_id
        for image_id in expected_ids
        if logged_counts[image_id] != feature_counts.get(image_id, 0)
    ]
    if mismatched_counts:
        raise RuntimeError(
            "JSONL/NPZ per-image mention counts differ for image IDs "
            f"{mismatched_counts[:10]}"
        )

    row_by_id = {int(row["image_id"]): row for row in rows}
    for index in range(mention_count):
        image_id = int(feature_image_ids[index])
        object_id = int(object_ids[index])
        mention_index = object_id - image_id * 128
        mentions = row_by_id[image_id].get("mentions", [])
        if not 0 <= mention_index < len(mentions):
            raise RuntimeError(
                f"Invalid object ID {object_id} for image {image_id}"
            )
        mention = mentions[mention_index]
        expected_values = {
            "word": str(mention["word"])[:63],
            "gen_pos": int(mention["rel_anchor"]),
            "label": int(bool(mention["hallucinated"])),
        }
        actual_values = {
            "word": str(arrays["word"][index]),
            "gen_pos": int(arrays["gen_pos"][index]),
            "label": int(arrays["label"][index]),
        }
        if actual_values != expected_values:
            raise RuntimeError(
                f"JSONL/NPZ mention mismatch at object ID {object_id}: "
                f"expected={expected_values}, actual={actual_values}"
            )

    return {
        "valid": True,
        "expected_images": len(expected_ids),
        "jsonl_rows_before_normalization": original_rows,
        "jsonl_rows": len(rows),
        "duplicate_extra_rows_removed": original_rows - len(rows),
        "duplicate_image_ids": duplicate_ids,
        "archive": str(archive) if archive else None,
        "mentions": mention_count,
        "grounded_mentions": int(
            np.sum(arrays["label"].astype(np.int32) == 0)
        ),
        "hallucinated_mentions": int(
            np.sum(arrays["label"].astype(np.int32) == 1)
        ),
        "images_with_mentions": len(feature_image_set),
        "no_rope_shape": list(arrays["no_rope"].shape),
        "feature_names": feature_names,
        "excluded_eval_overlap": [],
        "jsonl_sha256": _sha256(jsonl_path),
        "npz_sha256": _sha256(npz_path),
        "archive_sha256": _sha256(archive) if archive else None,
    }


def _registered_source_ids(
    object_cache: Path,
    exclude_manifest: Path,
    source_images: int,
) -> tuple[list[int], set[int]]:
    source = sorted(
        {
            int(json.loads(line)["image_id"])
            for line in object_cache.read_text().splitlines()
            if line.strip()
        }
    )[:source_images]
    excluded = {
        int(json.loads(line)["sample_id"])
        for line in exclude_manifest.read_text().splitlines()
        if line.strip()
    }
    return [
        image_id for image_id in source if image_id not in excluded
    ], excluded


def main() -> None:
    args = parse_args()
    source_ids, excluded_ids = _registered_source_ids(
        Path(args.object_cache),
        Path(args.exclude_manifest),
        args.source_images,
    )
    features_dir = Path(args.features_dir)
    shards = []
    all_images: set[int] = set()
    all_objects: set[int] = set()
    for shard_index in range(args.num_shards):
        expected = source_ids[shard_index :: args.num_shards]
        prefix = features_dir / f"features_shard{shard_index}"
        record = audit_shard(
            Path(str(prefix) + ".jsonl"),
            Path(str(prefix) + ".npz"),
            expected,
            excluded_ids,
            normalize_duplicates=args.normalize_duplicates,
        )
        with np.load(
            Path(str(prefix) + ".npz"), allow_pickle=False
        ) as cached:
            shard_images = set(map(int, cached["image_id"]))
            shard_objects = set(map(int, cached["object_id"]))
        if all_images & set(expected):
            raise RuntimeError(
                f"Image overlap involving shard {shard_index}"
            )
        if all_objects & shard_objects:
            raise RuntimeError(
                f"Object-ID overlap involving shard {shard_index}"
            )
        all_images.update(expected)
        all_objects.update(shard_objects)
        record["shard_index"] = shard_index
        record["npz_image_ids_subset_of_registered_shard"] = (
            shard_images <= set(expected)
        )
        shards.append(record)

    if all_images != set(source_ids):
        raise RuntimeError("Audited shards do not cover the source set")
    report = {
        "protocol": (
            "docs/iteration_protocols/"
            "image_caption_benchmark_matrix.md"
        ),
        "valid": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_cache": str(Path(args.object_cache)),
        "exclude_manifest": str(Path(args.exclude_manifest)),
        "source_images_before_exclusion": args.source_images,
        "registered_images_after_exclusion": len(source_ids),
        "excluded_eval_ids": len(excluded_ids),
        "excluded_eval_overlap": [],
        "num_shards": args.num_shards,
        "normalize_duplicates": args.normalize_duplicates,
        "total_mentions": sum(row["mentions"] for row in shards),
        "total_grounded_mentions": sum(
            row["grounded_mentions"] for row in shards
        ),
        "total_hallucinated_mentions": sum(
            row["hallucinated_mentions"] for row in shards
        ),
        "shards": shards,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
