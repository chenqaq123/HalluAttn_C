#!/usr/bin/env python3
"""Audit and atomically merge registered caption-generation shards."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shards_glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--id_key",
        choices=["sample_id", "clip_id"],
        required=True,
    )
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing shard metadata sidecar: {metadata_path}"
        )
    return json.loads(metadata_path.read_text())


def _stable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "shard_idx",
            "completed_samples",
            "registered_samples_in_shard",
        }
    }


def merge(
    *,
    manifest_path: Path,
    shard_paths: list[Path],
    output_path: Path,
    id_key: str,
) -> dict[str, Any]:
    if not shard_paths:
        raise ValueError("No shard files matched")
    manifest_rows = _jsonl(manifest_path)
    manifest_ids = [row[id_key] for row in manifest_rows]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError(f"Manifest has duplicate {id_key} values")

    metadata_rows = [_metadata(path) for path in shard_paths]
    num_shards_values = {
        int(metadata["num_shards"]) for metadata in metadata_rows
    }
    if len(num_shards_values) != 1:
        raise ValueError("Shard metadata disagree on num_shards")
    num_shards = num_shards_values.pop()
    if len(shard_paths) != num_shards:
        raise ValueError(
            f"Expected {num_shards} shard files, found {len(shard_paths)}"
        )
    shard_indices = [int(metadata["shard_idx"]) for metadata in metadata_rows]
    if sorted(shard_indices) != list(range(num_shards)):
        raise ValueError(
            f"Shard indices are incomplete or duplicated: {shard_indices}"
        )

    stable = _stable_metadata(metadata_rows[0])
    for metadata in metadata_rows[1:]:
        if _stable_metadata(metadata) != stable:
            raise ValueError("Shard metadata disagree outside shard-local fields")
    if Path(stable["manifest"]).resolve() != manifest_path.resolve():
        raise ValueError("Shard metadata manifest does not match --manifest")

    by_id: dict[Any, dict[str, Any]] = {}
    shard_audit = []
    for path, metadata in zip(shard_paths, metadata_rows):
        rows = _jsonl(path)
        expected = int(metadata["registered_samples_in_shard"])
        if len(rows) != expected or int(metadata["completed_samples"]) != expected:
            raise ValueError(
                f"Incomplete shard {path}: rows={len(rows)}, expected={expected}"
            )
        for row in rows:
            row_id = row[id_key]
            if row_id in by_id:
                raise ValueError(f"Duplicate generated {id_key}: {row_id}")
            by_id[row_id] = row
        shard_audit.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "rows": len(rows),
                "shard_idx": int(metadata["shard_idx"]),
            }
        )

    generated_ids = set(by_id)
    expected_ids = set(manifest_ids)
    missing = expected_ids - generated_ids
    extra = generated_ids - expected_ids
    if missing or extra:
        raise ValueError(
            f"Generated IDs do not match manifest: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row_id in manifest_ids:
            handle.write(json.dumps(by_id[row_id]) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)

    merged_metadata = {
        **stable,
        "shard_idx": None,
        "num_shards": num_shards,
        "registered_samples": len(manifest_ids),
        "completed_samples": len(by_id),
        "merge": {
            "manifest_sha256": _sha256(manifest_path),
            "output_sha256": _sha256(output_path),
            "preserved_manifest_order": True,
            "id_key": id_key,
            "shards": sorted(
                shard_audit, key=lambda item: item["shard_idx"]
            ),
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_temporary = metadata_path.with_suffix(
        metadata_path.suffix + ".tmp"
    )
    metadata_temporary.write_text(
        json.dumps(merged_metadata, indent=2) + "\n"
    )
    os.replace(metadata_temporary, metadata_path)
    return merged_metadata


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path) for path in glob.glob(args.shards_glob))
    payload = merge(
        manifest_path=Path(args.manifest),
        shard_paths=paths,
        output_path=Path(args.output),
        id_key=args.id_key,
    )
    print(
        json.dumps(
            {
                "completed_samples": payload["completed_samples"],
                "num_shards": payload["num_shards"],
                "output_sha256": payload["merge"]["output_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
