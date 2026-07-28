#!/usr/bin/env python3
"""Prepare open-caption benchmark data, including THRONE's O365 subset.

Objects365 does not publish the THRONE 5,110-image sample as a standalone
archive. This utility streams each official validation patch once and writes
only the registered subset images, avoiding extraction of all 80,000 images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

OBJECTS365_BASE = (
    "https://dorc.ks3-cn-beijing.ksyun.com/data-set/"
    "2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation",
        default="/tmp/sinkdetect-zhiyuan_objv2_val.json",
    )
    parser.add_argument(
        "--subset",
        default="/tmp/sinkdetect-throne-audit/throne/objects365_subset.txt",
    )
    parser.add_argument(
        "--fix_names",
        default="/tmp/sinkdetect-Objects365_names_fix.csv",
    )
    parser.add_argument(
        "--output_root",
        default="/home/chenguanxu/common_dataset/THRONE",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_fix_names(path: str | Path) -> dict[int, tuple[str, str]]:
    mapping = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        category_id, old_name, new_name = line.split(",", maxsplit=2)
        mapping[int(category_id)] = (old_name, new_name)
    return mapping


def make_fixname_annotation(
    annotation: dict,
    mapping: dict[int, tuple[str, str]],
) -> dict:
    output = dict(annotation)
    categories = []
    for category in annotation["categories"]:
        category = dict(category)
        category_id = int(category["id"])
        if category_id not in mapping:
            raise ValueError(f"Missing fix-name entry for category {category_id}")
        expected, replacement = mapping[category_id]
        if category["name"].strip() != expected.strip():
            raise ValueError(
                f"Category {category_id} mismatch: annotation="
                f"{category['name']!r}, mapping={expected!r}"
            )
        category["name"] = replacement
        categories.append(category)
    output["categories"] = categories
    return output


def registered_images(
    annotation: dict,
    subset_path: str | Path,
) -> list[dict]:
    subset_ids = {
        int(value)
        for value in Path(subset_path).read_text().split()
        if value.strip()
    }
    image_by_id = {int(image["id"]): image for image in annotation["images"]}
    missing = sorted(subset_ids - set(image_by_id))
    if missing:
        raise ValueError(f"{len(missing)} THRONE IDs missing from annotation")
    selected = [image_by_id[image_id] for image_id in sorted(subset_ids)]
    if len(selected) != 5110:
        raise ValueError(f"Expected 5,110 THRONE images, got {len(selected)}")
    return selected


def patch_url(relative_parent: str) -> str:
    parent = Path(relative_parent)
    parts = parent.parts
    if (
        len(parts) != 3
        or parts[0] != "images"
        or parts[1] not in {"v1", "v2"}
        or not parts[2].startswith("patch")
    ):
        raise ValueError(f"Unexpected Objects365 patch path: {relative_parent}")
    return f"{OBJECTS365_BASE}/{parts[0]}/{parts[1]}/{parts[2]}.tar.gz"


def _stream_one_patch(
    relative_parent: str,
    wanted: dict[str, Path],
    retries: int,
) -> None:
    missing = {
        basename: target
        for basename, target in wanted.items()
        if not target.is_file() or target.stat().st_size == 0
    }
    if not missing:
        print(f"{relative_parent}: already complete ({len(wanted)})", flush=True)
        return
    url = patch_url(relative_parent)
    for attempt in range(1, retries + 1):
        started = time.time()
        found: set[str] = set()
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "SinkDetect-research/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        basename = Path(member.name).name
                        if basename not in missing:
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        target = missing[basename]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(target.suffix + ".part")
                        with temporary.open("wb") as destination:
                            shutil.copyfileobj(source, destination, 1024 * 1024)
                        os.replace(temporary, target)
                        found.add(basename)
            unresolved = [
                name
                for name, target in wanted.items()
                if not target.is_file() or target.stat().st_size == 0
            ]
            if unresolved:
                raise ValueError(
                    f"{relative_parent}: {len(unresolved)} registered files "
                    "not found in archive"
                )
            print(
                f"{relative_parent}: extracted {len(found)} new / "
                f"{len(wanted)} total in {time.time() - started:.1f}s",
                flush=True,
            )
            return
        except Exception as error:
            print(
                f"{relative_parent}: attempt {attempt}/{retries} failed: {error}",
                flush=True,
            )
            if attempt == retries:
                raise


def validate_images(selected: list[dict], output_root: Path) -> dict:
    missing = []
    empty = []
    total_bytes = 0
    for image in selected:
        relative = Path(image["file_name"])
        target = output_root / "objects365" / relative
        if not target.is_file():
            missing.append(int(image["id"]))
        elif target.stat().st_size == 0:
            empty.append(int(image["id"]))
        else:
            total_bytes += target.stat().st_size
    return {
        "registered": len(selected),
        "present": len(selected) - len(missing) - len(empty),
        "missing_ids": missing,
        "empty_ids": empty,
        "total_bytes": total_bytes,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    annotation_path = Path(args.annotation)
    subset_path = Path(args.subset)
    fix_names_path = Path(args.fix_names)
    annotation = json.loads(annotation_path.read_text())
    selected = registered_images(annotation, subset_path)
    by_patch: dict[str, dict[str, Path]] = defaultdict(dict)
    for image in selected:
        relative = Path(image["file_name"])
        parent = str(relative.parent)
        basename = relative.name
        if basename in by_patch[parent]:
            raise ValueError(f"Duplicate basename inside patch: {relative}")
        by_patch[parent][basename] = output_root / "objects365" / relative

    if not args.validate_only:
        metadata = output_root / "metadata"
        metadata.mkdir(parents=True, exist_ok=True)
        shutil.copy2(subset_path, metadata / "objects365_subset.txt")
        shutil.copy2(annotation_path, metadata / "zhiyuan_objv2_val.json")
        shutil.copy2(fix_names_path, metadata / "Objects365_names_fix.csv")
        fixed = make_fixname_annotation(annotation, load_fix_names(fix_names_path))
        fixed_path = metadata / "zhiyuan_objv2_val_fixname.json"
        fixed_path.write_text(json.dumps(fixed, separators=(",", ":")) + "\n")
        for parent in sorted(by_patch):
            _stream_one_patch(parent, by_patch[parent], args.retries)

    validation = validate_images(selected, output_root)
    manifest = {
        "benchmark": "THRONE-Objects365",
        "subset_count": len(selected),
        "patch_count": len(by_patch),
        "source_annotation": str(annotation_path),
        "source_annotation_sha256": sha256_file(annotation_path),
        "subset_sha256": sha256_file(subset_path),
        "fix_names_sha256": sha256_file(fix_names_path),
        "validation": validation,
    }
    print(json.dumps(manifest, indent=2), flush=True)
    if validation["missing_ids"] or validation["empty_ids"]:
        raise SystemExit(2)
    if not args.validate_only:
        manifest_path = output_root / "metadata" / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

