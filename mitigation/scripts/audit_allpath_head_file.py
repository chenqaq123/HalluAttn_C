#!/usr/bin/env python3
"""Validate a frozen AllPath head file and disclose sign-set overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--heads_per_layer", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _flat(grouped: dict[str, list[int]]) -> set[tuple[int, int]]:
    return {
        (int(layer), int(head))
        for layer, heads in grouped.items()
        for head in heads
    }


def main() -> None:
    args = parse_args()
    source = Path(args.heads)
    payload = json.loads(source.read_text())
    good = _flat(payload["good_heads"])
    hallu = _flat(payload["hallu_heads"])
    overlap = good & hallu
    selected_scores = [
        float(item["score"])
        for values in payload["selections"].values()
        for item in values
    ]
    all_heads = good | hallu
    bounds_pass = all(
        0 <= layer < args.layers
        and 0 <= head < args.heads_per_layer
        for layer, head in all_heads
    )
    finite_pass = bool(np.isfinite(selected_scores).all())
    # FormulaAttentionPort was frozen before head identification and gives
    # good/in_scale priority by excluding a good head from the hallu branch.
    effective_hallu = hallu - good
    audit = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "architecture": {
            "layers": args.layers,
            "heads_per_layer": args.heads_per_layer,
        },
        "selection_lengths": {
            name: len(values)
            for name, values in payload["selections"].items()
        },
        "unique_good_heads": len(good),
        "unique_hallu_heads_before_priority": len(hallu),
        "good_hallu_overlap": [
            {"layer": layer, "head": head}
            for layer, head in sorted(overlap)
        ],
        "overlap_count": len(overlap),
        "overlap_resolution": (
            "pre-existing FormulaAttentionPort good/in_scale priority"
        ),
        "effective_good_heads": len(good),
        "effective_hallu_heads": len(effective_hallu),
        "bounds_pass": bounds_pass,
        "selected_scores_finite": finite_pass,
        "audit_pass": bounds_pass and finite_pass,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(audit, indent=2))
    if not audit["audit_pass"]:
        raise RuntimeError("AllPath head-file audit failed")


if __name__ == "__main__":
    main()
