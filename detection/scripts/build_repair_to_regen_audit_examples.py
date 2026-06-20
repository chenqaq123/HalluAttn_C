#!/usr/bin/env python3
"""Build closed-loop audit inputs comparing repaired captions to regeneration.

The existing closed-loop audit script expects a reference caption named
``vanilla_caption`` and a candidate caption named ``gated_caption``. For
verification-in-loop caption expansion, the reference is the deterministic
claim-local repair and the candidate is a regenerated caption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regen_examples_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_examples.json",
    )
    parser.add_argument(
        "--audit_compatible_json",
        default="detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_audit_compatible_examples.json",
    )
    parser.add_argument(
        "--output_json",
        default="detection/baselines/results/tdev_caption_verified_expansion_100/repair_to_detail_regen_audit_examples.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    regen_rows = read_json(Path(args.regen_examples_json))
    audit_rows = {
        int(row["image_id"]): row
        for row in read_json(Path(args.audit_compatible_json))
    }
    out = []
    for row in regen_rows:
        image_id = int(row["image_id"])
        audit_row = audit_rows[image_id]
        repaired_caption = str(row.get("repaired_caption", "")).strip()
        regenerated_caption = str(row.get("regenerated_caption", "")).strip()
        out.append(
            {
                "image_id": image_id,
                "image_path": row["image_path"],
                "denied_items": audit_row.get("denied_items", []),
                "num_denied_sequences": audit_row.get("num_denied_sequences", 0),
                "vanilla_source": "claim_local_repair",
                "vanilla_caption": repaired_caption,
                "gated_caption": regenerated_caption,
                "caption_differs_from_reference": int(repaired_caption != regenerated_caption),
                "reference_comparison_note": "controlled detail regeneration vs claim-local repair",
                "gate_events": row.get("gate_events", []),
            }
        )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "num_examples": len(out)}, indent=2))


if __name__ == "__main__":
    main()
