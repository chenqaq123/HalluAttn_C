#!/usr/bin/env python3
"""Audit active PAI length/EOS behavior on the frozen 20-image subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.scripts.run_open_caption_baseline import (
    _load_jsonl,
    _port_config,
)
from mitigation.src.formula_ports import (
    FormulaAttentionPort,
    generate_pai_greedy,
)
from mitigation.src.generation_audit import continuation_audit
from mitigation.src.open_caption_models import OpenCaptionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=["llava_next", "qwen25", "qwen3"], required=True
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def _sha256(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    captioner = OpenCaptionModel(
        args.kind,
        args.model_path,
        args.device,
        max_new_tokens=args.max_new_tokens,
    )
    rows = []
    for record in tqdm(
        _load_jsonl(args.manifest)[: args.limit],
        desc=f"pai-active-audit-{args.kind}",
    ):
        with Image.open(record["image_path"]) as source:
            inputs = captioner.prepare_image(
                source.convert("RGB"), record["prompt"]
            )
        prompt_length = int(inputs["input_ids"].shape[-1])
        visual_positions = captioner.visual_positions(
            inputs, media_type="image"
        )
        vanilla = captioner.generate(inputs)
        config, _ = _port_config(
            "pai",
            captioner.decoder.layers,
            visual_positions,
            identity=False,
            allpath_path=None,
        )
        with FormulaAttentionPort(
            captioner.decoder.modeling_module,
            captioner.decoder.attention_classes,
            config,
        ) as port:
            unconditional = captioner.prepare_text_only(record["prompt"])
            pai, decode = generate_pai_greedy(
                captioner.model,
                captioner.tokenizer,
                inputs,
                unconditional,
                port,
                max_new_tokens=args.max_new_tokens,
                guidance_scale=2.0,
                plausibility_cutoff=0.1,
            )
        vanilla_audit = continuation_audit(
            vanilla,
            prompt_length,
            captioner.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        pai_audit = continuation_audit(
            pai,
            prompt_length,
            captioner.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        rows.append(
            {
                "sample_id": record["sample_id"],
                "vanilla_caption": captioner.decode_continuation(
                    vanilla, prompt_length
                ),
                "pai_caption": captioner.decode_continuation(
                    pai, prompt_length
                ),
                "vanilla_sha256": _sha256(vanilla),
                "pai_sha256": _sha256(pai),
                "token_identical": bool(torch.equal(vanilla, pai)),
                "vanilla": vanilla_audit,
                "pai": pai_audit,
                "attention_audit": port.audit.to_dict(),
                "decode_audit": decode,
            }
        )
        del inputs, vanilla, pai, unconditional
        torch.cuda.empty_cache()
    payload = {
        "kind": args.kind,
        "model_path": str(Path(args.model_path).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "samples": len(rows),
        "parameters": {
            "attention_alpha": 0.5,
            "guidance_scale": 2.0,
            "plausibility_cutoff": 0.1,
            "layers": "[2,L)",
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": {
            "changed_outputs": sum(not row["token_identical"] for row in rows),
            "vanilla_premature_eos": sum(
                row["vanilla"]["premature_eos"] for row in rows
            ),
            "pai_premature_eos": sum(
                row["pai"]["premature_eos"] for row in rows
            ),
            "vanilla_mean_content_tokens": float(
                np.mean([row["vanilla"]["content_tokens"] for row in rows])
            ),
            "pai_mean_content_tokens": float(
                np.mean([row["pai"]["content_tokens"] for row in rows])
            ),
            "all_attention_ports_called": all(
                row["attention_audit"].get("calls", 0) > 0
                and row["attention_audit"].get("modified_calls", 0) > 0
                and row["attention_audit"].get("max_abs_tensor_delta", 0.0) > 0
                for row in rows
            ),
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload["summary"], indent=2))
    if not payload["summary"]["all_attention_ports_called"]:
        raise RuntimeError("Active PAI attention port was not exercised")


if __name__ == "__main__":
    main()
