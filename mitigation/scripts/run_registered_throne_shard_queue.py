#!/usr/bin/env python3
"""Run one physical-GPU shard of the frozen THRONE caption matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/chenguanxu/venvs/qwen3vl/bin/python"
MANIFEST = (
    "/tmp/sinkdetect_datasets/open_caption_manifests/"
    "throne-objects365.jsonl"
)

MODELS = {
    "llava_next": {
        "path": "/tmp/sinkdetect_models/llama3-llava-next-8b-hf",
        "directory": "llava_next",
        "heads": "experiments/registered_caption/llava_next/allpath_heads.json",
        "sensor": "experiments/registered_caption/llava_next/sensor.pt",
        "thresholds": (
            "experiments/registered_caption/llava_next/thresholds.json"
        ),
        "pai_gate": (
            "experiments/registered_caption/llava_next/"
            "pai_active_20_generationmix.json"
        ),
        "replay": (
            "experiments/registered_caption/llava_next/chair/"
            "vanilla_replay_20.json"
        ),
        "rollback": (
            "experiments/registered_caption/llava_next/chair/"
            "rollback_identity_5.json"
        ),
    },
    "qwen25": {
        "path": "/home/chenguanxu/models/Qwen2.5-VL-7B-Instruct",
        "directory": "qwen25",
        "heads": "experiments/registered_caption/qwen25/allpath_heads.json",
        "sensor": "experiments/registered_caption/qwen25/sensor.pt",
        "thresholds": (
            "experiments/registered_caption/qwen25/thresholds.json"
        ),
        "pai_gate": (
            "experiments/registered_caption/qwen25/"
            "pai_active_20_generationmix.json"
        ),
        "replay": (
            "experiments/registered_caption/qwen25/chair/"
            "vanilla_replay_20.json"
        ),
        "rollback": (
            "experiments/registered_caption/qwen25/"
            "rollback_identity_5.json"
        ),
    },
    "qwen3": {
        "path": "/home/chenguanxu/models/Qwen3-VL-8B-Instruct",
        "directory": "qwen3",
        "heads": "experiments/registered_caption/qwen3/allpath_heads.json",
        "sensor": "experiments/registered_caption/qwen3/sensor.pt",
        "thresholds": (
            "experiments/registered_caption/qwen3/thresholds.json"
        ),
        "pai_gate": (
            "experiments/registered_caption/qwen3/"
            "pai_active_20_generationmix.json"
        ),
        "replay": (
            "experiments/registered_caption/qwen3/chair/"
            "vanilla_replay_20.json"
        ),
        "rollback": (
            "experiments/registered_caption/qwen3/chair/"
            "rollback_identity_5.json"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical_gpu", type=int, required=True)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument(
        "--decoder_gate",
        default=(
            "experiments/registered_caption/"
            "formula_decoder_validation.json"
        ),
    )
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--max_attempts", type=int, default=5)
    return parser.parse_args()


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def validate_gate_files(decoder_gate: str | Path) -> None:
    decoder = _json(decoder_gate)
    if decoder.get("schema") != "sinkdetect-formula-decoder-validation-v1":
        raise ValueError("Unexpected formula decoder gate schema")
    for kind in ("qwen25", "llava_next_video"):
        audit = decoder.get("vcd_identity", {}).get(kind, {})
        if audit.get("samples") != 5 or audit.get("identical") != 5:
            raise ValueError(f"Decoder gate lacks a passing {kind} audit")
    if not decoder.get("unit_tests_pass"):
        raise ValueError("Decoder gate does not record passing unit tests")

    for kind, config in MODELS.items():
        replay = _json(config["replay"])
        if replay.get("kind") != kind:
            raise ValueError(f"{kind}: replay kind mismatch")
        if replay.get("samples") != 20 or replay.get("identical") != 20:
            raise ValueError(f"{kind}: vanilla replay gate failed")
        rollback = _json(config["rollback"])
        if rollback.get("kind") != kind:
            raise ValueError(f"{kind}: rollback kind mismatch")
        if rollback.get("samples") != 5:
            raise ValueError(f"{kind}: rollback gate must use five samples")
        if not rollback.get("all_token_identical"):
            raise ValueError(f"{kind}: rollback token identity failed")
        if not rollback.get("shared_control_checked"):
            raise ValueError(f"{kind}: shared control was not checked")
        if not rollback.get("all_shared_control_identical"):
            raise ValueError(f"{kind}: rollback source-control gate failed")

        pai = _json(config["pai_gate"])
        summary = pai.get("summary", {})
        if pai.get("kind") != kind or pai.get("samples") != 20:
            raise ValueError(f"{kind}: active PAI gate mismatch")
        if not summary.get("all_attention_ports_called"):
            raise ValueError(f"{kind}: PAI port was not exercised")
        if int(summary.get("changed_outputs", 0)) == 0:
            raise ValueError(f"{kind}: active PAI changed no outputs")
        for key in ("heads", "sensor", "thresholds"):
            if not (ROOT / config[key]).is_file():
                raise FileNotFoundError(ROOT / config[key])


def gates_ready(decoder_gate: str | Path) -> bool:
    required = [ROOT / decoder_gate]
    for config in MODELS.values():
        required.extend(
            ROOT / config[key]
            for key in (
                "heads",
                "sensor",
                "thresholds",
                "pai_gate",
                "replay",
                "rollback",
            )
        )
    return all(path.is_file() for path in required)


def gpu_memory_mib(index: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip())


def wait_for_gpu(index: int, poll_seconds: float) -> None:
    while gpu_memory_mib(index) >= 1000:
        time.sleep(poll_seconds)


def formula_command(
    kind: str, config: dict[str, str], shard_idx: int, num_shards: int
) -> list[str]:
    directory = (
        ROOT
        / "experiments"
        / "registered_caption"
        / config["directory"]
        / "throne"
    )
    command = [
        PYTHON,
        str(ROOT / "mitigation/scripts/run_open_caption_formula_suite.py"),
        "--kind",
        kind,
        "--model_path",
        config["path"],
    ]
    for method in ("vanilla", "pai", "clearsight", "diml", "allpath"):
        command.extend(
            [
                "--job",
                method,
                MANIFEST,
                str(directory / f"{method}_shard{shard_idx}.jsonl"),
            ]
        )
    command.extend(
        [
            "--device",
            "cuda:0",
            "--shard_idx",
            str(shard_idx),
            "--num_shards",
            str(num_shards),
            "--max_new_tokens",
            "512",
            "--seed",
            "927",
            "--allpath_heads",
            str(ROOT / config["heads"]),
            "--pai_gate",
            str(ROOT / config["pai_gate"]),
        ]
    )
    return command


def ours_command(
    kind: str, config: dict[str, str], shard_idx: int, num_shards: int
) -> list[str]:
    output = (
        ROOT
        / "experiments"
        / "registered_caption"
        / config["directory"]
        / "throne"
        / f"ours_shard{shard_idx}.jsonl"
    )
    return [
        PYTHON,
        str(ROOT / "mitigation/scripts/run_registered_qwen_rollback.py"),
        "--kind",
        kind,
        "--media",
        "image",
        "--model_path",
        config["path"],
        "--manifest",
        MANIFEST,
        "--sensor",
        str(ROOT / config["sensor"]),
        "--thresholds",
        str(ROOT / config["thresholds"]),
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--shard_idx",
        str(shard_idx),
        "--num_shards",
        str(num_shards),
        "--max_new_tokens",
        "512",
        "--max_actions",
        "5",
        "--seed",
        "927",
    ]


def run_retry(
    command: list[str],
    *,
    physical_gpu: int,
    poll_seconds: float,
    max_attempts: int,
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["TRANSFORMERS_OFFLINE"] = "1"
    for attempt in range(1, max_attempts + 1):
        wait_for_gpu(physical_gpu, poll_seconds)
        result = subprocess.run(command, cwd=ROOT, env=environment)
        if result.returncode == 0:
            return
        print(
            f"attempt {attempt}/{max_attempts} failed with "
            f"exit {result.returncode}; waiting for GPU {physical_gpu}",
            flush=True,
        )
    raise RuntimeError(f"Command failed {max_attempts} times: {command}")


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("Invalid shard")
    if not 0 <= args.physical_gpu < 8:
        raise ValueError("physical GPU must be in [0, 8)")
    while not gates_ready(args.decoder_gate):
        time.sleep(args.poll_seconds)
    validate_gate_files(args.decoder_gate)
    for kind, config in MODELS.items():
        run_retry(
            formula_command(
                kind, config, args.shard_idx, args.num_shards
            ),
            physical_gpu=args.physical_gpu,
            poll_seconds=args.poll_seconds,
            max_attempts=args.max_attempts,
        )
        run_retry(
            ours_command(kind, config, args.shard_idx, args.num_shards),
            physical_gpu=args.physical_gpu,
            poll_seconds=args.poll_seconds,
            max_attempts=args.max_attempts,
        )


if __name__ == "__main__":
    main()
