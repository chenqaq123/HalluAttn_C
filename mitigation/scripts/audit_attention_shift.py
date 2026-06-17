#!/usr/bin/env python3
"""Collect and summarize attention-routing audits for mitigation methods.

The audit asks whether an intervention changes attention in the intended
direction, and whether that change corresponds to better yes/no discrimination.
It is a small-subset diagnostic, not a replacement for the full POPE/CHAIR run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "mitigation"))
sys.path.insert(0, str(PROJECT_ROOT / "detection" / "src"))

from src.evaluation import normalize_yes_no, pope_metrics, read_jsonl, write_json


METHODS = ("vanilla", "pai", "clearsight", "visattnsink", "spin")


def _default_layers(method: str) -> tuple[int, int]:
    if method == "clearsight":
        return 9, 15
    if method in {"pai", "visattnsink"}:
        return 2, 32
    if method == "spin":
        return 0, 32
    return 2, 32


def _install_audit_wrappers(model, method: str, device, args: argparse.Namespace):
    from src.interventions import AttentionIntervention, InterventionConfig, get_llm_layers

    active_start, active_end = _default_layers(method)
    if args.start_layer is not None:
        active_start = args.start_layer
    if args.end_layer is not None:
        active_end = args.end_layer

    audit_start = args.audit_start_layer
    audit_end = args.audit_end_layer
    layers = get_llm_layers(model)
    active_end = min(active_end, len(layers))
    audit_end = min(audit_end, len(layers))

    active_config = InterventionConfig(
        method=method,
        start_layer=active_start,
        end_layer=active_end,
        pai_alpha=args.pai_alpha,
        vaf_enhance=args.vaf_enhance,
        vaf_suppress=args.vaf_suppress,
        vas_tau=args.vas_tau,
        vas_rho=args.vas_rho,
        vas_visual_mass=args.vas_visual_mass,
        vas_keep=args.vas_keep,
        spin_routed_heads=args.spin_routed_heads,
        spin_small_num_mask=args.spin_small_num_mask,
    )
    noop_config = InterventionConfig(
        method="vanilla",
        start_layer=audit_start,
        end_layer=audit_end,
        vas_tau=args.vas_tau,
    )

    for layer_idx in range(audit_start, audit_end):
        original = layers[layer_idx].self_attn
        config = active_config if method != "vanilla" and active_start <= layer_idx < active_end else noop_config
        adapter = AttentionIntervention(original.config, layer_idx, config)
        adapter.load_state_dict(original.state_dict())
        adapter.to(device=device, dtype=next(original.parameters()).dtype)
        adapter.audit_enabled = True
        layers[layer_idx].self_attn = adapter

        def _capture(_module, hook_args, idx=layer_idx):
            current = get_llm_layers(model)[idx].self_attn
            if isinstance(current, AttentionIntervention):
                current.capture_sink_candidates(hook_args[0])

        layers[layer_idx].register_forward_pre_hook(_capture)
    return active_config


def _reset_audit(model) -> None:
    from src.interventions import AttentionIntervention, get_llm_layers

    for layer in get_llm_layers(model):
        if isinstance(layer.self_attn, AttentionIntervention):
            layer.self_attn.reset_audit()


def _collect_audit_records(model) -> list[dict[str, Any]]:
    from src.interventions import AttentionIntervention, get_llm_layers

    records: list[dict[str, Any]] = []
    for layer in get_llm_layers(model):
        if isinstance(layer.self_attn, AttentionIntervention):
            records.extend(layer.self_attn.audit_records)
    return records


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return mean(finite) if finite else None


def _summarize_attention(records: list[dict[str, Any]], layers: set[int] | None = None) -> dict[str, Any]:
    selected = [row for row in records if layers is None or int(row["layer"]) in layers]
    out: dict[str, Any] = {
        "records": len(selected),
        "layers": sorted({int(row["layer"]) for row in selected}),
    }
    for key in [
        "pre_visual_mass",
        "post_visual_mass",
        "delta_visual_mass",
        "pre_prefix_mass",
        "post_prefix_mass",
        "delta_prefix_mass",
        "pre_sink_mass",
        "post_sink_mass",
        "delta_sink_mass",
        "pre_nonvisual_mass",
        "post_nonvisual_mass",
        "delta_nonvisual_mass",
    ]:
        out[key] = _finite_mean([row.get(key) for row in selected])
    out["mean_num_sinks"] = _finite_mean([row.get("num_sinks") for row in selected])
    return out


def _prepare(model, processor, image, prompt: str, device):
    import torch

    from sinkdetect.sink_utils import find_vis_bounds
    from src.interventions import set_visual_bounds

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)
    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(processor, "image_token_index", 32000)
    vis_start, vis_end = find_vis_bounds(inputs["input_ids"][0], image_token_id)
    set_visual_bounds(model, vis_start, vis_end)
    return inputs, vis_start, vis_end


def _generate(model, processor, inputs, max_new_tokens: int) -> str:
    import torch

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    del output_ids
    return text


def _confusion(pred: str, gold: str) -> str:
    if pred == "invalid":
        return "invalid"
    if pred == "yes" and gold == "yes":
        return "tp"
    if pred == "yes" and gold == "no":
        return "fp"
    if pred == "no" and gold == "no":
        return "tn"
    return "fn"


def collect(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm

    from sinkdetect.utils import load_model_and_processor
    from src.data import iter_pope_records, load_pope_image
    from src.interventions import get_llm_layers

    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)
    model, processor = load_model_and_processor(
        args.model_path,
        device,
        attn_implementation="eager",
        cache_dir=args.cache_dir or None,
    )
    intervention = _install_audit_wrappers(model, args.method, device, args)

    records = list(iter_pope_records(
        args.pope_dir,
        args.coco_path,
        args.pope_split,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        limit=args.limit,
    ))

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = {json.loads(line)["question_id"] for line in f if line.strip()}

    active_layers = set(range(intervention.start_layer, intervention.end_layer)) if args.method != "vanilla" else set()
    with output_path.open("a", encoding="utf-8") as out:
        for step, record in enumerate(tqdm(records, desc=f"audit:{args.pope_split}:{args.method}:shard{args.shard_idx}"), start=1):
            if record["question_id"] in existing:
                continue
            image = load_pope_image(record)
            question = record["question"] + " Please just answer yes or no."
            prompt = f"<image>\nUSER: {question}\nASSISTANT:"
            inputs, vis_start, vis_end = _prepare(model, processor, image, prompt, device)
            image.close()
            _reset_audit(model)
            text = _generate(model, processor, inputs, args.max_new_tokens)
            pred = normalize_yes_no(text)
            gold = str(record["label"]).lower()
            attention_records = _collect_audit_records(model)
            payload = {
                "question_id": record["question_id"],
                "image": record.get("image", ""),
                "question": record["question"],
                "label": gold,
                "method": args.method,
                "split": args.pope_split,
                "text": text,
                "prediction": pred,
                "correct": pred == gold,
                "confusion": _confusion(pred, gold),
                "visual_span": [vis_start, vis_end],
                "intervention": asdict(intervention),
                "audit_layers": list(range(args.audit_start_layer, min(args.audit_end_layer, len(get_llm_layers(model))))),
                "active_layers": sorted(active_layers),
                "attention_all": _summarize_attention(attention_records),
                "attention_active": _summarize_attention(attention_records, active_layers) if active_layers else None,
                "attention_by_layer": [
                    _summarize_attention(attention_records, {layer})
                    for layer in sorted({int(row["layer"]) for row in attention_records})
                ],
            }
            out.write(json.dumps(payload) + "\n")
            out.flush()
            del inputs
            if step % 16 == 0:
                gc.collect()
                torch.cuda.empty_cache()


def _load_method_rows(audit_dir: Path, method: str) -> list[dict[str, Any]]:
    path = audit_dir / method / "attention_audit.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def _attention_for_layers(row: dict[str, Any], layers: set[int]) -> dict[str, Any]:
    records = []
    for layer_payload in row.get("attention_by_layer", []):
        layer_list = layer_payload.get("layers", [])
        if layer_list and int(layer_list[0]) in layers:
            records.append(layer_payload)
    out: dict[str, Any] = {"records": sum(int(item.get("records", 0)) for item in records)}
    for key in [
        "pre_visual_mass",
        "post_visual_mass",
        "delta_visual_mass",
        "pre_prefix_mass",
        "post_prefix_mass",
        "delta_prefix_mass",
        "pre_sink_mass",
        "post_sink_mass",
        "delta_sink_mass",
    ]:
        out[key] = _finite_mean([item.get(key) for item in records])
    return out


def _method_summary(rows: list[dict[str, Any]], vanilla_by_id: dict[Any, dict[str, Any]] | None = None) -> dict[str, Any]:
    method = rows[0]["method"] if rows else ""
    pope_rows = [{"label": row["label"], "text": row["text"]} for row in rows]
    metrics = pope_metrics(pope_rows)
    summary: dict[str, Any] = {
        "method": method,
        **metrics,
        "mean_all_post_visual_mass": _finite_mean([row["attention_all"].get("post_visual_mass") for row in rows]),
        "mean_all_delta_visual_mass": _finite_mean([row["attention_all"].get("delta_visual_mass") for row in rows]),
        "mean_all_post_prefix_mass": _finite_mean([row["attention_all"].get("post_prefix_mass") for row in rows]),
        "mean_all_delta_prefix_mass": _finite_mean([row["attention_all"].get("delta_prefix_mass") for row in rows]),
        "mean_all_post_sink_mass": _finite_mean([row["attention_all"].get("post_sink_mass") for row in rows]),
        "mean_all_delta_sink_mass": _finite_mean([row["attention_all"].get("delta_sink_mass") for row in rows]),
        "tp_mean_post_visual_mass": _finite_mean([row["attention_all"].get("post_visual_mass") for row in rows if row["confusion"] == "tp"]),
        "fp_mean_post_visual_mass": _finite_mean([row["attention_all"].get("post_visual_mass") for row in rows if row["confusion"] == "fp"]),
    }
    active_layers = set(rows[0].get("active_layers", [])) if rows else set()
    if active_layers:
        active_values = [_attention_for_layers(row, active_layers) for row in rows]
        summary.update({
            "active_layers": f"{min(active_layers)}-{max(active_layers) + 1}",
            "mean_active_post_visual_mass": _finite_mean([item.get("post_visual_mass") for item in active_values]),
            "mean_active_delta_visual_mass": _finite_mean([item.get("delta_visual_mass") for item in active_values]),
            "mean_active_post_prefix_mass": _finite_mean([item.get("post_prefix_mass") for item in active_values]),
            "mean_active_delta_prefix_mass": _finite_mean([item.get("delta_prefix_mass") for item in active_values]),
            "mean_active_post_sink_mass": _finite_mean([item.get("post_sink_mass") for item in active_values]),
            "mean_active_delta_sink_mass": _finite_mean([item.get("delta_sink_mass") for item in active_values]),
        })
    else:
        summary["active_layers"] = ""

    if vanilla_by_id and active_layers:
        deltas = []
        prefix_deltas = []
        sink_deltas = []
        for row in rows:
            vanilla = vanilla_by_id.get(row["question_id"])
            if vanilla is None:
                continue
            method_attn = _attention_for_layers(row, active_layers)
            vanilla_attn = _attention_for_layers(vanilla, active_layers)
            method_visual = method_attn.get("post_visual_mass")
            vanilla_visual = vanilla_attn.get("post_visual_mass")
            if method_visual is not None and vanilla_visual is not None:
                deltas.append(method_visual - vanilla_visual)
            method_prefix = method_attn.get("post_prefix_mass")
            vanilla_prefix = vanilla_attn.get("post_prefix_mass")
            if method_prefix is not None and vanilla_prefix is not None:
                prefix_deltas.append(method_prefix - vanilla_prefix)
            method_sink = method_attn.get("post_sink_mass")
            vanilla_sink = vanilla_attn.get("post_sink_mass")
            if method_sink is not None and vanilla_sink is not None:
                sink_deltas.append(method_sink - vanilla_sink)
        summary["matched_delta_active_visual_vs_vanilla"] = _finite_mean(deltas)
        summary["matched_delta_active_prefix_vs_vanilla"] = _finite_mean(prefix_deltas)
        summary["matched_delta_active_sink_vs_vanilla"] = _finite_mean(sink_deltas)
    return summary


def summarize(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rows_by_method = {method: _load_method_rows(audit_dir, method) for method in methods}
    vanilla_by_id = {row["question_id"]: row for row in rows_by_method.get("vanilla", [])}
    summaries = [
        _method_summary(rows, vanilla_by_id=vanilla_by_id)
        for method, rows in rows_by_method.items()
        if rows
    ]
    payload = {
        "audit_dir": str(audit_dir),
        "methods": methods,
        "summaries": summaries,
    }
    write_json(audit_dir / "attention_audit_summary.json", payload)
    fieldnames = sorted({key for row in summaries for key in row.keys()})
    with (audit_dir / "attention_audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {audit_dir / 'attention_audit_summary.json'}")
    print(f"Wrote {audit_dir / 'attention_audit_summary.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attention-shift audit for mitigation methods")
    p.add_argument("--mode", choices=["collect", "summarize"], default="collect")
    p.add_argument("--method", choices=METHODS, default="vanilla")
    p.add_argument("--methods", default="vanilla,pai,clearsight,visattnsink")
    p.add_argument("--model_path", default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--coco_path", default="/home/chenguanxu/common_dataset/coco-2014-dataset")
    p.add_argument("--pope_dir", default="")
    p.add_argument("--pope_split", default="random")
    p.add_argument("--output_file", default="")
    p.add_argument("--audit_dir", default="")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=4)
    p.add_argument("--audit_start_layer", type=int, default=2)
    p.add_argument("--audit_end_layer", type=int, default=32)
    p.add_argument("--start_layer", type=int)
    p.add_argument("--end_layer", type=int)
    p.add_argument("--pai_alpha", type=float, default=0.2)
    p.add_argument("--vaf_enhance", type=float, default=1.15)
    p.add_argument("--vaf_suppress", type=float, default=0.95)
    p.add_argument("--vas_tau", type=float, default=20.0)
    p.add_argument("--vas_rho", type=float, default=0.5)
    p.add_argument("--vas_visual_mass", type=float, default=0.2)
    p.add_argument("--vas_keep", type=float, default=0.6)
    p.add_argument("--spin_routed_heads", type=float, default=0.8)
    p.add_argument("--spin_small_num_mask", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "summarize":
        if not args.audit_dir:
            raise ValueError("--audit_dir is required in summarize mode")
        summarize(args)
        return
    if not args.pope_dir:
        raise ValueError("--pope_dir is required in collect mode")
    if not args.output_file:
        raise ValueError("--output_file is required in collect mode")
    collect(args)


if __name__ == "__main__":
    main()
