#!/usr/bin/env python3
"""Extract disjoint COCO features for the registered LLaVA-NeXT sensor."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "detection" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "detection" / "src"))

from sinkdetect.chair import evaluate_chair

from mitigation.src.allpath_identification import first_chair_mentions
from mitigation.src.registered_llava_probe import (
    RegisteredLlavaCaptioner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
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
        "--coco_root",
        default="/home/chenguanxu/common_dataset/coco-2014-dataset",
    )
    parser.add_argument(
        "--chair_pkl",
        default="/home/chenguanxu/nas/VLM/Hallucination/pas/data/chair_coco.pkl",
    )
    parser.add_argument(
        "--pas_src",
        default="/home/chenguanxu/nas/VLM/Hallucination/pas/src",
    )
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--source_images", type=int, default=3000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def arrays(
    rows: dict[str, list], layers: int, heads: int
) -> dict[str, np.ndarray]:
    return {
        "object_id": np.asarray(rows["object_id"], dtype=np.int64),
        "image_id": np.asarray(rows["image_id"], dtype=np.int64),
        "token_pos": np.asarray(rows["token_pos"], dtype=np.int32),
        "gen_pos": np.asarray(rows["gen_pos"], dtype=np.int32),
        "label": np.asarray(rows["label"], dtype=np.int32),
        "word": np.asarray(rows["word"], dtype="U64"),
        "no_rope": (
            np.asarray(rows["no_rope"], dtype=np.float32)
            if rows["no_rope"]
            else np.empty((0, layers, 2, heads, 1), dtype=np.float32)
        ),
        "first_logp": np.asarray(
            rows["first_logp"], dtype=np.float64
        ),
        "feature_names": np.asarray(["visual_mass"], dtype="U32"),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.pas_src)
    evaluator = pickle.load(open(args.chair_pkl, "rb"))
    source_ids = sorted(
        {
            int(json.loads(line)["image_id"])
            for line in Path(args.object_cache).read_text().splitlines()
            if line
        }
    )[: args.source_images]
    excluded = {
        int(json.loads(line)["sample_id"])
        for line in Path(args.exclude_manifest).read_text().splitlines()
        if line
    }
    source_ids = [
        image_id for image_id in source_ids if image_id not in excluded
    ]
    selected = source_ids[
        args.shard_idx :: args.num_shards
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    device = torch.device(args.device)
    captioner = RegisteredLlavaCaptioner(
        args.model_path,
        device,
        kind="llava_next",
        max_new_tokens=args.max_new_tokens,
    )
    prefix = Path(args.output_prefix)
    jsonl_path = Path(str(prefix) + ".jsonl")
    npz_path = Path(str(prefix) + ".npz")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "object_id",
        "image_id",
        "token_pos",
        "gen_pos",
        "label",
        "word",
        "no_rope",
        "first_logp",
    )
    values: dict[str, list] = {key: [] for key in keys}
    done = set()
    if jsonl_path.is_file() and npz_path.is_file():
        cached = np.load(npz_path, allow_pickle=True)
        for key in keys:
            values[key] = list(cached[key])
        done = {
            int(json.loads(line)["image_id"])
            for line in jsonl_path.read_text().splitlines()
            if line
        }
    pending = [
        image_id for image_id in selected if image_id not in done
    ]
    print(
        f"LLaVA-NeXT image shard {args.shard_idx}/{args.num_shards}: "
        f"{len(pending)} pending"
    )

    with jsonl_path.open("a") as output:
        for image_id in tqdm(pending, desc="llava-next-image-features"):
            image_path = (
                Path(args.coco_root)
                / "val2014"
                / f"COCO_val2014_{image_id:012d}.jpg"
            )
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                inputs = captioner.prompt_inputs(
                    image, "Describe this image in detail."
                )
            prompt_length = int(inputs["input_ids"].shape[-1])
            full_ids = captioner.generate_ids(inputs)
            generated = full_ids[0, prompt_length:].tolist()
            caption = captioner.caption_text(
                full_ids, prompt_length
            )
            try:
                evaluations, _ = evaluate_chair(
                    evaluator,
                    image_ids=[image_id],
                    captions=[caption],
                )
                mentions = first_chair_mentions(
                    captioner.tokenizer,
                    evaluator,
                    evaluations[0],
                    generated,
                )
            except ZeroDivisionError:
                mentions = []
            anchors = [
                prompt_length + int(mention["rel_anchor"])
                for mention in mentions
            ]
            keep = [
                anchor + 1 < full_ids.shape[-1] for anchor in anchors
            ]
            mentions = [
                mention
                for mention, selected_value in zip(mentions, keep)
                if selected_value
            ]
            anchors = [
                anchor
                for anchor, selected_value in zip(anchors, keep)
                if selected_value
            ]
            if mentions:
                features, first_logp = captioner.replay_features(
                    inputs, full_ids, anchors
                )
                for index, mention in enumerate(mentions):
                    values["object_id"].append(
                        int(image_id) * 128 + index
                    )
                    values["image_id"].append(int(image_id))
                    values["token_pos"].append(int(anchors[index]))
                    values["gen_pos"].append(
                        int(mention["rel_anchor"])
                    )
                    values["label"].append(
                        int(bool(mention["hallucinated"]))
                    )
                    values["word"].append(mention["word"][:63])
                    values["no_rope"].append(
                        features[index][..., None]
                    )
                    values["first_logp"].append(
                        float(first_logp[index])
                    )
            output.write(
                json.dumps(
                    {
                        "image_id": int(image_id),
                        "caption": caption,
                        "mentions": [
                            {
                                "word": mention["word"],
                                "rel_anchor": int(
                                    mention["rel_anchor"]
                                ),
                                "hallucinated": bool(
                                    mention["hallucinated"]
                                ),
                            }
                            for mention in mentions
                        ],
                    }
                )
                + "\n"
            )
            output.flush()
            os.fsync(output.fileno())
            atomic_npz(
                npz_path,
                arrays(
                    values,
                    captioner.layer_count,
                    captioner.head_count,
                ),
            )
            del inputs, full_ids
            torch.cuda.empty_cache()
    atomic_npz(
        npz_path,
        arrays(values, captioner.layer_count, captioner.head_count),
    )
    print(
        f"wrote {jsonl_path} and {npz_path}: "
        f"{len(values['label'])} mentions"
    )


if __name__ == "__main__":
    main()
