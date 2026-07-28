#!/usr/bin/env python3
"""Extract model-specific video-native allocation features on Vript-HAL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mitigation.src.registered_llava_probe import (
    RegisteredLlavaCaptioner,
)
from mitigation.src.registered_qwen_probe import (
    RegisteredQwen25Captioner,
    RegisteredQwen3Captioner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        required=True,
        choices=["llava_next_video", "qwen25", "qwen3"],
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--manifest",
        default="/tmp/sinkdetect_datasets/VRIPT-HAL/frame_manifest.jsonl",
    )
    parser.add_argument(
        "--annotations",
        default="/tmp/sinkdetect_datasets/VRIPT-HAL/HAL_annotations.jsonl",
    )
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--cosine_threshold", type=float, default=0.5)
    return parser.parse_args()


def records(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [
            json.loads(line) for line in handle if line.strip()
        ]


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


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    import spacy
    from sentence_transformers import SentenceTransformer, util

    manifest = records(args.manifest)
    annotation = {
        row["clip_id"]: row for row in records(args.annotations)
    }
    ordered_ids = sorted(row["clip_id"] for row in manifest)
    index_by_id = {
        clip_id: index for index, clip_id in enumerate(ordered_ids)
    }
    selected = [
        row
        for index, row in enumerate(manifest)
        if index % args.num_shards == args.shard_idx
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    nlp = spacy.load("en_core_web_lg")
    sbert = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    def gt_words(clip_id: str) -> list[str]:
        values = []
        for content in annotation[clip_id]["content"]:
            for token in nlp(content):
                if token.pos_ not in {"VERB", "NOUN", "PROPN"}:
                    continue
                lemma = token.lemma_.lower().strip()
                if lemma and lemma not in nlp.Defaults.stop_words:
                    values.append(lemma)
        return sorted(set(values))

    device = torch.device(args.device)
    if args.kind == "qwen25":
        captioner = RegisteredQwen25Captioner(
            args.model_path,
            device,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.kind == "qwen3":
        captioner = RegisteredQwen3Captioner(
            args.model_path,
            device,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        captioner = RegisteredLlavaCaptioner(
            args.model_path,
            device,
            kind=args.kind,
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
            json.loads(line)["clip_id"]
            for line in jsonl_path.read_text().splitlines()
            if line
        }
    pending = [
        row for row in selected if row["clip_id"] not in done
    ]
    print(
        f"{args.kind} shard {args.shard_idx}/{args.num_shards}: "
        f"{len(pending)} pending"
    )

    with jsonl_path.open("a") as output:
        for row in tqdm(pending, desc=f"native-{args.kind}"):
            clip_id = row["clip_id"]
            inputs = captioner.video_prompt_inputs(
                row["video_path"],
                row["prompt"],
                system=row.get("system"),
            )
            prompt_length = int(inputs["input_ids"].shape[-1])
            full_ids = captioner.generate_ids(inputs)
            generated = full_ids[0, prompt_length:].tolist()
            caption = captioner.caption_text(
                full_ids, prompt_length
            )
            mentions = captioner.noun_mentions(nlp, generated)
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
            clip_index = index_by_id[clip_id]
            labels: list[int] = []
            if mentions:
                ground_truth = gt_words(clip_id)
                ground_embedding = (
                    sbert.encode(
                        ground_truth, convert_to_tensor=True
                    )
                    if ground_truth
                    else None
                )
                words = [mention["word"] for mention in mentions]
                word_embedding = sbert.encode(
                    words, convert_to_tensor=True
                )
                features, first_logp = captioner.replay_features(
                    inputs, full_ids, anchors
                )
                for index, mention in enumerate(mentions):
                    if ground_embedding is None:
                        grounded = False
                    else:
                        cosine = util.cos_sim(
                            ground_embedding, word_embedding[index]
                        ).squeeze(-1)
                        grounded = bool(
                            (cosine > args.cosine_threshold).sum() > 0
                        )
                    label = 0 if grounded else 1
                    labels.append(label)
                    values["object_id"].append(
                        clip_index * 512 + index
                    )
                    values["image_id"].append(clip_index)
                    values["token_pos"].append(int(anchors[index]))
                    values["gen_pos"].append(
                        int(mention["rel_anchor"])
                    )
                    values["label"].append(label)
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
                        "clip_id": clip_id,
                        "caption": caption,
                        "mentions": [
                            {
                                "word": mention["word"],
                                "rel_anchor": int(
                                    mention["rel_anchor"]
                                ),
                                "hallucinated": bool(label),
                            }
                            for mention, label in zip(mentions, labels)
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
