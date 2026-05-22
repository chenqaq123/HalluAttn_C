"""
CHAIR evaluation and token-level hallucination labeling.

Adapted from PAS's src/pas/evaluate/chair.py and src/pas/utils/coco.py.
Provides CHAIR evaluation, token-level object mention labeling, and
first-mention extraction for AUROC-based detection evaluation.
"""

import json
import os
import pickle
import re
import tempfile
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm.auto import tqdm


# ─── CHAIR Evaluator Loading ─────────────────────────────────────────────────

def load_chair_evaluator(chair_pkl_path: str):
    """Load a pre-built CHAIR evaluator from a pickle file.

    The pkl file contains a CHAIR object with MSCOCO object lists,
    synonym dictionaries, and COCO ground-truth annotations.
    """
    with open(chair_pkl_path, "rb") as f:
        evaluator = pickle.load(f)
    return evaluator


def evaluate_chair(
    evaluator,
    data: Optional[list[dict]] = None,
    image_ids: Optional[list[int]] = None,
    captions: Optional[list[str]] = None,
    json_path: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Evaluate COCO captions using CHAIR.

    Either `data` or both `image_ids` and `captions` must be provided.

    If ``json_path`` is None, a unique temp file is used so that concurrent
    shards/runs never write to the same file (previously hard-coded to
    ``/tmp/sinkdetect_chair.json``, which corrupted under parallelism).
    The file is written atomically (write to ``<path>.tmp`` then ``os.replace``).

    Returns:
        (eval_dicts, overall_metrics)
        eval_dicts: list of per-sentence CHAIR results
        overall_metrics: dict with CHAIRi, CHAIRs, etc.
    """
    if data is None and (image_ids is None or captions is None):
        raise ValueError("Either `data` or both `image_ids` and `captions` must be provided.")

    if data is None:
        data = [
            {"image_id": iid, "caption": cap}
            for iid, cap in zip(image_ids, captions)
        ]

    if json_path is None:
        fd, json_path = tempfile.mkstemp(prefix="sinkdetect_chair_", suffix=".json")
        os.close(fd)

    tmp_path = f"{json_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, json_path)

    chair_dicts = evaluator.compute_chair(json_path, "image_id", "caption")
    return chair_dicts["sentences"], chair_dicts["overall_metrics"]


# ─── Token-Level Labeling ────────────────────────────────────────────────────

def find_all_word_positions(
    output_tokens: Sequence[str],
    words: Sequence[str],
) -> list[int]:
    """Find the starting token index for each word in an ordered list.

    Performs whole-word search (won't match "ball" inside "baseball").
    Words must appear in the same relative order in the text.
    """
    if not words:
        return []

    full_text = "".join(output_tokens)
    token_start_chars = []
    current_pos = 0
    for token in output_tokens:
        token_start_chars.append(current_pos)
        current_pos += len(token)

    start_positions = []
    search_offset = 0

    for word in words:
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
        match = pattern.search(full_text, pos=search_offset)
        if not match:
            raise ValueError(f"Word '{word}' not found in sequence after previous word.")
        match_start_char = match.start()
        token_index = bisect_right(token_start_chars, match_start_char) - 1
        start_positions.append(token_index)
        search_offset = match.end()

    return start_positions


def add_token_labels(
    eval_dicts: list[dict],
    sequences: Sequence[Sequence[int]],
    processor,
    evaluator,
) -> list[dict]:
    """Annotate each eval dict with per-object token positions and hallucination labels.

    Modifies eval_dicts in place, adding an ``object_mentions`` key to each entry
    that records the token index, word, and hallucination status of each detected
    COCO object.

    Args:
        eval_dicts: Output from evaluate_chair().
        sequences: Token ID sequences (one per eval dict), typically the
                   generated portion of output_ids.
        processor: LlavaProcessor instance.
        evaluator: CHAIR evaluator instance.

    Returns:
        The modified eval_dicts (same objects, modified in place).
    """
    from pas.evaluate.chair import caption_to_words

    for eval_info, sequence in zip(tqdm(eval_dicts, desc="Labeling tokens"), sequences):
        output_tokens = processor.tokenizer.convert_ids_to_tokens(sequence)

        SPECIAL_SPACE_CHAR = processor.tokenizer.tokenize("Please")[0][0]

        non_hallu_words = set(eval_info["mscoco_gt_words"])

        _, _, _, double_words, original_words = caption_to_words(evaluator, eval_info["caption"])

        words_to_find, labels = [], []
        word_index = 0
        for word in eval_info["mscoco_generated_words"]:
            while word_index < len(double_words) and evaluator.inverse_synonym_dict.get(
                double_words[word_index], None
            ) != word:
                word_index += 1
            if word_index < len(double_words):
                org_word = original_words[word_index].replace(" ", SPECIAL_SPACE_CHAR)
                words_to_find.append(org_word)
                labels.append(False if word in non_hallu_words else True)
                word_index += 1
            else:
                raise ValueError(f"Word '{word}' not found in {double_words}")

        word_token_idxs = find_all_word_positions(output_tokens, words_to_find)
        object_mentions = [
            {
                "word": word,
                "hallucinated": label,
                "token_idx": token_idx - 1,  # position before the word starts
            }
            for word, label, token_idx in zip(
                eval_info["mscoco_generated_words"], labels, word_token_idxs
            )
        ]

        eval_info["object_mentions"] = object_mentions

    return eval_dicts


def find_first_mentions(eval_dict: dict) -> list[dict]:
    """Find the first mention of each unique object word.

    Args:
        eval_dict: A single entry from eval_dicts with 'object_mentions'.

    Returns:
        List of dicts with {word, pos, hallucinated, prev_word, prev_pos}.
    """
    object_mentions = eval_dict.get("object_mentions", [])
    mentioned_words = set()
    results = []
    prev_word, prev_pos = None, None

    for mention in object_mentions:
        if mention["word"] not in mentioned_words:
            mentioned_words.add(mention["word"])
            results.append(
                {
                    "word": mention["word"],
                    "pos": mention["token_idx"],
                    "hallucinated": mention["hallucinated"],
                    "prev_word": prev_word,
                    "prev_pos": prev_pos,
                }
            )
        prev_word, prev_pos = mention["word"], mention["token_idx"]

    return results
