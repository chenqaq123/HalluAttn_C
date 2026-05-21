# Pipeline

The data flow from a raw COCO image to a row in `metrics.json::roc_auc`.
For the *why* behind each score, see [design.md](design.md); for the *what*
of each score, see [scores.md](scores.md).

## Stage 1 — Captioning

`scripts/caption.py`, parallelised by `scripts/run_parallel.sh`.

1. Load `num_samples` images from COCO val2014 (deterministic by seed).
2. For each image, render the PAS-style chat template:
   ```
   {system} USER: <image>\n{user_instruction} ASSISTANT:
   ```
   where `{system}` = LLaVA-1.5 chat system prompt and `{user_instruction}`
   = "Please help me describe the image in detail.".
3. Greedy `model.generate(max_new_tokens=512)` with LLaVA-1.5-7B in `fp16`.
4. Record `{image_id, caption, output_ids, prompt_end_idx}` per image.

Output: `experiments/<exp>/generation_shard{i}.json` per shard, then merged
into `generation.json` by `scripts/merge_shards.py --mode caption`.

The 4 shards split `data[shard_idx::num_shards]`, each on its own GPU.

## Stage 2 — Detection

`scripts/detect.py`, parallelised by `scripts/run_parallel.sh`.

For each image:

1. **CHAIR evaluation** (once, on the full shard subset). Runs PAS's CHAIR
   evaluator to get `mscoco_generated_words` and `mscoco_gt_words` per
   caption.
2. **Token-level labelling**. For each generated COCO-object word, locate
   its starting token in `tokenizer.encode(caption)` (the *re-encoded*
   caption, not the raw `output_ids` — matches PAS exactly). Record
   `(word, pos = token_idx − 1, hallucinated)`. We then keep only the
   first mention of each unique object word.
3. **Forward pass** with `DetectionAdapter` injected into every layer's
   `self_attn`. Adapter:
   - Computes pre-RoPE key L2 norms; detects sink positions per layer via
     `mean − 1.5·std` thresholding on the visual span, clamped to `[3, 30]`.
   - Returns the original softmax attention.
   - In the same forward, stores three purification variants:
     `sink_only_*` zeroes sink columns but skips top-mass masking;
     `topmass_only_*` applies top-mass masking but keeps sinks;
     `purified_*` applies both.
   - Stores both on the layer for post-hoc extraction.
4. **Dynamic visual bounds**. `vis_start, vis_end` are found per image by
   scanning `input_ids` for runs of the image-token id. Pushed into every
   layer's adapter before the forward (`_set_adapter_bounds`).
5. **Score computation**. For each object mention's `token_pos`:
   - **CVG / Concentration / CLC family** on `A`, `A^S`, `A^T`, and `A'`
     (see [design.md §5](design.md#5-scoring-families)).
   - If `--compute_no_rope_attention` is enabled, the same shape scores on
     `no_rope_*` branches recomputed from pre-RoPE Q/K.
   - Optional per-head and cross-image null variants.
   - PAS-family scalar attention-mass scores are not emitted by current runs;
     PAS is treated as an external baseline.
6. AUROC over `(scores, labels)` per shard → `metrics_shard{i}.json`.
   Per-mention scores → `raw_scores_shard{i}.npz`.

Output (per shard): `metrics_shard{i}.json`, `raw_scores_shard{i}.npz`.
If `--save_shape_cache` is enabled, each shard also writes
`shape_cache_shard{i}.npz`, containing the per-mention visual rows needed to
recompute CVG / Concentration / CLC without another model forward pass.

## Merge

`scripts/merge_shards.py --mode detect`:

- Concatenates all per-shard `raw_scores_shard{i}.npz` arrays.
- If present, concatenates all per-shard `shape_cache_shard{i}.npz` arrays into
  `shape_cache.npz`.
- Re-computes AUROC on the **pooled** data (this is the headline number,
  not the per-shard AUROC).
- Aggregates CHAIRi across shards by weighted sum on object counts.
- Writes `metrics.json` and `raw_scores.npz`.

## What lives where

```
SinkDetect/
├── scripts/
│   ├── caption.py            # Stage 1 worker
│   ├── detect.py             # Stage 2 worker
│   ├── merge_shards.py       # Cross-shard merge + global AUROC
│   ├── run.sh                # Single-GPU runner (logical GPU 0-3)
│   └── run_parallel.sh       # 4-way data-parallel runner
├── src/sinkdetect/
│   ├── adapter.py            # DetectionAdapter (drop-in LlamaAttention)
│   ├── sink_utils.py         # auto_detect_sinks + purify_attention
│   ├── grounding.py          # CVG / Concentration / CLC primitives
│   ├── scoring.py            # compute_all_scores (all score families)
│   ├── chair.py              # CHAIR loading + token-level labelling
│   └── utils.py              # model loading, partition_tokens, prompt
├── experiments/<exp>/
│   ├── logs/                 # per-shard stage1/2 logs
│   ├── generation.json       # captions (merged)
│   ├── generation_shard{i}.json
│   ├── metrics.json          # AUROC table on pooled data
│   ├── metrics_shard{i}.json
│   ├── raw_scores.npz        # per-mention scores (merged)
│   ├── raw_scores_shard{i}.npz
│   ├── shape_cache.npz       # optional: cached rows for fast shape-score recompute
│   └── shape_cache_shard{i}.npz
└── docs/                     # this directory
```

## Shape-score cache

Use `SAVE_SHAPE_CACHE=1 bash scripts/run_parallel.sh` or pass
`--save_shape_cache` to `scripts/detect.py` to store the cache. Recompute shape
metrics later with:

```bash
python scripts/recompute_shape_from_cache.py \
  --cache experiments/<exp>/shape_cache.npz
```

This is useful for changing CVG signs, KL/JSD variants, concentration formulas,
CLC aggregation, and fusion analysis. It does **not** avoid rerunning Stage 2
when changing the top-mass ratio, sink detector, prompt, or model, because those
change the cached rows themselves.

## Sharding contract

- Stage 1 shards split `coco_data[i::N]` after deterministic sampling with
  the seed. Same seed → same global set → reproducible.
- Stage 2 shards split `gen_data[i::N]` over the merged caption list.
- Output filenames carry a `_shard{i}` suffix when `num_shards > 1` so the
  merge step can pick them up.
- Per-shard AUROCs in `metrics_shard{i}.json` are **debug-only**. The
  headline AUROC is in `metrics.json` after merge.
