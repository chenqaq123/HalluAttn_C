# SinkDetect — Sink-Purified Attention for Hallucination Detection

> 📑 **Docs**: design rationale in [docs/design.md](docs/design.md);
> per-score reference in [docs/scores.md](docs/scores.md); pipeline
> walk-through in [docs/pipeline.md](docs/pipeline.md).

Token-level hallucination detection for LVLMs (LLaVA-1.5-7B) by mining the
shape of the attention distribution over image tokens. Active signals:

1. **Shape scores** — CVG, concentration, and cross-layer consistency over
   object-token visual attention distributions.
2. **Ablations** — raw, sink-only, top-mass-only, purified, and optional
   no-RoPE branches.
3. **No-RoPE branch** — when enabled, recompute attention weights from
   pre-RoPE Q/K to test whether removing positional rotation improves the
   shape metrics.

The pipeline mirrors PAS (CVPR 2026) for apples-to-apples comparison on
COCO val2014 with CHAIR labels.

## Layout

```
SinkDetect/
├── scripts/
│   ├── caption.py     # Stage 1: vanilla LLaVA-1.5 greedy captioning
│   ├── detect.py      # Stage 2: forward pass + score + AUROC
│   └── run.sh         # One-click runner (Stage 1 → Stage 2)
├── src/sinkdetect/
│   ├── adapter.py     # DetectionAdapter — drop-in LlamaAttention replacement
│   ├── sink_utils.py  # sink detection + attention purification
│   ├── scoring.py     # per-mention score computation + AUROC
│   ├── chair.py       # CHAIR evaluation + token-level labeling
│   └── utils.py       # model loading, token partitioning, prompts
└── experiments/<exp>/ # outputs: generation.json, metrics.json, raw_scores.npz
```

## Requirements

- Sibling `pas/` project (we import `pas.evaluate.chair.caption_to_words` and
  reuse the cached `pas/data/chair_coco.pkl`).
- LLaVA-1.5-7B HuggingFace checkpoint.
- COCO val2014 with `annotations/instances_val2014.json` and the
  `val2014/COCO_val2014_*.jpg` images.

## One-click run

```bash
# Single GPU (logical index 0, with CUDA_VISIBLE_DEVICES locked to 0-3)
bash scripts/run.sh

# 4-way data parallel on GPUs 0,1,2,3
bash scripts/run_parallel.sh
```

Both runners set `CUDA_VISIBLE_DEVICES=0,1,2,3` by default so the pipeline can
never touch GPUs 4-7. Outputs go to `experiments/coco_llava_7b/`.

### Parallel mode details

`run_parallel.sh` launches `NUM_SHARDS` (default 4) worker processes, each
loading its own copy of LLaVA-1.5-7B onto one logical GPU. Each worker takes a
deterministic stride (`data[shard_idx::num_shards]`) of the input list. After
all workers finish, `scripts/merge_shards.py` concatenates results and
recomputes AUROC on the pooled set.

Per-shard logs land in `experiments/<exp>/logs/stage{1,2}_shard{i}.log`.
If any worker fails, the runner prints the last 30 lines of its log and aborts.

```bash
# Smoke test on small data
NUM_SAMPLES=200 bash scripts/run_parallel.sh

# Sweep ratio
for r in 0.3 0.5 0.7; do
  EXP_NAME=coco_llava_7b_r${r/./} RATIO=$r bash scripts/run_parallel.sh
done

# Use only 2 of the 4 GPUs (e.g. if 2,3 are occupied)
CUDA_VISIBLE_DEVICES=0,1 NUM_SHARDS=2 bash scripts/run_parallel.sh
```

### Common overrides

```bash
# small smoke test
NUM_SAMPLES=200 LIMIT=200 bash scripts/run.sh

# sweep the purification ratio
EXP_NAME=coco_llava_7b_r03 RATIO=0.3 bash scripts/run.sh

# custom paths
MODEL_PATH=/path/to/llava-1.5-7b-hf \
COCO_PATH=/path/to/coco-2014 \
CHAIR_PKL=/path/to/chair_coco.pkl \
bash scripts/run.sh

# regenerate captions even if generation.json exists
FORCE_REGEN=1 bash scripts/run.sh

# cache raw + no-RoPE branches for shape-only analysis
SAVE_SHAPE_CACHE=1 COMPUTE_NO_ROPE_ATTENTION=1 bash scripts/run_parallel.sh
```

All knobs (with defaults):

| var | default | meaning |
|---|---|---|
| `EXP_NAME` | `coco_llava_7b` | output subdir under `experiments/` |
| `MODEL_PATH` | LLaVA-1.5-7B HF cache path | local HF model dir |
| `COCO_PATH` | `/data/common_dataset/coco-2014-dataset/` | COCO root |
| `CHAIR_PKL` | `../pas/data/chair_coco.pkl` | pre-built CHAIR pickle |
| `NUM_SAMPLES` | 5000 | images for stage 1 |
| `MAX_NEW_TOKENS` | 512 | caption length cap |
| `RATIO` | 0.5 | top-mass kept on visual tokens during purify |
| `START_LAYER` / `END_LAYER` | 0 / 32 | which decoder layers to instrument |
| `LIMIT` | 0 (all) | cap detect-stage iterations |
| `DEVICE` | 0 | CUDA device index |
| `SEED` | 42 | sampling seed for stage 1 |
| `SAVE_SHAPE_CACHE` | 0 | save `shape_cache.npz` for fast CVG/Concentration/CLC recomputation |
| `COMPUTE_NO_ROPE_ATTENTION` | 0 | also score/cache no-RoPE attention branches from pre-RoPE Q/K |

## Manual two-stage usage

```bash
# Stage 1
python scripts/caption.py \
    --model_path /path/to/llava-1.5-7b-hf \
    --coco_path  /path/to/coco-2014 \
    --output_path experiments/coco_llava_7b/generation.json \
    --num_samples 5000

# Stage 2
python scripts/detect.py \
    --model_path /path/to/llava-1.5-7b-hf \
    --coco_path  /path/to/coco-2014 \
    --generation_json experiments/coco_llava_7b/generation.json \
    --output_dir experiments/coco_llava_7b \
    --chair_pkl  ../pas/data/chair_coco.pkl \
    --ratio 0.5
```

## Counterfactual Visual Grounding (CVG)

SinkDetect now focuses on the **shape** of the object query's visual attention
distribution relative to a content-independent null. PAS-style attention-mass
scores are treated as external baselines and are not emitted by current runs.
Three active sub-families:

| family | per-layer keys (also `global_*`) | high score ⇒ hallucination |
|---|---|---|
| `cvg_*` | `cvg_kl_instr_layer_{l}`, `cvg_kl_uniform_layer_{l}`, `cvg_jsd_instr_layer_{l}` | object attention close to the null (instruction-token average) ⇒ no extra grounding effort |
| `conc_*` | `conc_entropy_layer_{l}`, `conc_top{1,5,10}_mass_layer_{l}`, `conc_max_over_mean_layer_{l}` | diffuse visual attention (high entropy / low top-k mass) ⇒ no specific region drives the prediction |
| `clc_*` | `clc_gen_jsd`, `clc_mean_pairwise_jsd`, `clc_gen_jsd_midlate`, `clc_argmax_agree` | cross-layer attention shapes disagree ⇒ no coherent grounding target |

All distributions are **sink-stripped then renormalized** before scoring, so
massive-activation sinks cannot dominate either the test distribution or the
null. See [src/sinkdetect/grounding.py](src/sinkdetect/grounding.py) for the
exact formulas.

To iterate on shape metrics without another model forward pass, run detection
with caching:

```bash
SAVE_SHAPE_CACHE=1 bash scripts/run_parallel.sh
python scripts/recompute_shape_from_cache.py \
  --cache experiments/coco_llava_7b/shape_cache.npz
```

The cache stores per-mention object→visual rows, instruction-null→visual rows,
sink masks, and labels for `orig`, `sink_only`, `topmass_only`, and `purified`
branches. It is meant for changing scoring formulas and AUROC/fusion analysis;
changing the top-mass ratio or sink detector still requires rerunning Stage 2.

The instruction-token null is free: it's just the row-average of the same
attention matrix over instruction-token query positions. Since the
instruction string is fixed across all images, that row-average carries no
content from any specific object word — it's an empirical estimate of
sink + RoPE + generic-prompt bias for this image.

## What the scores mean

Per-mention scores are collected in `raw_scores.npz`. The active families:

- `cvg_*`, `conc_*`, `clc_*` — shape scores on raw attention.
- `sink_only_cvg_*`, `sink_only_conc_*`, `sink_only_clc_*` — same shape scores
  after sink removal but without visual top-mass masking.
- `topmass_only_cvg_*`, `topmass_only_conc_*`, `topmass_only_clc_*` — same
  shape scores after visual top-mass masking but before sink removal.
- `purified_cvg_*`, `purified_conc_*`, `purified_clc_*` — same shape scores
  after sink removal + visual top-mass masking.
- `no_rope_cvg_*`, `no_rope_conc_*`, `no_rope_clc_*` — same shape scores on
  attention recomputed from pre-RoPE Q/K. When enabled, corresponding
  `no_rope_sink_only_*`, `no_rope_topmass_only_*`, and `no_rope_purified_*`
  branches are also emitted.

A label of `1` = hallucinated mention (CHAIR), `0` = grounded. `metrics.json`
sorts AUROCs descending so the top entries are the strongest detectors in your
run.

## Implementation notes

- The LLaVA-1.5 prompt includes a system prefix, so image tokens do **not** sit
  at positions 1–576. `sink_utils.find_vis_bounds` locates them dynamically per
  sample and `_set_adapter_bounds` pushes the result into every layer adapter
  before the forward pass.
- `DetectionAdapter` is a drop-in `LlamaAttention` replacement — it copies the
  pretrained projection weights so the LM forward pass is bit-identical to the
  unmodified model. The purified attention is computed off the standard softmax
  output and stored on the adapter; nothing is fed back into the residual.
- Token labels are computed from a re-encoding of the caption (not from the raw
  `output_ids`) to match PAS exactly. This avoids "image"/"person" in the
  instruction text colliding with CHAIR object names.
