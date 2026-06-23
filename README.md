# SinkDetect

SinkDetect is a LLaVA-1.5-7B hallucination study with two separately scoped
tracks:

1. **Detection**: token-level object hallucination detection on COCO/CHAIR,
   with attention-shape diagnostics and position-confound controls.
2. **Mitigation**: controlled evaluation of attention interventions on
   POPE/CHAIR, with audits for answer-prior and caption-style shifts.

The code repository and paper repository are intentionally separate. This outer
repository manages code and project docs; `paper_repo/` is ignored here and is
managed by its own Git repository.

## Documentation Map

| Doc | Purpose |
|---|---|
| [docs/project_structure.md](docs/project_structure.md) | current repository layout and active entrypoints |
| [docs/design.md](docs/design.md) | research framing, method design, hypotheses, and open questions |
| [docs/aaai2027_paper_plan.md](docs/aaai2027_paper_plan.md) | paper-facing experiment and writing plan |
| [detection/docs/pipeline.md](detection/docs/pipeline.md) | detection data flow, sharding, cache, and merge contracts |
| [detection/docs/scores.md](detection/docs/scores.md) | exact score names and sign conventions |
| [detection/docs/results_summary.md](detection/docs/results_summary.md) | compact baseline result summary |
| [detection/baselines/README.md](detection/baselines/README.md) | detection baseline set and runner usage |
| [mitigation/README.md](mitigation/README.md) | mitigation track commands and audits |
| [mitigation/docs/evaluation_protocol.md](mitigation/docs/evaluation_protocol.md) | POPE/CHAIR interpretation protocol |

## Repository Layout

```text
SinkDetect/
├── detection/          # detection pipeline, baselines, diagnostics, source
├── mitigation/         # mitigation interventions, runners, audits, source
├── docs/               # project-level design and planning docs
├── paper/              # local paper build artifacts for this repo
├── paper_repo/         # separate Git repo for the paper; ignored by this repo
├── ref/                # local reference papers
└── reviews/            # local review material
```

Generated outputs are ignored by Git, especially `experiments/`,
`detection/baselines/results/`, and `mitigation/results/`.

## Requirements

- LLaVA-1.5-7B HuggingFace checkpoint.
- COCO val2014 images and `annotations/instances_val2014.json`.
- Sibling PAS checkout for CHAIR utilities and cached labels:
  `../pas/data/chair_coco.pkl`.
- GPUs large enough to load one LLaVA-1.5-7B worker per shard.

Set paths through environment variables or runner flags. The legacy defaults in
scripts reflect the local lab filesystem and should be overridden on another
machine.

## Detection Quick Start

Generate captions and run the legacy full-attention detector:

```bash
bash detection/scripts/legacy/run_parallel.sh
```

Run the current attention-row cache path for faster score iteration:

```bash
bash detection/scripts/run_row_cache_parallel.sh
python detection/scripts/recompute_from_row_cache.py \
  --cache experiments/coco_llava_7b_rows/attention_row_cache.npz \
  --ratio 0.5
```

Run controlled/adapted detection baselines:

```bash
bash detection/baselines/run_parallel_baselines.sh
python detection/baselines/analyze_controls.py \
  --result_dir detection/baselines/results/coco_llava_7b_baselines
```

For the full detection flow and output contract, use
[detection/docs/pipeline.md](detection/docs/pipeline.md). For score definitions,
use [detection/docs/scores.md](detection/docs/scores.md).

## Mitigation Quick Start

Run the four-GPU mitigation evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
POPE_DIR=$HOME/common_dataset/pope \
MITIGATION_EXP_NAME=coco_llava_7b_attention_only \
bash mitigation/scripts/run_parallel_mitigation.sh
```

Audit behavioral shifts after generation:

```bash
python mitigation/scripts/audit_results.py \
  --result_root mitigation/results/coco_llava_7b_attention_only
```

Run the attention-routing audit:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
POPE_DIR=$HOME/common_dataset/pope \
ATTN_AUDIT_EXP_NAME=coco_llava_7b_attention_audit \
LIMIT=120 \
bash mitigation/scripts/run_parallel_attention_audit.sh
```

See [mitigation/README.md](mitigation/README.md) for method scope and output
layout.

## Current Interpretation

The project treats high aggregate detection or mitigation scores cautiously.
Detection baselines are evaluated against generation-position controls. Mitigation
methods are evaluated against answer-prior shifts on POPE and caption-style
changes on CHAIR. Paper claims should stay tied to those controlled diagnostics.
