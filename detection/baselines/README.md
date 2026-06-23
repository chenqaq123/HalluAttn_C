# Detection Baselines

This directory contains the paper-facing token-level hallucination detection
baselines. The runner evaluates controlled/adapted implementations on the same
generated captions and CHAIR-derived object labels used by SinkDetect.

## Baseline Set

| Baseline | Category | Main score used for comparison |
|---|---|---|
| NLL | Logit / uncertainty | negative log-likelihood of the object token |
| Entropy | Logit / uncertainty | entropy of the next-token distribution |
| IC | Representation | internal confidence / image-object compatibility |
| GLSim | Representation | global-local visual/semantic similarity |
| SVAR | Global attention | summed visual attention ratio |
| PAS | Prelim attention | layer-0 Prelim Attention Score |
| Beyond-ADS | Fine-grained patch grounding | attention dispersion score |
| Beyond-CGC | Fine-grained patch grounding | cross-modal grounding consistency |
| Beyond-ADS+CGC | Fine-grained patch grounding | combined ADS+CGC detector |

PAS/SVAR are attention-mass baselines. IC/GLSim are representation baselines.
NLL/Entropy are logit uncertainty baselines. Beyond-ADS/CGC test finer patch
localization and semantic consistency assumptions.

## Reporting Rules

1. Use each paper's final detector as the main comparison score.
2. Report overall AUROC for comparability with prior work.
3. Report position-controlled metrics for this project: within-bin AUROC,
   matched-pair AUROC, and residual AUROC.
4. Keep all methods on the same generated captions and CHAIR object labels.
5. Report Beyond-ADS, Beyond-CGC, and Beyond-ADS+CGC separately because their
   components test different grounding assumptions.

Compact current results live in
[`detection/docs/results_summary.md`](../docs/results_summary.md). Avoid copying
large result tables back into this file. NLL, Entropy, PAS, and SVAR are direct
score implementations; IC, GLSim, and Beyond-ADS/CGC are adapted/paper-level
implementations and should be described that way in paper text.

## Unified Runner

Default inputs match the current SinkDetect experiment:

- `experiments/coco_llava_7b/generation.json`
- `experiments/coco_llava_7b_rows/attention_row_cache.npz`
- `experiments/coco_llava_7b_rows/row_cache_scores.npz`

Default outputs go to `detection/baselines/results/coco_llava_7b/`:

- `object_cache.jsonl`
- `baseline_scores.npz`
- `baseline_scores.csv`
- `metrics.json`
- `run_config.json`

Smoke test without loading the VLM:

```bash
python detection/baselines/run_all_baselines.py \
  --skip_model_baselines \
  --limit 20 \
  --output_dir detection/baselines/results/smoke_skip_model
```

Full one-GPU run; by default this aborts if the teacher-forced input does not
match the original `generation.json::output_ids` at every evaluated object
position:

```bash
python detection/baselines/run_all_baselines.py --device 0
```

Multi-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
BASELINE_EXP_NAME=coco_llava_7b_baselines \
bash detection/baselines/run_parallel_baselines.sh
```

The parallel runner writes shard outputs under
`detection/baselines/results/<BASELINE_EXP_NAME>/shard{i}/`, then merges scores
into `detection/baselines/results/<BASELINE_EXP_NAME>/`. Sharding is by image,
so all object mentions from the same image stay on one worker.

## Control Analysis

After a baseline run has produced `object_cache.jsonl` and
`baseline_scores.npz`, run:

```bash
python detection/baselines/analyze_controls.py \
  --result_dir detection/baselines/results/coco_llava_7b_baselines
```

It writes:

- `controlled_analysis/controlled_metrics.json`
- `controlled_analysis/controlled_summary.csv`
- `controlled_analysis/position_drift.csv`

The same-word matched metric is the main object-category control: it asks
whether a score can distinguish grounded vs. hallucinated mentions of the same
object class at nearby generation positions.

## Implementation Notes

- PAS formulas follow the sibling `../pas` checkout, especially
  `../pas/scripts/compute_scores.py`.
- GLSim follows the official `deeplearning-wisc/glsim` formulation at a high
  level, adapted to teacher-forced captions already generated in this project.
- Beyond Global Scores currently has no official implementation available in
  the checked sources, so this runner marks ADS+CGC as
  `paper_reimplementation` in `run_config.json`.
