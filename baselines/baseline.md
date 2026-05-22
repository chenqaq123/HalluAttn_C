# Baselines for Token-Level LVLM Object Hallucination Detection

This file lists the baselines we should compare against in the paper-facing
experiments. When a paper proposes multiple analyses or ablations, we keep only
the final method used as its main experimental detector.

## Baseline Set

| Baseline | Category | Main score used for comparison | Notes |
|---|---|---|---|
| NLL | Logit / uncertainty | Negative log-likelihood of the object token | Higher uncertainty implies higher hallucination risk. |
| Entropy | Logit / uncertainty | Entropy of the next-token distribution at the object step | Training-free and does not use hidden states or attention. |
| IC | Representation | Internal confidence / image-object compatibility from model representations | Representation-based baseline used in PAS. |
| GLSim | Representation | Global-local similarity between visual representations and object semantics | Representation-based image-object compatibility baseline. |
| SVAR | Global attention | Summed Visual Attention Ratio from object token to image tokens | Strong global image-attention baseline. |
| PAS | Prelim attention | Layer-0 Prelim Attention Score | Final PAS detector; do not separately list PAS ablations such as instruction/image/BOS attention or MI variants. |
| Beyond-ADS | Fine-grained patch grounding | Attention Dispersion Score | Tests whether object-to-patch attention is diffuse. |
| Beyond-CGC | Fine-grained patch grounding | Cross-modal Grounding Consistency | Tests whether object and attended patch representations are semantically aligned. |
| Beyond-ADS+CGC | Fine-grained patch grounding | Combined ADS+CGC detector | Final combined detector; report alongside ADS and CGC to diagnose which component contributes. |

## Grouping

### Logit-Level Baselines

- **NLL**: uses the model's likelihood for the generated object token.
- **Entropy**: uses uncertainty of the output distribution at the object-token
  step.

These are cheap and broadly applicable, but they can miss confident
hallucinations.

### Representation-Level Baselines

- **IC**: estimates whether internal image representations support the object.
- **GLSim**: compares global/local visual representations against object
  semantics.

These use hidden representations rather than attention mass alone.

### Attention-Level Baselines

- **SVAR**: global image-token attention. It asks how much the object token
  attends to the image as a whole.
- **PAS**: prelim-token attention. It asks how much the object token depends on
  previously generated tokens; higher prelim dependence indicates higher
  hallucination risk.

PAS is the primary baseline for our project because it is the strongest recent
training-free attention-based detector and is directly comparable to our
attention-row pipeline.

### Fine-Grained Grounding Baselines

- **Beyond-ADS**: attention dispersion, testing whether object-to-patch
  attention is localized or diffuse.
- **Beyond-CGC**: cross-modal grounding consistency, testing whether the object
  representation aligns with the attended patch representations.
- **Beyond-ADS+CGC**: combined detector using both signals.

We report all three scores. ADS and CGC are useful diagnostic baselines because
they reveal whether the combined detector is helped by attention localization,
representation consistency, or both.

## Reporting Rules

1. Use each paper's final detector as the main baseline when the paper reports
   one.
2. For Beyond Global Scores, report ADS, CGC, and ADS+CGC separately because
   the two components test different grounding assumptions.
3. Report overall AUROC for comparability with prior work.
4. Also report position-controlled metrics for our analysis:
   - position-only AUROC;
   - within-bin AUROC;
   - matched-pair AUROC;
   - residual AUROC.
5. When possible, evaluate all baselines on the same generated captions and
   CHAIR-derived object labels.

## Unified Runner

The v1 reproduction runner lives at:

```bash
python baselines/run_all_baselines.py
```

Default inputs match the current SinkDetect experiment:

- `experiments/coco_llava_7b/generation.json`
- `experiments/coco_llava_7b_rows/attention_row_cache.npz`
- `experiments/coco_llava_7b_rows/row_cache_scores.npz`

Outputs are written to `baselines/results/coco_llava_7b/`:

- `object_cache.jsonl`
- `baseline_scores.npz`
- `baseline_scores.csv`
- `metrics.json`
- `run_config.json`

Smoke test without loading the VLM:

```bash
python baselines/run_all_baselines.py \
  --skip_model_baselines \
  --limit 20 \
  --output_dir baselines/results/smoke_skip_model
```

Full baseline run on one GPU:

```bash
python baselines/run_all_baselines.py --device 0
```

Multi-GPU parallel run:

```bash
bash baselines/run_parallel_baselines.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
BASELINE_EXP_NAME=coco_llava_7b_baselines \
bash baselines/run_parallel_baselines.sh
```

For eight visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_SHARDS=8 \
BASELINE_EXP_NAME=coco_llava_7b_baselines \
bash baselines/run_parallel_baselines.sh
```

The parallel runner writes each shard under
`baselines/results/<BASELINE_EXP_NAME>/shard{i}/`, then merges all object-level
scores into `baselines/results/<BASELINE_EXP_NAME>/`. By default it shards by
image, so all object mentions from the same image stay on one GPU and the model
does not repeat the same image forward on multiple cards.

Implementation notes:

- PAS formulas follow the local sibling repo `../pas`, especially
  `scripts/compute_scores.py`.
- GLSim formulas follow the official repository
  `deeplearning-wisc/glsim`; v1 adapts its global-local cosine computation to
  the teacher-forced captions already generated in this project.
- Beyond Global Scores currently has no official implementation available in
  the checked sources, so v1 marks ADS+CGC as `paper_reimplementation` in
  `run_config.json`.

## Primary References

- PAS: *Prelim Attention Score for Detecting Object Hallucinations in Large
  Vision-Language Models*, CVPR 2026.
- Beyond Global Scores: *Beyond the Global Scores: Fine-Grained Token
  Grounding as a Robust Detector of LVLM Hallucinations*, CVPR 2026.
