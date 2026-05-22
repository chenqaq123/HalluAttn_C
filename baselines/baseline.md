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

---

## Experimental Results

**Setup**: LLaVA-1.5-7B, COCO val2014, 16 426 object mentions (4 009
hallucinated / 12 417 grounded). Position-only AUROC = 0.8304.

All baselines evaluated on the same generated captions and CHAIR-derived
labels. Bin width = 10 generated-token positions; bins with < 10 samples or
single-class labels are excluded from within-bin averaging.

### Summary Table (grouped by category, chronological within group)

Methods are grouped by signal type and ordered by proposal date within each
group. **Bold** marks the best value per column across all methods.

**Logit / uncertainty** (classical baselines, predate the others)

| Method | Proposed | Overall | Within-bin | Matched-pair | Residual |
|---|---|---|---|---|---|
| NLL | classical | 0.7114 | 0.6360 | 0.6524 | 0.6297 |
| Entropy | classical | 0.7206 | 0.6366 | 0.6545 | **0.6335** |

**Representation**

| Method | Proposed | Overall | Within-bin | Matched-pair | Residual |
|---|---|---|---|---|---|
| IC | 2024-10 (arXiv 2410.02762) | 0.7761 | **0.6856** | **0.7029** | 0.6331 |
| GLSim | 2025-08 (NeurIPS 2025) | 0.7230 | 0.5412 | 0.5564 | 0.5651 |
| GLSim (global) | 2025-08 (NeurIPS 2025) | 0.6198 | 0.5063 | 0.5049 | 0.5111 |
| GLSim (local) | 2025-08 (NeurIPS 2025) | 0.7724 | 0.5636 | 0.5931 | 0.6006 |

**Attention**

| Method | Proposed | Overall | Within-bin | Matched-pair | Residual |
|---|---|---|---|---|---|
| SVAR | 2024-11 (arXiv 2411.16724) | 0.8336 | 0.5748 | 0.5759 | 0.5570 |
| PAS | 2025-11 (CVPR 2026) | **0.8349** | 0.5928 | 0.5833 | 0.5694 |

**Fine-grained patch grounding** (Beyond Global Scores, CVPR 2026)

| Method | Proposed | Overall | Within-bin | Matched-pair | Residual |
|---|---|---|---|---|---|
| Beyond-ADS | 2026 (CVPR 2026) | 0.5033 | 0.5109 | 0.5148 | 0.5012 |
| Beyond-CGC | 2026 (CVPR 2026) | 0.6917 | 0.5208 | 0.5196 | 0.5181 |
| Beyond-ADS+CGC | 2026 (CVPR 2026) | 0.6926 | 0.5226 | 0.5235 | 0.5212 |

### Per-Bin Within-Bin AUROC

Each cell is the AUROC computed within a single generation-position bin (bin
width = 10). N/A = fewer than 2 classes in the bin.

| Bin | N | H | Rate | NLL | Entropy | IC | GLSim | GLSim-g | GLSim-l | SVAR | PAS | B-ADS | B-CGC | B-A+C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–9 | 4595 | 100 | 0.022 | 0.5611 | 0.5568 | 0.5987 | 0.5123 | 0.5653 | 0.4475 | 0.5672 | 0.6229 | 0.4934 | 0.5588 | 0.5477 |
| 10–19 | 2577 | 158 | 0.061 | 0.6760 | 0.6782 | 0.7586 | 0.4120 | 0.3999 | 0.4546 | 0.5576 | 0.5683 | 0.5190 | 0.4193 | 0.4237 |
| 20–29 | 1113 | 145 | 0.130 | 0.6109 | 0.6139 | 0.7284 | 0.4530 | 0.3924 | 0.5402 | 0.5867 | 0.5924 | 0.5241 | 0.4903 | 0.5035 |
| 30–39 | 901 | 215 | 0.239 | 0.6419 | 0.6311 | 0.7000 | 0.5587 | 0.5023 | 0.6049 | 0.5716 | 0.5686 | 0.5359 | 0.4934 | 0.5066 |
| 40–49 | 878 | 215 | 0.245 | 0.6412 | 0.6328 | 0.7302 | 0.6299 | 0.5326 | 0.6842 | 0.5833 | 0.5775 | 0.5050 | 0.5709 | 0.5724 |
| 50–59 | 882 | 312 | 0.354 | 0.6309 | 0.6346 | 0.7108 | 0.6332 | 0.5452 | 0.6829 | 0.5558 | 0.5589 | 0.4933 | 0.5644 | 0.5636 |
| 60–69 | 997 | 428 | 0.429 | 0.7009 | 0.7151 | 0.6922 | 0.6199 | 0.5238 | 0.6756 | 0.5719 | 0.5764 | 0.5084 | 0.5448 | 0.5517 |
| 70–79 | 1186 | 559 | 0.471 | 0.6969 | 0.6951 | 0.7315 | 0.6492 | 0.5482 | 0.7072 | 0.6088 | 0.6063 | 0.5303 | 0.5481 | 0.5558 |
| 80–89 | 1073 | 573 | 0.534 | 0.6671 | 0.6782 | 0.6969 | 0.6087 | 0.5261 | 0.6635 | 0.5861 | 0.5891 | 0.5282 | 0.5263 | 0.5372 |
| 90–99 | 833 | 471 | 0.565 | 0.7057 | 0.6997 | 0.7051 | 0.6142 | 0.5139 | 0.6780 | 0.5954 | 0.5968 | 0.5165 | 0.5329 | 0.5413 |
| 100–109 | 649 | 399 | 0.615 | 0.6514 | 0.6703 | 0.7001 | 0.5839 | 0.4879 | 0.6628 | 0.5938 | 0.5944 | 0.4786 | 0.5300 | 0.5230 |
| 110–119 | 357 | 206 | 0.577 | 0.6877 | 0.6894 | 0.6774 | 0.5652 | 0.4991 | 0.6112 | 0.5855 | 0.5940 | 0.5494 | 0.4946 | 0.5108 |
| 120–129 | 203 | 116 | 0.571 | 0.6507 | 0.6651 | 0.6761 | 0.5342 | 0.4379 | 0.6161 | 0.5787 | 0.5859 | 0.5676 | 0.4599 | 0.4806 |
| 130–139 | 91 | 56 | 0.615 | 0.6837 | 0.7031 | 0.6597 | 0.6276 | 0.5367 | 0.6724 | 0.5602 | 0.5648 | 0.4781 | 0.5638 | 0.5673 |
| 140–149 | 47 | 28 | 0.596 | 0.6222 | 0.5977 | 0.6410 | 0.6823 | 0.6447 | 0.6805 | 0.4850 | 0.4765 | 0.4831 | 0.5921 | 0.6053 |

### Observations

1. **Position confound is severe.** Position-only AUROC (0.8304) exceeds
   every baseline's overall AUROC except PAS (0.8349) and SVAR (0.8336).
   Methods with high overall AUROC but low within-bin AUROC are primarily
   detecting position, not hallucination.

2. **IC dominates position-controlled metrics.** IC achieves the highest
   within-bin (0.6856) and matched-pair (0.7029) AUROC, and maintains
   0.66–0.76 across nearly all bins. Its signal is genuinely
   position-independent.

3. **Entropy and NLL are the second tier.** Both show stable within-bin
   AUROC around 0.63–0.64, with consistent per-bin performance across all
   position ranges.

4. **Attention-mass methods (PAS, SVAR) are position-confounded.** Despite
   overall AUROC > 0.83, their within-bin AUROC drops to 0.57–0.59,
   indicating most of their discriminative power comes from the position
   prior.

5. **GLSim (local) shows a late-position pattern.** Weak in early bins
   (0.45–0.54) but reaches 0.68–0.71 in later bins. The combined GLSim
   averages out to 0.54 within-bin.

6. **Beyond-ADS/CGC are near random.** All three Beyond variants hover
   around 0.50–0.52 within-bin, offering essentially no position-independent
   hallucination signal. ADS alone is 0.50 overall.

7. **GLSim (global) is near random.** Within-bin 0.5063 indicates the
   global visual similarity carries no token-level hallucination signal
   after position control.
