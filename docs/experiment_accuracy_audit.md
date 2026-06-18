# Experiment Accuracy Audit

This audit records the current evidence that the main experimental conclusions
are supported by saved artifacts. It should be updated whenever results are
regenerated.

## Detection Baselines

Result root:

```text
detection/baselines/results/coco_llava_7b_baselines/
```

Object cache and label counts match `metrics.json`:

| Quantity | Value |
|---|---:|
| object mentions | 16,426 |
| hallucinated | 4,009 |
| grounded | 12,417 |
| position-only AUROC | 0.830414 |

Independent recomputation from `baseline_scores.npz` matches the saved
`metrics.json` values exactly for the core table:

| Method | Overall | Within-bin | Residual | Interpretation |
|---|---:|---:|---:|---|
| PAS | 0.834859 | 0.592847 | 0.569433 | high global score is mostly position-correlated |
| SVAR | 0.833643 | 0.574802 | 0.556970 | same position-confound pattern as PAS |
| IC | 0.776101 | 0.685611 | 0.633147 | strongest controlled baseline among current methods |
| GLSim-local | 0.772392 | 0.563597 | 0.600581 | some residual signal, weak within-bin |
| Entropy | 0.720592 | 0.636642 | 0.633548 | stable uncertainty baseline |
| NLL | 0.711388 | 0.635994 | 0.629718 | stable uncertainty baseline |
| Beyond-ADS | 0.503325 | 0.510928 | 0.501202 | near random |
| Beyond-CGC | 0.691734 | 0.520838 | 0.518110 | mostly position/weak controlled signal |
| Beyond-ADS+CGC | 0.692573 | 0.522593 | 0.521225 | combined score remains weak after controls |

The stronger post-hoc controls support the same conclusion. Same-object-word
matched AUROC separates stable non-attention baselines from position-confounded
attention-mass baselines:

| Method | Same-word AUROC | Same-word pairs | Linear residual | Cubic residual |
|---|---:|---:|---:|---:|
| PAS | 0.569094 | 123,375 | 0.580939 | 0.557904 |
| SVAR | 0.565447 | 123,375 | 0.572552 | 0.545911 |
| IC | 0.716166 | 123,375 | 0.636844 | 0.633306 |
| Entropy | 0.711562 | 123,375 | 0.633428 | 0.633959 |
| NLL | 0.697613 | 123,375 | 0.628125 | 0.629749 |
| Beyond-ADS+CGC | 0.494565 | 123,375 | 0.526758 | 0.520328 |

Caveat: IC, GLSim, and Beyond-ADS/CGC are adapted or paper-level
implementations, not bit-level official reruns. Paper text should use that
scope explicitly.

## Per-Head TDEV-Lite Probe Audit

`detection/scripts/diagnose_per_head.py` now emits machine-readable audit JSON
for the per-head attention-shape diagnostic. The run reuses the existing
per-head row-cache shards and evaluates held-out logistic probes with image-grouped
folds, so object mentions from the same image do not cross train/test.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/diagnose_per_head.py \
  --cache_glob 'experiments/coco_llava_7b_rows/per_head_row_cache_shard*.npz' \
  --generation_json experiments/coco_llava_7b/generation.json \
  --epochs 80 \
  --output_json detection/baselines/results/per_head_probe/per_head_probe_audit.json
```

Result file:

```text
detection/baselines/results/per_head_probe/per_head_probe_audit.json
```

Key controlled metrics on the same 16,426 CHAIR object mentions:

| Probe | Overall AUROC | Within-bin AUROC | Matched-pair AUROC | Residual AUROC |
|---|---:|---:|---:|---:|
| position-only | 0.830 | 0.577 | 0.546 | 0.522 |
| mean-head probe | 0.812 | 0.545 | 0.537 | 0.526 |
| all-layer per-head image-CV | 0.771 | 0.640 | 0.628 | 0.587 |
| layer 22 per-head image-CV | 0.837 | 0.684 | 0.670 | 0.609 |
| layer 31 per-head image-CV | 0.855 | 0.726 | 0.719 | 0.650 |

This supports a TDEV-lite practicality direction: target-grounding signal exists
inside late-layer per-head attention features and is washed out by mean-head or
early/all-layer aggregation. The result is still a supervised diagnostic probe
with `epochs=80`, not a final training-free method or an external-detector
replacement. It should be used to justify the next LH-Shape/TDEV-lite detector
rather than as the paper's deployed mitigation result.

### LH-Shape Split-Selected Ablation

`detection/scripts/evaluate_lh_shape.py` tests a lighter, more interpretable
readout than the logistic probe: choose top-k late-layer head/feature dimensions
inside each train image fold using position-residualized AUROC, orient them on
train data, then average their z-scored values on held-out images. It also
reports training-free late-layer mean-head shape features.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_lh_shape.py \
  --cache_glob 'experiments/coco_llava_7b_rows/per_head_row_cache_shard*.npz' \
  --generation_json experiments/coco_llava_7b/generation.json \
  --output_dir detection/baselines/results/lh_shape
```

Result files:

```text
detection/baselines/results/lh_shape/lh_shape_metrics.json
detection/baselines/results/lh_shape/lh_shape_metrics.csv
```

Key results:

| Score | Overall AUROC | Within-bin AUROC | Matched-pair AUROC | Residual AUROC |
|---|---:|---:|---:|---:|
| training-free layer31 mean heads, neg top1 mass | 0.527 | 0.521 | 0.530 | 0.518 |
| split-selected layer31 top1 | 0.635 | 0.583 | 0.596 | 0.586 |
| split-selected layer31 top5 | 0.643 | 0.597 | 0.606 | 0.592 |
| split-selected layers22+31 top5 | 0.644 | 0.593 | 0.603 | 0.590 |
| supervised layer31 logistic probe, reference | 0.855 | 0.726 | 0.719 | 0.650 |

This is negative evidence for a naive training-free or top-k-average LH-Shape
method. The late-head signal exists, but simple head selection and averaging do
not preserve enough of the supervised probe's strength. The next TDEV-lite
implementation should use a tiny regularized linear readout or distill the
layer31 probe into a fixed, low-dimensional score, then evaluate it with the
same image-grouped and position-controlled protocol.

### Calibrated LH-Shape Linear Readout

`detection/scripts/evaluate_lh_shape_linear.py` implements that next readout: a
small L2-regularized linear/logistic model over late-layer per-head
attention-shape features, trained with image-grouped folds and evaluated on
held-out images. The output also recomputes core CHAIR detection baselines on
the same 16,426 rows using the same metric implementation.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_lh_shape_linear.py \
  --cache_glob 'experiments/coco_llava_7b_rows/per_head_row_cache_shard*.npz' \
  --generation_json experiments/coco_llava_7b/generation.json \
  --baseline_scores_csv detection/baselines/results/coco_llava_7b_baselines/baseline_scores.csv \
  --output_dir detection/baselines/results/lh_shape_linear
```

Result files:

```text
detection/baselines/results/lh_shape_linear/lh_shape_linear_metrics.json
detection/baselines/results/lh_shape_linear/lh_shape_linear_metrics.csv
```

Key comparison:

| Score | Overall AUROC | Within-bin AUROC | Matched-pair AUROC | Residual AUROC |
|---|---:|---:|---:|---:|
| LH-Shape linear, layers 22+31 | 0.869 | 0.755 | 0.745 | 0.664 |
| LH-Shape linear, layer 31 | 0.856 | 0.728 | 0.721 | 0.651 |
| IC | 0.776 | 0.690 | 0.703 | 0.633 |
| Entropy | 0.721 | 0.641 | 0.655 | 0.633 |
| NLL | 0.711 | 0.639 | 0.652 | 0.629 |
| PAS | 0.835 | 0.605 | 0.583 | 0.576 |
| SVAR | 0.834 | 0.583 | 0.576 | 0.561 |
| GLSim-local | 0.772 | 0.561 | 0.593 | 0.600 |

This is positive evidence for a practical internal TDEV-lite path: a calibrated
late-head readout beats the strongest training-free CHAIR baselines under
position-controlled metrics. It remains supervised calibration and should be
reported separately from unsupervised detectors. The next required check is
whether the same calibrated readout transfers to semantic-neighbor POPE gating or
serves as a cheap prefilter before TDEV-region.

### LH-Shape to TDEV-Region Cascade Audit

`detection/scripts/evaluate_lh_shape_tdev_cascade.py` tests a practical use of
TDEV-lite that does not claim to replace OWLv2: use the calibrated LH-Shape
linear readout as a cheap prefilter, then call OWLv2/TDEV-region only on the
selected object mentions. The simulation keeps the full TDEV-region deletion
threshold fixed, so missed candidates directly count as lost TDEV deletions.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_lh_shape_tdev_cascade.py \
  --cache_glob 'experiments/coco_llava_7b_rows/per_head_row_cache_shard*.npz' \
  --generation_json experiments/coco_llava_7b/generation.json \
  --baseline_scores_csv detection/baselines/results/coco_llava_7b_baselines/baseline_scores.csv \
  --owlv2_scores_csv detection/baselines/results/owlv2_region_detection/owlv2_region_detection_scores.csv \
  --output_dir detection/baselines/results/lh_shape_tdev_cascade
```

Result files:

```text
detection/baselines/results/lh_shape_tdev_cascade/lh_shape_tdev_cascade_metrics.json
detection/baselines/results/lh_shape_tdev_cascade/lh_shape_tdev_cascade_metrics.csv
```

For the full TDEV hybrid-positive top-5% deletion set, LH-Shape prefiltering has
this conservative retention behavior:

| Prefilter | OWLv2 call rate | Hallucination coverage in candidates | Retained full-TDEV hallucination deletions | Cascade hallucination reduction | Grounded loss |
|---|---:|---:|---:|---:|---:|
| LH-Shape linear 22+31 | 25% | 0.639 | 0.706 | 0.120 | 0.002 |
| IC | 25% | 0.525 | 0.670 | 0.114 | 0.003 |
| position-only | 25% | 0.566 | 0.627 | 0.107 | 0.002 |
| LH-Shape linear 22+31 | 50% | 0.923 | 0.934 | 0.159 | 0.006 |
| position-only | 50% | 0.903 | 0.949 | 0.162 | 0.005 |
| LH-Shape linear 22+31 | 75% | 0.990 | 0.996 | 0.170 | 0.009 |

Interpretation: LH-Shape can save about 25% of OWLv2 calls while preserving
nearly all top-5% TDEV hallucination deletions, and it is strongest among the
25% candidate prefilters. However, position-only and PAS become competitive at
50%-75% call rates. This is a useful practicality ablation, not proof of POPE
transfer or a standalone mitigation method.

### POPE Per-Head Transfer Cache Entry Point

The current artifacts do not contain POPE question-token per-head features, so
LH-Shape transfer to semantic-neighbor POPE gating remains unproven. A new cache
entry point now exists at `mitigation/scripts/cache_pope_per_head_rows.py`. It
aligns POPE rows with `semantic_neighbor_rows.csv`, locates the target object
phrase in the LLaVA question prompt, and caches per-head visual-attention shape
features for the target phrase tokens. The label convention is `target_absent=1`,
matching the hallucination-risk direction used by CHAIR detection.

Tokenizer-only validation passed on the first 30 random-split POPE rows with
zero target-location failures. Static checks also pass:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python -m py_compile \
  mitigation/scripts/cache_pope_per_head_rows.py
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/cache_pope_per_head_rows.py --help
```

A real low-memory GPU smoke now passes using the script's bitsandbytes 8-bit
loading path. The smoke used GPU 1 with about 10.9GB free memory, `limit=2`, and
layer 31 only:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/cache_pope_per_head_rows.py \
  --model_path /home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --audit_csv mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv \
  --splits random \
  --limit 2 \
  --per_head_layers 31 \
  --output_dir /tmp/pope_llava_7b_per_head_smoke \
  --device 1 \
  --load_in_8bit
```

Smoke output check: `rows_requested=2`, `rows_cached=2`, `missing_targets=0`,
`present_rows=1`, `absent_rows=1`, and feature shape `(2, 1, 32, 4)` for
`(rows, layers, heads, features)`.

Once the real cache exists, `mitigation/scripts/evaluate_pope_lh_shape_transfer.py`
trains the calibrated LH-Shape readout on calibration splits, chooses an
absence threshold by MCC, and reports target-absence metrics by split and
semantic-neighbor subset:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/evaluate_pope_lh_shape_transfer.py \
  --cache_glob 'experiments/pope_llava_7b_per_head/pope_per_head_row_cache*.npz' \
  --output_dir mitigation/results/pope_lh_shape_transfer \
  --layer_sets '31;22,31' \
  --train_splits random \
  --eval_splits random,popular,adversarial
```

Validation status: `py_compile`, `--help`, a synthetic `/tmp` cache
end-to-end smoke, and the real 2-row 8-bit POPE cache smoke all pass. The
transfer evaluator can read the real smoke cache, but the smoke is too small for
a paper-facing transfer number.

A 120-row-per-split pilot was then generated with 8-bit LLaVA, layers 22 and 31,
and all three POPE splits. Cache quality checks pass: `rows_cached=360`,
`missing_targets=0`, `present_rows=180`, `absent_rows=180`, and feature shape
`(360, 2, 32, 4)`. The cache metrics file is:

```text
experiments/pope_llava_7b_per_head_120/pope_per_head_row_cache_metrics.json
```

Important leakage check: fixed random-split calibration is inflated because
popular/adversarial share 60 positive image-target-label rows with random in the
120-row pilot. Therefore the trusted pilot uses `--image_cv_folds 5`, not the
fixed split-calibrated numbers.

Image-grouped OOF command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/evaluate_pope_lh_shape_transfer.py \
  --cache experiments/pope_llava_7b_per_head_120/pope_per_head_row_cache.npz \
  --output_dir mitigation/results/pope_lh_shape_transfer_120_imagecv \
  --layer_sets '31;22,31' \
  --eval_splits random,popular,adversarial \
  --epochs 120 \
  --image_cv_folds 5 \
  --seed 0 \
  --include_prompt_baselines
```

Result files:

```text
mitigation/results/pope_lh_shape_transfer_120_imagecv/pope_lh_shape_transfer_metrics.json
mitigation/results/pope_lh_shape_transfer_120_imagecv/pope_lh_shape_transfer_metrics.csv
```

Pilot image-CV summary:

| Score | Macro MCC | Absent TPR | Present FPR | AUROC |
|---|---:|---:|---:|---:|
| LH-Shape POPE, layers 22+31 | 0.408 | 0.800 | 0.400 | 0.739 |
| LH-Shape POPE, layer 31 | 0.219 | 0.683 | 0.467 | 0.710 |
| prompt token position | -0.042 | 0.294 | 0.333 | 0.499 |
| target token span length | -0.042 | 0.294 | 0.333 | 0.499 |
| target character length | 0.086 | 0.417 | 0.333 | 0.508 |

For layers 22+31, related-present absent TPR is 0.775 on random, 0.875 on
popular, and 0.778 on adversarial, but the present-object FPR is still 0.400.
Interpretation: internal LH-Shape transfers beyond prompt-only confounds, but it
is not yet a clean POPE gate. It is better viewed as a TDEV-lite triage signal
or a feature to combine with target-vs-neighbor evidence.

The same pilot was used to test LH-Shape as a triage layer before OWLv2/TDEV
calls. `mitigation/scripts/evaluate_pope_lh_tdev_cascade.py` joins the LH-Shape
OOF scores with existing OWLv2 hybrid predictions and simulates two policies:
call full TDEV on top-q absent-risk rows, or call TDEV only when vanilla already
answered `yes` and LH-Shape marks the row high risk.

Cascade command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/evaluate_pope_lh_tdev_cascade.py \
  --lh_predictions_csv mitigation/results/pope_lh_shape_transfer_120_imagecv/pope_lh_shape_transfer_predictions.csv \
  --tdev_predictions_csv mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_predictions.csv \
  --output_dir mitigation/results/pope_lh_tdev_cascade_120 \
  --score_names lh_shape_pope_layers_22_31,lh_shape_pope_layers_31,prompt_token_pos,target_char_len \
  --call_rates 0.10,0.25,0.50,0.75,1.00
```

Result files:

```text
mitigation/results/pope_lh_tdev_cascade_120/pope_lh_tdev_cascade_metrics.json
mitigation/results/pope_lh_tdev_cascade_120/pope_lh_tdev_cascade_metrics.csv
```

Pilot cascade summary on the same 360 rows:

| Method | TDEV calls | Call savings | MCC | TPR | FPR | Related FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla base | 0 | 100% | 0.739 | 0.850 | 0.111 | 0.149 |
| full TDEV hybrid | 360 | 0% | 0.796 | 0.867 | 0.072 | 0.097 |
| LH 22+31 all-selected 50% | 180 | 50% | 0.773 | 0.867 | 0.094 | 0.127 |
| LH 22+31 all-selected 75% | 270 | 25% | 0.790 | 0.867 | 0.078 | 0.104 |
| LH 22+31 base-yes-selected 75% | 130 | 64% | 0.780 | 0.850 | 0.072 | 0.097 |

Interpretation: the best practical use of LH-Shape in this pilot is not as a
standalone gate, but as a suppress-only triage layer. Calling TDEV on high-risk
vanilla-yes rows recovers the full TDEV FPR at 130/360 calls, but it cannot
recover the full hybrid TPR because it does not invoke the rescue branch for
vanilla-no rows. This supports the paper's practicality story while preserving
TDEV-region as the primary verifier.

The full 9,000-row POPE per-head cache has now been generated with 8-bit LLaVA
and the same layers 22 and 31. This full run supersedes the 120-row pilot for
all POPE transfer claims. Cache quality checks pass: `rows_cached=9000`,
`missing_targets=0`, `present_rows=4500`, `absent_rows=4500`, each POPE split
has 3,000 rows, and feature shape is `(9000, 2, 32, 4)`.

Full cache metrics:

```text
experiments/pope_llava_7b_per_head_full/pope_per_head_row_cache_metrics.json
```

Full image-grouped OOF transfer command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/evaluate_pope_lh_shape_transfer.py \
  --cache experiments/pope_llava_7b_per_head_full/pope_per_head_row_cache.npz \
  --output_dir mitigation/results/pope_lh_shape_transfer_full_imagecv \
  --layer_sets '31;22,31' \
  --eval_splits random,popular,adversarial \
  --epochs 120 \
  --image_cv_folds 5 \
  --seed 0 \
  --include_prompt_baselines
```

Full transfer result files:

```text
mitigation/results/pope_lh_shape_transfer_full_imagecv/pope_lh_shape_transfer_metrics.json
mitigation/results/pope_lh_shape_transfer_full_imagecv/pope_lh_shape_transfer_metrics.csv
mitigation/results/pope_lh_shape_transfer_full_imagecv/pope_lh_shape_transfer_predictions.csv
```

Full image-CV transfer summary:

| Score | Macro MCC | Absent TPR | Present FPR | AUROC |
|---|---:|---:|---:|---:|
| LH-Shape POPE, layers 22+31 | 0.559 | 0.764 | 0.205 | 0.853 |
| LH-Shape POPE, layer 31 | 0.488 | 0.792 | 0.306 | 0.812 |
| prompt token position | 0.128 | 0.242 | 0.141 | 0.518 |
| target token span length | 0.128 | 0.242 | 0.141 | 0.518 |
| target character length | 0.183 | 0.270 | 0.124 | 0.540 |

For layers 22+31, split-level MCC is 0.605 on random, 0.587 on popular, and
0.486 on adversarial. Related-present absent TPR is 0.782 on random, 0.770 on
popular, and 0.679 on adversarial. Interpretation: the full run confirms that
internal LH-Shape transfers well beyond prompt-only controls, but its operating
point is still too false-positive-prone to replace TDEV-region as the primary
POPE verifier.

Full LH-Shape to TDEV cascade command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/evaluate_pope_lh_tdev_cascade.py \
  --lh_predictions_csv mitigation/results/pope_lh_shape_transfer_full_imagecv/pope_lh_shape_transfer_predictions.csv \
  --tdev_predictions_csv mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/hybrid_predictions.csv \
  --output_dir mitigation/results/pope_lh_tdev_cascade_full \
  --score_names lh_shape_pope_layers_22_31,lh_shape_pope_layers_31,prompt_token_pos,target_char_len \
  --call_rates 0.10,0.25,0.50,0.75,1.00
```

Full cascade result files:

```text
mitigation/results/pope_lh_tdev_cascade_full/pope_lh_tdev_cascade_metrics.json
mitigation/results/pope_lh_tdev_cascade_full/pope_lh_tdev_cascade_metrics.csv
```

Full cascade summary on all 9,000 POPE rows:

| Method | TDEV calls | Call savings | MCC | TPR | FPR | Related FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla base | 0 | 100% | 0.730 | 0.813 | 0.087 | 0.114 |
| full TDEV hybrid | 9,000 | 0% | 0.763 | 0.806 | 0.051 | 0.069 |
| LH 22+31 all-selected 50% | 4,500 | 50% | 0.746 | 0.812 | 0.071 | 0.096 |
| LH 22+31 all-selected 75% | 6,750 | 25% | 0.764 | 0.813 | 0.056 | 0.075 |
| LH 22+31 base-yes-selected 25% | 1,013 | 89% | 0.747 | 0.808 | 0.068 | 0.091 |
| LH 22+31 base-yes-selected 50% | 2,025 | 78% | 0.754 | 0.801 | 0.056 | 0.075 |
| LH 22+31 base-yes-selected 75% | 3,037 | 66% | 0.753 | 0.796 | 0.051 | 0.069 |

Interpretation: the full cascade supports LH-Shape as a cheap suppress-only
triage layer before TDEV-region. With 2,025/9,000 detector calls,
`base_yes_selected` reduces FPR from 0.087 to 0.056 and related-present FPR from
0.114 to 0.075, close to full TDEV's 0.051/0.069. It does not recover the full
hybrid rescue behavior, so the paper should not report it as a standalone POPE
mitigator.

## Detection Reproducibility Guard

A real smoke run exposed a processor-version mismatch: current transformers
expanded LLaVA-1.5 image placeholders to 575 tokens, while the existing
`generation.json` uses 576 image tokens. The loader now fills the missing
processor metadata from the model config and sets `num_additional_image_tokens`
to 1 for this checkpoint. A GPU smoke run with `--limit 2` and
`--alignment_policy error` then completed and wrote metrics successfully.

Smoke output root:

```text
/tmp/sinkdetect_baseline_alignment_smoke
```

## Mitigation POPE

Result root:

```text
mitigation/results/coco_llava_7b_attention_only/
```

Strict yes/no parsing was applied offline to every saved POPE prediction. All
36,000 POPE outputs start with an explicit `yes` or `no`, so the stricter parser
has `invalid=0` for every method/split and leaves all saved metrics unchanged.

Macro pattern from saved metrics:

| Split | Method | Yes rate | MCC | Interpretation |
|---|---|---:|---:|---|
| adversarial | vanilla | 0.479333 | 0.665902 | anchor |
| adversarial | PAI attention-only | 0.474667 | 0.664854 | no reliable gain |
| adversarial | ClearSight | 0.524667 | 0.646120 | yes-rate increases, discrimination drops |
| adversarial | VisAttnSink | 0.486000 | 0.656257 | no reliable gain |
| popular | vanilla | 0.445333 | 0.740439 | anchor |
| popular | PAI attention-only | 0.442000 | 0.738318 | no reliable gain |
| popular | ClearSight | 0.479333 | 0.737964 | yes-rate increases, no MCC gain |
| popular | VisAttnSink | 0.450667 | 0.732910 | no reliable gain |
| random | vanilla | 0.425000 | 0.785554 | anchor |
| random | PAI attention-only | 0.421667 | 0.783677 | no reliable gain |
| random | ClearSight | 0.454667 | 0.789920 | modest MCC gain with yes-rate increase |
| random | VisAttnSink | 0.429667 | 0.779080 | no reliable gain |

A two-sample real POPE smoke run using the strict parser also completed with
`invalid=0`.

### Current Runtime Pilot

After committing the shared mitigation runtime stack, a fresh real-model pilot
was run with the local paths supplied for this workspace:

```text
model: /home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9
COCO:  /home/chenguanxu/common_dataset/coco-2014-dataset
POPE:  /home/chenguanxu/common_dataset/pope
GPUs:  CUDA_VISIBLE_DEVICES=1,5, NUM_SHARDS=2
```

Result root:

```text
mitigation/results/pope_random_limit100_runtime_audit/
```

The run covers 100 POPE-random rows and verifies that the tracked runner,
dataset loader, local checkpoint, strict yes/no parser, shard merge, and method
comparison all execute end-to-end. All methods have `invalid=0`, and
`compare_methods.py` confirms matched sample IDs against vanilla.

| Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.870 | 0.744 | 0.820 | 0.080 | 0.450 | anchor |
| PAI attention-only | 0.870 | 0.744 | 0.820 | 0.080 | 0.450 | 0.000 |
| ClearSight | 0.860 | 0.721 | 0.840 | 0.120 | 0.480 | -0.020 |
| VisAttnSink | 0.890 | 0.781 | 0.860 | 0.080 | 0.470 | +0.040 |

This pilot is not the paper-facing full-data result, but it validates the
current tracked runtime against the local model/data environment. It also
matches the broader conclusion that unselective attention interventions are
not reliable mitigation: PAI is unchanged, ClearSight increases FPR more than
TPR, and the small VisAttnSink gain needs full-split and semantic-neighbor
confirmation before it can be treated as a real effect.

### VCD-Greedy Full POPE Audit

`vcd` is a controlled greedy port of Visual Contrastive Decoding: it uses the
official original/noisy-image contrastive logit form and diffusion noise
schedule, but keeps greedy decoding to match the rest of this repository's
POPE/CHAIR generation setup.

Result root:

```text
mitigation/results/pope_full_vcd_greedy_audit/
```

The run covers all 9,000 POPE rows across random, popular, and adversarial
splits. Vanilla anchors reproduce the same operating point as the existing full
run, with only one-row-scale differences from regenerated deterministic outputs.
VCD-greedy has `invalid=0` and matched sample IDs, but does not improve POPE; it
slightly raises TPR while raising FPR more, so MCC and accuracy drop on every
split.

| Split | Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| random | vanilla | 0.889 | 0.786 | 0.814 | 0.037 | 0.425 | anchor |
| random | VCD-greedy | 0.886 | 0.780 | 0.817 | 0.045 | 0.431 | -0.005 |
| popular | vanilla | 0.869 | 0.742 | 0.814 | 0.077 | 0.445 | anchor |
| popular | VCD-greedy | 0.862 | 0.728 | 0.817 | 0.092 | 0.454 | -0.013 |
| adversarial | vanilla | 0.834 | 0.668 | 0.813 | 0.145 | 0.479 | anchor |
| adversarial | VCD-greedy | 0.827 | 0.654 | 0.815 | 0.162 | 0.489 | -0.014 |
| macro | vanilla | 0.864 | 0.732 | 0.814 | 0.086 | 0.450 | anchor |
| macro | VCD-greedy | 0.858 | 0.720 | 0.816 | 0.100 | 0.458 | -0.011 |

Semantic-neighbor subset audit shows the same failure mode. VCD-greedy increases
FPR on related-present negatives in all three splits:

| Split | Method | Related-present FPR | Plain-absent FPR | Related-minus-plain gap |
|---|---|---:|---:|---:|
| random | vanilla | 0.055 | 0.014 | 0.041 |
| random | VCD-greedy | 0.061 | 0.024 | 0.036 |
| popular | vanilla | 0.099 | 0.035 | 0.064 |
| popular | VCD-greedy | 0.118 | 0.042 | 0.076 |
| adversarial | vanilla | 0.162 | 0.053 | 0.109 |
| adversarial | VCD-greedy | 0.178 | 0.075 | 0.103 |

This is negative evidence for deterministic VCD under the current greedy POPE
protocol. It strengthens the current paper claim that reducing language-prior
reliance is not enough: without target-discriminative visual evidence, a method
can still amplify yes answers on related but absent targets.

### TDEV Two-Stage Calibration Sensitivity

`mitigation/scripts/audit_two_stage_calibration.py` checks whether the positive
OWLv2 two-stage TDEV result depends on one cherry-picked calibration objective.
It reuses:

```text
mitigation/results/semantic_neighbor_audit/owlv2_tdev_zero/owlv2_tdev_predictions.csv
mitigation/results/coco_llava_7b_attention_only/
```

and writes:

```text
mitigation/results/semantic_neighbor_audit/owlv2_two_stage_calibration_sweep/
```

The tested grid is representative rather than exhaustive: MCC and
semantic-penalty objectives, related-present penalties `1.0` and `2.0`, TPR
floors `0.75` to `0.90`, and a compact threshold grid.

Key result:

| Mode | Macro MCC range | Macro TPR range | Macro FPR range | Adv. related FPR range | Adv. related-minus-plain gap |
|---|---:|---:|---:|---:|---:|
| direct | 0.7765-0.7774 | 0.884-0.906 | 0.1067-0.1300 | 0.2256-0.2516 | 0.1774-0.1858 |
| gate | 0.7514 | 0.7931 | 0.0509 | 0.1038 | 0.0775 |

This supports the existing wording that the two-stage rule is a constructive
TDEV direction, not a final mitigation claim. Direct prediction keeps aggregate
MCC high but still fails many semantic-neighbor negatives. The gate is robust
across the tested calibration settings and reduces related-present FPR, but it
does so by acting as a conservative precision filter over vanilla predictions.

### TDEV Hybrid Gate-Plus-Rescue Audit

`mitigation/scripts/evaluate_hybrid_region_rule.py` tests whether the
precision-oriented two-stage gate can recover recall by adding a strict rescue
branch for vanilla `no` answers. It reuses the same OWLv2 prediction CSV and
vanilla POPE outputs as the calibration sensitivity audit.

Result root:

```text
mitigation/results/semantic_neighbor_audit/owlv2_hybrid_region_rule/
```

Random-split semantic-penalty calibration selects:

```text
low=0.04, high=0.12, margin=-0.20, rescue_high=0.50, rescue_margin=-0.10
```

Compared with the two-stage gate, the hybrid rule improves recall and MCC while
leaving FPR effectively unchanged:

| Method | Macro MCC | Macro TPR | Macro FPR | Adv. MCC | Adv. related FPR | Adv. plain FPR |
|---|---:|---:|---:|---:|---:|---:|
| Two-stage gate | 0.751 | 0.793 | 0.051 | 0.705 | 0.104 | 0.026 |
| Hybrid gate+rescue | 0.763 | 0.806 | 0.051 | 0.717 | 0.105 | 0.026 |

This is positive evidence for an asymmetric TDEV design, but not final evidence
for a deployable mitigation method. The current implementation is a post-hoc
POPE verifier using an external OWLv2 detector; it still needs validation on
CHAIR-style generated captions and a practical integration path.

`mitigation/scripts/audit_hybrid_calibration.py` then checks representative
calibration sensitivity over MCC versus semantic-penalty objectives and TPR
floors `0.75` to `0.90`.

Result root:

```text
mitigation/results/semantic_neighbor_audit/owlv2_hybrid_calibration_sweep/
```

The hybrid result is stable in aggregate: macro MCC stays within
`0.7629-0.7649`. Semantic-penalty calibration consistently selects the lower-FPR
operating point (`macro FPR=0.0513`, adversarial related FPR `0.1046`), while
plain-MCC calibration selects a slightly higher-recall point (`TPR=0.8182`) with
higher semantic-neighbor FPR (`0.1195`). This confirms that the improvement is
not a single-threshold artifact, but the semantic-aware objective is needed to
preserve the desired failure-mode constraint.

### OWLv2 CHAIR Post-Hoc Score Audit

`detection/scripts/evaluate_owlv2_region_posthoc_scores.py` reuses the saved
CHAIR OWLv2 per-mention scores and evaluates additional TDEV-style score
variants. It does not rerun OWLv2.

Result root:

```text
detection/baselines/results/owlv2_region_posthoc_scores/
```

Best controlled CHAIR detection results from the post-hoc table:

| Score | Overall AUROC | Within-bin AUROC | Matched-pair AUROC | Residual AUROC |
|---|---:|---:|---:|---:|
| OWLv2 target absence | 0.865 | 0.842 | 0.847 | 0.711 |
| OWLv2 two-stage absence | 0.872 | 0.849 | 0.851 | 0.707 |
| Hybrid MCC positive branch | 0.874 | 0.853 | 0.855 | 0.708 |
| Target absence + 0.25 neighbor dominance | 0.874 | 0.852 | 0.854 | 0.722 |

This supports the TDEV mechanism across tasks: target-region absence is the main
signal, and neighbor dominance can improve position-residualized detection. It
also prevents overclaiming the POPE hybrid rule: CHAIR object mentions only test
the positive-claim verification branch, not the no-answer rescue branch.


### TDEV Caption Object-Mention Filter Proxy

`detection/scripts/evaluate_tdev_caption_filter.py` simulates a conservative
caption-side mitigation using the existing CHAIR object-mention labels and
OWLv2/TDEV scores. This is not a regenerated-caption result. It asks how many
hallucinated object mentions would be removed by abstaining on high-risk object
mentions, and how many grounded mentions would be lost.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_tdev_caption_filter.py
```

Baseline object-mention counts from the same CHAIR cache:

| Quantity | Value |
|---|---:|
| images with object mentions | 4,977 |
| object mentions | 16,426 |
| hallucinated mentions | 4,009 |
| grounded mentions | 12,417 |
| mention hallucination rate | 0.244 |
| image hallucination rate | 0.492 |

Representative operating points:

| Score / policy | Removed mentions | Hallucination reduction | Grounded loss | Removal precision | Remaining mention hallu. rate |
|---|---:|---:|---:|---:|---:|
| hybrid positive branch, top 5% | 821 | 0.170 | 0.011 | 0.832 | 0.213 |
| hybrid MCC positive branch, top 10% | 1,643 | 0.326 | 0.027 | 0.795 | 0.183 |
| hybrid positive branch, grounded-loss cap 1% | 732 | 0.152 | 0.010 | 0.831 | 0.217 |
| hybrid MCC positive branch, grounded-loss cap 2% | 1,274 | 0.256 | 0.020 | 0.805 | 0.197 |
| target absence + neighbor dominance 0.25, grounded-loss cap 5% | 2,623 | 0.500 | 0.050 | 0.764 | 0.145 |

This gives caption-side mitigation evidence at the object-mention selection
level: TDEV scores rank hallucinated mentions far above their base rate
(24.4%). The result should be described as an abstention/filtering proxy, not a
regenerated-caption result.

### TDEV Caption Text-Edit Proxy

`detection/scripts/evaluate_tdev_caption_edit.py` takes the high-risk object
mentions selected by TDEV, deletes the matched object phrase from the saved
caption text, and writes edited captions. It uses PAS CHAIR `synonyms_txt` plus
simple plural matching for phrase deletion. The accounting below is
"deletion-only": a selected mention only counts as removed if a phrase was
actually deleted from the caption.

Reproducibility commands:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_tdev_caption_edit.py \
  --run_chair

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_tdev_caption_edit.py \
  --score hybrid_mcc_positive_branch_absence \
  --top_frac 0.10 \
  --output_dir detection/baselines/results/tdev_caption_edit_hybrid_mcc_top10 \
  --run_chair
```

Result roots:

```text
detection/baselines/results/tdev_caption_edit/
detection/baselines/results/tdev_caption_edit_hybrid_mcc_top10/
```

Representative text-edit operating points:

| Score / policy | Selected | Actually deleted | Delete success | Deleted hallucinated | Deleted grounded | Hallu. reduction | Grounded loss | Remaining mention hallu. rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid positive branch, top 5% | 821 | 820 | 0.999 | 682 | 138 | 0.170 | 0.011 | 0.213 |
| hybrid MCC positive branch, top 10% | 1,643 | 1,624 | 0.988 | 1,290 | 334 | 0.322 | 0.027 | 0.184 |

The same script now reruns the official PAS CHAIR evaluator when `--run_chair`
is supplied. The scope is the 4,977 images with CHAIR object mentions in the
OWLv2 score CSV, so the vanilla anchor differs slightly from the 5,000-caption
mitigation table scope.

Official PAS CHAIR rerun:

| Policy | CHAIRi | CHAIRs | Total object mentions | Total hallucinated mentions | Mean words | Delta CHAIRi | Delta CHAIRs |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanilla anchor | 0.133984 | 0.492063 | 38,027 | 5,095 | 89.541 | anchor | anchor |
| hybrid positive branch, top 5% | 0.118630 | 0.450472 | 37,208 | 4,414 | 89.242 | -0.015353 | -0.041591 |
| hybrid MCC positive branch, top 10% | 0.104762 | 0.404661 | 36,416 | 3,815 | 88.975 | -0.029222 | -0.087402 |

This validates the deletion-accounting direction under official CHAIR: the
caption edit reduces both instance-level and sentence-level hallucination while
shortening captions by less than one word on average. The exact official CHAIR
deltas differ slightly from the deletion-only counts because CHAIR retokenizes,
POS-tags, and lemmatizes the edited caption text. Qualitatively, this
deterministic deletion proxy still leaves occasional ungrammatical fragments
when the deleted object was the syntactic subject; the paper should not present
it as a fluent caption rewriter or decoding-time method.

### SPIN Adversarial Subset Audit

`spin` is a controlled HuggingFace port of Image-Guided Head Suppression. The
port has passed an 8-row POPE-random smoke test and a larger POPE-adversarial
120-row subset run, both with strict yes/no parsing and `invalid=0`. The subset
is not paper-facing full-data evidence, but it is useful for deciding whether
SPIN should be prioritized as a full baseline.

Result roots:

```text
mitigation/results/pope_spin_smoke/
mitigation/results/pope_spin_adversarial_120/
mitigation/results/semantic_neighbor_audit/spin_adversarial_120_subset_eval/
```

On the adversarial 120-row subset, the default SPIN setting behaves like a
strong yes-prior shift. A milder hyperparameter check keeps 95% of heads and
uses a 0.5 suppression factor for non-routed heads; it avoids the collapse but
still does not improve target discrimination.

| Method | Setting | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| vanilla | anchor | 0.833 | 0.667 | 0.850 | 0.183 | 0.517 | anchor |
| SPIN | default 0.8/0.1 | 0.508 | 0.092 | 1.000 | 0.983 | 0.992 | -0.650 |
| SPIN | mild 0.95/0.5 | 0.833 | 0.670 | 0.883 | 0.217 | 0.550 | 0.000 |

The semantic-neighbor subset makes the failure mode explicit:

| Method | Setting | Related-present FPR | Plain-absent FPR | Related-minus-plain gap |
|---|---|---:|---:|---:|
| vanilla | anchor | 0.204 | 0.000 | +0.204 |
| SPIN | default 0.8/0.1 | 1.000 | 0.833 | +0.167 |
| SPIN | mild 0.95/0.5 | 0.241 | 0.000 | +0.241 |

This does not prove official SPIN fails under all settings: it is a controlled
HF port, a single split, and only 120 rows. It does show that the local port does
not rescue the current mechanism claim. The default setting collapses into a yes
prior, while the mild setting merely raises TPR and FPR together.

### DAMRO Adversarial Subset Audit

`damro` is a controlled HuggingFace port of DAMRO's CLS-selected outlier-token
contrastive decoding. It first passed a 4-row POPE-random smoke test with
`invalid=0`, then ran on the same POPE-adversarial 120-row subset used for the
SPIN audits.

Result roots:

```text
mitigation/results/pope_damro_smoke/
mitigation/results/pope_damro_adversarial_120/
mitigation/results/semantic_neighbor_audit/damro_adversarial_120_subset_eval/
```

On the adversarial 120-row subset, DAMRO improves recall but increases false
positives more, so discrimination weakens:

| Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.833 | 0.667 | 0.850 | 0.183 | 0.517 | anchor |
| DAMRO | 0.817 | 0.639 | 0.883 | 0.250 | 0.567 | -0.033 |

Semantic-neighbor FPR shows that the extra false positives are concentrated on
related-present negatives:

| Method | Related-present FPR | Plain-absent FPR | Related-minus-plain gap |
|---|---:|---:|---:|
| vanilla | 0.204 | 0.000 | +0.204 |
| DAMRO | 0.278 | 0.000 | +0.278 |

This is still subset evidence rather than a full baseline table. It does show
that the controlled greedy DAMRO port does not fix the semantic-neighbor failure
mode: suppressing CLS-selected outlier influence is not the same as verifying the
queried object against associated evidence.


### Qwen2.5-VL All-Splits Replication Audit

The second-model POPE replication now covers Qwen2.5-VL-7B-Instruct on all
three full POPE splits. The generation artifacts passed the following
consistency checks:

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/audit_qwen25vl_replication.py \
  --output_json mitigation/results/semantic_neighbor_audit/qwen25vl_replication_audit.json
```

| Split | Rows | Unique IDs | Yes labels | No labels | Invalid outputs | Semantic audit missing | Fixed-TDEV missing base |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 3000 | 3000 | 1500 | 1500 | 0 | 0 | 0 |
| popular | 3000 | 3000 | 1500 | 1500 | 0 | 0 | 0 |
| adversarial | 3000 | 3000 | 1500 | 1500 | 0 | 0 | 0 |

`predictions.jsonl` and `shard0.jsonl` have the same content by question ID for
all three splits; the row order differs because `merge_evaluate.py` writes
`predictions.jsonl` sorted by `question_id`.

Independent recomputation from the saved JSON/CSV files matches
`docs/multimodel_replication_audit.md`:

| Rule | Macro MCC | Macro TPR | Macro FPR | Macro yes rate | Macro related FPR | Macro plain FPR |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL vanilla | 0.765125 | 0.785556 | 0.033333 | 0.409444 | 0.040982 | 0.010432 |
| Qwen2.5-VL + fixed TDEV hybrid | 0.768808 | 0.781556 | 0.027111 | 0.404333 | 0.033728 | 0.007824 |

This confirms that the paper-safe Qwen claim is all-splits output-level
replication: the semantic-neighbor gap persists on a stronger model, and the
fixed LLaVA-selected TDEV verifier reduces false positives without Qwen-specific
recalibration. It is not evidence about Qwen internal attention because the
current attention adapter is LLaVA-specific.

## Mitigation CHAIR

CHAIR caption metrics support the same scoped conclusion: current attention
ports do not reliably reduce object hallucination under caption-style controls.

| Method | CHAIRi | CHAIRs | Mean words | Object mentions | Hallucinated mentions | Main delta vs vanilla |
|---|---:|---:|---:|---:|---:|---|
| vanilla | 0.133984 | 0.489800 | 89.4534 | 7.6054 | 1.0190 | anchor |
| PAI attention-only | 0.134189 | 0.500400 | 90.1726 | 7.7458 | 1.0394 | slightly worse CHAIR and more hallucinated mentions |
| ClearSight | 0.135051 | 0.493200 | 87.6646 | 7.4920 | 1.0118 | shorter/sparser captions, no CHAIR improvement |
| VisAttnSink | 0.142621 | 0.522000 | 90.9278 | 7.8502 | 1.1196 | richer captions with more hallucinated mentions |

## Current Supported Conclusions

1. Global attention-mass detector performance is heavily position-confounded.
2. Mean-over-head attention shape and fine-grained attention baselines do not
   recover robust target verification under the current controls.
3. Non-attention uncertainty and representation baselines retain more
   position-controlled signal than PAS/SVAR/Beyond in the current run.
4. Current attention-only mitigation ports do not provide reliable object-level
   mitigation under POPE/CHAIR diagnostics; a new 100-row runtime pilot also
   shows no robust attention-only improvement.
5. Controlled VCD-greedy also fails the target-verification test on full POPE:
   it raises FPR more than TPR and worsens related-present negative FPR.
6. The controlled SPIN port has negative adversarial-subset signals: the default
   setting collapses to FPR 0.983, while a mild setting keeps MCC near vanilla
   but raises TPR and FPR equally and worsens related-present FPR.
7. The controlled DAMRO port also has a negative adversarial-subset signal: it
   raises FPR more than TPR and worsens related-present FPR.
8. Qwen2.5-VL all-splits replication supports the main semantic-neighbor and
   fixed-verifier conclusions beyond LLaVA-1.5, with the caveat that this is
   output-level plus external region verification rather than Qwen internal
   attention evidence.
9. Claims must remain scoped: these are controlled ports/adapted baselines, not
   proof that every attention-based method or every official method fails.

## Missing Evidence Before ICML Submission

- Official-code parity checks on small subsets for PAI, ClearSight, and
  VisAttnSink where feasible.
- Optional third-model replication or a Qwen-specific internal-attention
  adapter; the required second-model POPE output-level replication is complete.
- Full regeneration of POPE/CHAIR mitigation results with the now-tracked
  runtime stack, or a documented hash-level equivalence check against the
  existing full run.
- If paper claims compare directly to official VCD, a stochastic decoding parity
  check on a small subset; the controlled greedy VCD full split is complete.
- Head-selection or head-specific mitigation baselines as positive
  counterexamples to unselective attention amplification.
