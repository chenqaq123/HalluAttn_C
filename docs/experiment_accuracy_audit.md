# Experiment Accuracy Audit

This audit records the current evidence that the main experimental conclusions
are supported by saved artifacts. It should be updated whenever results are
regenerated.


## Latest Recheck: Result Quality and Claim Scope

Date: 2026-06-18

After the concern that the current numbers are modest and may have drifted away
from the original `looking is not grounding` finding, the core result files were
rechecked from saved predictions and score caches.

Commands rerun:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/baselines/analyze_controls.py \
  --result_dir detection/baselines/results/coco_llava_7b_baselines \
  --output_dir /tmp/coco_llava_7b_baselines_control_verify

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/build_pope_internal_external_ablation.py \
  --output_dir /tmp/pope_internal_external_ablation_verify

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/build_pope_mechanism_alignment.py \
  --output_dir /tmp/pope_mechanism_alignment_verify \
  --examples_per_split 6

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/audit_qwen25vl_replication.py

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  scripts/build_tdev_ablation_summary.py
```

The reruns support the numerical conclusions but not a strong standalone
mitigation claim:

| Check | Recomputed result | Claim implication |
|---|---:|---|
| CHAIR position-only AUROC | 0.830414 | generation position is a strong confound |
| PAS controlled AUROC | overall 0.834859, within-bin 0.592847, same-word 0.569094 | high global attention score is not target grounding |
| SVAR controlled AUROC | overall 0.833643, within-bin 0.574802, same-word 0.565447 | same proxy failure as PAS |
| POPE vanilla -> full TDEV | MCC 0.730 -> 0.763, FPR 0.087 -> 0.051 | real but modest verification gain |
| POPE LH-alone | MCC 0.495, TPR 0.432 | contradicted as standalone mitigation |
| POPE LH->TDEV base-yes 50% | 2,025 calls, MCC 0.754, FPR 0.056 | useful routing into verifier |
| POPE mechanism alignment | neighbor dominance in 96.0% of vanilla related FPs; TDEV fixes 39.8% | matches associated-evidence failure, but is incomplete |
| Qwen2.5-VL replication audit | errors=[], fixed TDEV MCC 0.765125 -> 0.768808 | cross-model direction holds but gain is small |

Conclusion: the experiments are internally consistent. The paper should not be
written as "we found a strong detector." The defensible claim is that semantic-
neighbor target verification diagnoses and partially repairs the precise proxy
failure exposed by `looking is not grounding`.

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

A compact internal-vs-external ablation was then generated by
`mitigation/scripts/build_pope_internal_external_ablation.py`. This table keeps
vanilla, LH-alone suppression, LH-to-TDEV triage, prompt-only triage controls,
and full TDEV under one metric implementation and matched selection budgets.

Result files:

```text
mitigation/results/pope_internal_external_ablation_full/pope_internal_external_ablation.csv
mitigation/results/pope_internal_external_ablation_full/pope_internal_external_ablation.json
mitigation/results/pope_internal_external_ablation_full/pope_internal_external_ablation.md
```

Key macro results:

| Method | Score | Selection | TDEV calls | MCC | TPR | FPR | Related FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| vanilla base | anchor | 0.00 | 0 | 0.730 | 0.813 | 0.087 | 0.114 |
| LH-alone base-yes suppress | LH 22+31 | 0.50 | 0 | 0.495 | 0.432 | 0.018 | 0.026 |
| LH-to-TDEV base-yes | LH 22+31 | 0.25 | 1,013 | 0.747 | 0.808 | 0.068 | 0.091 |
| LH-to-TDEV base-yes | LH 22+31 | 0.50 | 2,025 | 0.754 | 0.801 | 0.056 | 0.075 |
| LH-to-TDEV base-yes | prompt token position | 0.50 | 2,025 | 0.741 | 0.805 | 0.070 | 0.091 |
| LH-to-TDEV base-yes | target character length | 0.50 | 2,025 | 0.743 | 0.802 | 0.066 | 0.087 |
| full TDEV hybrid | anchor | 1.00 | 9,000 | 0.763 | 0.806 | 0.051 | 0.069 |

This is the cleanest current practicality result. LH-alone suppression is too
aggressive: it nearly eliminates false positives but cuts TPR to 0.432 and MCC
to 0.495. The useful behavior comes from routing high-risk vanilla-yes cases to
TDEV-region. At the same 2,025-call budget, LH-Shape routing beats prompt-only
position/length controls on MCC and semantic-neighbor FPR, so the gain is not
only a prompt artifact.

A mechanism-alignment table was also generated to reconnect these aggregate
metrics to the original `looking is not grounding` motivation. It selects
related-present negatives where vanilla is a false positive, semantic-neighbor
evidence exceeds target evidence, and TDEV corrects the answer.

Result files:

```text
mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_examples.csv
mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_examples.md
mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_summary.json
```

Summary over full POPE related-present negatives: vanilla related-present FPR is
0.114; neighbor evidence exceeds target evidence in 98.6% of related-present
negatives and 96.0% of vanilla related-present false positives; TDEV corrects
39.8% of those vanilla related-present false positives. This supports the
associated-evidence interpretation while preserving the caveat that TDEV is a
partial verifier, not a full hallucination solution.

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

### TDEV Caption Neutral-Rewrite Proxy

`detection/scripts/evaluate_tdev_caption_rewrite.py` tests a less destructive
variant of the text-edit proxy. It selects the same high-risk TDEV object
mentions as the top-5% deletion run, but replaces the matched object phrase with
a neutral placeholder: `something` for objects and `someone` for person-like
phrases. It also supports `--rewrite_policy generic_noun`, which replaces
selected object phrases with broad class nouns such as `an item`, `a surface`,
or `an appliance`. These variants keep sentence structure closer to the
original caption while removing the specific object claim. They are still
deterministic post-processing, not neural fluent rewriters.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_tdev_caption_rewrite.py \
  --output_dir detection/baselines/results/tdev_caption_rewrite_neutral \
  --top_frac 0.05 \
  --run_chair

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/evaluate_tdev_caption_rewrite.py \
  --rewrite_policy generic_noun \
  --output_dir detection/baselines/results/tdev_caption_rewrite_generic \
  --top_frac 0.05 \
  --run_chair
```

Result root:

```text
detection/baselines/results/tdev_caption_rewrite_neutral/
detection/baselines/results/tdev_caption_rewrite_generic/
```

Official PAS CHAIR comparison on the same 4,977-image object-mention scope:

| Policy | CHAIRi | CHAIRs | Mean words | Mean object mentions | Mean hallucinated mentions |
|---|---:|---:|---:|---:|---:|
| vanilla anchor | 0.133984 | 0.492063 | 89.541 | 7.641 | 1.024 |
| neutral rewrite, top 5% | 0.118630 | 0.450472 | 89.407 | 7.476 | 0.887 |
| generic noun rewrite, top 5% | 0.118630 | 0.450472 | 89.530 | 7.476 | 0.887 |
| deletion edit, top 5% | 0.118630 | 0.450472 | 89.242 | 7.476 | 0.887 |
| deletion edit, top 10% | 0.104762 | 0.404661 | 88.975 | 7.317 | 0.767 |

Interpretation: neutral and generic-noun rewrites give the same CHAIRi/CHAIRs
improvement as top-5 deletion because CHAIR no longer counts the replaced object
phrase. The generic-noun variant changes length least (`-0.011` mean words),
which rules out pure caption shortening as the explanation, but it sometimes
replaces a wrong object with an underspecified phrase such as `an item` or
`an appliance`. This is evidence that TDEV can select useful local correction
targets, not evidence that the current system performs fluent visual correction.
It should remain appendix/proxy evidence until a fluent regeneration or
decoding-time implementation is added.

### TDEV Caption Sentence-Gate Prototype

`detection/scripts/evaluate_tdev_caption_clause_gate.py` tests a stricter local
claim-suppression proxy for the proposed target-discriminative decoding gate. It
selects the same high-risk top-5% TDEV object mentions, but instead of replacing
the matched object with a placeholder or generic noun, it removes the sentence
that contains the unsupported object claim. The script also retains an optional
`--gate_unit clause` mode, but the clause heuristic produced dangling fragments
in manual inspection; the reported result uses the default sentence gate.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python   detection/scripts/evaluate_tdev_caption_clause_gate.py   --output_dir detection/baselines/results/tdev_caption_sentence_gate_top5   --top_frac 0.05   --run_chair
```

Result root:

```text
detection/baselines/results/tdev_caption_sentence_gate_top5/
```

Official PAS CHAIR comparison on the same 4,977-image object-mention scope:

| Policy | CHAIRi | CHAIRs | Mean words | Mean object mentions | Mean hallucinated mentions |
|---|---:|---:|---:|---:|---:|
| vanilla anchor | 0.133984 | 0.492063 | 89.541 | 7.641 | 1.024 |
| generic noun rewrite, top 5% | 0.118630 | 0.450472 | 89.530 | 7.476 | 0.887 |
| sentence gate, top 5% | 0.116529 | 0.439622 | 86.528 | 7.273 | 0.847 |
| deletion edit, top 10% | 0.104762 | 0.404661 | 88.975 | 7.317 | 0.767 |

Interpretation: sentence-level claim suppression is more grammatical than the
failed clause heuristic and avoids placeholder artifacts, but it is too coarse.
It improves CHAIRi/CHAIRs slightly beyond the top-5 generic rewrite while
removing about three words per caption on average. The stronger top-10 word-level
deletion result still achieves lower CHAIR with a smaller length penalty. This
means the target-discriminative gate should be implemented at decoding time or at
object-phrase granularity, not by deleting completed sentences after generation.

### TDEV Decode-Gate Token Feasibility

`detection/scripts/audit_tdev_decode_gate_feasibility.py` checks whether the
same TDEV-selected object mentions can be intercepted before text is finalized.
It does not load the 7B model; it loads the LLaVA tokenizer, expands CHAIR object
synonyms, and matches object token spans near each saved `gen_pos` in
`owlv2_region_detection_scores.csv`.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_tdev_decode_gate_feasibility.py
```

Result root:

```text
detection/baselines/results/tdev_decode_gate_feasibility/
```

Key result:

| Scope | Mentions | Hallucinated | Near-`gen_pos` token-span match | Caption-anywhere match |
|---|---:|---:|---:|---:|
| all object mentions | 16,426 | 4,009 | 0.9888 | 0.9920 |
| hallucinated mentions | 4,009 | 4,009 | 0.9823 | 0.9843 |
| TDEV top-5% selected | 821 | 683 | 0.9793 | 0.9854 |
| TDEV top-5% hallucinated | 683 | 683 | 0.9766 | 0.9824 |

Interpretation: token coverage is not the bottleneck. The weak caption-side
results above come from using post-hoc deletion/rewrite proxies, not from an
inability to identify object claims in the decode stream. Because only 42.5% of
COCO object words have a single-token realization under the LLaVA tokenizer and
many synonyms share early BPE action tokens, the implementation should be a
short prefix-state object-phrase `LogitsProcessor`, not a naive banned-first-token
list.

### TDEV Decode-Gate Processor Smoke Test

`detection/src/sinkdetect/decode_gate.py` implements a reusable prefix-state
`ObjectPhraseGateLogitsProcessor`. The processor does not score vision evidence;
it receives per-image denied object phrase token sequences from an upstream TDEV
decision and suppresses the next token that would start or continue one of those
phrases.

`detection/scripts/smoke_tdev_decode_gate.py` validates this logic without
loading the 7B model. It takes the TDEV-selected matched rows from
`mention_token_matches.csv`, simulates the model trying to emit each matched
object token sequence, and verifies that the processor sets each next token to
`-inf`.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/smoke_tdev_decode_gate.py
```

Result root:

```text
detection/baselines/results/tdev_decode_gate_smoke/
```

Key result: all `804` tested TDEV-selected matched mentions pass the smoke test;
all `1,333` simulated phrase-generation steps are blocked (`step_block_rate =
1.0`). This proves the generation hook and prefix-state object-phrase mechanism
are functional for the selected token spans. It does **not** yet measure caption
quality, fluency, or whether rerunning LLaVA with the gate improves CHAIR. The
next experiment must run actual gated generation on a small COCO subset.

### TDEV Decode-Gate Caption Smoke Test

`detection/scripts/run_tdev_decode_gate_caption_smoke.py` runs bounded LLaVA
generation smoke tests. It supports three phrase sources:

- `surface`: oracle-input integration, using high-risk TDEV-scored vanilla
  caption mentions and their matched generated surface forms.
- `synonyms`: broader CHAIR synonym variants for the same denied vanilla claims.
- `closed_loop`: image-level unsupported COCO objects are precomputed from OWLv2
  target-vs-neighbor evidence, then narrow aliases are blocked during decoding.

The smoke tests verify generation integration with HuggingFace
`generate(logits_processor=...)`. They are not final caption-quality results.

Reproducibility command for the saved surface-form smoke:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source surface
```

Result root:

```text
detection/baselines/results/tdev_decode_gate_caption_smoke/
```

Key result: the one-image generated-vs-generated surface smoke completed on GPU
5 with `images_with_gate_events = 1` and `captions_differing_from_reference =
1`. For image `391158`, the TDEV-denied hallucinated claims were `person` and
`dining table`; the matched surface forms were `people` and `table`, producing 4
surface-form token sequences. The gated caption removed `two people` and avoided
`table/counter`, but introduced a new `bottle` claim. This proves the decode hook
can change generation, but it also shows that original-phrase suppression alone
can route the model into another unsupported object.

### TDEV Closed-Loop Claim Audit

`detection/scripts/audit_decode_gate_closed_loop_example.py` audits object claims
introduced by a gated caption smoke run. It reruns CHAIR on generated vanilla and
gated captions, identifies object words that appear only in the gated caption,
and scores those introduced claims with OWLv2 target-vs-neighbor evidence.

Reproducibility command for the surface-gate failure audit:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --device cuda:5
```

Result root:

```text
detection/baselines/results/tdev_decode_gate_closed_loop_example/
```

Key result: the surface-gated caption removed denied CHAIR objects `person` and
`dining table`, but introduced a hallucinated CHAIR object `bottle`. The
closed-loop TDEV check would reject this new claim: OWLv2 target score for
`bottle` is `0.0266`, the best semantic neighbor is `cup` at `0.3573`, the margin
is `-0.3307`, and the two-stage verifier predicts absent (`two_stage_present =
0`). This directly supports the next method design: a generation-time gate must
verify newly routed object continuations, not only suppress the original denied
phrase.

### TDEV Closed-Loop Decode-Gate Smoke Test

The first closed-loop smoke test uses the same script with
`--deny_phrase_source closed_loop`. Before LLaVA generation, it scores the image
against the COCO object universe plus semantic neighbors with OWLv2, applies the
same two-stage target-vs-neighbor present rule, and blocks narrow aliases for
objects predicted absent.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_smoke
```

Result root:

```text
detection/baselines/results/tdev_decode_gate_caption_closed_loop_smoke/
```

Key result: the run completed on GPU 5 with `images_with_gate_events = 1`,
`captions_differing_from_reference = 1`, and 320 blocked token sequences. The
same vanilla caption mentions `person`, `cup`, and `dining table`; the
closed-loop gated caption contains only `train` CHAIR objects and no new COCO
object claim.

Follow-up audit command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_audit
```

Audit root:

```text
detection/baselines/results/tdev_decode_gate_caption_closed_loop_audit/
```

Audit result: `introduced_words = []` and `introduced_hallucinated_words = []`.
This is the first end-to-end evidence that closed-loop target-vs-neighbor object
verification can block the substitution failure exposed by the surface gate.
However, the generated caption becomes conservative and train-only, so the method
is still a feasibility prototype.

### TDEV Soft Closed-Loop Decode-Gate Smoke Test

`ObjectPhraseGateLogitsProcessor` now supports a soft mode that subtracts a
finite logits penalty from unsupported object-phrase continuations instead of
setting them to `-inf`. The smoke script exposes this as `--gate_mode soft` and
`--soft_penalty` while preserving hard blocking as the default.

Reproducibility commands:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --gate_mode soft \
  --soft_penalty 4.0 \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_smoke

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --gate_mode soft \
  --soft_penalty 1.0 \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_smoke
```

Audit command for the saved `soft_penalty=1.0` result:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_audit
```

Result roots:

```text
detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_smoke/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_audit/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_smoke/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_audit/
```

Key result: both soft penalties (`4.0` and `1.0`) complete with
`images_with_gate_events = 1`, 320 unsupported-object token sequences, and no
new COCO object claim in the CHAIR audit. But both generations remain
train-only, close to the hard closed-loop output. This is useful negative
evidence: the conservative behavior is not fixed by a simple finite penalty.
The next implementation should narrow when object claims enter the gate, for
example by triggering verification only on candidate object continuations or by
building an allow/deny policy around supported objects, rather than pre-penalizing
nearly every absent COCO object from the first decoding step.

### TDEV Candidate and Top-Risk Closed-Loop Gate Ablations

Two narrower closed-loop ablations test whether the train-only behavior can be
fixed without changing the object-claim policy.

The first ablation delays blocking until the generated suffix already matches at
least one token of a denied phrase. This is exposed as
`--min_prefix_len_to_block 1`; the default remains `0`, which preserves the prior
hard and soft runs.

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --gate_mode hard \
  --min_prefix_len_to_block 1 \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_candidate_smoke

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_candidate_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_candidate_audit
```

Result: the candidate-prefix gate records 21 gate events, but
`captions_differing_from_reference = 0`. It leaves the vanilla caption unchanged,
including the `person`, `cup`, and `dining table` CHAIR objects. This means pure
prefix-triggering is too late for this failure: many object claims are either
single-token or decided by the first content token.

The second ablation keeps first-token blocking but limits each image to the top
30 absent objects by TDEV closed-loop score.

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --gate_mode hard \
  --closed_loop_max_denied 30 \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_top30_smoke

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_top30_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_top30_audit
```

Result: the top-risk gate reduces denied token sequences from 320 to 136 and the
denied object list to 30, including `dining table`, `person`, and `bottle`. The
CHAIR audit again reports no introduced COCO object claim and removes `person`,
`cup`, and `dining table`, but the caption remains train-only. This is a better
cost profile than the full deny set, but still not a natural captioning method.

Combined interpretation: simple softening, delaying, or top-k trimming does not
solve the quality tradeoff. The next method should verify object-token candidates
at the point where object continuations compete, then allow visually supported
objects and suppress unsupported ones. That is closer to a deployable
closed-loop verifier than a static precomputed deny list.

### TDEV Single-Token-First Closed-Loop Gate

The next ablation targets the concrete source of over-conservatism seen in the
full closed-loop gate: first-token blocking often suppresses broad tokenizer
subwords such as `d`, `des`, or `c` for multi-token object aliases. The
`first_token_policy=single_token_only` setting keeps first-token blocking only
for object aliases whose tokenized surface is already a single token; multi-token
object phrases are blocked only at phrase-completion steps.

Reproducibility command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --gate_mode hard \
  --first_token_policy single_token_only \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_smoke

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_audit
```

Result root:

```text
detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_smoke/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_audit/
```

Key result: this is the best current one-image tradeoff. The gated caption is no
longer train-only: it removes the vanilla `person` and `dining table` CHAIR
objects, keeps the supported `cup`, and the CHAIR audit reports
`introduced_words = []` and `introduced_hallucinated_words = []`. Manual
inspection still finds the phrase `bottled drink`, which CHAIR does not map to
COCO `bottle`; therefore this should be treated as promising integration
evidence, not a final quality claim.

### TDEV Variant-Leak Audit for Object-Like Paraphrases

`detection/config/object_variant_aliases.json` records a small, versioned set of
denied-object paraphrases used by both the gate and the audit. The audit script
now reports `gated_variant_hits` and `introduced_variant_leaks`, so CHAIR misses
such as `bottled drink` are no longer only manual observations.

Rechecking the single-token-first output with the variant audit flags the known
leak automatically:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_variant_audit
```

Result: `introduced_variant_leaks = [{"word": "bottle", "alias": "bottled drink"}]`.
This corrects the earlier CHAIR-only reading: the caption has no introduced COCO
object under CHAIR, but it does introduce an object-like bottle paraphrase.

The same alias file was then fed back into the single-token-first gate. This
blocks `bottled drink`, but exposes the brittleness of static alias chasing:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/run_tdev_decode_gate_caption_smoke.py \
  --image_ids 391158 \
  --max_new_tokens 160 \
  --device 5 \
  --generate_vanilla \
  --deny_phrase_source closed_loop \
  --gate_mode hard \
  --first_token_policy single_token_only \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_smoke

/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_audit
```

After adding `bottleneck`, the model routes to `bottling machine`; CHAIR maps this
to an introduced hallucinated `bottle` and OWLv2/TDEV rejects it with target
score `0.0266`, best neighbor `cup` score `0.3573`, and margin `-0.3307`.

After adding `bottling machine` and `bottling`, the model routes again, this time
to `bottletop`. CHAIR and the alias list do not catch it. A root-based audit fixes
that coverage gap:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_root_audit
```

Result: `introduced_root_leaks = [{"word": "bottle", "root": "bottl", "token": "bottletop"}]`.

The audit script now also includes an open-vocabulary candidate pass that does
not depend on CHAIR categories, the alias list, or the root table. It extracts
new object-like content candidates from the gated caption and verifies them with
the same OWLv2/TDEV thresholds:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  detection/scripts/audit_decode_gate_closed_loop_example.py \
  --examples_json detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_smoke/gated_generation_examples.json \
  --output_dir detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_open_vocab_audit \
  --device cuda:5
```

Result: `introduced_open_vocab_candidates = ["bottletop"]`, with target score
`0.0775` and `two_stage_present = 0`, so `unsupported_open_vocab_claims =
["bottletop"]`. This catches the same leak without using the hand-written
`bottl` root.

Re-running the same open-vocabulary audit on all four alias-chasing variants
produces the following summary:

```text
| Run | Route phrase | Open-vocab found | Raw phrase present | Raw score | Auto target | Auto map score | Auto present | Best neighbor |
|---|---|---:|---:|---:|---|---:|---:|---|
| single_token_first | bottled drink | 1 | 1 | 0.1457 | bottle | 1.0000 | 0 | cup |
| variant_alias_v1 | bottleneck | 1 | 1 | 0.1081 | bottle | 1.0000 | 0 | cup |
| variant_alias_v2 | bottling machine | 1 | 1 | 0.1596 | bottle | 1.0000 | 0 | cup |
| variant_alias_v3 | bottletop | 1 | 0 | 0.0775 | bottle | 1.0000 | 0 | cup |
```

Artifacts:

```text
detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_open_vocab_audit/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_open_vocab_audit/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v2_open_vocab_audit/
detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_open_vocab_audit/
detection/baselines/results/tdev_decode_gate_open_vocab_route_summary/
detection/baselines/results/tdev_decode_gate_open_vocab_auto_map_summary/
```

The important conclusion is sharper than the earlier one-image reading.
Open-vocabulary candidate discovery catches each route, but raw phrase scoring is
not sufficient: `bottled drink`, `bottleneck`, and `bottling machine` all look
present if they are treated as literal detector prompts. A lexical
candidate-to-denied-target matcher, using threshold `0.80` and not reading the
variant/root labels, maps all four route phrases to the canonical denied target
`bottle`. TDEV then rejects all four variants with the same target-vs-neighbor
evidence (`bottle` score `0.0266`, best neighbor `cup` score `0.3573`). Static
alias expansion is not a robust method, and CHAIR-only caption evaluation is too
weak for this setting. The paper-facing caption method should therefore use
open-vocabulary object-like candidate discovery plus candidate-to-target mapping
before applying TDEV target-vs-neighbor evidence, rather than relying on a
growing hand-written deny list or raw phrase verification alone.

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
