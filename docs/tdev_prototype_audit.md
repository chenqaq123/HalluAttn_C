# TDEV Prototype Audit

This note records the first executable target-discriminative verifier prototype.
The goal is to test whether the ICML method direction is practical, not to
claim a finished mitigation method.

## Prototype

`mitigation/scripts/evaluate_clip_tdev_pope.py` implements **TDEV-zero** with a
global CLIP image embedding:

```text
margin(o) = sim(image, "a photo of a o")
          - max_c sim(image, "a photo of a c")
```

where `c` ranges over the top COCO co-occurrence neighbors of target object
`o`. The default direct verifier predicts `yes` when `margin(o) > 0`.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/home/chenguanxu/common_model/huggingface \
TRANSFORMERS_OFFLINE=1 \
python mitigation/scripts/evaluate_clip_tdev_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --audit_csv mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --output_dir mitigation/results/semantic_neighbor_audit/clip_tdev_zero \
  --device cuda:0 \
  --batch_size 128
```

The local CLIP checkpoint is:

```text
/home/chenguanxu/common_model/huggingface/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
```

The run covers all 9,000 POPE rows.

## Direct TDEV-zero Result

At threshold `margin > 0`, TDEV-zero is a conservative verifier:

| Split | Accuracy | MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.640 | 0.356 | 0.329 | 0.050 | 0.040 | 0.062 |
| popular | 0.631 | 0.330 | 0.329 | 0.067 | 0.069 | 0.062 |
| adversarial | 0.634 | 0.339 | 0.329 | 0.061 | 0.053 | 0.101 |
| macro | 0.635 | 0.342 | 0.329 | 0.059 | 0.055 | 0.068 |

This sharply reduces false positives on adversarial related-present negatives
relative to vanilla (`16.4% -> 5.3%`), but recall is too low for direct use as a
POPE answerer.

## Vanilla Gate Result

`mitigation/scripts/evaluate_tdev_gate.py` evaluates the margin as a verifier
gate over vanilla predictions. It selects a threshold on the random split by
maximizing MCC and then applies that threshold to all splits.

Run:

```bash
python mitigation/scripts/evaluate_tdev_gate.py \
  --tdev_predictions_csv mitigation/results/semantic_neighbor_audit/clip_tdev_zero/clip_tdev_predictions.csv \
  --result_root mitigation/results/coco_llava_7b_attention_only \
  --output_dir mitigation/results/semantic_neighbor_audit/clip_tdev_zero_gate \
  --threshold_mode calibrate_mcc \
  --calibration_split random
```

Selected threshold:

```text
-0.09525295346975327
```

| Split | Accuracy | MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.889 | 0.788 | 0.811 | 0.033 | 0.049 | 0.012 |
| popular | 0.867 | 0.739 | 0.811 | 0.077 | 0.099 | 0.035 |
| adversarial | 0.833 | 0.667 | 0.810 | 0.144 | 0.160 | 0.053 |
| macro | 0.863 | 0.730 | 0.811 | 0.084 | 0.111 | 0.027 |

The calibrated gate is essentially tied with vanilla macro MCC (`0.730` vs
`0.731`) and only slightly lowers adversarial related-present FPR (`16.4% ->
16.0%`). Global CLIP margin therefore is not strong enough as the final method.

## Interpretation

The prototype is useful as a lower-bound method and sanity check:

- Target-vs-neighbor scoring is implementable with local models and cached
  POPE/COCO artifacts.
- A global-image margin can suppress false positives, but at a large recall
  cost when used directly.
- As a gate, global CLIP margin does not materially improve vanilla on hard
  related-present negatives.

The ICML method should therefore move from global CLIP TDEV to
**region/head-conditioned TDEV**:

1. score target-vs-neighbor evidence on patches or object boxes rather than the
   whole image;
2. weight regions by selected late-layer object-query heads rather than mean
   attention;
3. calibrate a gate on held-out data with an explicit objective such as MCC or
   fixed-TPR FPR reduction;
4. report semantic-neighbor FPR as the primary stress metric.


## Patch-Level Evidence Modes

The same script also supports patch-level CLIP evidence:

```bash
python mitigation/scripts/evaluate_clip_tdev_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --audit_csv mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --output_dir mitigation/results/semantic_neighbor_audit/clip_tdev_patch_margin_max \
  --device cuda:0 \
  --batch_size 128 \
  --evidence_mode patch_margin_max
```

Two patch variants were checked:

- `patch_max`: `max_patch target_score - max_patch max_neighbor_score`;
- `patch_margin_max`: `max_patch(target_score - max_neighbor_score)` on the
  same patch.

Full-data direct results:

| Mode | Macro MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---:|---:|---:|---:|---:|
| global | 0.342 | 0.329 | 0.059 | 0.055 | 0.068 |
| patch_max | 0.149 | 0.212 | 0.104 | 0.092 | 0.129 |
| patch_margin_max | 0.133 | 0.881 | 0.781 | 0.770 | 0.807 |

Calibrated vanilla-gate results, using random split MCC for threshold selection:

| Mode | Macro MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---:|---:|---:|---:|---:|
| global gate | 0.730 | 0.811 | 0.084 | 0.111 | 0.027 |
| patch_max gate | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 |
| patch_margin_max gate | 0.731 | 0.813 | 0.086 | 0.113 | 0.028 |

The patch-level CLIP variants do not materially improve the vanilla gate.
`patch_margin_max` confirms that unconstrained patch evidence is too permissive:
it finds some patch where the target barely beats neighbors for many absent
queries, producing very high FPR. This rules out naive CLIP patch similarity as
the final method.

The next useful prototype should add LVLM-specific selectivity rather than only
changing CLIP pooling:

1. use LLaVA late object-query heads to weight patches before computing the
   target-vs-neighbor margin;
2. restrict candidate regions with object proposals/boxes or high-evidence
   connected patches;
3. evaluate at fixed TPR or with an abstention budget, because direct yes/no
   replacement is too brittle.

## Attention-Weighted Detection TDEV

`detection/scripts/evaluate_attention_weighted_tdev.py` tests whether the cached LLaVA object-token attention row can rescue CLIP patch evidence on the CHAIR object-mention detection task. The script resizes each cached 24x24 visual attention row to CLIP 7x7 patch grid and computes:

```text
margin(o) = sum_p a_p sim(p, o) - max_c sum_p a_p sim(p, c)
```

where `c` is a COCO co-occurrence neighbor. Detection scores are `-margin`, so larger values mean more likely hallucinated.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/home/chenguanxu/common_model/huggingface \
TRANSFORMERS_OFFLINE=1 \
python detection/scripts/evaluate_attention_weighted_tdev.py \
  --row_cache experiments/coco_llava_7b_rows/attention_row_cache.npz \
  --object_cache detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --output_dir detection/baselines/results/attention_weighted_tdev \
  --device cuda:0 \
  --batch_size 128
```

Full-data result on 16,426 object mentions:

| Score | Overall AUROC | Within-bin AUROC | Matched-pair AUROC | Residual AUROC |
|---|---:|---:|---:|---:|
| attention-weighted TDEV mean | 0.582 | 0.560 | 0.551 | 0.524 |
| attention-weighted patch-margin mean | 0.581 | 0.551 | 0.542 | 0.517 |
| best cached layer | 0.590 | 0.562 | 0.552 | 0.527 |

This is a negative result. Mean-over-head early-layer attention from the current row cache does not make CLIP target-vs-neighbor evidence competitive with the audited baselines. It is far below IC/entropy/NLL under controlled metrics and should not be reported as a successful TDEV method.

The result narrows the next implementation target: TDEV needs late object-query/head-specific evidence or proposal-constrained regions. Simply combining CLIP patch similarities with cached average attention is insufficient.

## OWLv2 Region-Evidence Baseline

`mitigation/scripts/evaluate_owlv2_tdev_pope.py` evaluates a stronger region-evidence baseline with the local OWLv2 checkpoint:

```text
/home/chenguanxu/common_model/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d
```

For each image, the script queries all 80 COCO object names used by the semantic-neighbor audit and stores each object's highest OWLv2 box score. Two scores are then evaluated:

- `target_score`: highest OWLv2 region score for the queried object;
- `tdev_margin`: `target_score - max_neighbor_score` over COCO co-occurrence neighbors.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/home/chenguanxu/common_model/huggingface \
TRANSFORMERS_OFFLINE=1 \
python mitigation/scripts/evaluate_owlv2_tdev_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --audit_csv mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --output_dir mitigation/results/semantic_neighbor_audit/owlv2_tdev_zero \
  --device cuda:0 \
  --batch_size 4
```

Additional calibrated direct-score evaluations are produced by:

```bash
python mitigation/scripts/evaluate_direct_score.py \
  --predictions_csv mitigation/results/semantic_neighbor_audit/owlv2_tdev_zero/owlv2_tdev_predictions.csv \
  --output_dir mitigation/results/semantic_neighbor_audit/owlv2_target_score_direct \
  --score_field target_score \
  --threshold_mode calibrate_mcc \
  --calibration_split random

python mitigation/scripts/evaluate_direct_score.py \
  --predictions_csv mitigation/results/semantic_neighbor_audit/owlv2_tdev_zero/owlv2_tdev_predictions.csv \
  --output_dir mitigation/results/semantic_neighbor_audit/owlv2_margin_direct_cal \
  --score_field tdev_margin \
  --threshold_mode calibrate_mcc \
  --calibration_split random
```

Full POPE results with random-split MCC calibration:

| Score / use | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Adversarial related FPR |
|---|---:|---:|---:|---:|---:|---:|
| OWLv2 `target_score` direct | 0.777 | 0.911 | 0.134 | 0.184 | 0.025 | 0.281 |
| OWLv2 `tdev_margin` direct | 0.445 | 0.359 | 0.013 | 0.010 | 0.019 | 0.010 |
| OWLv2 `target_score` gate over vanilla | 0.738 | 0.812 | 0.078 | 0.105 | 0.017 | 0.156 |
| OWLv2 `tdev_margin` gate over vanilla | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 | 0.164 |

Interpretation:

- OWLv2 `target_score` is a strong open-vocabulary detector baseline and beats vanilla macro MCC, but its adversarial related-present FPR rises to 28.1%. This confirms that even region-level object evidence is vulnerable to semantically associated objects.
- OWLv2 target-vs-neighbor margin is a high-precision verifier, reducing related-present FPR to about 1%, but recall falls to 35.9%. It is not a direct POPE answerer.
- As a gate over vanilla, OWLv2 `target_score` gives only a small improvement over vanilla and CLIP gates. It is useful as a baseline and design clue, not yet an ICML-level solution.

The constructive method should combine the useful part of OWLv2-style target localization with a less recall-destructive discriminative check, possibly via proposal-level target/neighbor calibration, abstention, or LVLM-conditioned object queries.

## Calibrated Region Verifier Variants

Two additional calibrated verifier variants test whether OWLv2 region evidence can be converted into a more useful method.

`mitigation/scripts/evaluate_neighbor_penalty_score.py` searches:

```text
score = target_score - alpha * best_neighbor_score
```

with `alpha` in `[0, 1.5]`. Random-split MCC calibration selects `alpha=0.0` for both direct prediction and vanilla gating, meaning the best linear penalty is just the raw target detector score. This is a negative result for simple linear target-minus-neighbor scoring.

`mitigation/scripts/evaluate_two_stage_region_rule.py` evaluates a semantic-aware two-stage rule:

```text
yes if target_score > high_threshold
   or target_score > low_threshold and tdev_margin > margin_threshold
```

The rule keeps high-confidence target detections while requiring a neighbor check for medium-confidence detections. With random-split calibration using objective `MCC - 2 * related_FPR` and `TPR >= 0.85`, the selected direct thresholds are:

```text
low_threshold=0.10, high_threshold=0.16, margin_threshold=-0.15
```

For vanilla gating, the selected thresholds are:

```text
low_threshold=0.04, high_threshold=0.12, margin_threshold=-0.20
```

Full POPE results:

| Method | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Adv. MCC | Adv. related FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| OWLv2 `target_score` direct | 0.777 | 0.911 | 0.134 | 0.184 | 0.025 | 0.673 | 0.281 |
| OWLv2 `tdev_margin` direct | 0.445 | 0.359 | 0.013 | 0.010 | 0.019 | 0.441 | 0.010 |
| Two-stage direct | 0.769 | 0.850 | 0.083 | 0.111 | 0.021 | 0.701 | 0.167 |
| OWLv2 `target_score` gate | 0.738 | 0.812 | 0.078 | 0.105 | 0.017 | 0.673 | 0.156 |
| Two-stage gate | 0.751 | 0.793 | 0.051 | 0.069 | 0.012 | 0.705 | 0.104 |

Interpretation:

- A linear neighbor penalty is insufficient; MCC calibration chooses no neighbor penalty.
- The two-stage semantic-aware rule improves the tradeoff. Direct prediction reduces adversarial related-present FPR from 28.1% to 16.7% while improving adversarial MCC from 0.673 to 0.701, at the cost of TPR dropping from 0.911 to 0.850.
- As a gate, the two-stage rule improves macro MCC from 0.738 to 0.751 and reduces macro related-present FPR from 10.5% to 6.9%, but TPR drops from 0.812 to 0.793.

This is the first positive method-shaped result. It is not yet sufficient as the final ICML method because recall still drops, but it supports a concrete direction: semantic-neighbor-aware calibration over region evidence rather than raw detector score or hard target-vs-neighbor margin.

## OWLv2 Region Evidence on CHAIR Detection

`detection/scripts/evaluate_owlv2_region_detection.py` evaluates the same OWLv2 region evidence on the CHAIR object-mention hallucination detection task. For each generated object mention, it scores the mentioned object and its COCO co-occurrence neighbors with OWLv2 and reports standard detection metrics with the same position controls used for existing baselines.

Run:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/home/chenguanxu/common_model/huggingface \
TRANSFORMERS_OFFLINE=1 \
python detection/scripts/evaluate_owlv2_region_detection.py \
  --object_cache detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --output_dir detection/baselines/results/owlv2_region_detection \
  --device cuda:0 \
  --batch_size 4
```

Full-data result on 16,426 object mentions:

| Score | Overall AUROC | Within-bin AUROC | Matched-pair AUROC | Residual AUROC |
|---|---:|---:|---:|---:|
| OWLv2 target absence | 0.865 | 0.842 | 0.847 | 0.711 |
| OWLv2 two-stage absence | 0.872 | 0.849 | 0.851 | 0.707 |
| OWLv2 margin absence | 0.779 | 0.745 | 0.732 | 0.639 |
| OWLv2 two-stage binary | 0.763 | 0.745 | 0.747 | 0.635 |
| OWLv2 neighbor presence | 0.519 | 0.483 | 0.471 | 0.470 |

This is a strong positive cross-task result. The region-evidence detector is far stronger than the previous best controlled detection baselines: IC reaches within-bin AUROC 0.686 and matched-pair AUROC 0.703, while entropy/NLL are around 0.636/0.655. OWLv2 target and two-stage evidence reach about 0.84-0.85 on the same controlled metrics.

Interpretation:

- Region-level target evidence directly addresses the position confound that breaks attention-shape scores.
- The two-stage score slightly improves overall, within-bin, and matched-pair AUROC over raw target absence, though residual AUROC is similar.
- Neighbor presence alone is not predictive, which supports the earlier finding that the useful signal is target evidence plus calibrated discrimination, not merely the existence of related objects.
- Runtime is high: the full detection audit took about 36 minutes without image-score caching. Any practical method should cache OWLv2 image-object scores or use a cheaper proposal module.

This result upgrades the method direction: semantic-neighbor-aware region verification has evidence on both POPE mitigation/gating and CHAIR detection. The remaining ICML method challenge is practical, recall-preserving calibration rather than finding a signal from scratch.

## OWLv2 Image-Score Cache

OWLv2 region scoring is the current practical bottleneck. The full CHAIR detection audit took about 36 minutes because it recomputed image-object scores inside the analysis script. The OWLv2 POPE and CHAIR scripts now support a reusable NPZ cache with canonical COCO image keys, object names, and an `images x objects` score matrix.

Cache options:

```bash
# Save scores while running CHAIR detection.
python detection/scripts/evaluate_owlv2_region_detection.py \
  --object_cache detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --output_dir detection/baselines/results/owlv2_region_detection \
  --save_image_score_cache detection/baselines/results/owlv2_region_detection/owlv2_image_scores.npz

# Reuse the same scores without loading OWLv2.
python detection/scripts/evaluate_owlv2_region_detection.py \
  --object_cache detection/baselines/results/coco_llava_7b_baselines/object_cache.jsonl \
  --neighbors_json mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --output_dir detection/baselines/results/owlv2_region_detection_cached \
  --image_score_cache detection/baselines/results/owlv2_region_detection/owlv2_image_scores.npz
```

The same `--image_score_cache` and `--save_image_score_cache` options are available in `mitigation/scripts/evaluate_owlv2_tdev_pope.py`.

Verification:

- CHAIR detection smoke test with saved cache and cache reload produced identical metrics.
- POPE smoke test with saved cache and cache reload produced identical subset metrics.

This makes the region-evidence method more practical: expensive image scoring can be amortized once, while calibration, gating, semantic-neighbor slicing, and table generation become cheap deterministic post-processing.

