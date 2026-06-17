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
