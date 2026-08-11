# Registered protocol: VAR-MLP versus SinkDetect allocation probe

**Registration date:** 2026-08-11  
**Status:** revised and frozen before any comparison result was serialized or
inspected  
**Purpose:** determine whether the deployed SinkDetect sensor has a
position-controlled detection advantage over the VAR-MLP feature from *Devils
in Middle Layers*, and attribute any difference to attention source or layer
breadth rather than to data splits or classifier capacity.

## 1. Question

On the same LLaVA-1.5-7B captions and CHAIR mention labels, does the SinkDetect
post-row, all-layer, No-RoPE allocation representation outperform the
published VAR-MLP post-row, native-attention, layers-5--18 representation when
both use the same fixed image-level training/test split, training-only
preprocessing, classifier, position-conditional loss, and width-10 evaluation?

This is a detector comparison only. It does not run rollback generation and
cannot establish that either detector verifies candidate objects.

## 2. Frozen data and feature identity

- Caption universe: all 5,000 records in the frozen COCO val2014 generation
  manifest. Of these, 4,977 images contain at least one eligible COCO-80 object
  mention; the other 23 remain in the image split but cannot contribute a
  mention-level detection example.
- Mention population: the existing 16,426 first-unique COCO/CHAIR object
  mentions, including 4,009 hallucinated and 12,417 grounded mentions.
- Model/captions: frozen `llava-hf/llava-1.5-7b-hf` caption cache already used
  by the allocation and position-prior audits. No caption is regenerated.
- Split: sort all 5,000 manifest image IDs, apply
  `numpy.random.default_rng(17).permutation`, assign the first 4,000 image IDs
  to training and the remaining 1,000 to test. This label-blind split is shared
  without modification by every representation and baseline. It contains
  13,132 eligible training mentions from 3,985 images and 3,294 eligible test
  mentions from 992 images; 15/8 train/test images have no eligible mention.
- Join key/order: `object_id`; `image_id`, `gen_pos`, label, word, and token
  position must agree exactly between the raw/no-RoPE VVE shards and the
  retained allocation cache.
- Native VAR source: feature `visual_mass` from the retained `raw` tensor. The
  extractor's recorded maximum raw-mass reconstruction error must be zero.
- SinkDetect source: feature `visual_mass` from the retained `no_rope` tensor.
- Temporal row: post only for every row. In the frozen anchor convention this
  is the query row after the object's first subtoken has entered the prefix.
- Visual mass is the sum of the causal attention probability assigned to all
  visual-token positions for one layer/head. For native softmax attention this
  is the published per-head VAR because the denominator over visible keys is
  one.

No test label may select a layer, head, sign, feature, threshold, model seed, or
hyperparameter.

## 3. Frozen representations

The two named primary methods are:

1. **VAR-MLP / native / L5--18:** concatenate native post-RoPE VAR for every
   head in literal zero-based layers 5 through 18 inclusive (`14 x 32 = 448`
   raw inputs). This matches the paper's feature equation and principal layer
   range; the evaluation/training protocol is deliberately SinkDetect's, not
   the author's random mention-level split.
2. **SinkDetect / No-RoPE / L0--31:** concatenate post-row No-RoPE visual mass
   for every head in all 32 layers (`32 x 32 = 1,024` raw inputs).

Two bridge rows form a source-by-breadth factorial and must be reported whether
positive or negative:

3. **native / L0--31**;
4. **No-RoPE / L5--18**.

The registered contrasts are:

- named-method difference: SinkDetect minus VAR-MLP;
- source effect at L5--18: No-RoPE minus native;
- source effect at L0--31: No-RoPE minus native;
- breadth effect under native attention: L0--31 minus L5--18;
- breadth effect under No-RoPE attention: L0--31 minus L5--18;
- source-by-breadth difference-in-differences.

## 4. Shared fixed-split readout and capacity controls

The primary **method-faithful-input** table preserves each representation's
native dimension but holds the readout fixed: training-set standardization,
one-hidden-layer width-256 GELU MLP, scalar hallucination logit, AdamW
(`lr=1e-3`, `weight_decay=1e-4`), batch size 128, ten epochs. Every
representation uses the single frozen 4,000/1,000 image split above. Fit seeds
are `17`, `927`, and `1701`; average their test logits before scoring.

Because the raw input dimensions differ, a co-primary **capacity-matched**
table transforms every representation to exactly 64 inputs using training-only
PCA when the raw dimension exceeds 64 and zero padding otherwise. The PCA is
fit on training rows only, using deterministic randomized-PCA seed `20260811`
and four power iterations. All other training settings and test examples are
identical. Method superiority requires a favorable
named-method contrast in both native-dimension and 64-D tables; disagreement
is reported as a capacity-sensitive result.

Training uses the existing width-10 position-conditional `size` weights: inside
each training position bin the two labels receive equal total loss weight,
while bins retain weight proportional to their sample count. Position, length,
word/category identity, and confidence are not input features.

The shared GELU readout intentionally tests the feature representations under
one protocol. It is labelled **VAR-MLP feature with controlled readout**, not
an exact reproduction of the author's ReLU/grid-search training script.

## 5. Frozen evaluation

Primary metric: size-weighted width-10 within-position-bin AUROC on the fixed
test set, over bins with at least 30 test examples and both labels.

Mandatory secondary outputs:

- overall AUROC, average precision, matched-pair AUROC, and training-set
  residual AUROC evaluated on the test set;
- the complete per-position-bin AUROC vector;
- literal position-balanced AUROC and zero-logit balanced accuracy on the same
  1,000 seed-1701 draws for every row;
- macro average over valid position-bin AUROCs and the fixed late range
  positions 80--139 (the latter is weighted by the registered bin sample
  counts);
- score correlation with generation position and the caption-length proxy;
- three-seed range before logit averaging.

Uncertainty for all six registered contrasts uses 10,000 paired image-cluster
bootstrap draws, seed `20260811`. Holm correction is applied jointly across
the six contrasts within each capacity table. Report raw percentile intervals
and corrected p-values; do not replace the primary metric with whichever
secondary metric is favorable.

## 6. Reproduction and interpretation gates

Before interpreting the comparison:

1. the reconstructed No-RoPE all-layer post feature must match the retained
   post-only cache within `1e-6` elementwise after object-ID alignment;
2. the split manifest must contain exactly 5,000 unique images, with exactly
   4,000 assigned to train and 1,000 assigned to test, and no image overlap;
3. all 16,426 mentions must map to exactly one side of the image split and all
   test predictions must be finite;
4. no fit, standardizer, PCA, threshold, layer/head choice, or sign choice may
   consume test labels.

Interpretation:

- A SinkDetect feature advantage requires a positive named-method paired
  interval excluding zero in both native and 64-D tables.
- If only the native-dimension row is positive, call it a readout-capacity
  advantage, not a feature advantage.
- If the bridge rows remove the difference, attribute it to source or breadth
  as indicated; do not claim a new detector architecture.
- If the matched difference is inconclusive or negative, describe SinkDetect's
  sensor as a rigorously calibrated VAR-like probe and place any method claim
  solely on the downstream rollback policy.
- High within-bin performance remains natural-distribution posterior sensing,
  not candidate-object verification. The completed same-prefix actionability
  audit retains precedence over this comparison.

## 7. Required artifacts

Write ignored artifacts under `experiments/var_mlp_matched/`: config and cache
hashes, the exact 5,000-image split manifest, aligned feature manifest,
per-seed and ensembled test predictions, native/64-D metric tables, paired
bootstrap contrasts, per-bin tables, and a machine-readable summary. Append
the result and decision to the three living research documents without
overwriting history.

## 8. Revision history

The first committed draft specified five-fold image-grouped OOF evaluation.
Before the interrupted bootstrap could write or expose any result artifact,
the user replaced that setting with one shared fixed train/test split. The
interrupted in-memory fits are discarded and are not evidence. This revision
therefore freezes the requested simpler design before the formal fixed-split
run.
