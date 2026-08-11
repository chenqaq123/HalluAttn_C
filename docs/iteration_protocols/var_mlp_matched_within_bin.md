# Registered protocol: minimal fixed-split VAR-MLP versus ours

**Registration date:** 2026-08-11  
**Status:** frozen before the minimal formal run  
**Purpose:** compare the VAR-MLP feature and our allocation feature under one
shared, simple train/test protocol, without cross-validation or auxiliary
capacity/statistical analyses.

## 1. Question

On the same LLaVA-1.5-7B captions, CHAIR mention labels, fixed training images,
fixed test images, readout, optimizer, and evaluation, what are the overall and
width-10 within-position-bin detection results of VAR-MLP and ours?

This is a supervised detector comparison. It does not evaluate rollback
generation and does not by itself establish target-object verification.

## 2. Fixed data split

- Caption universe: all 5,000 records in the frozen COCO val2014 generation
  manifest `experiments/coco_llava_7b/generation.json`.
- Mentions: 16,426 first-unique COCO/CHAIR object mentions, comprising 4,009
  hallucinated and 12,417 grounded mentions from 4,977 images.
- The 23 images without a recognized COCO-80 mention remain part of the image
  split but cannot contribute a mention-level detection example.
- Split algorithm: sort the 5,000 image IDs, permute them with
  `numpy.random.default_rng(17)`, use the first 4,000 images for training and
  the remaining 1,000 for test.
- Resulting eligible rows: 13,132 training mentions from 3,985 images and 3,294
  test mentions from 992 images. The other 15 training and 8 test images have
  no eligible mention.
- Both methods use exactly these same image IDs. Mentions from one image never
  cross the split.

No test label may select a feature, layer, head, sign, threshold, seed, or
hyperparameter.

## 3. Two frozen representations

1. **VAR-MLP feature:** native post-RoPE visual-attention ratio, post temporal
   row, every head from zero-based layers 5--18 inclusive: `14 x 32 = 448`
   inputs.
2. **Ours:** No-RoPE visual-attention mass, post temporal row, every head from
   all zero-based layers 0--31: `32 x 32 = 1,024` inputs.

Visual-attention ratio/mass is the attention probability summed over all visual
tokens for one layer and head. No position, caption length, object category,
word identity, confidence, or label-derived feature is appended.

Only these two native representations are evaluated. There is no PCA,
dimensionality matching, layer/source bridge row, or post-hoc feature choice.

## 4. Shared training

Both representations use the same one-hidden-layer width-256 GELU MLP and
scalar hallucination logit. Training is fixed to:

- model seed 17;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size 128;
- 10 epochs;
- training-set-only feature standardization;
- width-10 position-conditional `size` loss weights, which equalize the two
  labels inside each eligible training position bin while retaining bin weight
  proportional to its training sample count.

Each method is fitted once. A balanced-accuracy threshold is selected from its
training predictions only and then frozen for the test set.

This is labelled **VAR-MLP feature with our controlled shared readout**, not an
exact reproduction of the authors' separate ReLU/grid-search training code.

## 5. Single deterministic test evaluation

Report once on the 3,294 fixed test mentions:

1. overall AUROC;
2. primary width-10 within-position-bin AUROC: compute AUROC independently in
   every test bin with at least 30 mentions and both labels, then average bin
   AUROCs weighted by their test sample counts;
3. ordinary accuracy at the training-selected threshold;
4. balanced accuracy at the same threshold;
5. the complete per-bin AUROC/count table.

There is no five-fold OOF, repeated position-balanced resampling, image
bootstrap, confidence interval, hypothesis test, or PCA table. The difference
`ours - VAR-MLP` is reported descriptively for overall and within-bin AUROC.

## 6. Integrity gates and artifacts

Before interpretation:

1. the generation manifest must contain 5,000 unique image IDs split into
   exactly 4,000 train and 1,000 test IDs with no overlap;
2. all 16,426 mentions must map to exactly one split;
3. the reconstructed No-RoPE feature must match the retained allocation cache
   within `1e-6` elementwise;
4. all train/test scores and thresholds must be finite;
5. neither fit nor threshold selection may consume test labels.

Write ignored artifacts under `experiments/var_mlp_matched/`:
`split_manifest.json`, `summary.json`, `summary.md`, and
`heldout_scores.npz`. Append the completed result and interpretation to the
living research documents without overwriting history.

## 7. Revision history

The first draft used five-fold OOF; the second added three seeds, a 64-D PCA
capacity table, 1,000 position-balanced draws, and 10,000 image bootstraps. The
user rejected those additions as unnecessary for the initial baseline. Both
interrupted runs wrote no formal result artifact. This third version is the
minimal experiment requested by the user and supersedes both earlier drafts.
