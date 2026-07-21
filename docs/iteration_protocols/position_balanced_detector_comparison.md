# Position-balanced detector comparison

**Date registered:** 2026-07-21  
**Backbone/data:** frozen LLaVA-1.5-7B, 16,426 COCO/CHAIR mentions  
**Status:** registered before computing the new per-bin comparison table

## 1. Purpose

Add representative previously reproduced detectors to the pre/post/full
No-RoPE mass position table. Every score must align to the same object IDs,
labels, generation positions, and 1,000 within-bin 1:1 downsampling draws used
by the temporal-row audit. No published number from a different dataset or test
distribution may be inserted.

## 2. Frozen detector columns

Label-free columns:

- PAS layer-0 visual-allocation score;
- faithful SVAR, summed over layers 5--18;
- IC logit-lens uncertainty;
- faithful TVER at layer 0;
- faithful Mask2Former SAE at its previously reported diagnostic layer 10.

Supervised columns:

- faithful ADS+CGC 64-dimensional logistic OOF reconstruction;
- local HaloProbe-style attention-plus-confidence OOF probe;
- No-RoPE mass pre-only, post-only, and full pre+post OOF probes.

The SAE layer is explicitly labeled as the previously selected best-layer
diagnostic, not an a-priori layer. ADS+CGC must reproduce its stored overall and
within-bin values before its per-bin scores are accepted.

## 3. Evaluation

Use position bins of width 10, seed 1701, and 1,000 draws. In each bin select
`min(n_grounded, n_hallucinated)` rows from each class without replacement.
Use the same selected indices for every detector. Report mean per-bin AUROC and
the 2.5--97.5 percentile interval in the artifact. The paper-facing table may
show means plus sample counts; intervals must be discussed for the two tail
bins.

Only AUROC is shared across all columns. Do not compare zero-threshold accuracy
for label-free scores because their raw scales have no common calibrated
threshold. This is a descriptive baseline table, not a method-selection gate.
