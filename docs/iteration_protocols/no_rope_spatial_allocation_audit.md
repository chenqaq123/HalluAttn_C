# No-RoPE spatial-allocation audit

**Date registered:** 2026-07-20  
**Backbone/data:** frozen LLaVA-1.5-7B, existing COCO-val2014 CHAIR cache  
**Status:** registered before feature extraction or detector fitting

## 1. Question and scope

The all-layer allocation probe currently gives each layer/head/row only total
visual attention mass. This audit asks whether its supervised signal improves
when the visual allocation retains a minimal two-dimensional shape statistic,
and whether deleting low-key-norm visual sink tokens helps or hurts that signal.

This is a post-hoc feasibility audit on the same COCO labels already used to
develop the detector. A positive result may nominate a spatial allocation
feature family, but it is not independent validation and does not establish
target-object evidence verification.

## 2. Frozen attention branches

For every first object mention, extract the same pre/post rows, all 32 decoder
layers, and all 32 query heads as the allocation control. Recompute each row over
the complete causal prefix from pre-RoPE Q/K:

    A_noRoPE = softmax(Q_preRoPE K_preRoPE^T / sqrt(d)).

The model forward, hidden states, caption, and logits are unchanged. No raw-RoPE
branch is a candidate: RoPE removal is mandatory by method definition, although
it does not remove response-depth information already present in hidden states.

For every layer, detect visual sinks with the repository's existing fixed rule:
mean pre-RoPE key L2 norm minus 1.5 standard deviations, clamped to 3--30 visual
tokens. Produce two otherwise identical branches:

- `no_rope`: retain every visual patch;
- `sink_clean`: zero detected sink keys in the complete attention row and
  renormalize over the remaining causal prefix.

The sink rule, threshold, and clamp may not be changed after seeing outcomes.

## 3. Frozen compact features

Let `m` be total mass on the 576 visual patches and let `p` be that visual slice
renormalized to sum to one. For each layer/head/row and each sink branch cache:

1. `visual_mass = m`;
2. `top10_mass`: mass of the largest `round(0.10 * 576) = 58` entries of `p`;
3. `top10_neighbor_coherence`: among those exact top-58 patches, the
   attention-mass-weighted fraction of available 8-neighbors that are also in
   the top-58 set.

The third score is in `[0,1]`. Grid-border patches divide by their actual number
of available neighbors, so borders do not receive a deterministic penalty. The
fixed top-10% support and 8-neighborhood follow the existing ADS spatial
convention. `top10_mass` is the equal-dimensional non-spatial concentration
control: connectivity is useful only if it improves over this control, not only
over total mass.

Also cache sink count and original attention mass assigned to detected sinks for
diagnosis. Neither enters a candidate detector.

## 4. Locked supervised comparisons

Use the existing five image-grouped folds, seed 17, 10-token position bins,
train-fold standardization, position-conditional class weighting, and the same
width-256 GELU MLP used by the allocation/VVE audit. Use one shared model seed
offset for every feature family. No position value, bin ID, word identity,
confidence, or value feature enters these probes.

Evaluate these ten fixed feature sets:

| Sink branch | 2,048-dimensional probes | 4,096-dimensional probes |
|---|---|---|
| no sink deletion | mass; top10; connectivity | mass+top10; mass+connectivity |
| sink deletion | mass; top10; connectivity | mass+top10; mass+connectivity |

Primary metric is size-weighted within-position-bin OOF AUROC. Report overall,
matched, residual, macro-bin, each 10-token bin, and the explicit 1:1
position-balanced AUROC/zero-threshold accuracy distribution. Paired
image-bootstrap intervals use 500 resamples.

## 5. Interpretation gates

- **Connectivity adds spatial information:** within a sink branch,
  `mass+connectivity` must beat the equal-dimensional `mass+top10` control by at
  least 0.010 within-bin AUROC and win in at least 8 of the 14 bins with at least
  30 examples per class.
- **Sink deletion is useful:** a sink-clean representation must beat its
  no-deletion counterpart by at least 0.010 within-bin AUROC and win in at least
  8/14 stable bins.
- **Candidate-family gate:** the best registered spatial fusion must beat the
  2,048-dimensional no-deletion mass probe by at least 0.020 within-bin AUROC,
  with a bootstrap interval excluding zero and no drop larger than 0.020 in any
  stable bin.

Failure closes the corresponding component for this formulation. Do not sweep
top-k fractions, neighborhood definitions, sink thresholds, layers, heads,
temporal rows, probe widths, seeds, or feature signs on these labels. A passing
result still requires a frozen implementation and independent model/dataset
transfer before becoming a paper method.
