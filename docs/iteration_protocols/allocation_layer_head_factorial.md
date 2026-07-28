# Registered protocol: layer-breadth × head-identity detector ablation

**Registration date:** 2026-07-28  
**Status:** frozen before confirmatory fitting  
**Hypothesis:** H3 in `research-state.yaml`

## Question

After controlling token position, feature dimension, classifier capacity, and
image split, does hallucination detection require information distributed
across decoder layers and differences between attention heads?

## Frozen data

Use the existing LLaVA-1.5-7B post-row No-RoPE visual-allocation cache with
16,426 first object mentions and all 32 decoder layers × 32 heads. Preserve:

- existing hallucinated/grounded labels and object matching;
- seed-17 image-grouped five-fold split;
- position and sequence-length covariates;
- the existing within-position-bin evaluation population.

No sample, layer, or head is selected using held-out labels. Cache provenance,
sample count, label prevalence, image count, missing tensors, and feature hashes
are written into the result metadata before fitting.

## Factorial representations

### Layer breadth

Evaluate:

1. every **single layer** `0..31`;
2. four fixed adjacent **8-layer bands**: `0–7`, `8–15`, `16–23`, `24–31`;
3. two fixed adjacent **16-layer halves**: `0–15`, `16–31`;
4. **all layers** `0–31`.

The “best single layer” is selected inside each training fold only. Its held-out
prediction is the primary single-layer comparator; an oracle chosen from
test-fold AUROC is forbidden and, if shown for diagnosis, labelled optimistic.

### Head treatment

For every layer breadth above construct:

1. **head-mean:** average the 32 heads within every retained layer before
   concatenating layer features;
2. **per-head identity:** retain each layer/head coordinate before
   concatenation;
3. **head-deviation:** subtract the within-layer head mean from every head
   before concatenation.

The third row tests whether between-head differences, rather than merely the
layer mean, carry the signal.

## Capacity and dimension controls

The primary comparison gives every representation exactly 64 learned input
dimensions:

- fit standardization on the training fold only;
- for raw dimensionality above 64, fit PCA to 64 components on the training
  fold only and apply it unchanged to the held-out fold;
- for raw dimensionality below 64, zero-pad after standardization;
- use the same MLP architecture, width 256, optimizer, stopping rule,
  class weighting, seed set, and training budget for every cell.

Report a secondary raw-dimension analysis using the existing probe recipe, but
base the mechanism claim on the 64-dimensional capacity-controlled grid.
Also fit position-only and position+length controls on the same folds.

## Frozen evaluation

- Primary metric: image-grouped out-of-fold **within-position-bin AUROC**.
- Secondary: global AUROC, average precision, balanced accuracy at a
  training-only threshold, literal position-balanced AUROC over 1,000
  resamples, and correlation of the score with token position and sequence
  length.
- Uncertainty: 10,000 image-cluster bootstrap replicates of paired prediction
  differences, seed `20260728`.
- Fit seeds: `17`, `927`, `1701`; average OOF probabilities before primary
  scoring and report seed range.
- Identical held-out examples must be compared in every paired cell.

## Pre-registered contrasts

1. **Cross-layer contrast:** all-layer/per-head minus training-selected
   single-layer/per-head.
2. **Layer breadth monotonicity:** all-layer/per-head versus each fixed
   8-layer band and each 16-layer half.
3. **Cross-head contrast:** all-layer/per-head minus all-layer/head-mean.
4. **Head-deviation retention:** all-layer/head-deviation minus
   all-layer/per-head.
5. **Interaction:** improvement from single layer to all layers under
   per-head identity minus the corresponding improvement under head-mean.

All contrasts use paired image-cluster bootstrap confidence intervals.
Multiplicity is handled with Holm correction across the five named
confirmatory contrast families; the uncorrected intervals are also shown.

## Interpretation gates

- Evidence for **cross-layer information** requires the all-layer/per-head
  representation to beat the training-selected single-layer/per-head
  comparator and every fixed 8-layer band after correction.
- Evidence for **cross-head information** requires all-layer/per-head to beat
  all-layer/head-mean after correction. If head-deviation retains at least 95%
  of the full per-head excess AUROC over the position+length control, this is
  additional evidence that head differences, not the shared head mean, drive
  performance.
- Evidence for the full requested mechanism requires both gates and a positive
  interaction contrast. Otherwise the conclusion must state which component is
  supported, inconclusive, or contradicted.
- A result that improves only global AUROC but not within-bin or literal
  position-balanced AUROC cannot support H3.

## Required outputs

1. A complete layer-breadth × head-treatment table with raw and 64-D input
   dimensions.
2. Per-layer AUROC curve with train-selected layers marked.
3. Paired contrast table with corrected p-values and bootstrap intervals.
4. Score-versus-position diagnostics and position-only controls.
5. Machine-readable OOF predictions, split manifests, config, environment, and
   cache hashes under ignored `experiments/`.
6. An appended entry in `docs/experiment_results.md` and an interpretation
   entry in `docs/iteration_log.md`.

