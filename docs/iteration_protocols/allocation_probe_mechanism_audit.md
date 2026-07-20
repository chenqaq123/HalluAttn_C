# No-RoPE Allocation Probe Mechanism Audit

**Date locked:** 2026-07-20  
**Status:** pre-registered before fitting analysis models or inspecting head importance  
**Backbone/data:** frozen LLaVA-1.5-7B, existing COCO/CHAIR VVE cache  
**Purpose:** explain why the all-layer/all-head supervised allocation probe is strong

## 1. Fixed Object of Study

The analyzed detector is the existing No-RoPE visual-allocation probe. For each
object mention it receives:

    m^(l,r,h) = sum_(j in V) alpha_no-rope^(l,r,h,j),

for all 32 decoder layers, pre/post rows, and 32 heads. The input therefore has
2,048 dimensions. The frozen reference readout is the width-256 GELU MLP, ten
epochs, size-weighted conditional loss, seed 17, and five image-grouped folds
used by the capacity audit. No target-value feature, confidence, object ID,
position, or label-derived test feature is added.

This is an explanatory audit of a supervised probe, not a training-free method
and not a head-selection optimization. All findings remain local to LLaVA/COCO
until independently reproduced.

## 2. Questions

1. Is the gain explained by a few stable layer-head pairs or by a distributed
   pattern?
2. Is information carried by the mean visual-mass trajectory across layers, or
   by differences among heads inside a layer?
3. Which depth ranges and temporal rows are necessary or sufficient?
4. Are the same groups useful across generation-position bins, especially in
   the late bins where balanced performance declines?

## 3. Reproduction Gate

Refit the exact reference detector and regenerate five-fold OOF scores. Its
within-bin AUROC must agree with the retained 0.8812 result within 0.002, and
the maximum absolute score difference from the retained OOF file must be
reported. Failure stops interpretation until the discrepancy is resolved.

The primary reporting object is the vector of per-bin AUROCs for positions
0--9 through 140--149 on the frozen OOF rows. Aggregate AUROCs may summarize but
must never replace this vector. Bins with fewer than 30 examples per class are
reported but excluded from mechanism gates.

## 4. Parameter and Local-Effect Attribution

For each fitted fold model, compute two feature scores after train-fold
standardization:

1. first-layer column norm `||W1[:,f]||_2`;
2. held-out mean absolute `gradient(logit,f) * standardized_input_f`.

Aggregate the two temporal features belonging to one `(layer, head)` by L2 norm.
Report fold-wise ranks, mean ranks, top-32 selection frequency, pairwise top-32
Jaccard, and rank correlation. Also aggregate by layer, head index, temporal row,
and fixed depth bands `{0--7, 8--15, 16--23, 24--31}`.

Weight norms alone cannot establish importance because correlated heads can
substitute for one another and hidden units can be reparameterized. A head is
called stable only if it is top-32 by parameter score in at least four of five
folds and is also in the top quartile by held-out local effect.

## 5. Held-Out Functional Tests

Without refitting the full model, replace standardized features by zero (their
train mean) and record OOF logits for:

- removing each layer separately;
- keeping each layer separately;
- removing each temporal row;
- keeping each fixed depth band separately.

Report the per-position AUROC vector for every intervention. A layer is called
necessary only if removing it lowers AUROC by at least 0.02 in at least three
stable bins and it ranks in the top attribution quartile in at least four folds.
A band is called sufficient only if keeping it retains at least 90% of the full
model's AUROC gain above 0.5 in at least 11 of 14 stable bins. Keep-only results
are model interventions, not independently trained detector comparisons.

## 6. Mean-vs-Head and Sparse-Head Refits

Using the unchanged folds and training recipe, fit three fixed representations:

- `layer_mean`: mean over 32 heads for each layer and row (64 dimensions);
- `head_deviation`: subtract that layer/row mean from every head (2,048
  dimensions);
- `full`: the original 2,048 allocation masses.

The result is mean-trajectory dominated only if `layer_mean` retains at least
90% of the full gain above 0.5 in 11 of 14 stable bins. It contains meaningful
head-identity structure only if `head_deviation` retains at least 75% in 11 of
14 bins.

For sparse refits, rank `(layer, head)` pairs using only the corresponding
training-fold full model's first-layer norms. Keep both temporal rows for fixed
`K = {8, 16, 32, 64, 128, 256}` and refit the same MLP. Compare each K with one
deterministic random-K subset per fold (`seed = 1701 + fold`). No test label or
test attribution enters selection.

A sparse-head explanation requires all of:

1. top-32 mean pairwise Jaccard across folds at least 0.20;
2. top-32 refits retain at least 90% of the full gain in 11 of 14 stable bins;
3. top-32 beats random-32 by at least 0.03 AUROC in at least 8 stable bins.

Otherwise describe the signal as distributed/redundant, even if individual
heads receive large weights.

## 7. Decision Boundary

This audit can support a mechanistic conclusion about where a supervised
allocation probe reads signal. It cannot revive allocation as the paper's
constructive method, claim causal verification, or select a deployment head set
on COCO. Any discovered head specialization must survive a modern backbone and
independent data before becoming a general claim.

Generated checkpoints, OOF scores, attribution arrays, and figures remain under
`experiments/allocation_probe_audit/` and are git-ignored.
