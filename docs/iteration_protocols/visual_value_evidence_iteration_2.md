# Visual Value Evidence, Iteration 2: Mechanism and Robustness Audit

**Date locked:** 2026-07-20  
**Status:** pre-registered before iteration-2 implementation or outcome inspection  
**Backbone/data:** frozen LLaVA-1.5-7B, existing COCO-val2014 CHAIR cache  
**Purpose:** validate VVE without changing its method center or tuning toward 0.90

**Pre-implementation wording amendment (2026-07-20):** The detector is described
directly as removing RoPE from Q/K before computing VVE. Raw post-RoPE VVE remains
only an ablation and the exact branch for intervention. This wording change does
not alter a feature, split, model, gate, or result.

**Pre-implementation position-uniform amendment (2026-07-20):** Inspection of
the already reported per-bin metrics showed that the original class-balanced
loss still gives each bin total weight proportional to its sample count. Bins
0--39 therefore supply 56% of the loss, and full-VVE AUROC falls from 0.9253 in
positions 0--39 to 0.7682 in 120--149. Before any iteration-2 fit, Stage A0
below freezes one equal-bin readout. This is a post-result COCO optimization and
requires unchanged external confirmation; it changes no VVE feature.

## 1. Fixed Method Center

The method remains **target-aligned visual value evidence**. For layer `l`, head
`h`, and object row `r`:

    c_vis^(l,h,r) = W_O^(l,h) sum_(j in V) alpha_(r,j)^(l,h) v_j^(l,h)

    gain_o^(l,h,r) = <LN_f(x_l,r + c_vis^(l,h,r)), w_o>
                       - <LN_f(x_l,r), w_o>.

The primary detector input is the all-layer, all-head `gain_o` trajectory at the
fixed pre/post rows. This is 2,048 dimensions for LLaVA-1.5-7B. The full five-
feature VVE tensor remains a diagnostic ceiling; visual mass, value norm,
target dot/cosine, confidence, attention controls, and intervention responses
must not be fused into a redesigned detector during this iteration.

This narrowing follows iteration 1, where target gain alone reached 0.8793
within-bin and the full tensor reached 0.8872. Because that comparison has
already been observed, target-gain-only is a post-result simplification on COCO,
not an independent confirmation. It must later be confirmed unchanged on a new
model/data setting.

The LVLM, generated captions, first-subtoken target definition, all 32 layers,
both temporal rows, MLP width, optimizer, folds, and evaluation metrics remain
unchanged. Except for the explicitly isolated Stage-A0 weighting audit, the
registered iteration-1 weighting remains the reference. The detector directly
removes RoPE from Q/K before
recomputing causal attention and target gain. Raw attention is retained only for
the performance ablation and exact intervention. RoPE removal controls the
explicit position prior; the claimed new signal remains visual value evidence.

## 2. Questions

Iteration 2 asks four questions only:

1. Does the result survive detector initialization rather than one favorable
   MLP seed?
2. Does target gain beat attention allocation when layer coverage, temporal
   rows, input dimension, classifier width, and folds are matched?
3. Does the signal require the correct emitted target direction and the visual
   source, rather than generic lexical or language-state information?
4. Does removing the attributed visual contribution cause the predicted target-
   logit change while leaving that layer's attention weights untouched?

No head, layer, feature, threshold, or fusion search is allowed.

## 3. Stage A0: Position-Uniform Readout

The iteration-1 weight for mention `i` in position bin `b` and class `y` is:

    w_size(i) = n_b / (2 n_(b,y)).

It balances classes inside each bin but leaves total bin weight proportional to
`n_b`. The one registered optimization is:

    w_uniform(i) = 1 / n_(b,y),

followed by one global normalization to mean weight one. Each valid `(bin,
class)` group therefore has equal total weight. A training-fold bin is valid
only when it has at least 30 mentions and both classes; other bins receive zero
fit weight. Feature standardization still uses all training images.

Run the original and uniform objectives with the same folds, model seed, width,
optimizer, epochs, and features for VVE-core, full VVE, allocation mass,
attention plus confidence, and confidence. No exponent, interpolation, minimum-
count, or bin-width sweep is permitted.

Primary metrics are equal-bin macro AUROC and position-matched AUROC. Report the
registered size-weighted within-bin metric for continuity, plus fixed aggregate
ranges `{0--39, 40--79, 80--119, 120--149}` and grounded score/position
Spearman.

The uniform objective advances only if all apply:

1. VVE-core and full VVE each improve macro-bin AUROC by at least 0.015;
2. their weighted within-bin and matched AUROC each fall by no more than 0.010;
3. their size-weighted mean AUROC over bins 80--139 improves by at least 0.015;
4. grounded score/position Spearman decreases by at least 0.05;
5. uniform VVE-core still beats uniformly trained attention plus confidence by
   at least 0.05 macro-bin AUROC.

If the gate passes, uniform weighting becomes the fixed readout objective for
the remaining robustness audits, while iteration-1 numbers remain the original
confirmatory result. If it fails, retain the original objective and report the
late-position degradation. Do not try square-root weighting, blended losses,
position inputs, bin-specific heads, or threshold calibration on COCO.

## 4. Stage A: Capacity- and Seed-Matched Readout

Keep image folds fixed with data seed 17. Train each readout with five fixed
model-initialization seeds `{17, 29, 43, 59, 71}`. Report every seed, mean,
standard deviation, and the mean of paired per-seed differences. Do not select
the best seed.

Use the same width-256 two-layer MLP and compare:

| Family | Features | Input dim. |
|---|---|---:|
| VVE-core | target gain, 32 layers x 2 rows x 32 heads | 2,048 |
| allocation-mass | visual mass on the identical layer/head/row grid | 2,048 |
| VVE-full diagnostic | five registered VVE features on that grid | 10,240 |
| attention-shape | five allocation-only shape statistics on that grid | 10,240 |
| confidence | two rows x five statistics | 10 |

The attention-shape features are fixed as visual mass, visual-distribution
entropy, top-1 mass, top-5 mass, and max-over-mean. They must be extracted for
all layers in the same forward/runtime as VVE. They contain no values,
unembedding direction, position, object ID, or label. VVE-core versus
allocation-mass is the exact input-dimension control; VVE-full versus attention-
shape is the full-capacity control.

Also fit a standardized linear logistic probe to VVE-core. It is a diagnostic
of whether the result requires a nonlinear high-capacity decoder, not a new
method variant.

Stage A passes only if:

1. VVE-core mean within-bin AUROC is at least 0.86 with seed SD at most 0.01;
2. every seed beats allocation-mass, and the mean advantage is at least 0.05;
3. VVE-full beats capacity-matched attention-shape by at least 0.03 on mean
   within-bin AUROC;
4. pre-only VVE-core remains at least 0.84;
5. leave-category-out VVE-core beats its matched attention control by at least
   0.03 on the seed mean.

Failure does not trigger feature tuning. It narrows iteration 1 to a high-
capacity in-domain probe result.

## 5. Stage B: Target and Source Specificity

Run fixed negative controls without changing the VVE-core readout.

### 5.1 Wrong-target direction

For each mention, construct a deterministic label-free derangement inside its
generation-position bin using seed 1701. The control target comes from another
image and must differ in both first-subtoken ID and canonical object word. Keep
the original image, attention, values, residual, and feature dimensions, but
replace `w_o` with the deranged target direction when computing dot/gain.

Train and evaluate the same five-seed probe on these features. Correct-target
VVE must exceed wrong-target VVE by at least 0.05 mean within-bin AUROC. This
control tests target-image correspondence; it is not fused into the method.

### 5.2 Language-source contribution

Replace the visual source span `V` with the generated-caption prefix visible at
the same causal row and compute the identical target-gain trajectory. Prompt
tokens are excluded. Use the same readout and folds. Visual-source VVE must beat
language-source gain by at least 0.03 within-bin, or the feature must be framed
as a generic source-conditioned internal probe rather than visual verification.

Report correct/wrong target and visual/language source results for ordinary
image OOF, same-position/same-category groups, and leave-category-out OOF.

## 6. Stage C: Attention-Preserving Value Intervention

Stage C runs only if Stages A and B pass. Use the raw branch because its visual
contribution is part of the model's actual attention output. At fixed layers
`{8, 16, 24, 31}`, run one intervention at a time at the pre-object row:

    attention_output_l,r <- attention_output_l,r - c_vis^(l,r),

where `c_vis^(l,r)` is the sum of all visual head contributions at that layer
and row. The layer's QK attention probabilities are computed normally and are
not changed; only the value-bearing visual contribution is removed after the
output projection. Later layers run normally from the intervened residual.

Use a deterministic, image-disjoint subset of 1,000 mentions, balanced by label
inside position bins with seed 1701. Detector fitting and any calibration use
the remaining images only. For each layer, record:

- predicted joint raw target gain;
- actual change in the final target logit at the pre row;
- whether the original top-1 token changes;
- a norm-matched random-direction subtraction control.

The mechanism gate requires pooled Spearman at least 0.50 between predicted and
actual target-logit change, at least 70% sign agreement, and a larger absolute
correlation than the random-direction control at every fixed layer. Grounded
versus hallucinated intervention effects are reported as analysis, not used to
choose layers or thresholds.

## 7. Decision

Passing all stages retains VVE-core and opens unchanged Qwen3-VL plus independent-
dataset confirmation. The method remains the target-gain trajectory; controls
and interventions establish validity but are not detector inputs.

If Stage A fails, report iteration 1 as a capacity-sensitive probe result. If
Stage B fails, remove the claim of target-specific visual evidence. If Stage C
fails, retain predictive detection numbers but describe target gain as a
correlational logit-lens feature, not a causal contribution.

Generated artifacts remain under `experiments/visual_value_evidence/` and are
git-ignored.
