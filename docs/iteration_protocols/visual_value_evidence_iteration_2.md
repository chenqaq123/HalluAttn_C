# Visual Value Evidence, Iteration 2: Mechanism and Robustness Audit

**Date locked:** 2026-07-20  
**Status:** pre-registered before iteration-2 implementation or outcome inspection  
**Backbone/data:** frozen LLaVA-1.5-7B, existing COCO-val2014 CHAIR cache  
**Purpose:** validate VVE without changing its method center or tuning toward 0.90

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
both temporal rows, no-RoPE detector branch, MLP width, optimizer, weighting,
folds, and evaluation metrics remain unchanged. Raw attention is retained for
the exact intervention. Removing RoPE is source hygiene, not a claimed source
of performance.

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

## 3. Stage A: Capacity- and Seed-Matched Readout

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

## 4. Stage B: Target and Source Specificity

Run fixed negative controls without changing the VVE-core readout.

### 4.1 Wrong-target direction

For each mention, construct a deterministic label-free derangement inside its
generation-position bin using seed 1701. The control target comes from another
image and must differ in both first-subtoken ID and canonical object word. Keep
the original image, attention, values, residual, and feature dimensions, but
replace `w_o` with the deranged target direction when computing dot/gain.

Train and evaluate the same five-seed probe on these features. Correct-target
VVE must exceed wrong-target VVE by at least 0.05 mean within-bin AUROC. This
control tests target-image correspondence; it is not fused into the method.

### 4.2 Language-source contribution

Replace the visual source span `V` with the generated-caption prefix visible at
the same causal row and compute the identical target-gain trajectory. Prompt
tokens are excluded. Use the same readout and folds. Visual-source VVE must beat
language-source gain by at least 0.03 within-bin, or the feature must be framed
as a generic source-conditioned internal probe rather than visual verification.

Report correct/wrong target and visual/language source results for ordinary
image OOF, same-position/same-category groups, and leave-category-out OOF.

## 5. Stage C: Attention-Preserving Value Intervention

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

## 6. Decision

Passing all stages retains VVE-core and opens unchanged Qwen3-VL plus independent-
dataset confirmation. The method remains the target-gain trajectory; controls
and interventions establish validity but are not detector inputs.

If Stage A fails, report iteration 1 as a capacity-sensitive probe result. If
Stage B fails, remove the claim of target-specific visual evidence. If Stage C
fails, retain predictive detection numbers but describe target gain as a
correlational logit-lens feature, not a causal contribution.

Generated artifacts remain under `experiments/visual_value_evidence/` and are
git-ignored.
