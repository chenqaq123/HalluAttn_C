# Visual Value Evidence, Iteration 1

**Date locked:** 2026-07-20  
**Status:** pre-registered before implementation or outcome inspection  
**Backbone/data:** frozen LLaVA-1.5-7B, existing COCO-val2014 CHAIR cache  
**Primary target:** position-controlled object-mention hallucination detection

## 1. Question

Attention allocation says where a decoder head reads, but not whether the
information carried from those locations supports the generated object. Test
whether the visual part of the attention output, projected into the target
object's logit direction, is a stronger position-controlled hallucination signal
than attention shape or decoder confidence alone.

The LVLM and its generated caption are never changed. The method reads one
teacher-forced forward. A supervised variant may train a small external detector
and must be reported separately from the label-free score.

## 2. Visual Value Evidence

For layer `l`, head `h`, query row `r`, and visual-token set `V`, decompose the
head output before the shared output projection:

    u_vis^(l,h,r) = sum_(j in V) alpha_(r,j)^(l,h) v_j^(l,h)
    c_vis^(l,h,r) = W_O^(l,h) u_vis^(l,h,r).

`W_O^(l,h)` is the column slice of the layer output projection belonging to
head `h`. This decomposition is exact because the output projection is linear.
Unlike visual-attention mass, `c_vis` contains the value content actually
transported into the residual stream.

Let `o` be the first generated subtoken of the labeled object mention, `w_o` its
unembedding direction, `x_l,r` the residual entering layer `l`, and `LN_f` the
frozen final language-model norm. Extract per-head features:

    mass       = sum_(j in V) alpha_(r,j)
    value_norm = ||c_vis||_2
    target_dot = <c_vis, w_o>
    target_cos = cos(c_vis, w_o)
    target_gain = <LN_f(x_l,r + c_vis), w_o>
                  - <LN_f(x_l,r), w_o>.

Also sum `c_vis` across heads and compute the joint normalized `target_gain`.
The primary interpretation is: larger positive target gain means that visual
content carried by the decoder directly supports the mentioned object.

No object candidate inventory is needed. The target is the model's own emitted
token, so extraction is open-vocabulary even though CHAIR provides COCO labels
for this feasibility evaluation.

## 3. Rows, Layers, and Branches

Extract all 32 decoder layers at two fixed rows:

- `pre`: the generating row immediately before the first object subtoken;
- `post`: the row after consuming that subtoken, using the same object target.

Two branches share the same values, output projection, residual, and target:

1. `raw`: exact post-RoPE model attention probabilities;
2. `no_rope`: recomputed pre-RoPE QK probabilities over the complete causal
   prefix, without visual-only renormalization.

The `no_rope` branch is primary. `raw` is a matched source-control ablation.
Sink deletion is excluded because iteration 1 showed that it removes useful
signal. Position, response length, repetition, and object identity are not
detector inputs.

## 4. Confidence Controls

At the `pre` row and following decoding row, cache fixed decoder statistics:

- selected-token log probability;
- predictive entropy;
- maximum log probability;
- selected-token margin over the strongest alternative;
- normalized selected-token rank.

These form a HaloProbe-style confidence control. Existing IC, NLL, entropy, and
the locked raw/no-RoPE attention probes remain external baselines.

## 5. Evaluation Contract

Use the existing 16,426 first-object mentions from 4,977 images. Evaluation is
five-fold image-grouped OOF with seed 17 and position-bin width 10. Every
normalizer and learned parameter is fit on train images only.

Report for every score:

- overall AUROC;
- size-weighted within-bin AUROC;
- position-matched pair AUROC;
- train-fold position-residual AUROC;
- grounded-only score/position Spearman;
- per-bin AUROC and macro-bin AUROC.

The strict detector excludes position and object identity. An object-word prior
may be reported only as a labeled shortcut ceiling. A same-position plus
same-object diagnostic and leave-object-category-out test are mandatory before
any method claim.

### Label-free score

Within each OOF train fold, robustly standardize each layer's joint no-RoPE
`pre` target gain without labels. The fixed hallucination score is the negative
median over layers 20--31. `post`, raw, mass-only, norm-only, and per-head
aggregations are ablations; their outcome cannot redefine the primary score.

### Supervised detector

Train a two-layer MLP on train-fold-standardized per-head evidence and confidence
features. Use hidden width 256, GELU, dropout 0.1, AdamW at `1e-3`, weight decay
`1e-4`, batch size 128, and 10 epochs. Training examples are weighted to balance
the two labels inside each position bin. No position value or bin ID enters the
network. Report evidence-only, confidence-only, and fused variants.

## 6. Stage-A Correctness Gates

Run 16 images before full extraction. Stage A passes only if:

1. every feature is finite and target IDs decode to the expected object start;
2. cached raw attention rows match model-returned rows within `1e-4`;
3. summed per-head visual contributions match the joint contribution within
   `1e-4` in float32;
4. visual plus non-visual source contributions reconstruct the bias-free
   attention output within `1e-4`;
5. raw and no-RoPE branches are non-identical;
6. extraction leaves model logits unchanged relative to an uninstrumented
   forward within `1e-6`.

Stage-A labels are used only to verify cache alignment, never to select layers,
signs, features, or thresholds.

## 7. Full-Data Decision Gates

The route advances to exact value-ablation experiments only if all apply:

1. fused strict no-RoPE detector reaches at least `0.85` within-bin AUROC;
2. it improves at least `0.03` over the re-evaluated attention-plus-confidence
   baseline under the identical classifier and folds;
3. no-RoPE evidence-only reaches at least `0.78` within-bin;
4. matched and residual AUROC both improve over the strong baseline;
5. the gain remains positive under same-object and leave-category-out controls.

`0.90` within-bin is a stretch target, not the pass threshold. If the fused
detector reaches 0.85 but not 0.90, the next iteration will test an
attention-preserving visual-value ablation. If it remains below 0.82 or fails to
beat the strong baseline, close this route rather than tuning on the full set.

## 8. Novelty Boundary

- HaloProbe combines fine-grained attention and confidence but does not project
  source-decomposed visual value content into the emitted target's logit.
- Owl/VTACR calls averaged visual/text attention weights "contribution"; it does
  not use value vectors or target-logit alignment.
- VISTA and Overthinking trace layerwise vocabulary logits but do not isolate
  the visual source of the residual update.
- Direct logit attribution and logit lens are established interpretability
  tools. The proposed contribution is their source-specific, object-level use
  for position-controlled hallucination detection, not either primitive alone.

Generated artifacts belong under `experiments/visual_value_evidence/` and remain
git-ignored.
