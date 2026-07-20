# No-RoPE allocation temporal-row audit

**Date registered:** 2026-07-20  
**Backbone/data:** frozen LLaVA-1.5-7B, existing COCO-val2014 CHAIR cache  
**Status:** registered before fitting the row-only probes

## 1. Question

The full No-RoPE allocation detector uses visual mass at all 32 layers, 32
heads, and two temporal rows. The `pre` row is immediately before the first
object subtoken enters the sequence; the `post` row has consumed that subtoken
and predicts the following token. This audit asks how much hallucination signal
each row can learn independently and whether their union is complementary.

The earlier mechanism audit zeroed one row in an already trained full readout.
That measures reliance of that model, not the best separately trained row-only
detector. This audit therefore refits each row from scratch.

## 2. Frozen feature sets and evaluator

Reuse the exact all-layer no-RoPE mass cache and evaluate only:

- `mass_pre`: `32 layers x 32 heads = 1,024` inputs from the pre row;
- `mass_post`: the equal-dimensional 1,024 post-row inputs;
- `mass_full`: the existing 2,048-dimensional pre+post representation.

Use the same five image folds, seed 17, train-fold standardization,
position-conditional class weights, ten epochs, batch size 128, and width-256
GELU MLP. Pre and post receive the same model seed offset. No position, object
identity, confidence, spatial shape, sink, or value feature is allowed.

Report overall, within-bin, matched, residual, macro-bin, every 10-token bin,
and 1,000 literal position-balanced draws. Use 500 image-bootstrap resamples for
`post-pre`, `full-post`, and `full-pre` differences.

## 3. Locked interpretation

- **Post dominance:** post must beat pre by at least 0.020 within-bin AUROC,
  have a bootstrap interval excluding zero, and win at least 8/14 stable bins.
- **Temporal complementarity:** full must beat the stronger row-only detector
  by at least 0.010, have an interval excluding zero, and win at least 8/14
  stable bins.
- If post dominates but complementarity fails, describe the fingerprint as
  primarily post-target reorganization. If both rows remain strong and full
  passes, describe a post-dominated but complementary temporal pattern.

This is a post-hoc explanation audit on reused labels. It cannot establish
causal object verification or authorize selecting temporal rows as a new method.
