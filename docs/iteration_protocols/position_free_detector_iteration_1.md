# Position-Free Detector Iteration 1: Clean Per-Head Feasibility

Date: 2026-07-20
Status: pre-registered; not started

## Question

After removing RoPE from the attention score and deleting attention-sink visual
tokens, can late-layer per-head attention support either:

1. a useful label-free object-hallucination score; or
2. a position-conditional supervised external classifier?

The LVLM is frozen. This iteration neither fine-tunes generation nor changes a
generated answer. It tests detection on already generated captions.

## Frozen data and baselines

- Model: LLaVA-1.5-7B, fp16, eager attention, unchanged weights and logits.
- Data: the existing COCO-val2014 caption cache with 16,426 first object mentions
  from 4,977 images; 4,009 mentions are CHAIR-labeled hallucinations.
- Prompt, token alignment, image preprocessing, and labels are exactly those in
  `experiments/coco_llava_7b_rows` and
  `detection/baselines/results/coco_llava_7b_baselines`.
- Position-only overall AUROC: 0.8304.
- Frozen label-free controls: IC 0.686, entropy 0.637, NLL 0.636 within-bin.
- Frozen supervised reference: raw post-RoPE layer-31 per-head shape probe 0.726
  within-bin under image-grouped OOF.

No caption regeneration, POPE evaluation, threshold fitting, or mitigation run
is authorized by this protocol.

## Position-removal contract

Removing RoPE is only the first control. This protocol separates three levels:

1. **Attention source:** capture Q/K before multimodal RoPE and recompute each
   selected head's causal attention. Do not use the recomputed weights in the
   model forward.
2. **Local contrast:** compare each object-generating row with nearby generated
   non-object rows from the same image and response neighborhood.
3. **Learning/evaluation:** train only on differences matched inside generation-
   position bins and report within-bin, matched-pair, and residual AUROC.

`gen_pos`, absolute token index, total response length, and position-bin ID are
never detector features.

## Feature extraction

Extract only decoder layers 22 and 31. For each first object mention, retain up
to eight nearest generated non-object token rows within a 16-token window. A
local reference is invalid when no such row exists; report and exclude invalid
rows identically for every branch.

For each layer/head, build three matched branches:

1. `raw`: ordinary post-RoPE visual attention;
2. `no_rope`: pre-RoPE QK attention;
3. `clean`: pre-RoPE QK attention after deleting sink visual keys.

Detect sinks independently at each layer from low pre-RoPE key L2 norms using
the repository's frozen `auto_detect_sinks` rule. The sink set is label-free.
For every branch, restrict to visual keys and L1-renormalize the object and local
rows separately.

Store only summarized float32 features, not full attention tensors. For object
distribution `p_t` and local-reference distribution `p_n`, save per head:

- `entropy_obj`;
- `top1_obj`, `top5_obj`, and `max_over_mean_obj`;
- `jsd_obj_local = JSD(p_t, p_n)`;
- `delta_top1 = top1(p_t) - top1(p_n)`;
- `delta_top5 = top5(p_t) - top5(p_n)`;
- `delta_entropy = entropy(p_n) - entropy(p_t)`.

Higher delta features mean more target-specific visual concentration. Assert
that raw-branch object-only shape features reproduce the existing cache to
`1e-4` on 100 rows before opening the new metrics.

## Label-free score

Use layer 31 only. Within each image-grouped fold, estimate an empirical CDF from
the unlabeled training-fold distribution for each head's `jsd_obj_local` and
positive `delta_top5`. Map test features through those frozen CDFs and define

    e_h = 0.5 * CDF_h(jsd_obj_local)
        + 0.5 * CDF_h(max(0, delta_top5))
    grounding_gain = mean of the largest 8 values among {e_h}_{h=1}^{32}
    score_unsup = -grounding_gain.

The top eight heads are chosen per mention by their current label-free evidence,
not by any learned head identity. Eight is the pre-registered top 25% of 32
heads. There is no learned weight or label-selected sign.

Matched ablations use exactly the same folds and CDF procedure:

- all-head mean instead of top eight;
- object-only shape without local differencing;
- layer 22 instead of 31;
- `raw`, `no_rope`, and `clean` branches.

## Position-conditional supervised probe

Use all eight layer-31 per-head features: 256 dimensions. Standardize each
dimension using the training images only. Inside each generation-position bin of
width 10, deterministically match every hallucinated training mention to a
grounded mention from another image with nearest exact position. Use seed 17 for
ties and do not reuse a grounded mention until the bin pool is exhausted.

For every matched pair form both ordered examples:

    d  = x_hallucinated - x_grounded, label 1
    -d,                              label 0.

Fit L2 logistic regression with no intercept (`C=1.0`) to these differences.
The single-mention test score is `w^T x`; only ranking metrics are reported.
This conditional objective estimates directions that distinguish labels within
the same response-depth stratum rather than fitting the global position prior.

Run five-fold image-grouped OOF with the same folds for:

- raw post-RoPE features;
- no-RoPE features;
- clean no-RoPE plus sink deletion;
- clean object-only shape features, excluding local-gain features.

No regularization, feature, layer, or head sweep is permitted in iteration 1.

## Metrics and leakage diagnostics

Primary metric: size-weighted within-bin AUROC with width 10.

Also report:

- overall AUROC;
- matched-pair AUROC with maximum position difference 5;
- OOF residual AUROC, subtracting train-fold score means by position bin;
- AUROC in every valid position bin and the fraction of bins above 0.55;
- Spearman correlation between detector score and `gen_pos`, overall and among
  grounded mentions only;
- number of valid local-reference rows and class balance by fold;
- paired image-bootstrap 95% intervals for branch/control differences.

The label-free and supervised variants must remain separate in every table.

## Advance/kill gates

The **label-free route** advances only if clean `score_unsup`:

1. reaches at least 0.70 within-bin AUROC;
2. exceeds IC by at least 0.01 within-bin;
3. exceeds IC on both matched-pair and residual AUROC;
4. beats the raw branch and all-head-mean ablation by at least 0.01 within-bin;
5. receives positive within-bin changes from both no-RoPE and sink deletion.

The **supervised route** advances only if the clean conditional probe:

1. reaches at least 0.75 within-bin AUROC;
2. beats both the frozen raw layer-31 probe and the matched raw conditional
   probe by at least 0.02 within-bin;
3. reaches at least 0.72 matched-pair and 0.68 residual AUROC;
4. beats its clean object-only ablation, showing that local grounding gain adds
   information beyond attention shape;
5. receives positive within-bin changes from both no-RoPE and sink deletion.

If only the supervised route passes, the project has a supervised external
detector, not an unsupervised method. If neither passes, close constructive
attention detection and retain the experiment as support for the negative
analysis. Do not tune head count, layer, local window, sink rule, feature signs,
or regularization after seeing the full result.

## Execution and reporting

Stage A is a 16-image extraction smoke that checks token alignment, raw
faithfulness, finite features, and model-logit identity. Stage B is one full
four-shard extraction followed by the locked offline evaluation above.

Commit implementation before Stage B. Append every Stage-B result, including
failed ablations and leakage diagnostics, to `docs/experiment_results.md` and
append the decision to `docs/iteration_log.md`. Feature caches, OOF predictions,
and bootstrap samples remain under ignored experiment storage.
