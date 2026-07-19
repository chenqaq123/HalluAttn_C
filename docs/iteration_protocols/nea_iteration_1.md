# NEA Iteration 1 Protocol: Null Routing Before Null Values

Date: 2026-07-19
Status: cancelled before start; superseded by EEA iteration 0

> Historical protocol only. No model inference or training was run under this
> protocol. NEA was cancelled because its learned null branch is functionally an
> absence-conditioned correction to the ordinary answer and is not sufficiently
> distinct from contrastive/calibrated decoding. The active protocol is
> [eea_iteration_0.md](eea_iteration_0.md).

## Question

Can a frozen Qwen3-VL attention head distinguish "some visual token supports the
queried object" from "the visual branch is allocated to a related but wrong
object" when it is given an explicit null competitor?

This iteration tests Null-Evidence Attention (NEA). It must establish transferable
null routing before any learned null value is allowed to change model behavior.

## Frozen Model Contract

- Model: `/home/chenguanxu/models/Qwen3-VL-8B-Instruct`.
- Runtime: Transformers 4.57.6, bf16, eager attention, deterministic
  200704-pixel image preprocessing.
- Prompt: the existing direct POPE yes/no format. The measured row is the final
  prompt row that predicts the first answer token.
- Candidate layers: text-decoder layers 18--29. No other layer may be added in
  this iteration.
- Backbone, vision encoder, projector, LM head, and ordinary Q/K/V/O projections
  remain frozen. No LoRA is permitted.

For each candidate layer/head, retain the original post-RoPE attention weights
`a`, visual-token set `V`, and original visual mass

    rho = sum_{i in V} a_i.

Capture the normalized query and visual keys immediately before multimodal RoPE.
Define token-count-normalized visual content energy and null logit

    e_vis  = logmeanexp_{i in V}(<q^0, k_i^0> / sqrt(d))
    z_null = <q^0, kappa_lh> / sqrt(d) + b_lh
    p_null = sigmoid(z_null - e_vis).

`kappa_lh` and `b_lh` are the only Stage-A trainable parameters. The normalization
by visual-token count prevents dynamic image resolution from becoming the score.

## Natural-Image Data

Use COCO train2017 instance annotations only. Sort the 80 categories by category
ID; every fourth category is a held-out target category (20), leaving 60 training
target categories. A held-out category may be visible as context but is never the
queried target in a training row.

For each target, construct three image types without editing any pixels:

1. **present:** the target has at least one annotated instance;
2. **related absent:** the target is absent and at least one of its top-five
   positive-NPMI neighbors is annotated as present;
3. **random absent:** the target and all top-five positive-NPMI neighbors are
   absent.

NPMI is computed from train2017 image-level category presence. An eligible
neighbor must co-occur with the target in at least 50 images. This threshold is
fixed from an annotation-only feasibility audit: at 100, `bear` and `hair drier`
have no eligible neighbor; at 50, all 80 categories have at least one. It is not
tuned against model outcomes. Each category is
balanced 1:0.5:0.5 across present, related-absent, and random-absent rows. Sample
at most one query per image in each split.

Freeze four manifests before model inference:

- `routing_train`: 1,200 rows from the 60 training target categories;
- `routing_transfer`: 800 rows from the 20 held-out target categories;
- `value_train`: 2,400 new rows from the 60 training target categories;
- `mechanism_test`: 800 new rows from the 20 held-out target categories.

Select held-out-category rows first. No image ID in either held-out manifest may
appear in a training manifest, and no image ID may occur in more than one
manifest. Store target, subtype, visible neighbor categories, image ID, and prompt
in the committed manifest; local image paths remain in ignored storage.

## Stage A: Routing Feasibility

Run the frozen model on `routing_train` and cache, for every candidate
layer/head, `q^0`, `e_vis`, `rho`, category, label, subtype, and image ID. Fit
`kappa_lh,b_lh` independently for every head with balanced binary cross-entropy,
where absent targets have null label 1 and present targets null label 0.

Use image-grouped five-fold cross-validation on `routing_train`. Rank head-layer
pairs by mean macro AUROC on present versus related-absent rows and freeze the top
eight pairs before opening `routing_transfer` outcomes.

Required controls for each selected head and their mean ensemble:

- **query-only:** logistic regression on `q^0`, without `e_vis`;
- **energy-only:** logistic regression on `e_vis`, without `q^0`;
- **post-RoPE mass:** scalar `rho`;
- **key-energy shuffle:** permute `e_vis` across images within target category
  while preserving `q^0`, labels, and subtype;
- **counterlabel image swap:** pair present and absent rows within each target,
  exchange their images, rerun the frozen model, and score the recomputed
  `q^0,e_vis,rho` against the original labels. This deliberately makes the
  supplied image contradict each label and tests the complete route rather than
  only the explicit energy term.

Evaluate the frozen eight-head ensemble once on `routing_transfer`, reporting
category-macro AUROC, balanced accuracy at the training-only threshold, subtype
metrics, paired bootstrap intervals by image, and per-category dispersion.

Stage A passes only if all conditions hold on held-out target categories:

1. present versus related-absent macro AUROC is at least 0.70;
2. full null odds exceed the better of query-only and energy-only AUROC by at
   least 0.03, with the paired bootstrap 95% interval above zero;
3. key-energy shuffling lowers AUROC by at least 0.03, and counterlabel image
   swapping lowers it by at least 0.10;
4. no more than five of the twenty categories have AUROC below 0.55.

If any condition fails, reject NEA before implementing the behavior-changing
operator. Do not add heads, layers, LoRA, box supervision, or fitted subtypes.

## Stage B: Mass-Preserving Null Values

Run Stage B only after Stage A passes. For each selected head, gate only the
original conditional visual contribution at the answer-prediction row:

    a_bar_i = a_i / rho
    g_null = p_null
    g_eff = alpha_lh * g_null
    o_NEA = o_text + rho * ((1 - g_eff) * sum_i a_bar_i v_i
                            + g_eff * nu_lh).

The original text contribution and `rho` are unchanged. Constrain
`0 <= alpha_lh <= 1`, initialize `alpha_lh=0`, and verify that initial NEA logits
match vanilla within `1e-5`. Project `alpha_lh` back to `[0,1]` after each
optimizer update. Parameterize each null value as a unit direction times a
non-negative scalar capped at that head's running median visual-value norm.

Train only the eight `kappa`, `b`, `alpha`, and bounded `nu` parameter sets on
`value_train`, initialized from Stage A. Use seed 17, AdamW, learning rate
`1e-3`, no weight decay, effective batch size 32, and one epoch. The frozen loss
is

    L = L_answer + 0.2 * L_route,

where `L_answer` is first-token Yes/No cross-entropy and `L_route` is balanced
BCE on the mean selected-head null probability. There is no hyperparameter sweep.

## Required Comparisons

Train or evaluate all controls on the same `value_train` rows:

1. frozen vanilla Qwen3-VL;
2. fixed zero-null route: `kappa=0`, `b=0`, `alpha=1`, and `nu=0`, with no
   training;
3. trained routing with `nu=0` (zero-value NEA);
4. one learned Yes/No logit-bias scalar;
5. a parameter-matched gated residual adapter on the same answer rows: for each
   selected head use `sigmoid(<q^0,r_lh>+c_lh) * u_lh` with one bounded strength,
   matching NEA's parameter count within 5% and with no access to visual keys;
6. full bounded-value NEA.

The residual adapter is the critical generic-steering control. Its rank and exact
parameter count must be committed before training.

## Evaluation and Gate

First evaluate all methods on the unopened `mechanism_test`, then on POPE-popular
through `lmms-eval`. Report POPE confusion metrics, balanced accuracy, MCC,
yes-rate, and negative subtypes from COCO annotations. Report trained-target and
held-out-target categories separately. POPE-adversarial is not run.

Full NEA advances only if all conditions hold:

1. On `mechanism_test`, null routing retains present-versus-related-absent macro
   AUROC at least 0.70, the key-energy-shuffle drop at least 0.03, and the
   counterlabel-image-swap drop at least 0.10.
2. On POPE-popular, related-present FPR falls by at least 0.02, TPR falls by no
   more than 0.005, balanced accuracy and MCC both increase, and absolute
   yes-rate change is at most 0.01.
3. At matched TPR, full NEA has lower related-present FPR and higher MCC than
   zero-value NEA, the logit-bias control, and the parameter-matched residual
   adapter.
4. At least ten net baseline errors are corrected, preventing another sparse
   four-action result from passing on aggregate rounding.
5. Null probability rises primarily on corrected related-present negatives; it
   must not rise comparably on newly broken positives.

If full NEA fails any gate, do not tune a behavior threshold or broaden the
backbone update. If it passes, freeze the method and run exactly one untouched
confirmation task: POPE-random or MME existence.

## Required Reporting

Commit the manifests and implementation before inference. Append Stage-A and
Stage-B results, including failures, to `docs/experiment_results.md`; append the
decision to `docs/iteration_log.md`. Feature caches, raw model outputs, adapter
weights, and `lmms-eval` artifacts remain under ignored experiment storage.
