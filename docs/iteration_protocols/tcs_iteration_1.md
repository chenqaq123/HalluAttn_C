# TCS Iteration 1 Protocol: Intervention Validity and Mechanism Pilot

Date: 2026-07-19
Status: cancelled before start; superseded by NEA iteration 1

> Design-review cancellation (2026-07-19): no images were edited and no model
> result was produced under this protocol. Four-view binary supervision already
> subsumes much of the proposed sensitivity/invariance behavior, while object
> removal introduces artifact and causal-validity risks. The active protocol is
> [nea_iteration_1.md](nea_iteration_1.md). The remainder is retained as the
> pre-registered historical design.

## Question

Can paired counterfactual supervision teach Qwen3-VL to depend on the queried
object rather than on a visible semantic neighbor, without obtaining an
apparent gain by shifting all answers toward `No`?

This is the first test of Target-Causal Selectivity (TCS). It is not a search
over XEV thresholds, proposals, crops, aliases, or verifier prompts. XEV is a
closed route.

## Model and Decision Score

- Base model: local `Qwen3-VL-8B-Instruct` checkpoint.
- Prompt: the existing POPE yes/no prompt contract, with the first generated
  answer restricted to `Yes` or `No` for margin measurement.
- Decision score:

      m(t, I) = log p(Yes | q_t, I) - log p(No | q_t, I)

- Vision encoder remains frozen. The training pilot uses decoder LoRA on
  attention `q/k/v/o` projections only, rank 16, alpha 32, dropout 0.05.
- Random seed 17; one epoch; learning rate `2e-5`; effective batch size 32
  triples. Any unavoidable implementation change must be recorded in an
  amended protocol before model outcomes are inspected.

## Category Split and Pair Selection

Use COCO train2017 instance annotations. Sort the 80 categories by COCO category
ID; every fourth category is held out as an intervention target (20 categories),
and the other 60 are training targets. Held-out categories may occur as
neighbors but never as the queried/removed target in training.

For each target `t`, eligible neighbors `n` must:

- have positive NPMI with `t`, computed from train2017 image-level presence;
- co-occur with `t` in at least 100 train2017 images;
- be among the five highest-NPMI eligible categories for `t`;
- have a union instance mask disjoint from the target union mask after the
  fixed inpainting dilation, so one edit does not directly erase the other.

An image is eligible only when the target and neighbor union masks each occupy
0.5%--30% of image area and every annotated instance of the removed category can
be included in the edit mask. Sample at most one `(t,n)` triple per source image.
Freeze the source image IDs, categories, masks, and natural-negative image IDs
in a committed manifest before any model run.

For each source image `I`, construct:

- `I^-t`: remove all instances of `t`, preserving `n`;
- `I^-n`: remove all instances of `n`, preserving `t`;
- `I^-b`: remove an annotated-object-free background region matched to the
  target mask area and aspect ratio;
- `I_abs`: a natural train2017 image containing `n` but no annotated `t`.

Use one pinned LaMa checkpoint and the same mask dilation and inpainting
configuration for all edited branches. Save checkpoint identity, code revision,
and edit seed in the manifest. Inpainting is data construction only and is not
used at inference.

## Stage A: Intervention Audit

Build 200 candidate triples, stratified across training and held-out target
categories. Before running Qwen3-VL, blindly inspect a fixed random sample of
100 triples and record four binary judgments:

1. target present in `I` and `I^-n`;
2. target absent in `I^-t`;
3. neighbor present in `I` and `I^-t`;
4. no conspicuous edit artifact that alone reveals which branch was edited.

The data pipeline advances only if at least 90/100 triples satisfy judgments
1--3 and at least 85/100 satisfy judgment 4. Otherwise repair the data pipeline
without examining model margins and pre-register a replacement audit.

On accepted triples, report frozen-model distributions of:

    Delta_t = m(t,I) - m(t,I^-t)
    Delta_n = m(t,I) - m(t,I^-n)
    Delta_b = m(t,I) - m(t,I^-b)
    TCS     = Delta_t - |Delta_n|

Report means, medians, bootstrap 95% intervals, answer-flip counts, and results
separately for training and held-out target categories. These measurements are
diagnostic; edit validity, not a favorable base-model effect, is the Stage-A
gate.

## Stage B: LoRA Mechanism Pilot

After Stage A passes, create up to 2,000 training triples from the 60 training
target categories and a disjoint 400-triple mechanism test from the 20 held-out
target categories. Train the following adapters from the identical base model:

1. **Natural SFT:** binary loss on `I` and `I_abs` only.
2. **Four-view SFT:** binary loss on `I`, `I^-t`, `I^-n`, and `I_abs`, with no
   paired margin regularizer.
3. **Target-only:** binary loss on `I`, `I^-t`, and `I_abs`, plus
   `0.5 * softplus(1 - Delta_t)`.
4. **Full TCS:** four-view binary loss plus
   `0.5 * softplus(1 - Delta_t) + 0.5 * smooth_l1(Delta_n, 0)`.

Training example order, optimizer, number of updates, and natural examples are
matched wherever their branches overlap. No hyperparameter sweep is allowed in
this iteration.

## Evaluation and Pre-Registered Gate

Primary mechanism evaluation is the 400-triple held-out-category test. Primary
behavior evaluation is POPE-popular through `lmms-eval`. Report POPE by label,
trained versus held-out target category, and related-present versus plain-absent
negative subtype. POPE-adversarial is not a development set and is not rerun in
this iteration.

Full TCS advances only if all conditions hold:

1. Its held-out-category mean TCS exceeds both four-view SFT and target-only,
   with a paired bootstrap 95% interval excluding zero.
2. On POPE-popular, related-present FPR decreases, TPR drops by at most 0.005,
   and both balanced accuracy and MCC exceed the frozen base model.
3. The predicted mechanism is selective: `Delta_t` increases relative to the
   base model while `|Delta_n|` does not increase, and the change is larger than
   the matched-background `|Delta_b|` change.
4. The overall yes-rate does not fall by more than 0.01. This prevents a global
   `No` shift from satisfying the FPR gate.

If target-only matches full TCS on held-out TCS, if four-view SFT explains the
gain, or if the gain is only a yes-rate reduction, reject the claimed paired
selectivity mechanism. Only after this gate passes may the unchanged adapter be
evaluated on one untouched confirmation task: POPE-random or MME existence.

## Required Reporting

Append the edit audit, data attrition, all frozen-model margins, all adapter
metrics, and the gate decision to `docs/experiment_results.md`. Record the route
decision in `docs/iteration_log.md`. Generated images, manifests with local
paths, checkpoints, and raw result files stay under ignored experiment storage.
