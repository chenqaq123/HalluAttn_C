# XEV Iteration 14 Protocol: Multiscale Alternative-Identity Certificate

Date: 2026-07-19
Status: pre-registered exploratory development protocol

## Motivation

Iteration 12 and 13 establish complementary verification failures. A
full-canvas clear-box view retains context but under-resolves small objects;
an exact crop enlarges object pixels but can remove the context needed to
interpret an imperfect or extremely small grounding box. Selecting one view
after observing its answer would turn scale into an unregistered confidence
signal.

The constructive goal is not to make another permissive target-presence judge.
XEV should issue a sparse absence certificate only when independent visual
verification shows that a target proposal consistently depicts a competing
object. Attention and target-conditioned grounding remain proposal mechanisms;
neither contributes evidence by its location, mass, returned label, or box
presence.

## Hypothesis

An absent queried target that is grounded onto a visible confounder will expose
the same competing identity over nearby object-to-context scales. A real target
may fail at one scale, but should either receive cycle-consistent target support
somewhere or lack a repeated directional conflict against one stable
alternative. Requiring the same target-free alternative at multiple scales
should therefore preserve correction precision while avoiding single-view true
positive vetoes.

## Frozen Certificate

### Proposal stages

1. Reproduce iteration 11 unchanged: no-RoPE attention, layers 18--29, three
   residual top-30%-mass connected proposals, open/closed C/N candidates, and
   strict reversed pairwise local conflicts. Attention only proposes regions.
2. Only for rows with at least one local conflict, ask Qwen3-VL for at most three
   normalized target-conditioned grounding boxes using the iteration-12 prompt.
   Ignore generated labels and non-coordinate fields.
3. Around each valid grounding box, construct exact centered crops at fixed
   linear scales `{1, 2, 4}`. Scale multiplies box width and height before image
   clipping. If clipping produces duplicate pixel boxes, evaluate the image view
   once and never count it as repeated-scale agreement.

### Per-view verification

For each unique crop:

1. without naming the target, list at most six concrete visible identities;
2. compute natural-zero Yes/No margins for the target and candidates;
3. compare the target against up to three positive-margin C alternatives using
   both reversed pairwise questions.

A view has target support only when a target alias appears in its target-free
inventory, target margin is positive, and no C alternative has target-first
margin `<0` with alternative-first margin `>0`.

A candidate is a view-level alternative only when it appears in the target-free
inventory at that view, is classified C, has positive crop margin, and satisfies
that same strict directional conflict. Closed-neighbor membership alone cannot
count toward scale agreement.

### Row-level action

Two candidate phrases match only under the existing deterministic lexical alias
matcher. A **multiscale alternative certificate** exists when the same candidate
is a view-level alternative at two or more distinct pixel crops belonging to the
same grounding box.

Veto baseline Yes only when all conditions hold:

    at least one iteration-11 local conflict exists
    at least one multiscale alternative certificate exists
    no crop has cycle-consistent target support

All margins use zero thresholds. Scale count is a structural replication
requirement, not a score. No attention value, grounding field, box geometry,
number of boxes, candidate-list length, or model confidence enters the action.

## Data and Evaluation

- Use fresh POPE-COCO adversarial rows **2640--2759**: 120 balanced rows / 20
  images. This window was untouched when the protocol was committed and becomes
  development-only after the run.
- Rows 2520--2639 motivate the fixed design but are not used for its gate.
- Rows 2760 onward remain untouched.
- Primary behavior is the full 120-row replay. Report local-conflict coverage,
  valid grounding coverage, per-scale target support, repeated-alternative
  certificates, actions, false-positive corrections, true-positive vetoes, and
  baseline/method TPR, FPR, balanced accuracy, and MCC.
- Diagnostics may report single-scale and any-scale target support, but may not
  replace the registered certificate.

## Development Gate

Advance unchanged to a separately committed confirmatory protocol only if the
multiscale certificate has:

- at least 2 actions;
- correction precision at least 2/3;
- no more than one true-positive veto;
- lower FPR than baseline;
- TPR loss at most 0.005;
- non-decreasing balanced accuracy and MCC.

If the gate fails, separate grounding coverage, target-support protection, and
repeated-alternative coverage. Do not tune scales, expansion geometry, minimum
scale count, candidate K, aliases, relation classes, or non-zero margins on this
window.
