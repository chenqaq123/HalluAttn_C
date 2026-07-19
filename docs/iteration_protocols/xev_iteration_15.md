# XEV Iteration 15 Protocol: Replicated Target-Negative Evidence

Date: 2026-07-19
Status: stage-1 mechanism gate passed; fresh stage 2 requires a separate protocol

## Motivation

Iteration 14 shows that a grounded scale pyramid contains useful target support
and alternative conflicts, but exact alternative phrases do not repeat reliably
across scales. The stable hypothesis across all views is the queried target, not
the generated candidate string. Replication should therefore apply to target
evidence, while a target-free competing identity remains an independent
certificate that the box depicts something else.

This change does not relax the semantic or directional verifier. One negative
target margin is insufficient, and repeated negative margins without an explicit
alternative are insufficient. Attention and target-conditioned grounding still
only determine where verification runs.

## Hypothesis

For an absent target grounded onto a confounder, target evidence will be negative
at multiple object-to-context scales, and at least one scale will expose a
target-free C alternative with a strict reversed-prompt identity switch. For a
real target, any-scale target support should block the action; otherwise the
conjunction should be substantially rarer than either negative margins or a
single local conflict alone.

## Frozen Certificate

Start only from a saved one-region strict directional C conflict:

    baseline predicts Yes
    local candidate relation is C
    local target-first margin < 0
    local alternative-first margin > 0

For the queried target, generate at most three grounding boxes and the same
unique centered `{1,2,4}` crop pyramid as iteration 14. At every view, retain the
same target-free six-item inventory, natural-zero margins, C-only relation gate,
top-3 positive candidate limit, and reversed pairwise questions.

For one grounding box, **replicated target-negative evidence** requires:

1. target margin `<0` in at least two distinct pixel crop views;
2. at least one view has a target-free open C candidate with positive margin,
   target-first margin `<0`, and alternative-first margin `>0`.

The alternative need not have the same phrase at another scale. Closed-neighbor
membership alone cannot satisfy condition 2. A row is vetoed only when such a
box exists and **no** pyramid view has cycle-consistent target support (target
alias, positive target margin, and no C conflict).

All thresholds are zero. The minimum two-view count is a structural replication
requirement. Target-conditioned labels, grounding response fields, box geometry,
attention values, inventory length, and proposal confidence do not enter the
decision.

## Data and Stages

- **Stage 1 mechanism audit:** reuse the ten saved strict directional C-conflict
  rows from iteration-4 POPE-adversarial rows 600--1079: seven real targets and
  three absent targets. These labels and rows are historical development data;
  this stage cannot support a benchmark or generalization claim.
- Do not rerun or alter the saved local attention conflicts. Grounding and
  pyramid inference are new, but their rule is frozen by this commit.
- **Stage 2:** only if stage 1 passes, pre-register the unchanged full pipeline
  on fresh rows 2760--3239 in a separate commit. Rows 2640--2759 are closed by
  iteration 14 and cannot validate this rule.

## Stage-1 Gate

Advance only if the grounded certificate:

- preserves at least 2 of the 3 absent-target corrections;
- vetoes at most 1 of the 7 real targets;
- has correction precision at least 2/3.

Report valid boxes, target-support protection, repeated-negative coverage,
open-C alternative coverage, and final actions separately. If the gate fails,
do not tune scales, minimum negative-view count, candidate K, relation classes,
aliases, or non-zero margins on these ten rows.

## Stage-1 Result

All ten saved conflict rows return valid boxes. Six rows receive any-scale target
support, two have target-negative margins in at least two scales, four have an
open C strict directional alternative somewhere, and two satisfy the joint
certificate. The two actions are absent `handbag -> backpack` and absent
`bicycle -> motorcycle`; both are corrections. No real target is vetoed, giving
2/2 correction precision and passing every stage-1 gate.

On the historical 480-row replay, baseline `TP/FP/TN/FN = 198/20/220/42`
becomes `198/18/222/42`: TPR remains 0.8250, FPR falls 0.0833 -> 0.0750,
balanced accuracy rises 0.8708 -> 0.8750, and MCC rises 0.7448 -> 0.7538.
These are development numbers. The real `truck` row without target support does
not form the joint certificate and remains unchanged. No parameter was changed
after inference.
