# XEV Iteration 15 Stage-2 Protocol: Fresh Replicated-Target Evaluation

Date: 2026-07-19
Status: superseded before result inspection; requested range exceeds dataset end

## Claim Under Test

The iteration-15 certificate can add high-precision object-hallucination
corrections without the true-positive collapse of broader local-attention
conflicts. Attention and target-conditioned grounding only propose where to
inspect. The action requires replicated negative target evidence, one explicit
target-free competing identity, and failure to find positive target support.

## Frozen Pipeline

### Local proposal

- Qwen3-VL-8B-Instruct, deterministic inference and the existing 200704-pixel
  preprocessing budget.
- One no-RoPE attention proposal from layers 18--29, top-30% connected mass.
- Up to six target-free open candidates, eight closed COCO-NPMI neighbors, and
  up to three positive-margin pairwise candidates.
- A row advances only when baseline predicts Yes and a C candidate has strict
  reversed-prompt direction: target-first margin `<0`, alternative-first margin
  `>0`. All thresholds are zero.

Attention values rank the proposal only. They never enter the certificate.

### Grounded certificate

Only for advancing rows:

1. request at most three normalized target-conditioned grounding boxes and
   ignore all generated labels/non-coordinate fields;
2. form unique centered crops at fixed scales `{1,2,4}`;
3. at each crop, run target-free six-item identity generation, target/candidate
   natural-zero margins, C-only filtering, and reversed pairwise verification.

One grounding box has replicated target-negative evidence when target margin is
negative in at least two distinct pixel views and at least one view has a
target-free open C candidate with positive margin, target-first `<0`, and
alternative-first `>0`. Veto baseline Yes only when such a box exists and no
crop has cycle-consistent target support: target alias named, positive target
margin, and no strict C conflict.

No component may use labels, attention mass, grounding labels, box count or
geometry, proposal confidence, inventory length, or non-zero fitted thresholds.

## Data

- POPE-COCO adversarial rows **2760--3239**: 480 rows / 80 images, expected 240
  yes and 240 no. This window is untouched at protocol commit time.
- Rows 600--1079 are historical development data for stage 1. Rows 2520--2759
  are closed development windows from iterations 9--14.
- Rows 3240 onward remain untouched.
- Run the full local stage on all 480 rows. Run grounding lazily on every strict
  directional C-conflict row, without label-based selection.

## Confirmatory Gate

Adopt the grounded extension only if it has:

- at least 3 actions;
- correction precision at least 2/3;
- no more than 1 true-positive veto;
- lower FPR than baseline;
- TPR loss at most 0.005;
- non-decreasing balanced accuracy and MCC.

Report the complete local-to-grounded funnel, valid-box coverage, target-support
protection, replicated-negative coverage, open-alternative coverage, final
actions, and all baseline/method confusion metrics. If any gate fails, do not
tune scales, candidate counts, relation classes, aliases, sign thresholds, or
minimum negative-view count on this window.

## Protocol Correction

The POPE adversarial split contains 3000 rows, so the requested index range
2760--3239 cannot contain 480 rows. The runner exposed this before analysis by
reporting a 240-row selection. The run was interrupted after nine rows, without
reading any saved case output or method result. This protocol is superseded by
`xev_iteration_15_stage2_amended.md`, which keeps the method unchanged and
pre-registers the actual 240-row remainder with sample-size-adjusted gates.
