# XEV Iteration 15 Stage-2 Amended Protocol: Final POPE Remainder

Date: 2026-07-19
Status: pre-registered confirmatory protocol; supersedes invalid 480-row range

## Amendment Scope

The POPE adversarial split has exactly 3000 rows. The only untouched remainder
after iteration 14 is index range **2760--2999**: 240 rows / 40 images / 120 yes
and 120 no, with question IDs 2761--3000. The original stage-2 protocol requested
rows beyond the dataset boundary and is invalid as a 480-row protocol.

The runner revealed the 240-row selection and was interrupted after nine rows.
No saved case output, conflict identity, label-conditioned result, aggregate
method metric, or gate outcome was inspected before this amendment. Those nine
raw rows may be reused by deterministic resume after this commit.

## Frozen Pipeline

The model and method are unchanged from the superseded protocol:

- Qwen3-VL-8B-Instruct with deterministic 200704-pixel preprocessing;
- one no-RoPE layers-18--29 top-30%-mass connected attention proposal;
- six target-free open candidates, eight closed COCO-NPMI neighbors, and at most
  three positive-margin pairwise candidates;
- advance only baseline-Yes rows with a strict directional C conflict;
- lazily generate at most three target grounding boxes, ignore labels/non-box
  fields, and form unique centered `{1,2,4}` crops;
- require target margin `<0` in at least two views of one box plus at least one
  target-free open C alternative with positive margin, target-first `<0`, and
  alternative-first `>0`;
- veto only when no crop has target alias recovery, positive target margin, and
  no strict C conflict.

All thresholds remain zero. Attention and grounding remain proposal-only.

## Corrected Data and Gate

- Evaluate all 240 rows at indices 2760--2999. There is no later POPE-adversarial
  row to reserve.
- Run grounding on every strict directional C-conflict row without label-based
  selection.
- Report the full proposal/verification funnel and all confusion metrics.

The sample-size-adjusted confirmatory gate requires:

- at least 2 actions;
- correction precision at least 2/3;
- no more than 1 true-positive veto;
- lower FPR than baseline;
- TPR loss at most 0.01, allowing at most one error among 120 positives;
- non-decreasing balanced accuracy and MCC.

The action and error counts are the proportional 240-row counterparts of the
invalid 480-row protocol. No model output informed this adjustment. If any gate
fails, do not tune the method on this final POPE-adversarial remainder.
