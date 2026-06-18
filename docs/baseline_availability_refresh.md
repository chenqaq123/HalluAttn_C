# Baseline Availability Refresh

Date: 2026-06-18

This refresh records the current status of the closest recent baselines for the
ICML version of **Looking Is Not Verifying**. The goal is to avoid two mistakes:
(1) overclaiming novelty against methods that occupy the same head/attention
steering space, and (2) reporting unfair local surrogates as official
reproductions.

## Current Decision

The closest new work is in training-free attention/head/region steering. These
papers strengthen the related-work pressure, but they do not change the current
experimental plan unless runnable code becomes available. The fair comparison is
not aggregate POPE accuracy alone; it is whether the method lowers
related-present false positives without simply shifting yes rate.

Do not spend the next phase implementing unofficial approximations of CAI, CAST, Focus Matters,
or Region-Aware Attention Recalibration. If code becomes available before paper
freeze, run a bounded semantic-neighbor audit. If code remains unavailable, cite
these papers as closest concurrent/related methods and state that the paper's
claim is an evaluation criterion plus TDEV positive control, not superiority over
unreleased implementations.

## Availability Table

| Method | Source status on 2026-06-18 | Why it matters | Current action |
|---|---|---|---|
| CAI: Caption-Sensitive Attention Intervention | arXiv page exists; no direct official code link found in the current arXiv/search check. | Very close conceptually: uses caption-query attention patterns to enhance visual attention. | P0 related work; audit only if official code appears. |
| CAST: Caption-Guided Visual Attention Steering | arXiv page exists; arXiv code/media section exposes generic code-finder links but no direct official GitHub link in the checked page. | Closest head-steering baseline: probes caption-guided heads and applies steering vectors; claims SOTA with low overhead. | P0 related work; audit semantic-neighbor FPR if code appears. |
| Region-Aware Attention Recalibration | arXiv page states that code will be public. No direct runnable code found in the current check. | Closest region/head recalibration baseline; explicitly targets CHAIR, POPE, and MME with training-free region-aware attention modulation. | P0 related work; monitor for code and then run the subset audit. |
| Focus Matters: Phase-Aware Suppression | arXiv page exists; no direct official code link found in the current arXiv/search check. | Training-free single-forward-pass visual-token/attention suppression with low-latency hallucination mitigation claims; relevant as another internal visual-routing baseline. | P1 related work; audit semantic-neighbor FPR if official code appears. |
| Dynamic Multimodal Activation Steering | arXiv page exists; no direct official code link found in the checked page. | Relevant activation/head steering baseline, but less directly tied to semantic-neighbor target verification. | P1 related work unless official code appears and is easy to run. |

## Required Audit If Code Appears

Run a bounded semantic-neighbor audit before full-scale reruns:

1. Use the same POPE rows and semantic-neighbor split already generated under
   `mitigation/results/semantic_neighbor_audit/`.
2. Evaluate at least the adversarial split first; full random/popular/adversarial
   audit only if the subset result is meaningful.
3. Report accuracy, MCC, TPR, FPR, yes rate, related-present FPR, plain-absent
   FPR, and the related-minus-plain gap.
4. Compare against vanilla, PAI-attention-only, ClearSight, VisAttnSink,
   VCD-greedy, raw OWLv2 target score, two-stage gate, and hybrid gate+rescue.
5. Do not claim failure from aggregate POPE alone. The required question is
   whether the method verifies the queried target rather than amplifying
   associated evidence.

## Paper Positioning

These baselines occupy the space of making visual attention or hidden states more
visually informative. That is not the same claim as TDEV. The paper should draw
the boundary as follows:

- CAI/CAST/Focus Matters/Region-Aware ask how to improve visual routing,
  visual-token filtering, or region-aware attention during inference.
- TDEV asks how to evaluate and operationalize target-vs-neighbor verification
  under semantic-neighbor negatives.
- Our current evidence does not prove those methods fail; it says that any such
  method should be tested under related-present FPR and target-vs-neighbor
  controls before being interpreted as object hallucination mitigation.

## Checked Sources

- CAI: https://arxiv.org/abs/2506.23590
- CAST: https://arxiv.org/abs/2605.04641
- Region-Aware Attention Recalibration: https://arxiv.org/abs/2605.24957
- Focus Matters: https://arxiv.org/abs/2604.03556
- Dynamic Multimodal Activation Steering: https://arxiv.org/abs/2602.21704
