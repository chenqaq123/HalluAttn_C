# Baseline Availability Refresh

Date: 2026-06-19

This refresh records the current status of the closest recent baselines for the
ICML version of **Looking Is Not Verifying**. The goal is to avoid two mistakes:
(1) overclaiming novelty against methods that occupy the same head/attention,
decoding, or detector/revision space, and (2) reporting unfair local surrogates
as official reproductions.

## Current Decision

The closest new work is still dominated by training-free attention/head/region
steering and decoding-time language-prior suppression. These papers strengthen
the related-work pressure, but they do not change the central evaluation axis:
aggregate POPE or CHAIR improvement is insufficient unless the method reduces
related-present false positives without simply shifting answer priors, shortening
captions, or replacing specific claims with vague text.

**Run NoLan next as a bounded P0 baseline if its public code integrates cleanly.**
It is a public-code decoding baseline and directly tests whether dynamic
language-prior suppression fixes the same failure mode. The first run should be
the adversarial semantic-neighbor subset; full all-split POPE/CHAIR reruns are
justified only if the subset lowers related-present FPR without a yes-rate or
caption-length shortcut. The compatibility caveat is recorded in
`docs/nolan_baseline_feasibility.md`: the official code targets an older
transformers/torch stack and monkey-patches sampling, so it needs a guarded port
or isolated environment before use.

Do not spend the next phase implementing unofficial approximations of CAI, CAST,
Focus Matters, BRACS, AIR, or Region-Aware Attention Recalibration. If official
code is directly available and easy to adapt, run the bounded semantic-neighbor
audit. If code remains unavailable or unclear, cite these papers as closest
concurrent/related methods and state that our claim is a diagnostic plus
verification criterion, not superiority over unreleased implementations.

## Availability Table

| Method | Source status on 2026-06-19 | Why it matters | Current action |
|---|---|---|---|
| NoLan: No-Language-Hallucination Decoding | Public GitHub repository found: `https://github.com/lingfengren/NoLan`; README says code released and supports LLaVA-1.5/InstructBLIP/Qwen-VL integration. | Closest runnable decoding baseline: suppresses language priors by comparing multimodal and text-only distributions. This directly tests whether language-prior suppression fixes related-object false positives. | P0 runnable candidate. Clone/inspect in `ref/`, then run adversarial semantic-neighbor subset before any full rerun. |
| AIR: Attention Imbalance Rectification | arXiv v2, 2026-06-14, says CVPR 2026 Findings and that code is available via a GitHub link, but a stable repo URL was not recovered from search in this refresh. | Strong attention-reallocation baseline across CHAIR, POPE, and MM-Vet; conceptually close to our attention-proxy critique. | P0 monitor. Do not implement a surrogate; find official repo and run subset audit if available. |
| BRACS: Barrier-Regulated Adaptive Closed-form Steering | arXiv page exists; no direct official code link found in the checked page. | Strong recent steering baseline; explicitly claims adaptive intervention when grounding deteriorates and reports CHAIR/POPE gains. | P1 related work unless code appears. Audit with semantic-neighbor FPR if runnable. |
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

These baselines occupy three neighboring spaces:

- CAI/CAST/Focus Matters/Region-Aware/AIR/BRACS ask how to improve visual
  routing, hidden-state steering, visual-token filtering, or regional attention
  during inference.
- NoLan asks whether dynamic language-prior suppression is enough to reduce
  object hallucination without an explicit object verifier.
- TDEV asks how to evaluate and operationalize target-vs-neighbor verification
  under semantic-neighbor negatives.

Our current evidence does not prove those methods fail. It says that any such
method should be tested under related-present FPR and target-vs-neighbor controls
before being interpreted as object hallucination mitigation. This keeps the paper
connected to the original motivation: **looking is not grounding, and stronger
routing is not the same as verifying the queried target.**

## Checked Sources

- NoLan: https://arxiv.org/abs/2602.22144 and https://github.com/lingfengren/NoLan
- AIR: https://arxiv.org/abs/2603.24058
- BRACS: https://arxiv.org/abs/2605.29881
- CAI: https://arxiv.org/abs/2506.23590
- CAST: https://arxiv.org/abs/2605.04641
- Region-Aware Attention Recalibration: https://arxiv.org/abs/2605.24957
- Focus Matters: https://arxiv.org/abs/2604.03556
- Dynamic Multimodal Activation Steering: https://arxiv.org/abs/2602.21704
