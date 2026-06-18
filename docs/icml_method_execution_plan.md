# ICML Method Execution Plan

Date: 2026-06-18

This note turns the current audits into an execution plan for a complete ICML
submission. It assumes the current result gates are authoritative:

- `mitigation/scripts/audit_qwen25vl_replication.py` for Qwen2.5-VL all-splits
  replication;
- `scripts/build_tdev_ablation_summary.py` for the paper-facing TDEV ablation
  table and machine-readable summary.

## Current Paper-Safe Claims

1. **The failure mode is real.** Related-present negatives are consistently
   harder than plain-absent negatives under LLaVA and Qwen2.5-VL.
2. **Attention/decoding shortcuts are insufficient.** Mean attention, attention
   redistribution, VCD-greedy, local SPIN settings, and local DAMRO settings do
   not pass semantic-neighbor controls.
3. **Target-discriminative region evidence works better than generic grounding.**
   Raw OWLv2 target evidence is strong but over-fires on related-present
   negatives; target-vs-neighbor verification reduces those errors.
4. **The strongest current method shape is asymmetric.** Use TDEV mainly to
   verify unsafe positive claims, and only rescue `no` answers under very high
   target evidence.
5. **The claim must stay scoped.** OWLv2 is an evidence backend, not the
   contribution. Qwen evidence is output-level plus model-independent region
   verification, not Qwen internal attention evidence.

## Method to Present

Working name: **TDEV**, Target-Discriminative Evidence Verification.

The method should be framed as a decision criterion plus a practical verifier:

```text
grounded(o, image) := E(target=o) is high enough
                      and E(o) exceeds evidence for semantic neighbors N(o)
```

For POPE-style yes/no claims:

- positive gate: suppress a model `yes` unless target evidence is high or
  medium target evidence has a sufficient target-vs-neighbor margin;
- conservative rescue: flip a model `no` only when target evidence and margin
  are both very strong.

For caption object mentions:

- use target absence and neighbor dominance as a continuous hallucination score;
- evaluate under within-bin, matched-pair, and residual controls, not only
  overall AUROC.

This gives the paper a constructive method while keeping the main insight
backend-agnostic.

## Practicality Path

The ICML version should show two tiers:

| Tier | Role | Current status | Why it matters |
|---|---|---|---|
| TDEV-region | strongest verifier using cached OWLv2 image-object scores | positive POPE and CHAIR evidence exists | establishes the target-discrimination criterion |
| TDEV-lite | cheaper internal/proposal backend | not implemented yet | answers the practicality concern that OWLv2 is an external detector |

TDEV-lite should not be a weaker restatement of CLIP margin; those variants are
already negative. The best next implementation target is a late-query/internal
probe inspired by HALP, but trained/evaluated under semantic-neighbor controls:

1. Cache late query-token states or selected visual-token states at object
   decision points.
2. Train a small probe on held-out split labels or target-vs-neighbor evidence
   labels.
3. Report whether the probe reduces related-present FPR at fixed TPR.
4. Treat this as a practicality ablation unless it matches TDEV-region.

## Baseline Priority

| Priority | Baseline | Action |
|---|---|---|
| P0 | OPERA | Keep guarded integration; only report numbers if the environment passes `check_opera_support.py`. |
| P0 | CAI/CAST | Treat as the most relevant head-selection baseline. As of the current check, arXiv pages expose no direct code link; run a subset semantic-neighbor audit if code appears. |
| P0 | Region-Aware Attention Recalibration | Closest region/head recalibration baseline. The arXiv page says code will be public; monitor and audit related-present negatives when available. |
| P1 | HALP-style probe | If official code is unavailable, implement a local late-query probe as TDEV-lite rather than as a direct paper-to-paper reproduction. |
| P1 | Official VCD/GLSim parity | Optional parity checks; current controlled versions are enough for mechanism claims if wording stays scoped. |
| P2 | Woodpecker/UNIHD/Volcano | Discuss as high-latency tool/revision systems unless the paper needs a broad post-hoc correction comparison. |

## Next Experiments

1. **Caption mitigation beyond the proxy.** The object-mention filter proxy now
   shows useful hallucinated-mention removal at low grounded-mention loss. The
   next step is a real caption edit or decoding integration followed by CHAIR
   re-evaluation after text changes.
2. **TDEV-lite probe.** Build a small cached-feature probe on LLaVA decision
   states and evaluate it against related-present negatives.
3. **Third-model sanity check.** If compute allows, run only vanilla plus fixed
   TDEV on InternVL or LLaVA-NeXT; do not rerun every baseline.
4. **Baseline availability check.** Re-check CAI/CAST/region-aware code before
   freezing experiments. If unavailable, explicitly mark them as closest
   concurrent related work and compare conceptually.

## Writing Position

The paper should not claim "external detection solves hallucination." A stronger
and more defensible ICML framing is:

> Hallucination mitigation fails when it optimizes visual attention, grounding,
> or language-prior reduction without checking whether the evidence uniquely
> supports the target object. TDEV operationalizes this missing target-vs-
> neighbor verification criterion and shows that it improves both yes/no
> mitigation and object-mention detection under semantic-neighbor controls.

## Sources Checked

- CAI, arXiv:2506.23590, https://arxiv.org/abs/2506.23590
- CAST, arXiv:2605.04641, https://arxiv.org/abs/2605.04641
- Region-Aware Attention Recalibration, arXiv:2605.24957,
  https://arxiv.org/abs/2605.24957
- HALP, arXiv:2603.05465, https://arxiv.org/abs/2603.05465
