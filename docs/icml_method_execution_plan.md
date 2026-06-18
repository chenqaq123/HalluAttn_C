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
| TDEV-lite | cheaper internal/proposal backend | calibrated late-head linear readout positive on CHAIR detection; POPE transfer pending | answers the practicality concern that OWLv2 is an external detector |

TDEV-lite should not be a weaker restatement of CLIP margin; those variants are
already negative. The current internal-probe audit shows that late-layer
per-head attention-shape features preserve position-controlled signal: layer 31
image-grouped CV reaches 0.726 within-bin AUROC and 0.719 matched-pair AUROC on
CHAIR object mentions, while the mean-head probe is only 0.545 within-bin. This
is evidence that a practical no-detector route exists, but it remains supervised
and diagnostic.

A first LH-Shape split-selected ablation is negative: train-fold selected
layer31 top-5 head/features reach only 0.597 within-bin AUROC and 0.606
matched-pair AUROC, far below the layer31 logistic probe. Training-free
late-layer mean-head features are near random after controls. This rules out the
simplest fixed-head averaging route.

The calibrated LH-Shape linear readout is positive. With image-grouped folds,
layers 22+31 reach 0.755 within-bin AUROC, 0.745 matched-pair AUROC, and 0.664
residual AUROC on CHAIR object mentions, ahead of IC (0.690/0.703/0.633) under
the same metric implementation. This gives the paper a practical internal
variant, but it is supervised calibration and must be separated from the
training-free baseline table.

Next implementation target:

1. Build a POPE question-token per-head cache before claiming LH-Shape transfer
   to semantic-neighbor yes/no gating; no such cache exists in the current
   artifacts.
2. Treat LH-Shape prefiltering as a CHAIR-side practicality result: it can save
   25% of OWLv2 calls while retaining 99.6% of full TDEV top-5 hallucination
   deletions at 75% call rate, but position-only/PAS are competitive at higher
   call rates.
3. Present LH-Shape as an internal calibrated TDEV-lite diagnostic and triage
   ablation unless POPE transfer passes the same semantic-neighbor controls.

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

1. **Caption mitigation beyond the proxy.** The object-mention filter proxy,
   deterministic text-edit proxy, and official post-edit PAS CHAIR rerun now
   all show useful hallucination reduction at low caption-length cost. On the
   4,977-image object-mention scope, the top-5% hybrid branch reduces CHAIRi
   from 0.1340 to 0.1186 and CHAIRs from 0.4921 to 0.4505; the top-10% hybrid
   MCC branch reduces CHAIRi to 0.1048 and CHAIRs to 0.4047. The next step is
   a fluent rewrite or decoding integration rather than raw phrase deletion.
2. **TDEV-lite transfer.** Calibrated LH-Shape linear readout is positive on
   CHAIR detection. The CHAIR cascade audit shows it can triage TDEV-region
   calls, but the gain is partly shared by position/PAS prefilters; POPE
   semantic-neighbor transfer still requires a question-token per-head cache.
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
