# ICML Method Execution Plan

Date: 2026-06-18

This note turns the current audits into an execution plan for a complete ICML
submission. It should be read together with `docs/claims_alignment_audit.md`,
which is the current hard gate on what the evidence does and does not support.
`docs/icml_evidence_matrix.md` is the paper-facing claim-to-evidence matrix and
should be used as the writing checklist before adding or strengthening claims.
`docs/icml_paper_blueprint.md` is now the structure-level writing blueprint:
use it to decide which results belong in the main paper, which belong in the
appendix, and which claims must stay downgraded.
The current result gates are authoritative:

- `mitigation/scripts/audit_qwen25vl_replication.py` for Qwen2.5-VL all-splits
  replication;
- `scripts/build_tdev_ablation_summary.py` for the paper-facing TDEV ablation
  table and machine-readable summary;
- `mitigation/scripts/build_pope_internal_external_ablation.py` for the current
  internal-vs-external routing/verification ablation.

The paper should be treated as a diagnostic-plus-verification paper, not as a
strong standalone mitigation paper. The motivation remains `looking is not
grounding`: visual routing can be meaningful while failing target-object
verification.


## Result-Quality Decision After Recheck

The latest artifact-level recheck confirms that the results are accurate but
modest. This changes the execution emphasis:

1. Do not spend the next phase trying to make LH-Shape look like a standalone
   detector; current POPE evidence contradicts that role.
2. Do not frame OWLv2/TDEV as a generic external-detector pipeline; related work
   already covers high-latency tool validation and correction.
3. Prioritize experiments and figures that expose the target-vs-neighbor
   failure: related-present examples, target evidence versus best-neighbor
   evidence, and attention/decoding baselines that still answer `yes`.
4. Treat TDEV as a verifier that partially repairs this failure, plus LH-Shape
   as a cost-saving router into that verifier.
5. The next publishability bottleneck is not another aggregate POPE point; it is
   a cleaner constructive story for caption-style correction and a fair positive
   control against recent head/region steering methods.

## Current Paper-Safe Claims

1. **The failure mode is real.** Related-present negatives are consistently
   harder than plain-absent negatives under LLaVA and Qwen2.5-VL.
2. **Looking is not grounding.** Attention-mass and hand-crafted attention-shape
   scores can track generation position, visual routing, or associated evidence
   without verifying the target object.
3. **Attention/decoding shortcuts are insufficient.** Mean attention, attention
   redistribution, VCD-greedy, local SPIN settings, and local DAMRO settings do
   not pass semantic-neighbor controls.
4. **Target-discriminative region evidence works better than generic grounding.**
   Raw OWLv2 target evidence is strong but over-fires on related-present
   negatives; target-vs-neighbor verification reduces those errors.
5. **The strongest current method shape is asymmetric but modest.** Use TDEV
   mainly to verify unsafe positive claims, and only rescue `no` answers under
   very high target evidence. Report it as reducing semantic-neighbor false
   positives, not as solving hallucination.
6. **The claim must stay scoped.** OWLv2 is an evidence backend, not the
   contribution. LH-Shape is supervised triage, not a standalone internal
   mitigation method. Qwen evidence is output-level plus model-independent
   region verification, not Qwen internal attention evidence.

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

This gives the paper a constructive verification criterion while keeping the
main insight backend-agnostic. It should not be described as a complete
hallucination solution; the current POPE gain is mostly a false-positive
reduction on semantic-neighbor cases.

## Practicality Path

The ICML version should show two tiers:

| Tier | Role | Current status | Why it matters |
|---|---|---|---|
| TDEV-region | strongest verifier using cached OWLv2 image-object scores | positive POPE and CHAIR evidence exists | establishes the target-discrimination criterion |
| TDEV-lite | cheaper internal/proposal backend | calibrated late-head linear readout positive on CHAIR detection; full POPE image-CV transfer is positive as a triage signal | answers the practicality concern that OWLv2 is an external detector |

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

Current implementation target:

1. The full POPE question-token per-head cache is complete on all 9,000 rows.
   Full image-CV transfer shows real signal above prompt-only baselines
   (layers 22+31: MCC 0.559, AUROC 0.853), but present-object FPR is still
   0.205, so it is not a deployable standalone POPE gate. As a suppress-only
   triage layer before TDEV-region, it is useful: `base_yes_selected` reaches
   FPR 0.056 and related-present FPR 0.075 with 2,025/9,000 detector calls,
   close to full TDEV's 0.051 and 0.069. A matched-budget ablation confirms the
   role split: LH-alone suppression collapses TPR to 0.432, while LH-routed
   TDEV at the same 2,025-call budget beats prompt-position and target-length
   routing controls on MCC and related-present FPR.
2. Treat LH-Shape prefiltering as a CHAIR-side practicality result: it can save
   25% of OWLv2 calls while retaining 99.6% of full TDEV top-5 hallucination
   deletions at 75% call rate, but position-only/PAS are competitive at higher
   call rates.
3. Present LH-Shape as an internal calibrated TDEV-lite diagnostic and triage
   ablation unless POPE transfer passes the same semantic-neighbor controls.

## Baseline Priority

The latest availability refresh is recorded in
`docs/baseline_availability_refresh.md`. Use that note as the current gate for
whether a recent method is runnable, a required baseline, or only related work.

| Priority | Baseline | Action |
|---|---|---|
| P0 | OPERA | Keep guarded integration; only report numbers if the environment passes `check_opera_support.py`. |
| P0 | CAI/CAST | Closest caption-query/head-steering baselines. As of the June 18, 2026 refresh, no direct official code link was found. Do not implement an unofficial surrogate; run a bounded semantic-neighbor audit only if official code appears. |
| P0 | Region-Aware Attention Recalibration | Closest region/head recalibration baseline. The arXiv page says code will be public, but no runnable code was found in the refresh. Monitor and audit related-present negatives when available. |
| P1 | Dynamic Multimodal Activation Steering | Relevant activation/head steering baseline; treat as related work unless official code appears and is easy to adapt to the POPE semantic-neighbor split. |
| P1 | HALP-style probe | If official code is unavailable, implement a local late-query probe as TDEV-lite rather than as a direct paper-to-paper reproduction. |
| P1 | Official VCD/GLSim parity | Optional parity checks; current controlled versions are enough for mechanism claims if wording stays scoped. |
| P2 | Woodpecker/UNIHD/Volcano | Discuss as high-latency tool/revision systems unless the paper needs a broad post-hoc correction comparison. |

If any P0 code appears, first run the adversarial semantic-neighbor subset and
report MCC, TPR, FPR, yes rate, related-present FPR, plain-absent FPR, and the
related-minus-plain gap. Full all-split reruns are only justified if the subset
result changes the paper conclusion.

## Paper Table Source Update

The next draft should use
`mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md`
as the main semantic-neighbor mitigation/control table. It is regenerated by:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  scripts/build_semantic_neighbor_control_table.py
```

The table is deliberately stricter than aggregate POPE reporting. It shows that
attention interventions and VCD-greedy do not remove the related-present FPR gap,
raw target-region evidence over-fires on related negatives, strict margin is too
low-recall, and hybrid target-vs-neighbor verification gives the best current
tradeoff. This should be the empirical bridge from the reproduced baseline
failure to the constructive TDEV method.

## Next Experiments

0. **Mechanism alignment figure/table.** Completed in
   `mitigation/results/pope_mechanism_alignment_full/`. The table and generated
   contact sheet select related-present cases where vanilla and attention-only
   interventions answer `yes`, but target evidence is weaker than semantic-
   neighbor evidence and TDEV answers `no`. This reconnects the solution to
   `looking is not grounding` and also keeps the claim modest: TDEV corrects
   39.8% of vanilla related-present false positives, not all failures. The
   contact sheet is regenerated by `mitigation/scripts/build_pope_mechanism_figure.py`.
1. **Caption mitigation beyond the proxy.** The object-mention filter proxy,
   deterministic text-edit proxy, neutral-rewrite proxy, and official PAS CHAIR
   reruns now all show useful hallucination reduction at low caption-length
   cost. On the 4,977-image object-mention scope, the top-5% neutral rewrite
   reduces CHAIRi from 0.1340 to 0.1186 and CHAIRs from 0.4921 to 0.4505 while
   reducing mean length by only 0.134 words; the top-10% deletion branch reduces
   CHAIRi to 0.1048 and CHAIRs to 0.4047. The next step is fluent constrained
   regeneration or decoding integration rather than deterministic placeholders.
2. **TDEV-lite transfer.** Calibrated LH-Shape linear readout is positive on
   CHAIR detection. The CHAIR cascade audit shows it can triage TDEV-region
   calls, but the gain is partly shared by position/PAS prefilters. The full
   POPE image-CV run confirms transfer above prompt-only controls, but the more
   defensible use remains suppress-only TDEV triage, not replacing the verifier.
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

## Novelty Decision After Latest Check

A June 18, 2026 literature refresh changes the positioning but not the current
experimental direction. Recent methods such as Region-Aware Attention
Recalibration/SADI, Conscious Gaze, and Energy-Guided Decoding occupy the
training-free internal-intervention space: they recalibrate attention, select
hidden states, or trigger visual-focus interventions without external detectors.
Therefore, the paper should not frame LH-Shape as the main novelty or as a
standalone attention-calibration competitor.

The defensible contribution is narrower and cleaner:

1. The semantic-neighbor stress protocol exposes failures that aggregate
   attention, yes-rate correction, and generic grounding can miss.
2. TDEV is a target-vs-neighbor verification criterion: evidence must uniquely
   support the queried object, not merely any semantically related object.
3. OWLv2 is one backend for this criterion; LH-Shape is a calibrated internal
   triage signal that reduces backend calls and provides practicality evidence.
4. Any paper table involving LH-Shape should report it as supervised internal
   triage/TDEV-lite, not as training-free mitigation.

The next concrete experiment choice from the literature refresh has now been
partly completed. The internal-vs-external ablation shows that LH-Shape is not a
standalone mitigator, but it is a useful router into target-vs-neighbor
verification: at 2,025/9,000 TDEV calls, LH-routed TDEV reaches MCC 0.754 and
related-present FPR 0.075, compared with prompt-position routing at 0.741/0.091
and target-length routing at 0.743/0.087. This supports the paper framing: do
not build another external detector pipeline next; strengthen the semantic-
neighbor verification criterion and its cheap routing story.

## Sources Checked

- CAI, arXiv:2506.23590, https://arxiv.org/abs/2506.23590
- CAST, arXiv:2605.04641, https://arxiv.org/abs/2605.04641
- Region-Aware Attention Recalibration / SADI, arXiv:2605.24957,
  https://arxiv.org/abs/2605.24957
- Conscious Gaze, arXiv:2512.05546, https://arxiv.org/abs/2512.05546
- Energy-Guided Decoding, arXiv:2507.07731, https://arxiv.org/abs/2507.07731
- HALP, arXiv:2603.05465, https://arxiv.org/abs/2603.05465
