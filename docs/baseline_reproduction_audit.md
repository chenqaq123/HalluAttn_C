# Baseline Reproduction Audit

This note records the current baseline coverage for the ICML paper and the next
baselines that should be reproduced under the same controlled metrics. It is a
workflow artifact, not a replacement for the paper's Related Work section.

## Current Experimental Coverage

Detection/verification baselines currently audited in paper tables:

- NLL and entropy uncertainty scores.
- IC and GLSim representation scores.
- SVAR, PAS, and Beyond-ADS/CGC attention/patch-grounding scores.
- SinkDetect attention-shape stress test.
- TDEV with OWLv2 region evidence for CHAIR mention detection.

Mitigation/gating baselines currently audited in paper tables:

- Vanilla LLaVA-1.5-7B.
- PAI attention-only component, excluding its CFG/contrastive branch.
- ClearSight VAF component.
- Visual Attention Sink redistribution component.
- VCD-greedy controlled decoding baseline on all POPE splits.
- TDEV direct and vanilla-gate rules on POPE semantic-neighbor splits.

Tracked mitigation baselines not yet in paper tables:

- DAMRO controlled outlier-token contrastive decoding port: POPE-random smoke
  and POPE-adversarial 120-row subset audit complete; full POPE/CHAIR audit
  pending.
- SPIN controlled head-suppression port: POPE-random smoke plus default and
  mild POPE-adversarial 120-row subset audits complete; full POPE/CHAIR audit
  pending.

Current evidence supports the paper's scoped claim: aggregate attention mass,
unselective attention intervention, and controlled VCD-greedy decoding do not
verify target-object presence; target-discriminative region evidence survives
the same controls. It does not yet prove that all decoding-time or post-hoc
mitigation methods fail.

## High-Priority End-to-End Baselines

These methods should be reproduced or explicitly scoped out before a final ICML
submission.

| Baseline | Source | Why it matters | Integration status |
|---|---|---|---|
| VCD | arXiv:2311.16922 | canonical visual contrastive decoding baseline against language-prior reliance | controlled greedy port implemented; full POPE splits complete; official sampling parity only needed for direct paper-to-paper comparison |
| OPERA | arXiv:2311.17911 | strong decoding baseline using over-trust penalty and rollback | not implemented locally; needs beam/search-time hook |
| DAMRO | arXiv:2410.04514 | CLS-selected ViT outlier-token contrastive decoding, close to our attention-shape audit | controlled greedy port implemented; adversarial 120-row subset is negative: TPR +0.033 but FPR +0.067, MCC 0.667 -> 0.639, related-present FPR 0.204 -> 0.278; full POPE/CHAIR audit pending |
| LURE | arXiv:2310.00754 | uses co-occurrence, uncertainty, and position factors aligned with our mechanism | not implemented locally; best used as post-hoc/revision or analysis baseline |
| Woodpecker | arXiv:2310.16045 | post-hoc visual validation/correction pipeline | not implemented locally; higher latency and external-tool dependence |
| Volcano | arXiv:2311.07362 | self-feedback guided revision baseline | not implemented locally; full model/data setup likely heavier |

## Head/Attention Positive Controls

These are especially relevant because they may pass some of our controls if they
select heads or regions more carefully than mean attention.

| Baseline | Source | Why it matters | Integration status |
|---|---|---|---|
| SPIN | arXiv:2505.16411 | image-guided dynamic head suppression; direct counterpoint to mean-head attention failure | controlled HF port implemented; default adversarial 120-row subset collapses to yes-rate 0.992/FPR 0.983; mild setting avoids collapse but gives Delta TPR - Delta FPR ~= 0 and related-present FPR 0.241; full POPE/CHAIR audit pending |
| CAI | arXiv:2506.23590 | caption-sensitive attention intervention; tests prompt-paired head selection | not implemented locally; code availability/compatibility to check |
| CAST | arXiv:2605.04641 | caption-guided visual attention steering; newer CAI-style method | not implemented locally; current as of 2026-06 |
| Region-Aware Attention Recalibration | arXiv:2605.24957 | region-aware inter-head recalibration; close to TDEV's region/neighbor motivation | not implemented locally; current as of 2026-06 |

## Recommended Next Reproduction Order

1. OPERA on the same subset if the official search-time logic ports cleanly to
   HuggingFace LLaVA-1.5.
2. Deprioritize full SPIN unless a stronger official-parity setting is needed;
   both default and mild adversarial 120-row checks fail to show target-
   discriminative gains.
3. Deprioritize full DAMRO unless official parity is required; its controlled
   adversarial subset raises false positives more than recall and worsens
   related-present FPR.
4. LURE-style factors as an analysis baseline: co-occurrence, uncertainty, and
   generation position are already available or cheap to compute in this repo.
5. Woodpecker/Volcano only if the paper needs a high-latency post-hoc correction
   comparison; they are less central to the allocation-vs-verification claim.

## Required Metrics for Any Added Baseline

Every added mitigation baseline should report the same metrics as the current
attention-only and TDEV audits:

- POPE split-level and macro accuracy, F1, MCC, yes rate, TPR, FPR, and
  `Delta TPR - Delta FPR` relative to vanilla.
- Semantic-neighbor negative FPR: all negative, related-present, plain-absent,
  and the related-minus-plain gap.
- CHAIR caption statistics where applicable: CHAIRi, CHAIRs, mean caption
  length, object mentions, and hallucinated mentions.
- If the method changes attention, an attention-shift selectivity audit on the
  adversarial split: visual mass change and TP/FP post-intervention visual mass.

A baseline should not be counted as mitigation evidence if it only improves F1
or recall by increasing yes rate, shortening captions, or reducing object
mentions without passing the semantic-neighbor and answer-prior controls.

## Current Code Implication

The mitigation generation stack is now tracked and smoke-tested. Commit
`fc74867` adds the shared POPE/CHAIR runtime, attention-only method hooks,
merge/evaluation scripts, and method comparison utilities. Commit `e4df8d2`
ensures explicit environment variables for model, data, and GPU selection take
precedence over stale `.env` defaults.

A 100-row POPE-random pilot using the local LLaVA checkpoint and COCO/POPE
paths completed for vanilla, PAI, ClearSight, and VisAttnSink with strict
yes/no parsing and matched sample IDs. The VCD-greedy port has now completed
all three POPE splits with `invalid=0` and matched sample IDs. It does not
improve the current conclusion: macro MCC drops from 0.732 to 0.720 and
related-present negative FPR rises on every split. The SPIN controlled port now
runs through the same tracked mitigation runtime. An 8-row POPE-random smoke run
finished with `invalid=0` under `mitigation/results/pope_spin_smoke/`. A larger
POPE-adversarial 120-row subset under
`mitigation/results/pope_spin_adversarial_120/` gives an early negative signal:
SPIN raises recall from 0.850 to 1.000, but raises FPR from 0.183 to 0.983,
yes-rate from 0.517 to 0.992, and drops MCC from 0.667 to 0.092. On the matched
semantic-neighbor subset, related-present negative FPR rises from 0.204 to
1.000 and plain-absent negative FPR from 0.000 to 0.833. A milder check with
`SPIN_ROUTED_HEADS=0.95` and `SPIN_SMALL_NUM_MASK=0.5` under
`mitigation/results/pope_spin_adversarial_120_mild/` avoids the collapse but
still gives no target-discriminative gain: TPR rises from 0.850 to 0.883 while
FPR rises from 0.183 to 0.217, so `Delta TPR - Delta FPR` is effectively zero;
related-present negative FPR rises from 0.204 to 0.241. These subset checks are
too small for paper tables, but they make full SPIN lower priority than DAMRO,
OPERA, or a documented official-parity check. The DAMRO controlled port has
now passed a 4-row POPE-random smoke run under `mitigation/results/pope_damro_smoke/`
with `invalid=0`; each prediction records `alpha=2.0`, `beta=0.1`, `topk=10`,
and the selected outlier token indices. This validates the local negative-branch
construction. A larger adversarial 120-row subset under
`mitigation/results/pope_damro_adversarial_120/` is negative: DAMRO raises TPR
from 0.850 to 0.883 but raises FPR from 0.183 to 0.250, drops MCC from 0.667
to 0.639, and raises related-present negative FPR from 0.204 to 0.278. The
subset is too small for a paper table, but it suggests DAMRO does not solve the
target-verification failure mode in the local controlled greedy port.
