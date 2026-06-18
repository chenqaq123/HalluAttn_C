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
- TDEV post-hoc CHAIR score variants: target absence, target-vs-neighbor
  margin, two-stage absence, hybrid positive branch, and neighbor-dominance
  continuous scores.
- LURE-style statistical analysis factors for CHAIR detection: generation
  position, NLL/entropy uncertainty, generated-caption co-occurrence support,
  and simple combinations.

Mitigation/gating baselines currently audited in paper tables:

- Vanilla LLaVA-1.5-7B.
- PAI attention-only component, excluding its CFG/contrastive branch.
- ClearSight VAF component.
- Visual Attention Sink redistribution component.
- VCD-greedy controlled decoding baseline on all POPE splits.
- TDEV direct, vanilla-gate, hybrid gate+rescue, and calibration-sensitivity
  rules on POPE semantic-neighbor splits.

Tracked mitigation baselines not yet in paper tables:

- DAMRO controlled outlier-token contrastive decoding port: POPE-random smoke
  and POPE-adversarial 120-row subset audit complete; full POPE/CHAIR audit
  pending.
- SPIN controlled head-suppression port: POPE-random smoke plus default and
  mild POPE-adversarial 120-row subset audits complete; full POPE/CHAIR audit
  pending.

Current evidence supports the paper's scoped claim: aggregate attention mass,
unselective attention intervention, controlled VCD-greedy decoding, SPIN subset
checks, and DAMRO subset checks do not verify target-object presence;
target-discriminative region evidence survives the same controls. It does not
yet prove that all decoding-time or post-hoc mitigation methods fail, and the
paper should avoid presenting OWLv2 itself as the contribution. The contribution
is the semantic-neighbor stress protocol and target-vs-neighbor verification
criterion.

## High-Priority End-to-End Baselines

These methods should be reproduced or explicitly scoped out before a final ICML
submission.

| Baseline | Source | Why it matters | Integration status |
|---|---|---|---|
| VCD | arXiv:2311.16922 | canonical visual contrastive decoding baseline against language-prior reliance | controlled greedy port implemented; full POPE splits complete; official sampling parity only needed for direct paper-to-paper comparison |
| OPERA | arXiv:2311.17911 | strong decoding baseline using over-trust penalty and rollback | guarded official-hook interface added; current environment must pass `mitigation/scripts/check_opera_support.py` before producing any OPERA numbers |
| DAMRO | arXiv:2410.04514 | CLS-selected ViT outlier-token contrastive decoding, close to our attention-shape audit | controlled greedy port implemented; adversarial 120-row subset is negative: TPR +0.033 but FPR +0.067, MCC 0.667 -> 0.639, related-present FPR 0.204 -> 0.278; full POPE/CHAIR audit pending |
| LURE | arXiv:2310.00754 | uses co-occurrence, uncertainty, and position factors aligned with our mechanism | analysis baseline implemented for CHAIR detection; not a revisor reproduction |
| Woodpecker | arXiv:2310.16045 | post-hoc visual validation/correction pipeline with external tools/open-set detection | not implemented locally; discuss as a detector/tool pipeline rather than a direct low-latency baseline |
| UNIHD/MHaluBench | arXiv:2402.03190 | unified hallucination detection with auxiliary tools | not implemented locally; discuss as broad tool-based detection, distinct from our semantic-neighbor decision criterion |
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

1. OPERA on the same adversarial or semantic-neighbor subset after the active
   Python environment reports `ready_for_official_opera=true` via
   `mitigation/scripts/check_opera_support.py`. A subset result is enough to
   decide whether full POPE is worth the cost, but OPERA should not block the
   method-side work while the local transformers fork is unavailable.
2. CAI/CAST and Region-Aware Attention Recalibration are the highest-priority
   head/region positive controls. As of the current check, CAI/CAST arXiv pages
   expose no direct code link and Region-Aware says code will be public; run a
   semantic-neighbor subset audit if usable code appears.
3. The local TDEV-lite supervised per-head diagnostic is implemented and emits
   `detection/baselines/results/per_head_probe/per_head_probe_audit.json`:
   layer-31 image-grouped CV reaches 0.726 within-bin AUROC and 0.719
   matched-pair AUROC. The split-selected LH-Shape average is a negative
   ablation (`detection/baselines/results/lh_shape/lh_shape_metrics.json`), but
   calibrated LH-Shape linear readout is positive
   (`detection/baselines/results/lh_shape_linear/lh_shape_linear_metrics.json`):
   layers 22+31 reach 0.755 within-bin and 0.745 matched-pair AUROC. The CHAIR
   cascade audit (`detection/baselines/results/lh_shape_tdev_cascade/lh_shape_tdev_cascade_metrics.json`)
   shows LH-Shape can prefilter OWLv2/TDEV-region calls, especially at a 25%
   candidate budget, but position-only and PAS are competitive at 50%-75% call
   rates. A POPE per-head cache script now exists
   (`mitigation/scripts/cache_pope_per_head_rows.py`) and a transfer evaluator
   is ready (`mitigation/scripts/evaluate_pope_lh_shape_transfer.py`). Static
   checks, synthetic evaluator smoke, and a real 2-row 8-bit POPE cache smoke
   pass. A 120-row-per-split image-CV pilot is positive but imperfect: layers
   22+31 reach MCC 0.408/AUROC 0.739 and beat prompt-only controls, while
   present-object FPR remains 0.400. The full GPU cache still needs to be
   generated before claiming semantic-neighbor transfer; keep HALP as related
   work or an official-code baseline if code becomes available.
4. Deprioritize full SPIN unless a stronger official-parity setting is needed;
   both default and mild adversarial 120-row checks fail to show target-
   discriminative gains.
5. Deprioritize full DAMRO unless official parity is required; its controlled
   adversarial subset raises false positives more than recall and worsens
   related-present FPR.
6. Discuss Woodpecker/UNIHD/Volcano as high-latency tool or revision pipelines
   unless the final paper needs an explicit post-hoc correction comparison. They
   are less central to the allocation-vs-verification claim.


## Current TDEV Paper-Facing Tables

`docs/tdev_ablation_summary.md` is regenerated by:

```bash
python scripts/build_tdev_ablation_summary.py
```

It currently consolidates the POPE semantic-neighbor ablation and CHAIR
object-mention detection ablation. The table is the authoritative paper-facing
checkpoint for TDEV numbers; individual result directories remain the source of
truth for full metrics.

Key current TDEV numbers:

- POPE hybrid gate+rescue: macro MCC 0.763, macro TPR 0.806, macro FPR 0.051,
  adversarial MCC 0.717, adversarial related-present FPR 0.105.
- CHAIR hybrid MCC positive branch: 0.874 overall AUROC, 0.853 within-bin AUROC,
  0.855 matched-pair AUROC.
- CHAIR target absence plus 0.25 neighbor dominance: 0.874 overall AUROC,
  0.852 within-bin AUROC, 0.854 matched-pair AUROC, and 0.722 residual AUROC.
- LURE-style position+uncertainty: 0.831 overall AUROC but only 0.641
  within-bin AUROC and 0.657 matched-pair AUROC. Generated-caption
  co-occurrence support alone is near random: 0.576 overall AUROC and 0.496
  within-bin AUROC.

## ICML Readiness Checklist

Current status for a credible ICML submission:

| Requirement | Status | Evidence / next action |
|---|---|---|
| Controlled failure diagnosis | mostly complete | position controls, semantic-neighbor subsets, attention/decoding negative audits |
| Paper-facing TDEV ablation | complete for LLaVA-1.5 | `docs/tdev_ablation_summary.md` |
| External-detector positioning | complete for current draft | `docs/tdev_detector_positioning.md`; do not pitch OWLv2 as the method |
| Cheap co-occurrence/position baseline | complete as analysis baseline | `detection/scripts/evaluate_lure_style_detection.py`; LURE-style factors remain far below TDEV under controls |
| Strong decoding baseline beyond VCD | partial | OPERA official-hook interface is wired; still need an environment with the OPERA transformers fork and an adversarial subset result |
| Caption-style mitigation evidence | proxy/text-edit and official CHAIR rerun complete on object-mention scope | CHAIR detection is strong; object-mention filtering proxy removes 17.0% of hallucinated mentions at 1.1% grounded loss for the top-5% hybrid branch. Official PAS CHAIR on the 4,977-image object-mention scope drops from CHAIRi 0.1340/CHAIRs 0.4921 to 0.1186/0.4505 for top-5%, and to 0.1048/0.4047 for top-10% hybrid MCC. LH-Shape cascade triage can save 25% of OWLv2 calls while retaining 99.6% of full top-5 hallucination deletions, but this remains a cache-only triage simulation, not a fluent rewriter or decoding-time result |
| Multi-model replication | full all-splits evidence | Qwen2.5-VL full POPE vanilla has macro MCC 0.765/FPR 0.033; fixed LLaVA-selected TDEV hybrid improves to macro MCC 0.769/FPR 0.027 with TPR 0.782 vs. 0.786. Macro related-present FPR drops from 0.041 to 0.034 (`docs/multimodel_replication_audit.md`). |

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

The mitigation generation stack is tracked and smoke-tested. OPERA is wired only as a guarded official beam-search hook: the local code records key-position and OPERA parameters but refuses to emit predictions if the active transformers install lacks `opera_decoding` and `opera_beam_search`. In the current `latentGuard` environment, `mitigation/scripts/check_opera_support.py` reports transformers `4.57.6`, `supports_opera_generate_flag=false`, `has_opera_beam_search=false`, and `ready_for_official_opera=false`; therefore no OPERA number should be reported from this environment yet. Commit
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
