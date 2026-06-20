# ICML Evidence Matrix

Date: 2026-06-20

This matrix is the current writing and experiment gate for the ICML version of
**Looking Is Not Verifying**. It links each paper claim to the strongest saved
evidence, the exact paper artifact that should carry it, and the remaining risk.
For a compact reader-facing baseline comparison, use
`docs/current_result_baseline_comparison.md`. The goal is to keep the paper
aligned with the reproduced baseline conclusion: `looking is not grounding`
because visual routing can lock onto related evidence without verifying the
queried target object.

## Claim-to-Evidence Matrix

| Claim | Status | Primary evidence | Paper artifact | Remaining risk / next action |
|---|---|---|---|---|
| C1. Global attention scores are position-confounded. | Supported | `detection/baselines/results/coco_llava_7b_baselines/controlled_analysis/controlled_summary.csv`: PAS overall `0.835` but within-bin `0.593`, same-word `0.569`; position-only AUROC `0.830`. | Table 1: controlled CHAIR detection baselines. | Wording must say attention mass is an unreliable proxy, not that all attention is useless. |
| C2. Semantic-neighbor negatives expose the actual failure mode. | Supported | `mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv`; mechanism summary: neighbor evidence exceeds target evidence in `96.0%` of vanilla related FPs. | Figure 1 + mechanism examples table. | Add visual examples or region overlays if time allows; current table is textual/score-based. |
| C3. Attention/VCD controls do not close the related-present gap; NoLan helps with a conservative tradeoff. | Supported with updated scope | `mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md`: PAI `0.110`, ClearSight `0.160`, VisAttnSink `0.123`, VCD `0.127` related FPR versus vanilla `0.114`; NoLan-compatible lowers related FPR to `0.076` but drops TPR to `0.778` and yes rate to `0.418`. | Main mitigation/control table. | CAI/CAST/Focus Matters/region-aware attention remain related-work positive controls unless code becomes runnable; do not claim all decoding methods fail. |
| C4. Generic region/object evidence is not enough. | Supported | Same table: OWLv2 target direct MCC `0.777` but related FPR `0.184` and adversarial related FPR `0.281`. | Main mitigation/control table, row group: region evidence. | Avoid calling OWLv2 target score a failed detector; it is strong aggregate evidence but non-discriminative under semantic neighbors. |
| C5. Target-vs-neighbor verification is the constructive criterion. | Supported, modest effect | Hybrid gate+rescue: MCC `0.763`, FPR `0.051`, related FPR `0.069`; strict margin related FPR `0.010` but TPR `0.359`. | Main TDEV table plus calibration ablation. | Need to frame as best current tradeoff, not solved hallucination. |
| C6. The criterion transfers to object-mention detection. | Supported | `docs/tdev_ablation_summary.md`: target absence + 0.25 neighbor dominance reaches overall `0.874`, within-bin `0.852`, matched-pair `0.854`, residual `0.722`. | CHAIR detection table. | Make clear this is post-hoc object-mention scoring, not fluent generation. |
| C7. Caption-side mitigation exists, but the useful route is claim acceptance plus constrained local repair plus verified candidate regeneration rather than fixed token suppression. | Partial, updated | Proxy rows show useful target selection: top-5 neutral rewrite CHAIRi `0.1340 -> 0.1186`; top-10 deletion CHAIRi `0.1048`. Generated smoke tests show fixed hard gating can route into substitute claims. On the 100-image high-risk set, claim-local repair reaches CHAIRi `0.0600`, `27` hallucinated mentions, `73.47%` retained vanilla grounded, and `0` content-light cases. Prompt-only regeneration is a negative control: concise lowers CHAIRi to `0.0538` by over-compressing to `17.75` words and `36.47%` retained vanilla grounded; detail reaches `39.04` words but worsens CHAIRi to `0.0714` and retains only `60.73%`. | Caption route summary plus appendix/prototype table. | Still high-risk selected and deterministic repair is only a modest gain. Next action is verification-in-loop regeneration or candidate selection, reported with content-preservation metrics, not length alone. |
| C8. TDEV-lite gives practicality but not standalone mitigation. | Supported with scope | LH-alone POPE MCC `0.495`, TPR `0.432`; LH->TDEV at 2,025/9,000 calls MCC `0.754`, FPR `0.056`, related FPR `0.075`. | Efficiency/practicality table. | Must be labeled supervised routing/triage; do not present it as a training-free attention method. |
| C9. Cross-model direction holds on Qwen2.5-VL. | Supported, small effect | `docs/multimodel_replication_audit.md`: Qwen vanilla macro MCC `0.765`, FPR `0.033`, related FPR `0.041`; fixed TDEV macro MCC `0.769`, FPR `0.027`, related FPR `0.034`. | Cross-model table. | Evidence is output-level plus model-independent OWLv2 verification, not Qwen internal attention evidence. |
| C10. Novelty is target-discriminative verification, not external detection, chain verification, or stronger visual routing. | Supported by related-work boundary | Woodpecker/LURE/R-CoV cover post-hoc claim extraction/verification/revision; CAI/CAST/Region-Aware/Focus Matters cover internal visual-routing or attention steering. Our local semantic-neighbor controls show why aggregate visual reliance is not enough. | Related work + limitation section. | State explicitly that TDEV may use an external verifier backend, but the contribution is the related-neighbor control and target-vs-neighbor decision criterion. |

## Proposed Main Tables and Figures

| Artifact | Purpose | Source / command | Status |
|---|---|---|---|
| Table 1: CHAIR controlled detection baselines | Establish `looking is not grounding` and position confound. | `detection/baselines/analyze_controls.py`; `docs/experiment_accuracy_audit.md`. | Ready. |
| Figure 1: Related-evidence failure examples | Show absent target, present semantic neighbor, attention/interventions still answer `yes`. | `mitigation/results/pope_mechanism_alignment_full/figure/pope_mechanism_alignment_contact_sheet.png`; generated by `mitigation/scripts/build_pope_mechanism_figure.py`. | Contact-sheet figure ready; region boxes remain optional. |
| Table 2: Semantic-neighbor POPE control table | Main bridge from baseline failure to method. | `/home/chenguanxu/miniconda3/envs/latentGuard/bin/python scripts/build_semantic_neighbor_control_table.py`. | Ready. |
| Table 3: TDEV POPE/CHAIR ablations | Show raw target evidence, margin, two-stage, hybrid, and CHAIR transfer. | `scripts/build_tdev_ablation_summary.py`; `docs/tdev_ablation_summary.md`. | Ready. |
| Table 4: TDEV-lite practicality | Show LH-alone fails, LH-routed TDEV saves calls and beats prompt controls. | `mitigation/results/pope_internal_external_ablation_full/`; CHAIR cascade metrics. | Ready but should be secondary. |
| Table 5: Cross-model Qwen replication | Show direction holds beyond LLaVA. | `mitigation/scripts/audit_qwen25vl_replication.py`; `docs/multimodel_replication_audit.md`. | Ready with scoped wording. |
| Table 6: Caption rewrite / correction | Show caption-side usefulness and limits. | `docs/caption_method_route_summary.md`; `detection/baselines/results/tdev_caption_candidate_pool_oracle_100/candidate_pool_oracle_metrics.json`. | Partial; repair works as a fallback, prompt-only regeneration and selector-only oracle are negative controls. |

## Current ICML Weak Points

Baseline availability is tracked separately in
`docs/baseline_availability_refresh.md`. That document is the current gate for
whether CAI, CAST, Region-Aware Attention Recalibration, or other recent
head/region steering methods are runnable baselines or related-work pressure.

1. **Caption mitigation is not yet a natural method.** The neutral and
   generic-noun rewrite proxies are less destructive than deletion and keep
   sentence length closer to vanilla, but they are still deterministic
   post-processing. The generated hard-gate smoke tests show a sharper failure:
   token suppression can produce substitute or escape object claims, so a larger
   deny list is not the right main path. Claim-local repair is the current safe
   fallback on the 100-image high-risk set (`0.0600` CHAIRi, `73.47%` retained
   vanilla grounded mentions, `0` content-light cases). Prompt-only regeneration
   is a negative control: concise lowers CHAIRi to `0.0538` by over-compressing
   to `17.75` words and `36.47%` retained vanilla grounded; detail reaches
   `39.04` words but worsens CHAIRi to `0.0714` and retains only `60.73%`.
   The candidate-pool oracle is also small: no-worse-than-repair selection keeps
   27 hallucinated mentions and only
   raises retained vanilla grounded mentions to `75.22%`. A deployable-style
   verified selector does not close the gap either: r0.75 worsens CHAIRi to
   `0.0621`, while strict r1.00 is repair-level (`0.0599` CHAIRi, `73.65%`
   retained vanilla grounded). Local-addition from whole-caption regeneration is
   also a negative control: o0.55 adds only 6 mostly redundant sentences and one
   hallucinated mention, while o0.40 adds nothing. The current positive route is
   verified atomic detail generation: raw atomic spans need verification (`31`
   hallucinated mentions), but verified atomic selection accepts 30 images, keeps
   hallucinated mentions at `27`, improves CHAIRi to `0.0558`, and raises retained
   vanilla grounded mentions to `76.79%`. The gain audit is the key sanity check:
   raw atomic spans add `+59` grounded and `+4` hallucinated mentions, while the
   verifier keeps `+34` grounded/object mentions with `+0` hallucinated mentions.
   For a stronger ICML story, scale and
   harden this verification-in-loop candidate generation: propose short missing-detail
   spans, extract object-like claims, map them to canonical targets, accept only
   claims passing target-vs-neighbor evidence, and fall back to local repair when
   no safe new detail exists. Report CHAIR together with retained grounded
   objects, object-mention retention, mean words, and content-light rate.
2. **Positive head/region baselines are not fully reproduced.** Current local
   ports cover PAI, ClearSight, VisAttnSink, VCD, SPIN subset, DAMRO subset, and
   NoLan-compatible all-splits. AIR official code is now accessible and is the
   next P0 attention-reallocation audit candidate; CAI/CAST, Focus Matters, and
   Region-Aware Attention Recalibration remain close competitors to audit if
   usable code appears.
3. **TDEV-region uses OWLv2.** This is acceptable as a backend for the criterion
   only if the paper repeatedly states that OWLv2 is not the contribution.
4. **Effect sizes are modest on strong models.** Qwen improves only slightly.
   This is still useful for a verifier paper if framed as false-positive
   reduction under semantic-neighbor controls.

## Baseline / Related-Work Boundary After Latest Check

Recent related work reinforces the chosen scope:

| Family | Examples | Boundary for this paper |
|---|---|---|
| Tool or chain verification | Woodpecker, LURE, UNIHD, R-CoV | They validate or rewrite with multi-step tools, statistical factors, or verification chains. We should not claim novelty as a generic post-hoc correction pipeline. |
| Caption/head steering | CAI, CAST | They increase visual attention through caption-query patterns. Our required test is whether the steered evidence is target-discriminative under related-present negatives. |
| Phase-aware visual-token suppression | Focus Matters | It filters/suppresses visual tokens from internal attention dynamics. Treat as positive related work; audit if code becomes public/runnable. |
| Region/head recalibration | Region-Aware Attention Recalibration | Closest low-cost internal mitigation direction. Treat as positive related work; audit if code becomes public/runnable. |
| Internal probes | HALP, local LH-Shape | Supports practicality of internal routing, but TDEV-lite must remain a triage/readout story unless it passes full semantic-neighbor controls. |
| Generic grounding/localization detectors | Fine-grained token grounding, raw OWLv2 target score | Localization/target evidence alone is insufficient; the discriminative margin versus semantic neighbors is the contribution. |

Checked sources:

- Woodpecker: https://arxiv.org/abs/2310.16045
- LURE: https://arxiv.org/abs/2310.00754
- UNIHD/MHaluBench: https://arxiv.org/abs/2402.03190
- R-CoV: https://arxiv.org/abs/2604.20696
- CAI: https://arxiv.org/abs/2506.23590
- CAST: https://arxiv.org/abs/2605.04641
- Focus Matters: https://arxiv.org/abs/2604.03556
- Region-Aware Attention Recalibration: https://arxiv.org/abs/2605.24957
- HALP: https://arxiv.org/abs/2603.05465
- Fine-Grained Token Grounding: https://arxiv.org/abs/2604.04863

## Next Concrete Work Order

1. **Upgrade faithful-concise caption prototype.** The deterministic neutral-rewrite
   proxy, generated hard-gate smoke tests, sentence acceptance, claim-local
   repair smoke, 20/40/100-image scaled checks, prompt-only regeneration
   negative controls, candidate-pool oracle, deployable-style verified selector,
   local-addition probes, and the first verified atomic-span prototype are
   complete. The remaining high-value gap is scaling and hardening
   verification-in-loop atomic detail generation, not selector-only
   reranking, evaluated against the same high-risk set with CHAIR, retained
   grounded objects, object retention, mean words, and content-light rate.
2. **Region-box mechanism figure.** The contact-sheet mechanism figure is
   complete. If time allows, add detector boxes or attention overlays for the
   same examples; this is optional because the score/evidence figure already
   makes `looking is not verifying` legible.
3. **Baseline availability refresh.** Re-check CAI/CAST/Region-Aware code just
   before paper freeze. If code exists, run a 120-row adversarial or full
   semantic-neighbor subset audit. If not, keep them as related work.
4. **Optional third model.** Run vanilla plus fixed TDEV only; do not rerun every
   baseline unless the first sanity check contradicts the current claim.

## Paper Thesis After Evidence Audit

The paper should be written around this thesis:

> LVLM hallucination methods often test whether visual information is attended,
> localized, or made more influential. These are insufficient because related
> visual evidence can support the wrong object claim. A claim is grounded only
> when evidence for the queried target exceeds evidence for plausible semantic
> alternatives. TDEV operationalizes this target-discriminative verification
> criterion and yields modest but consistent reductions in semantic-neighbor
> false positives, with an internal triage path for practicality.
