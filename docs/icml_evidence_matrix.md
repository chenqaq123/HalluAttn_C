# ICML Evidence Matrix

Date: 2026-06-18

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
| C3. Existing attention/decoding controls do not close the related-present gap. | Supported for local controlled ports | `mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md`: PAI `0.110`, ClearSight `0.160`, VisAttnSink `0.123`, VCD `0.127` related FPR versus vanilla `0.114`. | Main mitigation/control table. | CAI/CAST/Focus Matters/region-aware attention remain related-work positive controls unless code becomes runnable. |
| C4. Generic region/object evidence is not enough. | Supported | Same table: OWLv2 target direct MCC `0.777` but related FPR `0.184` and adversarial related FPR `0.281`. | Main mitigation/control table, row group: region evidence. | Avoid calling OWLv2 target score a failed detector; it is strong aggregate evidence but non-discriminative under semantic neighbors. |
| C5. Target-vs-neighbor verification is the constructive criterion. | Supported, modest effect | Hybrid gate+rescue: MCC `0.763`, FPR `0.051`, related FPR `0.069`; strict margin related FPR `0.010` but TPR `0.359`. | Main TDEV table plus calibration ablation. | Need to frame as best current tradeoff, not solved hallucination. |
| C6. The criterion transfers to object-mention detection. | Supported | `docs/tdev_ablation_summary.md`: target absence + 0.25 neighbor dominance reaches overall `0.874`, within-bin `0.852`, matched-pair `0.854`, residual `0.722`. | CHAIR detection table. | Make clear this is post-hoc object-mention scoring, not fluent generation. |
| C7. Caption-side mitigation exists but is still proxy/rewrite-based. | Partial | Top-5 neutral rewrite: CHAIRi `0.1340 -> 0.1186`, CHAIRs `0.4921 -> 0.4505`, mean words `89.54 -> 89.41`; top-5 generic noun rewrite keeps length closer (`89.53`) with the same CHAIR drop; sentence gate reaches CHAIRi `0.1165`, CHAIRs `0.4396` but mean words `86.53`; top-10 deletion: CHAIRi `0.1048`, CHAIRs `0.4047`. | Caption mitigation table/prototype evidence, appendix only unless a fluent decode-time gate is added. | Biggest ICML weakness remains that deterministic post-processing either uses generic nouns or deletes too much context. Next action is object-token/object-phrase decoding gate. |
| C8. TDEV-lite gives practicality but not standalone mitigation. | Supported with scope | LH-alone POPE MCC `0.495`, TPR `0.432`; LH->TDEV at 2,025/9,000 calls MCC `0.754`, FPR `0.056`, related FPR `0.075`. | Efficiency/practicality table. | Must be labeled supervised routing/triage; do not present it as a training-free attention method. |
| C9. Cross-model direction holds on Qwen2.5-VL. | Supported, small effect | `docs/multimodel_replication_audit.md`: Qwen vanilla macro MCC `0.765`, FPR `0.033`, related FPR `0.041`; fixed TDEV macro MCC `0.769`, FPR `0.027`, related FPR `0.034`. | Cross-model table. | Evidence is output-level plus model-independent OWLv2 verification, not Qwen internal attention evidence. |
| C10. Novelty is not external detection or chain verification. | Supported by related-work boundary | Woodpecker/UNIHD/R-CoV cover tool or chain verification; CAI/CAST/Region-Aware cover internal attention steering. | Related work + limitation section. | Need one paragraph explicitly separating TDEV from high-latency post-hoc verification and attention steering. |

## Proposed Main Tables and Figures

| Artifact | Purpose | Source / command | Status |
|---|---|---|---|
| Table 1: CHAIR controlled detection baselines | Establish `looking is not grounding` and position confound. | `detection/baselines/analyze_controls.py`; `docs/experiment_accuracy_audit.md`. | Ready. |
| Figure 1: Related-evidence failure examples | Show absent target, present semantic neighbor, attention/interventions still answer `yes`. | `mitigation/results/pope_mechanism_alignment_full/figure/pope_mechanism_alignment_contact_sheet.png`; generated by `mitigation/scripts/build_pope_mechanism_figure.py`. | Contact-sheet figure ready; region boxes remain optional. |
| Table 2: Semantic-neighbor POPE control table | Main bridge from baseline failure to method. | `/home/chenguanxu/miniconda3/envs/latentGuard/bin/python scripts/build_semantic_neighbor_control_table.py`. | Ready. |
| Table 3: TDEV POPE/CHAIR ablations | Show raw target evidence, margin, two-stage, hybrid, and CHAIR transfer. | `scripts/build_tdev_ablation_summary.py`; `docs/tdev_ablation_summary.md`. | Ready. |
| Table 4: TDEV-lite practicality | Show LH-alone fails, LH-routed TDEV saves calls and beats prompt controls. | `mitigation/results/pope_internal_external_ablation_full/`; CHAIR cascade metrics. | Ready but should be secondary. |
| Table 5: Cross-model Qwen replication | Show direction holds beyond LLaVA. | `mitigation/scripts/audit_qwen25vl_replication.py`; `docs/multimodel_replication_audit.md`. | Ready with scoped wording. |
| Table 6: Caption rewrite / correction | Show caption-side usefulness. | `detection/baselines/results/tdev_caption_rewrite_{neutral,generic}/chair_metrics.json` and `detection/baselines/results/tdev_caption_edit*/chair_metrics.json`. | Partial; deterministic rewrite proxies are ready, fluent regeneration still missing. |

## Current ICML Weak Points

Baseline availability is tracked separately in
`docs/baseline_availability_refresh.md`. That document is the current gate for
whether CAI, CAST, Region-Aware Attention Recalibration, or other recent
head/region steering methods are runnable baselines or related-work pressure.

1. **Caption mitigation is not yet a natural method.** The neutral and
   generic-noun rewrite proxies are less destructive than deletion and keep
   sentence length closer to vanilla, but they are still deterministic
   post-processing. The generic-noun version confirms that CHAIR can improve
   while object claims are replaced by broad nouns, so it should not be promoted
   as visual correction. For a stronger ICML story, implement constrained
   regeneration, a learned/LLM sentence-local rewrite, or a decoding-time object
   gate if local generation hooks are reliable.
2. **Positive head/region baselines are not fully reproduced.** Current local
   ports cover PAI, ClearSight, VisAttnSink, VCD, SPIN subset, and DAMRO subset.
   CAI/CAST, Focus Matters, and Region-Aware Attention Recalibration remain
   closest attention/visual-token competitors; run semantic-neighbor subset
   audits only if usable code appears.
3. **TDEV-region uses OWLv2.** This is acceptable as a backend for the criterion
   only if the paper repeatedly states that OWLv2 is not the contribution.
4. **Effect sizes are modest on strong models.** Qwen improves only slightly.
   This is still useful for a verifier paper if framed as false-positive
   reduction under semantic-neighbor controls.

## Baseline / Related-Work Boundary After Latest Check

Recent related work reinforces the chosen scope:

| Family | Examples | Boundary for this paper |
|---|---|---|
| Tool or chain verification | Woodpecker, UNIHD, R-CoV | They validate or rewrite with multi-step tools/chains. We should not claim novelty as a post-hoc correction pipeline. |
| Caption/head steering | CAI, CAST | They increase visual attention through caption-query patterns. Our required test is whether the steered evidence is target-discriminative under related-present negatives. |
| Phase-aware visual-token suppression | Focus Matters | It filters/suppresses visual tokens from internal attention dynamics. Treat as positive related work; audit if code becomes public/runnable. |
| Region/head recalibration | Region-Aware Attention Recalibration | Closest low-cost internal mitigation direction. Treat as positive related work; audit if code becomes public/runnable. |
| Internal probes | HALP, local LH-Shape | Supports practicality of internal routing, but TDEV-lite must remain a triage/readout story unless it passes full semantic-neighbor controls. |
| Generic grounding/localization detectors | Fine-grained token grounding, raw OWLv2 target score | Localization/target evidence alone is insufficient; the discriminative margin versus semantic neighbors is the contribution. |

Checked sources:

- Woodpecker: https://arxiv.org/abs/2310.16045
- UNIHD/MHaluBench: https://arxiv.org/abs/2402.03190
- R-CoV: https://arxiv.org/abs/2604.20696
- CAI: https://arxiv.org/abs/2506.23590
- CAST: https://arxiv.org/abs/2605.04641
- Focus Matters: https://arxiv.org/abs/2604.03556
- Region-Aware Attention Recalibration: https://arxiv.org/abs/2605.24957
- HALP: https://arxiv.org/abs/2603.05465
- Fine-Grained Token Grounding: https://arxiv.org/abs/2604.04863

## Next Concrete Work Order

1. **Fluent caption rewrite prototype.** The deterministic neutral-rewrite proxy is
   complete. The remaining high-value gap is a fluent constrained regeneration
   or sentence-local rewrite that preserves non-object content while removing
   unsupported object claims.
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
