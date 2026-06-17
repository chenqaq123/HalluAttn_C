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
- SPIN controlled head-suppression port smoke-tested on POPE-random; full audit pending.
- TDEV direct and vanilla-gate rules on POPE semantic-neighbor splits.

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
| DAMRO | arXiv:2410.04514 | attention outlier/background-token suppression, close to our attention-shape audit | not implemented locally; likely can reuse attention hook stack |
| LURE | arXiv:2310.00754 | uses co-occurrence, uncertainty, and position factors aligned with our mechanism | not implemented locally; best used as post-hoc/revision or analysis baseline |
| Woodpecker | arXiv:2310.16045 | post-hoc visual validation/correction pipeline | not implemented locally; higher latency and external-tool dependence |
| Volcano | arXiv:2311.07362 | self-feedback guided revision baseline | not implemented locally; full model/data setup likely heavier |

## Head/Attention Positive Controls

These are especially relevant because they may pass some of our controls if they
select heads or regions more carefully than mean attention.

| Baseline | Source | Why it matters | Integration status |
|---|---|---|---|
| SPIN | arXiv:2505.16411 | image-guided dynamic head suppression; direct counterpoint to mean-head attention failure | controlled HF port implemented; 8-row POPE-random smoke passes with strict yes/no parsing; full POPE/semantic-neighbor audit pending |
| CAI | arXiv:2506.23590 | caption-sensitive attention intervention; tests prompt-paired head selection | not implemented locally; code availability/compatibility to check |
| CAST | arXiv:2605.04641 | caption-guided visual attention steering; newer CAI-style method | not implemented locally; current as of 2026-06 |
| Region-Aware Attention Recalibration | arXiv:2605.24957 | region-aware inter-head recalibration; close to TDEV's region/neighbor motivation | not implemented locally; current as of 2026-06 |

## Recommended Next Reproduction Order

1. OPERA on the same subset if the official search-time logic ports cleanly to
   HuggingFace LLaVA-1.5.
2. Finish SPIN full POPE and semantic-neighbor audits now that the controlled
   head-suppression port runs under the local LLaVA stack.
3. DAMRO as the next attention-shape counterpoint, because existing local code
   already patches LLaMA attention modules.
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
runs through the same tracked mitigation runtime: an 8-row POPE-random smoke
run finished with `invalid=0` and produced audit files under
`mitigation/results/pope_spin_smoke/`. The smoke set is too small for paper
claims, but it confirms that OPERA/DAMRO-style follow-up baselines can build on
tracked infrastructure instead of uncommitted runner changes.
