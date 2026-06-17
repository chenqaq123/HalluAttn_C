# ICML Paper Strategy: From Looking to Verifying

Working title:

> Looking Is Not Verifying: Target-Discriminative Evidence for LVLM Object Hallucination

This document converts the current audit findings into an ICML-oriented paper
plan. The goal is not only to diagnose that attention shortcuts fail, but to
propose a practical target-discriminative verification method and evaluate it
against strong contemporary baselines.

## 1. What Is Already Supported

The current experiments support a scoped but strong negative result:

1. **Global attention detection is position-confounded.** PAS and SVAR reach
   high overall AUROC but drop sharply under within-bin, residual, and
   same-object controls.
2. **Mean-over-head attention shape is insufficient.** Sink removal, top-mass
   masking, no-RoPE rows, and Beyond-style attention dispersion do not recover
   robust target verification from mean attention.
3. **Attention is not empty.** Late-layer per-head diagnostics and recent
   head-selection papers suggest useful signal may live in specific heads or
   regions, not in aggregate attention mass.
4. **Unselective attention interventions are not enough.** PAI attention-only,
   ClearSight VAF, and Visual Attention Sink ports do not reliably improve
   POPE/CHAIR under answer-prior and caption-style audits.
5. **The plausible failure mode is semantic association.** A model can attend
   to real related evidence, such as toilet/bathroom for a sink query, while
   still failing to verify the target object.

The paper should not claim that attention is useless. The sharper claim is:

> Attention can route to meaningful visual evidence, but object hallucination
> requires target-discriminative verification: the evidence must support the
> queried object more than plausible semantic neighbors or co-occurring objects.

## 2. ICML-Level Gap

Current methods often optimize one of three proxies:

| Proxy | Representative methods | Problem exposed by our audit |
|---|---|---|
| More visual attention | PAI, ClearSight, VAS, CAI/CAST-style steering | Visual attention can increase without target selectivity |
| Less language-prior reliance | PAS, OPERA, VCD | Can reduce priors but may not verify the specific object |
| Attention localization/shape | Beyond-ADS, SinkDetect shape scores, DAMRO | Localized attention can still point to related non-target evidence |

The missing criterion is **counterfactual target discrimination**: if the same
attended visual evidence also supports a related absent object, attention is not
enough to call the claim grounded.

## 3. Proposed Method: Target-Discriminative Evidence Verification

Name placeholder: **TDEV** (Target-Discriminative Evidence Verification).

TDEV is a lightweight verifier/intervention run at object-decision points. It
can be used as a detector, abstention module, or decoding gate.

### 3.1 Inputs

For each generated object mention or yes/no object query:

- target object `o`;
- image visual tokens and patch/region representations;
- per-head object-query attention rows from selected late layers;
- a small semantic-neighbor set `N(o)` built from COCO co-occurrence and text
  embedding similarity.

### 3.2 Evidence Score

For each candidate object `c` in `{o} ∪ N(o)`, compute a visual evidence score:

```text
E(c) = max_region_or_patch  sim( visual_region, text_embed(c) )
```

Optionally weight regions by selected-head attention rather than mean attention:

```text
E_h(c) = sum_patch A_h(object_query, patch) * sim(visual_patch, text_embed(c))
```

The target-discriminative margin is:

```text
M(o) = E(o) - max_{c in N(o)} E(c)
```

A claim is grounded only if `M(o)` is positive after calibration. A high visual
attention score with low or negative margin means the model is looking at
plausible but non-target evidence.

### 3.3 Head Selection

Use the existing per-head diagnostic as a discovery tool, then test practical
selection rules:

1. **Unsupervised head filter:** keep heads whose visual rows have high
   target-vs-neighbor evidence margin on a small calibration set.
2. **Prompt-paired head filter:** compare attention under caption prompts vs
   object queries, inspired by CAI/CAST, but require target margin rather than
   just visual-mass increase.
3. **Per-sample dynamic head filter:** keep heads whose attended patches have
   high object-text compatibility and low neighbor compatibility.

This converts the current supervised probe into a publishable practical module.

### 3.4 Detection Use

Detection score:

```text
hallu_score = -M(o) + alpha * uncertainty + beta * position_residualized_prior
```

Report both training-free and calibrated variants:

- **TDEV-zero:** no learned weights, fixed margin threshold.
- **TDEV-cal:** logistic calibration on a held-out split, evaluated with
  image-disjoint splits and position controls.

### 3.5 Mitigation Use

For yes/no POPE:

- If target absent evidence margin is low, suppress `yes` or abstain.
- Report accuracy, MCC, yes-rate, TPR, FPR, and `Delta TPR - Delta FPR`.

For caption generation:

- When the next object noun has low margin, either suppress that noun, replace
  with a generic phrase, or request more visual evidence before emitting it.
- Keep caption length/object-count controls to avoid trivial short captions.

This is practical because it can run only at object-decision points and can use
cached visual/text embeddings plus a small neighbor set.

## 4. New Experiments Needed

### E1. Reproduce Current Audit Tables

Use the current audited results as Table 1:

- position-only vs PAS/SVAR/IC/Entropy/NLL/Beyond/SinkDetect;
- include overall, within-bin, same-word, residual;
- mark adapted/paper-level baselines explicitly.

### E2. Semantic-Neighbor Negative Audit

Construct POPE-style negative subsets:

1. random absent target;
2. co-occurring-object present;
3. semantic-neighbor present;
4. visually similar object present when possible.

Metrics:

- FPR by negative subset;
- target-vs-neighbor evidence margin;
- score gap between true-present and semantic-neighbor-negative examples;
- attention mass to related-object boxes if boxes are available.

Expected contribution:

> Attention methods fail hardest when related evidence is present, proving that
> the key missing ingredient is target discrimination rather than visual routing.

Current implementation status: `mitigation/scripts/build_semantic_neighbor_audit.py`
now builds this split from COCO val2014 annotations and POPE questions. On the
full 9,000-row POPE set, related-present negatives account for 56.0% of random,
65.3% of popular, and 84.8% of adversarial negatives, with zero unparsed targets.
This supports using the split as the first stress test for TDEV and existing
mitigation methods. Joining the split with existing attention-only POPE
predictions shows a consistent FPR gap: related-present negatives are much more
likely to receive false-positive `yes` answers than plain absent negatives for
vanilla, PAI, ClearSight, and VisAttnSink.

A first executable TDEV-zero prototype using global CLIP target-vs-neighbor
margin is now implemented. Direct `margin > 0` scoring lowers macro FPR to
5.9% but has only 32.9% TPR; as a calibrated gate over vanilla it ties vanilla
macro MCC (0.730 vs. 0.731) and barely reduces adversarial related-present FPR
(16.4% to 16.0%). This rules out whole-image CLIP margin as the final method.
A follow-up CLIP patch test further rules out naive patch pooling: `patch_max`
has lower macro MCC (0.149), while `patch_margin_max` has high TPR but unusably
high FPR (78.1%). As a calibrated vanilla gate, all CLIP-only variants tie
vanilla rather than fixing related-present false positives.

A detection-side attention-weighted prototype also produced a negative result:
resizing the cached mean-over-head LLaVA object attention rows to CLIP patches
and scoring target-vs-neighbor margins reaches only 0.582 overall AUROC, 0.560
within-bin AUROC, 0.551 matched-pair AUROC, and 0.524 residual AUROC on 16,426
CHAIR object mentions. This rules out the current early-layer average-attention
cache as a successful TDEV implementation.

A stronger OWLv2 region-evidence baseline changes the picture. Calibrating the
raw target object detection score on the random split gives macro MCC 0.777,
TPR 0.911, and FPR 0.134, which beats vanilla on aggregate but has high
adversarial related-present FPR (0.281). The target-vs-neighbor OWLv2 margin
nearly removes related-present false positives (macro related FPR 0.010) but
collapses recall to 0.359. A linear `target_score - alpha * neighbor_score`
search selects `alpha=0`, so simple soft margin scoring is insufficient.

The first positive method-shaped result is a semantic-aware two-stage verifier:
accept high-confidence target detections directly, but require a target-vs-
neighbor margin for medium-confidence detections. Direct two-stage verification
keeps macro MCC at 0.769 while reducing adversarial related-present FPR from
0.281 to 0.167 and improving adversarial MCC from 0.673 to 0.701. As a gate over
vanilla, the same idea improves macro MCC from 0.738 to 0.751 and lowers macro
related-present FPR from 0.105 to 0.069, but recall drops from 0.812 to 0.793.
A CHAIR object-mention detection audit strengthens the case for region evidence:
OWLv2 target absence reaches 0.865 overall AUROC, 0.842 within-bin AUROC, and
0.847 matched-pair AUROC, while the two-stage region score reaches 0.872,
0.849, and 0.851 respectively. This is far above the strongest previous
controlled baseline IC (0.776 overall, 0.686 within-bin, 0.703 matched-pair) and
confirms that the signal is not POPE-specific.

This is not yet the final ICML method, but it gives a concrete constructive
route: proposal-constrained region evidence plus semantic-neighbor-aware
calibration, with the remaining challenges being recall-preserving calibration
and cheaper proposal extraction. The first practicality issue is partly handled:
OWLv2 image-object scores can now be cached in a reusable NPZ file and reused by
both POPE and CHAIR audits, so expensive region scoring is amortized while
calibration and table generation become cheap deterministic post-processing.

### E3. TDEV Detection

Compare:

- NLL, Entropy;
- IC, GLSim;
- SVAR, PAS;
- Beyond-ADS/CGC;
- SinkDetect shape scores;
- HALP or a late-query probe if implementation is feasible;
- TDEV-zero and TDEV-cal.

Required controls:

- position-only baseline;
- within-bin AUROC;
- same-word matched AUROC;
- image-disjoint split;
- semantic-neighbor subset AUROC/FPR.

### E4. TDEV Mitigation

Compare:

- vanilla;
- PAI attention-only;
- ClearSight;
- Visual Attention Sink;
- VCD-greedy (completed; negative under full POPE and semantic-neighbor audits);
- OPERA;
- SPIN controlled head-suppression port (default and mild adversarial 120-row subsets negative; full audit lower priority);
- TDEV gate/intervention.

Report POPE and CHAIR with answer-prior and style controls. TDEV should be
judged by `MCC`, `Delta TPR - Delta FPR`, and CHAIR hallucinated mentions at
matched object-count/length, not only accuracy or CHAIRi.

### E5. Multi-Model Replication

Minimum ICML target:

- LLaVA-1.5-7B, existing full run;
- one newer open LVLM such as LLaVA-NeXT, Qwen2.5-VL-7B, or InternVL;
- optional third model if compute allows.

The key is not to rerun every baseline on every model; the main audit and TDEV
need at least one replication to show the phenomenon is not LLaVA-specific.

## 5. Baselines to Add or Discuss

### Detection / Verification

| Baseline | Why it matters | Feasibility |
|---|---|---|
| HALP | pre-generation internal probe; strong recent internal-state baseline | implement if code/checkpoints are available, otherwise discuss |
| HalLoc | token-level hallucination localization dataset/model | useful as related work, may not be apples-to-apples COCO object cache |
| GLSim official parity | representation similarity baseline already adapted locally | add small official-code parity if possible |
| LURE factors | co-occurrence, uncertainty, position are directly aligned with our story | use as analysis baseline/features |

### Mitigation

| Baseline | Why it matters | Feasibility |
|---|---|---|
| VCD | classic visual contrastive decoding; attacks language-prior reliance | controlled greedy port complete; official sampling parity optional |
| OPERA | attention over-trust penalty and rollback | high priority if code works with LLaVA |
| DAMRO | attention outlier/background-token suppression | useful attention-shape counterpoint |
| SPIN | image-guided dynamic head suppression; direct positive counterexample | controlled HF port implemented; default subset collapses to yes prior and mild subset ties vanilla without target-selective gains |
| CAI/CAST | caption-guided head/attention steering | strong recent head-specific baseline; code availability uncertain |
| Woodpecker/Volcano | post-hoc correction/self-feedback | useful but higher latency; include as practical comparison if time allows |

## 6. Why This Can Be ICML-Level

A pure negative audit is more likely AAAI/COLM-style. To target ICML, the paper
needs a method that resolves the diagnosed mechanism. TDEV can provide that if
it satisfies three criteria:

1. **Conceptual novelty:** target-discriminative verification against semantic
   counterfactuals, not just more visual attention.
2. **Practicality:** object-decision-point verifier, cached embeddings, small
   neighbor sets, no full external detector required in the default variant.
3. **Empirical strength:** improves controlled detection and mitigation metrics,
   especially semantic-neighbor negatives and `Delta TPR - Delta FPR`.

## 7. Immediate Next Work

1. Implement semantic-neighbor split construction from COCO annotations and text
   embedding similarity.
2. Add an audit script that reports FPR and score gaps for random vs
   semantic-neighbor negatives.
3. Prototype TDEV-zero using existing row-cache visual representations if
   available; otherwise extend row cache to save visual hidden states for key
   layers/patches.
4. Move to DAMRO/OPERA or official SPIN parity only if required; local SPIN
   default and mild subset checks do not show target-discriminative gains.
5. Run a small multi-model pilot before expanding all tables.

## 8. Current Paper Positioning Sentence

> We show that attention-based hallucination methods fail because they test
> whether a model looks at the image, not whether the attended evidence
> discriminates the target object from plausible semantic alternatives; we then
> introduce a target-discriminative verifier that converts attention and visual
> similarity into robust object-level evidence checks.

## 9. Research Sources Used for This Plan

Detection and evaluation:

- POPE: *Evaluating Object Hallucination in Large Vision-Language Models*,
  arXiv:2305.10355. Introduces object hallucination evaluation and reports
  co-occurrence/instruction effects.
- LURE: *Analyzing and Mitigating Object Hallucination in Large Vision-Language
  Models*, arXiv:2310.00754. Highlights co-occurrence, uncertainty, and object
  position as hallucination factors.
- PAS: *Prelim Attention Score for Detecting Object Hallucinations in Large
  Vision-Language Models*, arXiv:2511.11502. Recent attention-mass detector.
- GLSim: *Detecting Object Hallucinations in LVLMs via Global-Local
  Similarity*, arXiv:2508.19972. Recent representation-similarity detector.
- HalLoc: *Token-level Localization of Hallucinations for Vision Language
  Models*, arXiv:2506.10286. Token-level hallucination localization dataset and
  baseline.
- HALP: *Detecting Hallucinations in Vision-Language Models without Generating
  a Single Token*, arXiv:2603.05465. Pre-generation internal-state probe.

Mitigation and head/attention baselines:

- VCD: *Mitigating Object Hallucinations in Large Vision-Language Models
  through Visual Contrastive Decoding*, arXiv:2311.16922.
- OPERA: *Alleviating Hallucination in Multi-Modal Large Language Models via
  Over-Trust Penalty and Retrospection-Allocation*, arXiv:2311.17911.
- DAMRO: *Dive into the Attention Mechanism of LVLM to Reduce Object
  Hallucination*, arXiv:2410.04514.
- SPIN: *Mitigating Hallucinations in Vision-Language Models through
  Image-Guided Head Suppression*, arXiv:2505.16411.
- CAI/CAST: caption-guided visual attention steering/intervention,
  arXiv:2506.23590 and arXiv:2605.04641.
- Region-Aware Attention Recalibration, arXiv:2605.24957.
- Woodpecker: *Hallucination Correction for Multimodal Large Language Models*,
  arXiv:2310.16045.
- Volcano: *Mitigating Multimodal Hallucination through Self-Feedback Guided
  Revision*, arXiv:2311.07362.
