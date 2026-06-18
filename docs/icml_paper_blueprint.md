# ICML Paper Blueprint

Date: 2026-06-18

Working title:

> Looking Is Not Verifying: Target-Discriminative Evidence for LVLM Object Hallucination

This blueprint converts the current result-quality recheck into a paper plan.
It is deliberately conservative: the experiments support a strong diagnostic
claim and a modest verifier, not a state-of-the-art standalone hallucination
mitigation claim.

## Bottom Line

The results still match the original motivation, but only under the right
framing.

The original defect was not that LVLMs never use visual information. It was
that visual routing, attention mass, or generic object evidence can be
non-discriminative: the model may look at a real, semantically associated region
and still fail to verify the queried target object.

The current evidence supports this version:

```text
looking / visual routing / generic grounding  !=  target-object verification

target-object verification requires:
E(target) is high enough and E(target) > E(semantic neighbors)
```

The method contribution should therefore be written as **Target-Discriminative
Evidence Verification (TDEV)**: a criterion and verifier for semantic-neighbor
failure cases. OWLv2 is only the current evidence backend. LH-Shape is only a
supervised routing/practicality signal. Neither should be presented as a full
solution to object hallucination.

## Current Readiness Judgment

| Paper identity | Current status | Decision |
|---|---|---|
| Strong SOTA mitigation paper | Not supported | Do not write this version. The gains are real but too modest. |
| Detector pipeline paper | Misleading | Existing tool/revision systems already occupy this space. OWLv2 is a backend, not the novelty. |
| Diagnostic + verification-criterion paper | Supported | Write this version. It is aligned with `looking is not grounding`. |
| Practical lightweight verifier paper | Partial | Use LH-Shape as triage evidence, not as the main method. |

ICML viability depends on making the diagnostic and protocol feel necessary,
then showing that TDEV is the constructive positive control that partially
repairs the exact failure mode. The paper should be honest that the mitigation
effect is modest.

## Core Thesis

One-sentence thesis:

> LVLM hallucination methods often test whether visual information is attended,
> localized, or made more influential, but target-object grounding requires a
> stricter criterion: evidence for the queried target must dominate evidence
> for plausible semantic alternatives.

The paper should use the phrase **looking is not verifying** when moving from
motivation to method. `Looking is not grounding` remains the broader motivation;
the paper's sharper claim is that grounding requires target-discriminative
verification.

## Contributions

1. **Verification-centered diagnostic protocol.**
   Position-controlled CHAIR detection, answer-prior POPE metrics, and
   semantic-neighbor negative slices expose proxy failures that aggregate
   attention or yes-rate metrics miss.
2. **Mechanistic semantic-neighbor evidence.**
   Related-present false positives show the actual failure mode: associated
   evidence is often stronger than target evidence. Among vanilla related
   false positives, semantic-neighbor evidence exceeds target evidence in
   `96.0%` of cases.
3. **Target-Discriminative Evidence Verification.**
   TDEV formalizes the missing criterion and uses an asymmetric gate/rescue
   rule to reduce semantic-neighbor false positives. The strongest current POPE
   variant improves macro MCC from `0.730` to `0.763` and related FPR from
   `0.114` to `0.069`.
4. **Practicality and transfer evidence with scoped claims.**
   LH-Shape can route a subset of calls into TDEV, and Qwen2.5-VL shows the same
   direction with smaller gains. These are support results, not standalone
   solutions.

## Paper Structure

### Abstract

Lead with the proxy failure, not the method score. State that attention can
route to meaningful associated evidence while failing target verification.
Mention the key controlled results and the modest TDEV gain. Avoid any phrase
that implies hallucination is solved.

### 1. Introduction

The hook should be a semantic-neighbor example: the image contains a plausible
associated object, attention or model behavior looks visually grounded, but the
queried target is absent. This makes the failure more precise than "attention is
noisy."

Required introduction moves:

- attention and visual-influence proxies are widely used;
- those proxies do not require target discrimination;
- semantic-neighbor negatives expose plausible but wrong looking;
- TDEV is a constructive verifier for this missing criterion;
- reported gains are false-positive reductions under strict controls.

### 2. Related Work

Organize by failure boundary, not by chronology:

| Family | Examples | Boundary |
|---|---|---|
| Attention/decoding mitigation | PAI, ClearSight, VisAttnSink, VCD, DAMRO, OPERA | They can change visual routing or language-prior reliance without proving target-vs-neighbor selectivity. |
| Head/region steering | CAI, CAST, Region-Aware Attention Recalibration | Closest positive related work; audit if official code becomes available. The required comparison is related-present FPR, not only aggregate accuracy. |
| Tool/revision verification | Woodpecker, UNIHD, R-CoV, LURE-style revision | They are not the novelty target. Do not claim "external detector pipeline" novelty. |
| Grounding/localization detectors | OWLv2 and fine-grained token grounding | Useful backend evidence, but raw target evidence over-fires on related-present negatives. |
| Internal probes | IC, HALP-style probes, LH-Shape | Useful for practicality and diagnosis; supervised readouts must be labeled as such. |

### 3. Diagnostic Protocol

Define the controls before the method:

- position-controlled CHAIR object-mention detection;
- same-object and matched-pair controls;
- POPE answer-prior and yes-rate metrics;
- related-present versus plain-absent negatives;
- target evidence versus best semantic-neighbor evidence.

This section should make the evaluation rule explicit:

> A mitigation claim is weak if it improves aggregate accuracy while leaving
> related-present FPR high or merely changing the yes rate.

### 4. Findings: Why Looking Fails

This is the first main empirical section.

Evidence to include:

- position-only CHAIR AUROC `0.830`;
- PAS overall `0.835` but within-bin `0.593`, same-word `0.569`;
- SVAR overall `0.834` but within-bin `0.575`, same-word `0.565`;
- attention-only interventions do not close the semantic-neighbor FPR gap;
- mechanism figure showing absent target, present neighbor, predictions, and
  target/neighbor evidence scores.

The conclusion should be:

> The problem is not absence of visual routing. The problem is lack of
> target-discriminative verification.

### 5. Method: TDEV

Present TDEV as a criterion first and an implementation second.

Core rule:

```text
grounded(o, image) :=
  target evidence E(o) is sufficient
  and E(o) exceeds max evidence over semantic neighbors N(o)
```

For POPE:

- suppress unsafe model `yes` answers when target evidence is weak or dominated
  by semantic neighbors;
- rescue model `no` answers only under very high target evidence and margin;
- report hybrid gate+rescue as the best current tradeoff.

For CHAIR object mentions:

- score hallucination risk from target absence plus moderate neighbor
  dominance;
- report controlled AUROC, not only overall AUROC.

For practicality:

- LH-Shape routes calls into TDEV;
- it is supervised triage, not detector-free mitigation.

### 6. Experiments

Recommended main tables:

| Table | Role | Key current numbers |
|---|---|---|
| Table 1: CHAIR controlled detection | Establish position confound and attention proxy failure | position-only `0.830`; PAS within-bin `0.593`; IC within-bin `0.686`; TDEV CHAIR within-bin `0.852`. |
| Table 2: Semantic-neighbor POPE controls | Main mitigation stress test | vanilla related FPR `0.114`; PAI `0.110`; ClearSight `0.160`; VisAttnSink `0.123`; VCD `0.127`; hybrid TDEV `0.069`. |
| Table 3: TDEV ablations | Show target score, margin, two-stage, hybrid tradeoffs | target direct MCC `0.777` but related FPR `0.184`; strict margin FPR low but TPR `0.359`; hybrid MCC `0.763`. |
| Table 4: TDEV-lite routing | Practicality | LH-alone MCC `0.495`; LH-routed TDEV `2,025/9,000` calls, MCC `0.754`, related FPR `0.075`. |
| Table 5: Cross-model Qwen | Scoped transfer | vanilla MCC `0.765`; fixed TDEV MCC `0.769`; related FPR `0.041 -> 0.034`. |
| Appendix table: caption correction | Useful but secondary | neutral rewrite CHAIRi `0.1340 -> 0.1186`, CHAIRs `0.4921 -> 0.4505`. |

Recommended figures:

| Figure | Role | Status |
|---|---|---|
| Figure 1: mechanism contact sheet | Make associated-evidence failure visually legible | Ready. |
| Figure 2: result map | Show proxy metrics versus target-discriminative metrics | To draft from existing tables. |
| Appendix figure: caption rewrite examples | Only if fluent rewrite is added | Not ready. |

### 7. Discussion and Limitations

The discussion should explicitly downgrade the method claim:

- TDEV reduces semantic-neighbor false positives but does not solve
  hallucination.
- OWLv2 is an experimental backend.
- LH-Shape is a supervised triage/readout path.
- Caption correction is still proxy/rewrite-based unless a fluent regeneration
  method is added.
- Stronger concurrent attention/region steering baselines should be audited
  when runnable code is available.

## Claim Wording Gate

Use:

- "reduces semantic-neighbor false positives";
- "target-discriminative verification criterion";
- "OWLv2-backed instantiation";
- "supervised internal triage";
- "modest but consistent verifier gains";
- "attention mass is an unreliable grounding proxy under controls";
- "visual routing can be coherent but non-discriminative."

Avoid:

- "solves hallucination";
- "state-of-the-art mitigation";
- "external detector is the contribution";
- "attention is useless";
- "LH-Shape is a standalone mitigator";
- "grounding boxes are enough";
- "POPE MCC gain alone proves grounding."

## Last Experiments Before Paper Freeze

Required only if the paper aims for a stronger method identity:

1. **Fluent caption-side correction.**
   Replace deterministic deletion/neutral placeholders with sentence-local
   constrained regeneration or a decoding-time object gate. This is the biggest
   current method weakness.
2. **Matched positive baselines if code appears.**
   Re-check CAI, CAST, and Region-Aware Attention Recalibration close to paper
   freeze. If runnable code exists, evaluate the semantic-neighbor subset, not
   only aggregate POPE.
3. **Optional third model.**
   Run vanilla plus fixed TDEV only. Do not rerun every baseline unless the
   first sanity check contradicts the current claim.
4. **Optional region-box visualization.**
   Add boxes or overlays to the existing contact-sheet examples if it improves
   readability, but do not block the paper on this.

## Current Decision

Proceed as a diagnostic-plus-verification paper.

The current numbers are not good enough for a strong mitigation paper, but they
are coherent and useful for the original scientific point. The paper should make
the community's evaluation bar stricter: methods must prove target selectivity
under position, answer-prior, and semantic-neighbor controls before their gains
are interpreted as hallucination mitigation.
