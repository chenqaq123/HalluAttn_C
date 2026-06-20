# ICML Result Gap and Method Decision

Date: 2026-06-19

This note answers the current concern directly: the latest numbers are not
strong enough for a headline mitigation paper, and the caption-rewrite proxy does
not solve the original defect. The publishable direction is still coherent only
if we keep the paper centered on **looking is not grounding**: visual routing or
localized evidence can be meaningful while failing to verify the queried target
object.

## Current Result Diagnosis

The result profile is mixed, not weak everywhere.

| Question | Current evidence | Decision |
|---|---|---|
| Do the reproduced baselines expose the original defect? | Yes. PAS/SVAR have strong overall CHAIR AUROC but collapse under position and same-object controls; position-only reaches `0.830`. POPE related-present negatives have much higher FPR than plain-absent negatives. | Keep as the main diagnostic contribution. |
| Is the semantic-neighbor failure real? | Yes. In full POPE related-present negatives, neighbor evidence exceeds target evidence in `98.6%` of rows and in `96.0%` of vanilla related FPs. | This is the strongest bridge from motivation to method. |
| Does TDEV repair the exact failure? | Partly. Hybrid TDEV lowers macro related FPR from `0.114` to `0.069` and MCC from `0.730` to `0.763`; it corrects `39.8%` of vanilla related-present FPs. | Claim "reduces semantic-neighbor false positives", not "solves hallucination". |
| Does NoLan change the conclusion? | Yes, by narrowing it. NoLan-compatible lowers related FPR to `0.076` and FPR to `0.058`, but TPR drops to `0.778` and yes rate to `0.418`, with MCC tied to vanilla. | Do not claim decoding cannot help; claim prior suppression helps conservatively but still lacks explicit target-vs-neighbor verification. |
| Is raw detector evidence enough? | No. Direct OWLv2 target evidence has high aggregate MCC but related FPR `0.184` and adversarial related FPR `0.281`. | This supports target-vs-neighbor verification as the novelty, not external detection. |
| Is current caption correction a real solution? | No. Neutral/generic rewrites lower CHAIRi/CHAIRs, but generic noun replacement shows that CHAIR can improve by replacing claims with broad nouns. A sentence-gate prototype is more grammatical but too coarse: CHAIRi `0.1165`, CHAIRs `0.4396`, mean words `86.53`. | Keep caption correction in appendix/prototype evidence only. |
| Is LH-Shape a standalone practical method? | No. LH-alone suppress collapses TPR to `0.432`; LH-routed TDEV is useful only as triage. | Present as supervised routing into TDEV, not mitigation. |

## Why This Still Matches Looking Is Not Grounding

The original flaw was not merely that attention heatmaps are noisy. The sharper
flaw is that a model can route computation to real associated evidence and still
make a false target claim. This appears in three places:

1. **Detection:** global attention scores partly rank hallucinations by generation
   position rather than visual support.
2. **Mitigation:** attention interventions change visual routing or yes rate but
   do not selectively reduce related-present false positives; NoLan reduces them
   by becoming more conservative, not by verifying the target against neighbors.
3. **Semantic neighbors:** the image often contains plausible associated objects;
   the model can be visually engaged but non-discriminative.

The method must therefore answer a target-selectivity question:

```text
not enough:    visual evidence exists
required:      E(target) is sufficient and E(target) > E(semantic neighbors)
```

This is why the current external-detector-looking implementation is acceptable
only as a backend for the criterion. If the paper sounds like "we add OWLv2 to
check objects," it becomes unoriginal. If it sounds like "we prove target-vs-
neighbor selectivity is the missing condition and instantiate it with OWLv2 plus
internal triage," it remains aligned and defensible.

## External-Work Pressure

Recent work makes the method boundary important:

| Area | Representative work | Pressure on our paper |
|---|---|---|
| Tool/revision verification | Woodpecker, LURE, UNIHD, R-CoV | Do not claim novelty as a post-hoc detector or caption correction pipeline. |
| Caption/head steering | CAI, CAST | Do not claim visual-attention steering is unexplored; require semantic-neighbor FPR as the comparison axis. |
| Visual-token/region recalibration | Focus Matters, Region-Aware Attention Recalibration | Do not compete by generic attention intervention unless official code can be audited. |
| Internal hallucination probes | HALP and local LH-Shape | Use internal probes as practicality/routing evidence, but still require target-vs-neighbor verification. |
| Fine-grained grounding detectors | Fine-Grained Token Grounding and raw OWLv2 localization | Localization or patch grounding alone is not enough; the discriminative margin is the contribution. |

Current source anchors: Woodpecker `2310.16045`, LURE `2310.00754`, UNIHD
`2402.03190`, CAOS `2501.15046`, CAI `2506.23590`, CAST `2605.04641`, Focus
Matters `2604.03556`, Region-Aware Attention Recalibration `2605.24957`, HALP
`2603.05465`, and Fine-Grained Token Grounding `2604.04863`.

## Method Decision

The next method step should not be another generic caption rewrite. It should be
one of two target-discriminative routes.

### Route A: TDEV-Guided Claim Acceptance

Goal: turn TDEV from a post-hoc verifier into a practical object-claim
acceptance policy for faithful concise captioning. The current evidence says this
should not be framed as a pure token-ban decoder or as a length-preservation
method.

Mechanism:

1. Generate or keep a candidate sentence/span.
2. Extract object-like claims, including open-vocabulary route forms.
3. Map each claim to a canonical target when possible and score
   target-vs-neighbor evidence with the current backend.
4. Accept supported claims and reject unsupported claims.
5. When rejection would delete useful context, run a constrained local repair or
   regeneration step instead of leaving a hole or generic placeholder.

Why this fits: it intervenes exactly at unsupported object claims, so it remains
aligned with the associated-evidence failure mode. Unlike generic caption
rewrites, it does not claim success just because object mentions become rarer;
it explicitly tests whether each candidate claim is target-discriminative under
related evidence. A shorter caption is acceptable when the removed text consists
of unsupported object claims; the failure case is becoming generic, empty, or
losing the main supported scene content.

Latest caption-side prototype summary: `docs/caption_method_route_summary.md` is
generated from saved artifacts. Token feasibility is high: near-generation-position
matching succeeds for `98.88%` of all CHAIR object mentions and `98.23%` of
hallucinated mentions; the TDEV top-5% high-risk slice remains similarly
locatable (`97.93%`, and `97.66%` for selected hallucinated mentions). A
100-image prefilter audit selects 126 image-word pairs with `86.51%` CHAIR-label
hallucination precision, while keeping deny lists narrow (`1.26` words/image;
p95 `24` token sequences/image).

The generated-caption smoke tests clarify the policy risk. A 5-image t96 hard
gate still has CHAIRi `0.2500` and 8 hallucinated mentions because it routes into
complete substitute/escape claims. Sentence repair improves CHAIRi to `0.1786`
and 5 hallucinated mentions but misses complete substitutes. Sentence acceptance
reaches CHAIRi `0.1053` and 2 hallucinated mentions, showing that verifier-guided
claim rejection targets the right failure mode. It removes `24.60` words on
average, which is not automatically bad, but this remains a method-direction
diagnostic because we have not yet measured whether the retained caption keeps
the main supported scene content across a larger sample.

Risk: the next step must add constrained repair/regeneration after claim
rejection only when deletion would remove central supported content or leave an
incoherent fragment. Without this distinction, sentence acceptance may become too
generic; without claim acceptance, token suppression routes into new unsupported
claims.

### Route B: Internal TDEV-Lite Distillation

Goal: reduce reliance on OWLv2 while preserving the target-vs-neighbor criterion.

Mechanism:

1. Use existing OWLv2 TDEV decisions as teacher labels for target support and
   target-minus-neighbor margin.
2. Train or calibrate late-query/head features to predict the TDEV margin, not
   just hallucination labels.
3. Use the internal model only as a router or first-stage verifier; send uncertain
   or high-risk cases to the full evidence backend.
4. Evaluate with the same semantic-neighbor FPR, plain-absent FPR, and CHAIR
   position-controlled metrics.

Why this fits: it answers the practicality concern without pretending LH-Shape is
standalone. It also makes the paper less vulnerable to "this is just OWLv2".

Risk: current LH-alone POPE transfer is not reliable enough; the acceptable claim
is call reduction / triage, not detector replacement.

## Recommended Next Experiment Order

1. **Do Route A as claim acceptance plus constrained repair first.** Use the same
   high-risk CHAIR subset and report CHAIRi, CHAIRs, mean words, object mentions,
   rejected claims, repaired spans, and manual fluency/error examples. The success
   condition is lower hallucination while preserving the main supported scene
   content; mean-word reduction is acceptable when it removes unsupported detail,
   but empty or generic captions should be counted as failures.
2. **Then improve Route B if time allows.** Distill target-vs-neighbor margin, not
   binary hallucination, and compare against prompt-position and target-length
   routing controls.
3. **Keep CAI/CAST/Focus Matters/Region-Aware as monitored related work.** Only
   run them if official code is available; the required audit is related-present
   FPR under the existing semantic-neighbor split.
4. **Do not promote caption proxy rows.** They stay appendix-only unless Route A
   produces natural constrained generation.

## Paper Claim After This Decision

The paper should state:

> Attention, generic grounding, and language-prior suppression can make a false
> object claim visually plausible or less frequent, but they do not establish
> that the image supports the queried target rather than a semantic neighbor.
> TDEV operationalizes this missing target-discriminative verification criterion.
> Current results show a modest but consistent reduction in semantic-neighbor
> false positives and a practical triage path, while caption correction remains
> an open implementation step.

This is a publishable diagnostic-plus-verification paper if the writing keeps the
claim narrow. It is not yet a strong end-to-end hallucination mitigation paper.
