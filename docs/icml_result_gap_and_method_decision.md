# ICML Result Gap and Method Decision

Date: 2026-06-18

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
   do not selectively reduce related-present false positives.
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

### Route A: Target-Discriminative Decoding Gate

Goal: turn TDEV from a post-hoc verifier into a decoding-time object-claim gate.

Mechanism:

1. Detect candidate object claims during generation using the next-token object
   vocabulary or a short rolling noun-phrase parser.
2. For each candidate target, score target evidence and top semantic neighbors
   with the current evidence backend.
3. If target evidence is weak or neighbor-dominated, suppress only the object
   token/phrase branch rather than replacing the completed sentence afterward.
4. If the model emits a generic fallback or non-object continuation, keep the
   rest of the caption unchanged.

Why this fits: it intervenes exactly at unsupported object claims, so it is
closer to the original failure than post-hoc placeholder rewriting. It also gives
caption-side evidence without pretending generic nouns are visual correction.

Preliminary offline result: sentence-level suppression gives a cleaner output
than the failed clause heuristic but is too coarse (`-3.01` mean words for only
`-0.0175` CHAIRi). This confirms that the useful version of Route A must gate
object-token or object-phrase continuations during decoding, not delete completed
sentences after generation.

Token feasibility audit: this next step is technically plausible. On the saved
16,426 CHAIR object mentions, tokenizer-span matching near `gen_pos` succeeds for
`98.9%` overall and `98.2%` of hallucinated mentions. Among the TDEV top-5% high
risk mentions, `97.9%` match near `gen_pos`, including `97.7%` of selected
hallucinated mentions. The remaining risk is not coverage but decoding policy:
LLaVA's tokenizer often represents object phrases as multi-token BPE sequences,
so the gate must track short prefixes and suppress object-phrase continuations,
not just ban a first token globally.

Risk: it requires generation hooks and careful token/object mapping; if too
slow, report it as a small-scale proof of concept on the 4,977 CHAIR scope.

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

1. **Do Route A as a bounded proof of concept first.** Use a small CHAIR subset
   where current TDEV identifies high-risk object mentions. Measure CHAIRi,
   CHAIRs, mean words, object mentions, and manual fluency/error examples. The
   success condition is not only lower CHAIR; it must avoid generic placeholder
   artifacts.
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

> Attention and generic grounding can make a false object claim visually
> plausible, but they do not establish that the image supports the queried
> target rather than a semantic neighbor. TDEV operationalizes this missing
> target-discriminative verification criterion. Current results show a modest but
> consistent reduction in semantic-neighbor false positives and a practical
> triage path, while caption correction remains an open implementation step.

This is a publishable diagnostic-plus-verification paper if the writing keeps the
claim narrow. It is not yet a strong end-to-end hallucination mitigation paper.
