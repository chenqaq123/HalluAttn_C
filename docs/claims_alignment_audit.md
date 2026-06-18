# Claims Alignment Audit

Date: 2026-06-18

This note is a hard gate for paper claims after the full POPE LH-Shape and
TDEV-region audits. It answers a specific concern: whether the current results
still match the original motivation, **looking is not grounding**, or whether we
have drifted into a generic external-detector mitigation story.

## Bottom Line

The current evidence supports a diagnostic paper and a modest verification
method. It does **not** yet support a strong standalone mitigation claim.

The aligned thesis is:

> Looking is not grounding because visual routing can be coherent but
> non-discriminative. A model or method can attend to real associated evidence
> and still fail the target-object claim. The missing operation is target-vs-
> neighbor verification, not simply more visual attention or generic grounding.

TDEV-region is a constructive verifier for this criterion. LH-Shape is a
supervised internal triage signal. Neither should be described as a full
solution to hallucination.


A fresh result-quality recheck on 2026-06-18 confirms the same boundary. The
numbers are reproducible from saved artifacts, but they are not large enough to
support a headline claim that TDEV or LH-Shape is a strong standalone detector.
Full POPE TDEV improves MCC from 0.730 to 0.763 and lowers FPR from 0.087 to
0.051; LH-alone collapses TPR to 0.432. The important evidence is mechanistic:
among vanilla related-present false positives, semantic-neighbor evidence
exceeds target evidence in 96.0% of cases, and TDEV fixes 39.8% of them. This
keeps the paper aligned with `looking is not grounding`: the model often sees
plausible associated evidence, but does not verify the queried target.

## Claim Gate

| Candidate claim | Current status | Evidence | Paper wording |
|---|---|---|---|
| Attention mass is not a reliable grounding proxy | Supported | PAS/SVAR high overall AUROC but weak within-bin, matched, same-object, and residual AUROC; position-only AUROC is 0.830 | Use as a main finding |
| Hand-crafted attention shape fixes the proxy failure | Not supported | SinkDetect/naive LH-Shape averages are weak after controls | Use as negative evidence |
| Late-layer head features contain some target-grounding signal | Supported with scope | Supervised layer31 and layers22+31 image-CV probes are positive on CHAIR; POPE full image-CV AUROC is 0.853 | Report as internal diagnostic/triage, not training-free method |
| Attention intervention baselines solve the failure mode | Not supported | PAI/ClearSight/VisAttnSink/VCD/SPIN/DAMRO checks do not pass semantic-neighbor or answer-prior controls | Use as baseline motivation |
| TDEV-region substantially solves POPE hallucination | Too strong | Full TDEV improves macro MCC 0.730 -> 0.763 and related FPR 0.114 -> 0.069, but TPR drops 0.813 -> 0.806 | Say it reduces semantic-neighbor false positives, not solves hallucination |
| LH-Shape is a standalone POPE mitigator | Contradicted | LH-alone suppress reaches FPR 0.018 but TPR collapses to 0.432 and MCC to 0.495 | Do not claim |
| LH-Shape is useful as cheap triage into TDEV | Supported with scope | LH-routed TDEV at 2,025/9,000 detector calls reaches MCC 0.754, FPR 0.056, related FPR 0.075 | Use as practicality ablation |
| The contribution is an external detector pipeline | Misaligned | Existing Woodpecker/UNIHD/tool pipelines already cover detector-backed correction; our OWLv2 backend is not the contribution | Explicitly avoid |
| The contribution is target-discriminative verification under semantic-neighbor controls | Supported | Raw target evidence over-fires on related objects; TDEV target-vs-neighbor rules reduce related-present FPR on POPE and improve controlled CHAIR detection | Use as method criterion |

## How This Matches Looking Is Not Grounding

The reproduced baselines show two versions of the same proxy failure.

1. **Detection-side failure:** attention scores can rank hallucinations well in
   aggregate because they track generation position, not object evidence. This
   is the direct `looking is not grounding` result.
2. **Mitigation-side failure:** attention interventions can increase visual
   routing or yes rate without selectively improving true positives over false
   positives. This shows that changing where the model looks does not guarantee
   object verification.
3. **Semantic-neighbor failure:** related-present negatives are the most direct
   mechanism. The image contains plausible associated evidence, so a model can
   look at something meaningful while still answering the target claim wrongly.

TDEV is aligned only if presented as the missing verification operation:

```text
visual routing / objectness / generic grounding  !=  target-object verification
verification requires: evidence(target) > evidence(semantic neighbors)
```

This is why external OWLv2 evidence is acceptable as an experimental backend:
it tests whether the criterion itself repairs the failure mode. It is not proof
that an external detector is the paper contribution.

## Mechanism Alignment Evidence

`mitigation/scripts/build_pope_mechanism_alignment.py` now builds a compact
related-present example table from saved predictions. It selects POPE rows where
vanilla answers `yes` to an absent target, semantic-neighbor evidence is stronger
than target evidence, and TDEV flips the answer to `no`. The table also attaches
PAI, ClearSight, and VisAttnSink predictions to show whether attention-only
interventions fix the same target-verification failure.

Result files:

```text
mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_examples.csv
mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_examples.md
mitigation/results/pope_mechanism_alignment_full/pope_mechanism_alignment_summary.json
```

Key summary from all full POPE related-present negatives:

| Quantity | Value | Interpretation |
|---|---:|---|
| vanilla related-present FPR | 0.114 | baseline semantic-neighbor false-positive rate |
| neighbor evidence > target evidence among all related negatives | 0.986 | related-present rows usually contain stronger associated evidence than target evidence |
| neighbor evidence > target evidence among vanilla related FPs | 0.960 | most vanilla false positives fit associated-evidence grounding |
| TDEV correction rate among vanilla related FPs | 0.398 | TDEV fixes a substantial slice but not all failures |

The selected examples are intentionally mechanism-facing rather than score-
maximizing. For example, rows such as random `question_id=340` ask for an absent
`dog` while `bed|book` are present; vanilla, PAI, ClearSight, and VisAttnSink all
answer `yes`, while TDEV answers `no` because the best neighbor (`bed`) has much
higher evidence than the target (`0.534` vs `0.033`). This is exactly the
`looking is not grounding` failure: the image contains plausible visual evidence,
but not evidence for the queried target object.

## Paper-Facing Semantic-Neighbor Control Table

`scripts/build_semantic_neighbor_control_table.py` now builds a compact control
table that should be treated as a main paper-table candidate rather than an
auxiliary log. It pools split-level confusion counts from saved POPE metrics and
places aggregate behavior next to the actual stress test: related-present FPR
versus plain-absent FPR.

Result files:

```text
mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.csv
mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md
mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.json
```

Key table facts:

- Vanilla has macro related-present FPR `0.114` versus plain-absent FPR `0.028`.
- PAI, ClearSight, VisAttnSink, and VCD-greedy do not close this gap; ClearSight
  and VCD increase related-present FPR relative to vanilla.
- Raw OWLv2 target evidence has high aggregate MCC (`0.777`) but worsens
  related-present FPR to `0.184`, showing that generic target detection is not
  enough.
- Strict target-vs-neighbor margin nearly removes related-present FPs (`0.010`)
  but collapses TPR to `0.359`.
- Hybrid gate+rescue gives the current best tradeoff: MCC `0.763`, FPR `0.051`,
  related-present FPR `0.069`, and adversarial related-present FPR `0.105`.

This table makes the paper argument more precise: the result is not that an
external detector is strong, but that target-vs-neighbor verification improves
the specific semantic-neighbor failure mode exposed by `looking is not
grounding`.

## Current Result Quality

The method-shaped result is real but modest.

| Method | Calls | MCC | TPR | FPR | Related FPR | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| vanilla | 0 | 0.730 | 0.813 | 0.087 | 0.114 | anchor |
| full TDEV-region | 9,000 | 0.763 | 0.806 | 0.051 | 0.069 | lowers semantic-neighbor FPR, mild MCC gain |
| LH-alone suppress | 0 | 0.495 | 0.432 | 0.018 | 0.026 | not a usable standalone method |
| LH->TDEV triage | 2,025 | 0.754 | 0.801 | 0.056 | 0.075 | practical routing result, close to full TDEV FPR |
| prompt-position triage | 2,025 | 0.741 | 0.805 | 0.070 | 0.091 | control; weaker than LH routing |
| target-length triage | 2,025 | 0.743 | 0.802 | 0.066 | 0.087 | control; weaker than LH routing |

This is not strong enough to headline as a state-of-the-art mitigation method.
It is strong enough to support a paper whose main contribution is the diagnostic
protocol plus the target-discriminative verification criterion.

## Required Reframing For The Paper

Use this framing:

1. **Motivation:** Looking is not grounding; attention can select associated
   evidence and still fail target verification.
2. **Protocol:** Position controls and semantic-neighbor controls are necessary
   to test whether a method verifies the target object.
3. **Findings:** Attention detectors/interventions and several decoding
   baselines fail these controls or improve the wrong operating point.
4. **Constructive criterion:** TDEV asks whether evidence uniquely supports the
   target over plausible neighbors.
5. **Backend evidence:** OWLv2 instantiates the criterion and reduces related-
   object false positives. It is not the novelty by itself.
6. **Practicality:** LH-Shape shows internal late-head features can route a
   subset of calls to the verifier, but it is supervised triage and not a
   standalone fix.

Avoid this framing:

- "We solve hallucination with an external detector."
- "LH-Shape is an internal mitigation method."
- "Attention is useless."
- "Grounding boxes are enough."
- "The POPE MCC gain alone proves the method."

## Next Work That Would Strengthen The Paper

1. **Mechanism figure/table:** show related-present examples where attention or
   answer behavior is plausible but target evidence is absent. This reconnects
   the method visually to `looking is not grounding`.
2. **Caption-side correction beyond deletion:** current deletion/edit proxy is
   useful but too crude. A fluent rewrite or constrained regeneration would make
   the constructive method less like a detector report.
3. **Matched-budget attention baseline:** if CAI/CAST or region-aware attention
   code becomes available, run exactly the related-present audit. The question
   is not aggregate accuracy; it is whether they lower related-present FPR
   without simply changing yes rate.
4. **Paper table cleanup:** move LH-Shape to a practicality/efficiency table,
   not the main method table.
