# ICML Paper Coverage Audit

Date: 2026-06-18

This note checks whether the current paper draft covers the evidence already
accepted in `docs/icml_evidence_matrix.md`. It is a writing task list, not a new
claim gate. The current draft should remain a diagnostic-plus-verification paper:
strong mechanism and protocol in the main text, modest method claims, and weaker
practicality/caption results kept scoped.

## Current Draft Coverage

| Evidence claim | Current paper coverage | Decision |
|---|---|---|
| C1. Attention scores are position-confounded | Covered in detection protocol, detection tables, and figures. | Main text ready. |
| C2. Semantic-neighbor negatives expose the failure mode | Covered with associated-evidence audit and mechanism contact-sheet figure. | Main text ready. |
| C3. Attention/decoding controls do not close related-present gap | Covered for PAI-attn-only, ClearSight, VisAttnSink, and VCD-greedy. | Main text ready; do not imply CAI/CAST are reproduced. |
| C4. Generic region/object evidence is not enough | Covered in region-verifier POPE table: raw target score over-fires on related negatives. | Main text ready. |
| C5. TDEV target-vs-neighbor verification is constructive | Covered with hybrid gate-plus-rescue and scoped method text. | Main text ready. |
| C6. TDEV transfers to CHAIR object-mention detection | Partly covered through OWLv2 region-verifier detection table. | Main text ready, but make clear this is scoring/post-hoc detection. |
| C7. Caption-side mitigation exists but is proxy/rewrite-based | Not in current main draft except baseline CHAIR mitigation discussion. | Appendix only unless fluent rewrite/constrained regeneration is added. |
| C8. TDEV-lite/LH-Shape practicality | Not in current main draft. | Add a compact appendix or secondary table; label as supervised triage. |
| C9. Qwen2.5-VL replication | Not in current main draft. | Add a compact appendix table or short robustness paragraph. |
| C10. Novelty is not external detection or chain verification | Covered in related work and limitations. | Main text ready; keep wording conservative. |

## Alignment With the Initial Defect

The current results do still align with the original defect, but only under a
more conservative interpretation.

Initial defect: the model often has visual evidence for a semantically related
object or scene cue, while the asked or generated target object is not actually
verified. This is the `related-present negative` failure mode, not simply an
attention-score calibration problem and not simply an object-detector recall
problem.

What the current evidence supports:

1. The semantic-neighbor split isolates the defect: vanilla, attention-based
   controls, decoding controls, and raw region/object evidence all retain a
   higher related-negative false-positive rate.
2. Target-vs-neighbor verification reduces that specific related-negative
   false-positive rate while preserving most positive recall.
3. The mechanism contact sheet supports the diagnosis: many false positives
   contain plausible associated evidence, but not target-specific evidence.

What the current evidence does not support:

1. The method is not a strong standalone hallucination mitigation system.
2. Caption-side edits are currently proxy post-processing, not fluent
   generation-time correction.
3. LH-Shape is useful as a supervised triage/readout signal, not as a
   training-free replacement for TDEV.
4. The Qwen result is positive but small, so it should be robustness evidence,
   not a headline result.

Therefore the paper should not claim that the intervention fully fixes the
defect. The stronger and more defensible claim is that the benchmark exposes a
previously hidden target-vs-neighbor verification failure, and that explicit
target-discriminative evidence is a necessary ingredient for reducing it.

## Missing Paper Artifacts

### 1. Cross-model robustness table

Purpose: show that the semantic-neighbor/TDEV direction is not purely LLaVA-
specific.

Recommended placement: appendix or a short robustness paragraph near Discussion.

Use current scoped numbers only:

| Model | Method | Macro MCC | TPR | FPR | Related FPR | Plain FPR |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL | vanilla | 0.765 | 0.786 | 0.033 | 0.041 | 0.010 |
| Qwen2.5-VL | fixed TDEV hybrid | 0.769 | 0.782 | 0.027 | 0.034 | 0.008 |

Required wording: this is output-level plus model-independent OWLv2 evidence;
it is not Qwen internal attention evidence.

### 2. TDEV-lite practicality table

Purpose: answer the practicality concern that full TDEV uses external region
evidence on every question.

Recommended placement: appendix or secondary experiment subsection after the
main associated-evidence audit.

Use current scoped numbers:

| Method | TDEV calls | Macro MCC | TPR | FPR | Related FPR | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| full TDEV hybrid | 9,000 | 0.763 | 0.806 | 0.051 | 0.069 | strongest verifier |
| LH-alone suppress | 0 | 0.495 | 0.432 | 0.018 | 0.026 | not usable standalone |
| LH-routed TDEV | 2,025 | 0.754 | 0.801 | 0.056 | 0.075 | useful supervised triage |
| prompt-position routing | 2,025 | 0.741 | 0.805 | 0.070 | 0.091 | routing control |
| target-length routing | 2,025 | 0.743 | 0.802 | 0.066 | 0.087 | routing control |

Required wording: LH-Shape is supervised triage/readout evidence, not a
training-free mitigation method and not a replacement for TDEV.

### 3. Caption-side correction appendix

Purpose: show that target-discriminative object-mention scoring can support
caption correction, while not pretending we have a fluent decoder-integrated
method.

Recommended placement: appendix only until a fluent sentence-local rewrite or
constrained regeneration is implemented.

Use current scoped numbers:

| Caption variant | CHAIRi | CHAIRs | Mean words | Scope |
|---|---:|---:|---:|---|
| vanilla | 0.1340 | 0.4921 | 89.54 | 4,977-image object-mention scope |
| neutral rewrite top-5 | 0.1186 | 0.4505 | 89.41 | deterministic local placeholder rewrite |
| deletion top-10 | 0.1048 | 0.4047 | 88.98 | stronger but less natural edit |

Required wording: this is deterministic post-processing/proxy evidence, not
natural generation or decoding-time mitigation.

## Recommended Next Paper Edits

1. Add a compact appendix subsection named `Additional Robustness and
   Practicality Evidence`.
2. Include Qwen and TDEV-lite tables there, with conservative captions.
3. Include caption correction only as a separate appendix paragraph/table, or
   leave it out until fluent rewrite is ready.
4. Keep the main text focused on the diagnostic protocol, semantic-neighbor
   mechanism, and TDEV verifier. Do not make Qwen/TDEV-lite/caption results carry
   the headline claim.

## Current Readiness Judgment

The current paper draft is coherent for the main ICML story but incomplete as a
full evidence package. The missing pieces are not contradictions; they are
secondary support results that should be added carefully. The highest-value next
paper edit is a compact appendix table for Qwen robustness and TDEV-lite
practicality. The highest-value experimental edit remains fluent caption-side
correction, but the current deterministic rewrite should not be promoted to a
main method claim.
