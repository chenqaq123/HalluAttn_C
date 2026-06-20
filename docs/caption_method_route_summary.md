# Caption Method Route Summary

This note is generated from saved caption-side TDEV prototype artifacts. It keeps the method decision aligned with the latest evidence: hard token suppression is technically feasible but behaviorally incomplete; the next publishable route is verifier-guided claim acceptance with constrained repair or regeneration.

## Feasibility Evidence

| Check | Value | Reading |
|---|---:|---|
| CHAIR object mentions | 16426 | cached mention-level scope |
| Near-position tokenizer match, all mentions | 98.88% | object claims can usually be found near generation position |
| Near-position tokenizer match, hallucinated mentions | 98.23% | hallucinated claims are also gateable |
| Near-position match, TDEV top-5.00% mentions | 97.93% | high-risk claims remain token-locatable |
| Near-position match, selected hallucinated mentions | 97.66% | coverage is not the main bottleneck |
| Single-token COCO word fraction | 42.50% | multi-token phrase tracking is required |

## Multi-Image Prefilter Evidence

| Check | Value | Reading |
|---|---:|---|
| Images selected | 100 | bounded GPU evaluation target |
| Selected image-word pairs | 126 | small deny/verify workload |
| CHAIR hallucination precision | 86.51% | TDEV-selected claims are mostly true hallucinations |
| Hallucinated-mention coverage on selected images | 66.46% | useful but incomplete coverage |
| Mean denied words per image | 1.26 | deny lists are narrow |
| P95 token sequences per image | 24.00 | token workload is bounded |

## Generated-Caption Smoke Evidence

| Prototype | Images | CHAIRi | Hallucinated mentions | Mean words | Removed words | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| t96 hard gate | 5 | 0.2500 | 8 | 71.80 | 0.00 | hard gating creates substitute/escape claims |
| sentence repair | 5 | 0.1786 | 5 | 63.60 | 8.20 | fixes incomplete tails but misses complete substitute claims |
| sentence acceptance | 5 | 0.1053 | 2 | 47.20 | 24.60 | best hallucination reduction, but over-deletes content |

## Method Decision

The practical method should now be framed as **TDEV-guided claim acceptance**, not as a pure token-ban decoder. The saved runs show that object claims are usually token-locatable and deny lists are narrow, but hard token suppression alone routes the model into new unsupported claims or incomplete fragments. Sentence acceptance catches those unsupported substitutes, which is exactly the target-vs-neighbor criterion we want, but it removes too much text because it has no replacement generator.

The next implementation target is therefore:

1. Generate or keep a candidate sentence/span.
2. Extract object-like claims, including open-vocabulary route forms.
3. Map each claim to a canonical target when possible and score target-vs-neighbor evidence.
4. Accept supported claims, reject unsupported claims, and ask for a constrained local repair only when rejection would delete useful content.
5. Report both CHAIR and content-retention metrics; lower CHAIR alone is insufficient if mean words collapse.

This preserves the paper's motivation: the method does not merely make object claims less frequent, and it does not rely on visual routing as proof. It explicitly tests whether the candidate claim is target-discriminative under related evidence.

## Paper-Safe Scope

Current evidence supports a diagnostic-plus-verification paper with a bounded caption-side prototype. It does not yet support claiming a complete end-to-end caption mitigation method. For ICML, the strongest practical path is a small generated-caption experiment that compares hard gate, sentence repair, sentence acceptance, and constrained repair on the same high-risk image set.

## Source Artifacts

- `detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json`
- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`
