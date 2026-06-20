# Caption Method Route Summary

This note is generated from saved caption-side TDEV prototype artifacts. It keeps the method decision aligned with the latest evidence: hard token suppression is technically feasible but behaviorally incomplete; the next publishable route is verifier-guided claim acceptance with faithful concise captioning and constrained repair or regeneration.

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
| sentence acceptance | 5 | 0.1053 | 2 | 47.20 | 24.60 | strong hallucination reduction, but drops complete mixed sentences |
| claim-local repair | 5 | 0.0909 | 2 | 52.40 | 19.40 | best smoke tradeoff: preserves safe sentence prefixes before unsupported clauses |

## Concise-Faithfulness Audit

| Metric | Value | Reading |
|---|---:|---|
| retained vanilla grounded mentions | 70.83% | accepted captions keep most supported object mentions |
| hallucination reduction vs gated | 75.00% | accepted captions remove most gated hallucinated mentions |
| hallucination reduction vs vanilla | 77.78% | accepted captions improve over the original generated captions |
| object mention retention vs gated | 59.38% | shorter but not object-empty |
| generic/empty accepted captions | 0 | no accepted caption is empty/generic under the audit threshold |

## Claim-Local Repair Audit

| Metric | Value | Reading |
|---|---:|---|
| local repair actions | 2 | complete sentence prefixes saved before unsupported clauses |
| retained vanilla grounded mentions | 83.33% | improves content retention over sentence acceptance |
| hallucination reduction vs gated | 75.00% | keeps the same hallucination reduction as sentence acceptance |
| hallucination reduction vs vanilla | 77.78% | improves over original generated captions |
| object mention retention vs gated | 68.75% | less destructive than sentence acceptance |
| generic/empty repaired captions | 0 | no repaired caption is empty/generic under the audit threshold |

## Scaled Smoke Checks

| Scale | Prototype | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs gated | Empty/generic | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20-image | t96 hard gate | 20 | 0.1654 | 22 | 72.00 | -- | -- | -- | high-risk generated baseline |
| 20-image | sentence acceptance | 20 | 0.0674 | 6 | 48.85 | 72.81% | 66.92% | 1 | reduces hallucination but can delete too much |
| 20-image | claim-local repair | 20 | 0.0625 | 6 | 52.35 | 78.95% | 72.18% | 0 | preserves more supported content with the same hallucinated mention count |
| 40-image | t96 hard gate | 40 | 0.1544 | 42 | 74.08 | -- | -- | -- | maximum available iter2-prefilter set |
| 40-image | sentence acceptance | 40 | 0.0773 | 15 | 50.70 | 76.17% | 71.32% | 1 | same hallucination count as repair but less content retention |
| 40-image | claim-local repair | 40 | 0.0739 | 15 | 53.38 | 80.00% | 74.63% | 0 | best larger-scale prototype tradeoff |
| 100-image | t96 hard gate | 100 | 0.1172 | 73 | 72.13 | -- | -- | -- | broader high-risk generated baseline |
| 100-image | sentence acceptance | 100 | 0.0622 | 27 | 49.14 | 70.68% | 69.66% | 4 | strong hallucination reduction but visible-content loss remains |
| 100-image | claim-local repair | 100 | 0.0600 | 27 | 50.74 | 73.47% | 72.23% | 3 | slight CHAIR/content gain over acceptance, but generic cases remain |

The scaled checks support the direction but narrow the claim. On the 100-image high-risk set, sentence acceptance and claim-local repair both reduce hard-gated hallucinated mentions from 73 to 27. Claim-local repair is still slightly better than sentence acceptance: CHAIRi 0.0600 vs. 0.0622, mean words 50.74 vs. 49.14, retained vanilla grounded mentions 73.47% vs. 70.68%, and empty/generic cases 3 vs. 4. However, the repair gain is now modest, generic/empty cases are no longer zero, and object retention vs. vanilla remains only 64.01%. This is useful caption-side prototype evidence, not a complete caption mitigation result.

## Method Decision

The practical method should now be framed as **TDEV-guided claim acceptance for faithful concise captioning with constrained local repair/regeneration**, not as a pure token-ban decoder. The saved runs show that object claims are usually token-locatable and deny lists are narrow, but hard token suppression alone routes the model into new unsupported claims or incomplete fragments. Sentence acceptance catches those unsupported substitutes, which is exactly the target-vs-neighbor criterion we want. Its length reduction is not inherently bad: concise captions are preferable to long captions that keep inventing objects; the risk is only when shortening removes supported visible content or collapses into generic captions. The 100-image result shows that deterministic local repair is a useful but modest improvement over sentence deletion, so the remaining method gap is a controlled regeneration step with a generic-content guard when a safe prefix is not enough.

The next implementation target is therefore:

1. Generate or keep a candidate sentence/span.
2. Extract object-like claims, including open-vocabulary route forms.
3. Map each claim to a canonical target when possible and score target-vs-neighbor evidence.
4. Accept supported claims and reject unsupported claims; allow the caption to become shorter when unsupported detail is the only thing being removed.
5. Apply constrained local repair or regeneration when rejection would remove central visible content, when an unsupported claim sits in a detachable clause, or when the accepted caption would become generic.
6. Report CHAIR together with concise-faithfulness metrics: retained supported objects, mean words, object mentions, empty/generic-caption rate, and manual examples.

This preserves the paper's motivation: the method does not merely make object claims less frequent, and it does not rely on visual routing as proof. It explicitly tests whether the candidate claim is target-discriminative under related evidence. The desired behavior is not maximum caption length; it is concise but faithful captioning that keeps supported visual content and stops before unsupported object invention.

## Paper-Safe Scope

Current evidence supports a diagnostic-plus-verification paper with a bounded caption-side prototype. It does not yet support claiming a complete end-to-end caption mitigation method. For ICML, the strongest practical path is to add controlled regeneration and a generic-content guard, then evaluate it against the 100-image high-risk set with the same CHAIR and content-preservation audits.

## Source Artifacts

- `detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json`
- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_repair/sentence_repair_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_iter2_t96_claim_repair/claim_repair_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_audit_ov96/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_20_iter2_t96_claim_repair/claim_repair_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_audit_ov96/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_50_iter2_t96_claim_repair/claim_repair_metrics.json`
- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter_100_examples/multi_image_prefilter_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_prefilter/iterative_prefilter_summary.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_audit_ov96/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance/sentence_acceptance_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_concise_faithfulness/concise_faithfulness_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair/claim_repair_metrics.json`
