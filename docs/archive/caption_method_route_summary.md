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

| Scale | Prototype | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs gated | COCO-objectless / content-light | Reading |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20-image | t96 hard gate | 20 | 0.1654 | 22 | 72.00 | -- | -- | -- | high-risk generated baseline |
| 20-image | sentence acceptance | 20 | 0.0674 | 6 | 48.85 | 72.81% | 66.92% | 1 / 1 | reduces hallucination but can delete too much |
| 20-image | claim-local repair | 20 | 0.0625 | 6 | 52.35 | 78.95% | 72.18% | 0 / 0 | preserves more supported content with the same hallucinated mention count |
| 40-image | t96 hard gate | 40 | 0.1544 | 42 | 74.08 | -- | -- | -- | maximum available iter2-prefilter set |
| 40-image | sentence acceptance | 40 | 0.0773 | 15 | 50.70 | 76.17% | 71.32% | 1 / 1 | same hallucination count as repair but less content retention |
| 40-image | claim-local repair | 40 | 0.0739 | 15 | 53.38 | 80.00% | 74.63% | 0 / 0 | best larger-scale prototype tradeoff |
| 100-image | t96 hard gate | 100 | 0.1172 | 73 | 72.13 | -- | -- | -- | broader high-risk generated baseline |
| 100-image | sentence acceptance | 100 | 0.0622 | 27 | 49.14 | 70.68% | 69.66% | 4 / 1 | strong hallucination reduction but visible-content loss remains |
| 100-image | claim-local repair | 100 | 0.0600 | 27 | 50.74 | 73.47% | 72.23% | 3 / 0 | slight CHAIR/content gain over acceptance; content-light cases are removed |

The scaled checks support the direction but narrow the claim. On the 100-image high-risk set, sentence acceptance and claim-local repair both reduce hard-gated hallucinated mentions from 73 to 27. Claim-local repair is still slightly better than sentence acceptance: CHAIRi 0.0600 vs. 0.0622, mean words 50.74 vs. 49.14, retained vanilla grounded mentions 73.47% vs. 70.68%, and content-light cases 0 vs. 1. The old CHAIR-objectless counts, 3 vs. 4, mostly reflect non-COCO but descriptive objects such as roads, signs, poles, or watercraft. However, the repair gain is now modest and object retention vs. vanilla remains only 64.01%. This is useful caption-side prototype evidence, not a complete caption mitigation result.

## Controlled Regeneration Probe

| Prototype | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| claim-local repair | 100 | 0.0600 | 27 | 50.74 | 73.47% | 64.01% | 0 | current best deterministic fallback |
| controlled regen concise | 100 | 0.0538 | 12 | 17.75 | 36.47% | 31.72% | 0 | lowers CHAIR by over-compressing content |
| controlled regen detail | 100 | 0.0714 | 27 | 39.04 | 60.73% | 53.77% | 0 | recovers length but not the repair tradeoff |

Prompt-only regeneration is therefore not the solution. The concise prompt reduces CHAIRi by collapsing to short safe captions, while the detail-preserving prompt restores some length but has worse CHAIRi and lower grounded-content retention than claim-local repair. The next method should be verification-in-loop regeneration or candidate selection: regenerate only missing visible details, verify each new claim against target-vs-neighbor evidence, and keep the deterministic repaired caption as a fallback.

## Candidate Pool Oracle

This is an upper-bound analysis, not a deployable method: it uses CHAIR labels to choose among the already generated claim-local repair, concise regeneration, and detail regeneration candidates. It asks whether the current candidate pool contains useful alternatives that a future verifier could select without hallucination growth.

| Selector | Images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Selected candidates | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| oracle min hallucination | 100 | 0.0268 | 11 | 45.47 | 69.28% | 58.46% | 0 | repair 79, concise 14, detail 7 | best possible hallucination control still costs visible content |
| oracle no-worse-than-repair | 100 | 0.0586 | 27 | 50.27 | 75.22% | 65.58% | 0 | repair 80, concise 0, detail 20 | small upper-bound gain: detail candidates help 20 images but do not change the conclusion |

The no-worse-than-repair oracle is the relevant upper bound for a future verifier. It keeps hallucinated mentions at 27, improves CHAIRi only from 0.0600 to 0.0586, and raises retained vanilla grounded mentions from 73.47% to 75.22%. This means the existing regeneration candidates contain some recoverable detail, but the gain is too small to justify a selector-only paper claim. The next method needs better candidate generation plus target-vs-neighbor claim verification.

## Verified Candidate Selection Probe

Unlike the oracle above, this probe does not use CHAIR labels for selection. It accepts detail-regenerated captions only when the repaired-to-candidate closed-loop audit finds no unsupported introduced target/open-vocabulary claim, with a repaired-caption fallback.

| Selector | Images | Accepted detail | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| verified selector r0.75 | 100 | 40 | 0.0621 | 27 | 48.80 | 70.86% | 61.88% | 0 | verifier admits too many compressed candidates; worse than repair |
| verified selector r1.00 | 100 | 10 | 0.0599 | 27 | 50.75 | 73.65% | 64.15% | 0 | stricter guard collapses to repair-level behavior |

The deployable-style selector confirms the oracle warning. A loose word-ratio guard accepts 40 detail candidates and worsens CHAIR/content retention. A strict r1.00 guard accepts only 10 detail candidates and is statistically indistinguishable from claim-local repair: CHAIRi 0.0599 vs. 0.0600 and retained vanilla grounded 73.65% vs. 73.47%. The bottleneck is not just selection; the candidate generator must produce claim-local additions that preserve detail without rewriting away supported content.

## Verified Local-Addition Probe

This probe keeps the repaired caption and only appends low-overlap sentences from a detail-regenerated candidate whose introduced claims pass the closed-loop target-vs-neighbor audit. It tests whether whole-caption regeneration can serve as a source of local detail spans.

| Selector | Images | Images with additions | Added sentences | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| local additions o0.55 | 100 | 6 | 6 | 0.0605 | 28 | 51.99 | 73.82% | 65.86% | 0 | appends mostly paraphrases; adds one hallucinated mention |
| local additions o0.40 | 100 | 0 | 0 | 0.0600 | 27 | 50.74 | 73.47% | 64.01% | 0 | stricter overlap rejects all additions and returns to repair |

The local-addition probe narrows the fix. Whole-caption regeneration does not yield clean missing-detail spans: a loose overlap threshold appends only 6 mostly redundant sentences, increases hallucinated mentions from 27 to 28, and only nudges retained vanilla grounded from 73.47% to 73.82%; a stricter threshold appends nothing. The next generator must be explicitly trained or prompted to propose atomic missing-detail claims/spans, not full paraphrased captions.

## Atomic Detail Generation Probe

This is the first positive caption-side prototype after the negative controls. LLaVA is prompted to produce at most two short new visible-detail sentences from the image and repaired draft. The closed-loop verifier then keeps augmented captions only when introduced claims pass target-vs-neighbor verification.

| Prototype | Images | Accepted additions | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Object retention vs vanilla | Content-light | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw atomic augmentation | 100 | 52 images / 52 spans | 0.0604 | 31 | 55.14 | -- | -- | -- | adds detail but needs verifier; hallucinated mentions rise |
| verified atomic selection | 100 | 30 images | 0.0558 | 27 | 53.33 | 76.79% | 68.85% | 0 | improves repair without increasing hallucinated mentions |

Verified atomic selection changes the caption-side conclusion. Relative to claim-local repair, it keeps hallucinated mentions fixed at 27, improves CHAIRi from 0.0600 to 0.0558, raises mean words from 50.74 to 53.33, and raises retained vanilla grounded mentions from 73.47% to 76.79%. This is still a bounded 100-image high-risk prototype, but it is aligned with the paper motivation: looking is not enough, so generated details are only accepted when their object claims pass target-vs-neighbor verification.

### Atomic Overlap Sensitivity

This ablation re-parses the same raw atomic generations with different overlap thresholds before verifier selection. It checks whether the positive result depends on one hand-tuned duplicate filter.

| Variant | Raw candidates | Accepted images | CHAIRi | Hall. mentions | Mean words | Retained vanilla grounded | Delta grounded | Delta hallucinated | Content-light | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| verified atomic o0.50 | 47 images / 47 spans | 26 | 0.0560 | 27 | 52.96 | 76.79% | 32 | 0 | 0 | conservative filter |
| verified atomic o0.65 | 52 images / 52 spans | 30 | 0.0558 | 27 | 53.33 | 76.79% | 34 | 0 | 0 | default filter |
| verified atomic o0.85 | 71 images / 71 spans | 46 | 0.0541 | 27 | 54.63 | 77.49% | 49 | 0 | 0 | best current detail/faithfulness tradeoff |

The overlap sweep strengthens the claim without changing its scope. Stricter filtering still improves repair without adding hallucinated mentions, and the looser o0.85 variant accepts 46 images, adds 49 grounded mentions over repair, adds 0 hallucinated mentions, and raises retained vanilla grounded mentions to 77.49%. The result is still a high-risk 100-image prototype, not a full caption benchmark result.

### Atomic Gain Decomposition

| Gain audit | Raw augmented vs repair | Verified selected vs repair |
|---|---:|---:|
| accepted / fallback images | -- | 30 / 70 |
| delta object mentions | 63 | 34 |
| delta grounded mentions | 59 | 34 |
| delta hallucinated mentions | 4 | 0 |
| images with more grounded mentions | 46 | 25 |

The gain decomposition shows the verifier's role directly. Raw atomic spans add 59 grounded mentions but also 4 hallucinated mentions. Verified selection keeps 34 new grounded/object mentions across 25 images while adding 0 hallucinated mentions. The improvement is therefore not an empty-caption or repetition artifact; it is the verifier selecting useful atomic additions and rejecting risky ones.

## Method Decision

The practical method should now be framed as **TDEV-guided claim acceptance for faithful concise captioning with constrained local repair/regeneration**, not as a pure token-ban decoder. The saved runs show that object claims are usually token-locatable and deny lists are narrow, but hard token suppression alone routes the model into new unsupported claims or incomplete fragments. Sentence acceptance catches those unsupported substitutes, which is exactly the target-vs-neighbor criterion we want. Its length reduction is not inherently bad: concise captions are preferable to long captions that keep inventing objects; the risk is only when shortening removes supported visible content or truly collapses into content-light captions. The content-light audit shows that many CHAIR-objectless captions are still descriptive. The controlled-regeneration probe, candidate-pool oracle, deployable-style verified selector, and local-addition probe show why prompt-only rewriting plus selection is insufficient. The atomic-detail probe gives the current method direction: propose short missing-detail spans, verify each introduced claim against target-vs-neighbor evidence, and fall back to the deterministic repair when no safe new detail is found.

The next implementation target is therefore:

1. Generate or keep a candidate sentence/span.
2. Extract object-like claims, including open-vocabulary route forms.
3. Map each claim to a canonical target when possible and score target-vs-neighbor evidence.
4. Accept supported claims and reject unsupported claims; allow the caption to become shorter when unsupported detail is the only thing being removed.
5. Apply constrained local repair first; use regeneration only as a verified candidate source for missing visible details, and fall back to the repaired caption when regenerated claims fail target-vs-neighbor checks.
6. Report CHAIR together with concise-faithfulness metrics: retained supported objects, mean words, object mentions, empty/generic-caption rate, and manual examples.

This preserves the paper's motivation: the method does not merely make object claims less frequent, and it does not rely on visual routing as proof. It explicitly tests whether the candidate claim is target-discriminative under related evidence. The desired behavior is not maximum caption length; it is concise but faithful captioning that keeps supported visual content and stops before unsupported object invention.

## Paper-Safe Scope

Current evidence supports a diagnostic-plus-verification paper with a bounded caption-side prototype. It does not yet support claiming a complete end-to-end caption mitigation method. For ICML, the strongest practical path is verification-in-loop atomic detail generation, evaluated beyond the current 100-image high-risk set with CHAIR, content-light, and content-preservation audits. The prompt-only regeneration probe, candidate-pool oracle, verified selector, and local-addition probe should be reported as negative controls.

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
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_sentence_acceptance_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_decode_gate_prefilter_100_iter2_t96_claim_repair_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_controlled_regen_100_t80/controlled_regeneration_metrics.json`
- `detection/baselines/results/tdev_caption_controlled_regen_100_t80_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_controlled_regen_100_t80_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128/controlled_regeneration_metrics.json`
- `detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_controlled_regen_100_detail_t128_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_candidate_pool_oracle_100/candidate_pool_oracle_metrics.json`
- `detection/baselines/results/tdev_caption_candidate_pool_oracle_100_min_hallucination_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_candidate_pool_oracle_100_no_worse_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_verified_expansion_100_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_caption_verified_candidate_select_100/verified_candidate_selection_metrics.json`
- `detection/baselines/results/tdev_caption_verified_candidate_select_100_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_verified_candidate_select_100_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_verified_candidate_select_100_r10/verified_candidate_selection_metrics.json`
- `detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_verified_candidate_select_100_r10_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_verified_local_additions_100/verified_local_addition_metrics.json`
- `detection/baselines/results/tdev_caption_verified_local_additions_100_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_verified_local_additions_100_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_verified_local_additions_100_o04/verified_local_addition_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_gen_100/atomic_detail_generation_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_gen_100_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100/verified_atomic_detail_selection_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100/atomic_detail_gain_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o050/atomic_detail_reparse_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o050_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o050/verified_atomic_detail_selection_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o050_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o050_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100_o050/atomic_detail_gain_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o085/atomic_detail_reparse_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_reparse_100_o085_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o085/verified_atomic_detail_selection_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o085_preservation/caption_variant_preservation_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_select_100_o085_content_light/caption_content_light_metrics.json`
- `detection/baselines/results/tdev_caption_atomic_detail_gain_audit_100_o085/atomic_detail_gain_metrics.json`
