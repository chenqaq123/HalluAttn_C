# Current Result and Baseline Comparison

Date: 2026-06-19

This note is the current one-page answer to: *what are the results, and how do
they compare with baselines?* It should be read as a status summary, not as a
new source of numbers. The authoritative artifacts are listed in the last
section.

## Bottom Line

The current evidence supports a **diagnostic plus target-verification paper**.
It does not yet support a strong standalone caption-mitigation paper.

- Attention-only baselines and VCD-greedy do not close the semantic-neighbor
  false-positive gap. NoLan-compatible is a useful decoding exception: it lowers
  FPR and related FPR, but mainly by becoming more conservative and losing recall.
- AIR official code is now located and partially wired for the 120-row
  adversarial semantic-neighbor audit, but it is not yet a comparable result row;
  it still needs an isolated original-LLaVA AIR run.
- Raw object-region evidence is not enough: it improves aggregate MCC but
  over-fires when related objects are present.
- Target-vs-neighbor verification gives the best current POPE tradeoff and is
  much stronger on controlled CHAIR object-mention detection.
- Caption-side correction is now beyond pure text-edit proxy, but still only at
  smoke-test scale. The next required method step is TDEV-guided claim acceptance
  for faithful concise captioning, with constrained repair/regeneration only when
  deletion would remove central supported content or leave incoherent fragments.

## POPE Mitigation and Semantic-Neighbor Controls

Main comparison source:

```text
mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md
```

The regenerated all-split control table now includes NoLan-compatible:

| Method | Family | Macro MCC | TPR | FPR | Yes rate | Related FPR | Plain FPR | Gap | Adv. related FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | base | 0.730 | 0.813 | 0.087 | 0.450 | 0.114 | 0.028 | 0.086 | 0.164 |
| PAI attention-only | attention intervention | 0.728 | 0.808 | 0.084 | 0.446 | 0.110 | 0.028 | 0.082 | 0.159 |
| ClearSight VAF | attention intervention | 0.723 | 0.848 | 0.125 | 0.486 | 0.160 | 0.047 | 0.113 | 0.223 |
| VisAttnSink | attention intervention | 0.722 | 0.815 | 0.096 | 0.455 | 0.123 | 0.038 | 0.085 | 0.173 |
| VCD-greedy | contrastive decoding | 0.719 | 0.816 | 0.100 | 0.458 | 0.127 | 0.039 | 0.088 | 0.178 |
| NoLan-compatible | contrastive decoding | 0.731 | 0.778 | 0.058 | 0.418 | 0.076 | 0.018 | 0.058 | 0.107 |
| OWLv2 target direct | region evidence | 0.777 | 0.911 | 0.134 | 0.522 | 0.184 | 0.025 | 0.159 | 0.281 |
| OWLv2 margin direct | target-vs-neighbor verifier | 0.445 | 0.359 | 0.013 | 0.186 | 0.010 | 0.019 | -0.009 | 0.010 |
| Two-stage direct | target-vs-neighbor verifier | 0.769 | 0.850 | 0.083 | 0.466 | 0.111 | 0.021 | 0.090 | 0.167 |
| Two-stage gate | target-vs-neighbor verifier | 0.751 | 0.793 | 0.051 | 0.422 | 0.069 | 0.012 | 0.056 | 0.104 |
| Hybrid gate+rescue | target-vs-neighbor verifier | 0.763 | 0.806 | 0.051 | 0.429 | 0.069 | 0.012 | 0.057 | 0.105 |

Reading:

- Attention-only baselines and VCD-greedy stay close to vanilla or worsen the
  related-present false-positive rate.
- NoLan-compatible is the first positive decoding baseline in this refresh:
  related FPR drops from `0.114` to `0.076`, and FPR drops from `0.087` to
  `0.058`. The tradeoff is recall and answer prior: TPR drops from `0.813` to
  `0.778`, yes rate drops from `0.450` to `0.418`, and macro MCC is essentially
  tied with vanilla (`0.731` vs. `0.730`). It should be reported as a compatible
  local port, not official NoLan.
- Raw OWLv2 target evidence has the highest direct aggregate MCC, but it is not
  target-discriminative enough: related FPR rises to `0.184`.
- The strict margin rule shows the desired behavior on related negatives
  (`0.010` related FPR) but loses too much recall (`0.359` TPR).
- The hybrid gate+rescue rule remains the best current tradeoff: macro MCC is
  `0.763`, TPR is `0.806`, overall FPR is `0.051`, and related FPR is `0.069`.

This keeps the original conclusion but makes it sharper: language-prior
suppression can help, yet the unresolved failure is still target verification
under related visual evidence. AIR should be treated as the next official-code
stress test, not as an already measured row in this table.

## CHAIR Object-Mention Detection

Main comparison source:

```text
docs/tdev_ablation_summary.md
```

| CHAIR score | Overall AUROC | Within-bin | Matched-pair | Residual |
|---|---:|---:|---:|---:|
| Position only | 0.830 | 0.572 | 0.546 | 0.522 |
| Entropy | 0.721 | 0.637 | 0.655 | 0.634 |
| NLL | 0.711 | 0.636 | 0.652 | 0.630 |
| IC | 0.776 | 0.686 | 0.703 | 0.633 |
| LURE-style position+uncertainty | 0.831 | 0.641 | 0.657 | 0.635 |
| LURE-style all factors | 0.808 | 0.626 | 0.619 | 0.586 |
| LURE-style cooccurrence support | 0.576 | 0.496 | 0.481 | 0.480 |
| TDEV target absence + neighbor dominance | 0.874 | 0.852 | 0.854 | 0.722 |

Reading:

- Overall AUROC alone is misleading: position-only already reaches `0.830`.
- The controlled metrics are the important columns. Attention/position-style
  signals collapse there, while TDEV remains strong.
- This is the strongest current evidence that the target-vs-neighbor criterion
  is not just another position or language-prior proxy.

## Caption-Side Correction Status

Main comparison sources:

```text
paper/tables/table_appendix_caption_proxy.tex
detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json
```

| Caption variant | CHAIRi | CHAIRs | Mean words | Status |
|---|---:|---:|---:|---|
| vanilla | 0.1340 | 0.4921 | 89.54 | baseline |
| neutral rewrite top-5 | 0.1186 | 0.4505 | 89.41 | proxy rewrite |
| generic noun rewrite top-5 | 0.1186 | 0.4505 | 89.53 | proxy rewrite, not visual correction |
| sentence gate top-5 | 0.1165 | 0.4396 | 86.53 | more grammatical but too coarse |
| deletion top-10 | 0.1048 | 0.4047 | 88.97 | stronger edit, still post-hoc |

Reading:

- These rows show that TDEV scores select useful object-claim correction
  targets.
- They do not show a completed natural mitigation method, because CHAIR can
  improve when an object claim is replaced by a generic noun or removed.
- The decode-gate feasibility audit removes one implementation concern:
  `98.9%` of all object mentions and `97.7%` of TDEV top-5 selected
  hallucinated mentions match a tokenizer span near the saved generation
  position.
- A prefix-state `LogitsProcessor` smoke test now blocks all `804` tested
  TDEV-selected matched mentions and all `1,333` simulated phrase-generation
  steps.
- A surface-form generated-vs-generated LLaVA smoke test confirms the gate can
  change decoding through `generate(logits_processor=...)`. It removes denied
  `people/table` claims, but the same sample introduces a new `bottle` claim.
- A closed-loop audit shows that the new `bottle` claim is also unsupported:
  target score `0.0266`, best neighbor `cup` score `0.3573`, margin `-0.3307`,
  and two-stage TDEV predicts absent. This verifies integration and exposes why
  original-phrase suppression is insufficient.
- A follow-up one-image closed-loop smoke test precomputes unsupported COCO
  objects with OWLv2 target-vs-neighbor evidence and blocks 320 narrow alias
  token sequences. On image `391158`, it removes the `person`, `dining table`,
  and `cup` CHAIR objects from the generated caption and does not introduce any
  new COCO object claim in the audit. The cost is clear: the caption becomes
  conservative and train-only, so this is feasibility evidence for closed-loop
  object verification, not a final caption-quality result.
- A soft closed-loop variant now supports logits penalties instead of hard
  `-inf` blocking. On the same image, penalties `1.0` and `4.0` both avoid new
  COCO object claims but remain train-only.
- Two narrower-gate ablations clarify the next design. A prefix-triggered gate
  (`min_prefix_len_to_block=1`) is too late and leaves the vanilla caption
  unchanged, including `person/cup/dining table`. A top-risk closed-loop gate
  (`closed_loop_max_denied=30`) reduces denied sequences from 320 to 136 and
  still removes `person/cup/dining table` without new COCO claims, but remains
  train-only.
- The best current single-image tradeoff is `first_token_policy=single_token_only`:
  it avoids broad first-subtoken bans for multi-token object phrases. It removes
  `person/dining table`, keeps the supported `cup`, and the CHAIR audit reports
  no introduced COCO object claim.
- A new variant/open-vocabulary leak audit shows why this is still not a final
  quality result. The first single-token-first run contained `bottled drink`,
  which CHAIR missed but the variant audit flags as a `bottle` leak. Adding
  bottle variants to the gate blocks that form, but the model routes to
  `bottleneck`, then `bottling machine`, then `bottletop`. The open-vocabulary
  audit finds all four route phrases. However, raw phrase verification is not
  enough: `bottled drink`, `bottleneck`, and `bottling machine` are all judged
  present as literal OWLv2 prompts. A lexical candidate-to-denied-target mapper
  with threshold `0.80` maps each route to canonical target `bottle` without
  reading variant/root labels, and TDEV rejects all four (`bottle` target score
  `0.0266`, best neighbor `cup` score `0.3573`, two-stage present `0`). A
  threshold sweep keeps 4/4 route positives and maps 0/18 reference-unmapped
  nonroute candidates for thresholds `0.60` through `0.90`; stricter `0.95`
  and `1.00` settings lose the `bottled/bottling` routes. Thus `0.80` is a
  reasonable operating point rather than a knife-edge setting on this
  diagnostic. This is strong evidence that the next useful method step is
  open-vocabulary
  object-like phrase discovery plus candidate-to-target mapping and TDEV
  verification, not a larger hand-written alias list or raw phrase scoring
  alone.
- A full cached-caption mapper audit is a necessary caveat on that direction.
  Over 4,977 generated captions, the lightweight open-vocabulary extractor
  produced 157,476 candidates; 21,453 mapped to a canonical CHAIR/COCO object
  at threshold `0.80`. Of those mapped candidates, 21,142 were aligned with a
  CHAIR-recognized word in the same caption (`98.55%`), while 311 were extra
  mapped candidates (`1.45%`). This is not a human precision estimate; remaining
  extras are now dominated by vocabulary granularity cases such as `bear` versus
  `teddy bear`. The lexical mapper is therefore a stronger route-leak prototype
  and scaling diagnostic, but still not a paper-ready open-vocabulary object
  detector.
- A multi-image decode-gate prefilter audit now gives the next bounded GPU
  target. On the 100 highest-risk images selected from cached caption-mention
  TDEV scores, the prefilter selects 126 image-word deny candidates. CHAIR labels
  mark 109/126 as hallucinated (`86.51%` precision) and the selected candidates
  cover `66.46%` of hallucinated mentions on those images. The deny-list width
  is small: mean `1.26` unique denied words per image, p95 `2`, and with the
  LLaVA tokenizer mean `10.92` denied token sequences per image, p95 `24`. This
  supports a bounded multi-image generation run, but it is still a mention-score
  prefilter audit rather than generated-caption quality evidence.
- A five-image generated-caption smoke test confirms that the prefilter can be
  connected to real LLaVA decoding, but it also exposes the next failure mode.
  With `first_token_policy=single_token_only`, 1/5 captions changed: image
  `22596` removed the denied `bird` claim but routed to a new `person` claim,
  which CHAIR marks hallucinated and TDEV verifies as absent (`target_score`
  `0.0245`, `two_stage_present=0`). Allowing first-token blocking changes 2/5
  captions but keeps the same `bird -> person` substitution. A closed-loop top30
  deny-list also misses `person`; an all-unsupported-COCO deny-list includes it
  but leaves the caption truncated (`"two ch"`). A bounded second-pass expansion
  that adds only the TDEV-absent `person` substitute also removes the new COCO
  hallucination, but still ends with the same incomplete `"two ch"` fragment. The
  conclusion is negative but useful: dynamic replacement verification helps with
  substitution, but fixed token suppression is not enough. Soft penalty `1.0` is
  too weak and leaves the original `bird` claim unchanged on the stress image;
  soft penalty `4.0` behaves like hard blocking and still ends with `"two ch"`.
  An offline sentence-boundary repair on the iter2 5-image smoke removes
  incomplete trailing fragments in 4/5 captions, reduces CHAIRi from `0.1111` to
  `0.0588`, and reduces hallucinated mentions from 2 to 1, but shortens captions
  by 5 words on average under the 64-token smoke budget. A 96-token rerun shows
  that token budget alone is not the fix: hard gating now changes 5/5 captions
  and routes denied claims into complete substitute/escape forms such as
  `bottes`, `elephant`, `zebra`, `chickens`, and hallucinated `person`. Sentence
  repair still improves CHAIRi (`0.2500` to `0.1786`) but cannot remove complete
  substitute claims. Expanding the open-vocabulary audit limit from 32 to 96
  catches later substitutes such as `chickens`. An offline sentence-level
  candidate-acceptance proxy then reduces 96-token gated CHAIRi from `0.2500`
  to `0.1053` and hallucinated mentions from 8 to 2, while removing 24.6 words
  per caption on average. A concise-faithfulness audit clarifies that this is
  not merely empty-caption gaming: accepted captions retain `70.83%` of vanilla
  grounded object mentions, remove `75.00%` of gated hallucinated mentions, and
  have `0` generic/empty accepted captions under the current threshold. A new
  claim-local repair proxy keeps the same hallucinated mention count (`2`) while
  improving CHAIRi to `0.0909`, mean words to `52.40`, and retained vanilla
  grounded mentions to `83.33%` by trimming only detachable unsupported clauses
  when a safe sentence prefix remains. The next caption method should therefore
  scale verifier-guided claim acceptance with constrained local repair, not fixed
  token suppression or pure deletion.

## Paper-Safe Claim

The strongest safe claim is:

> Attention-only baselines and VCD-greedy do not reliably distinguish target
> evidence from related-object evidence. NoLan-compatible shows that
> language-prior suppression can reduce semantic-neighbor false positives, but it
> does so with a recall/yes-rate tradeoff and without explicit target-vs-neighbor
> verification. TDEV operationalizes that missing criterion and gives the best
> current POPE tradeoff, with strong controlled object-mention detection
> evidence. Caption mitigation remains a prototype until the closed-loop
> object-claim gate is scaled and softened enough to preserve useful detail.

## Authoritative Artifacts

- `mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md`
- `mitigation/results/semantic_neighbor_audit/nolan_adversarial_full_subset_eval/nolan_adversarial_full_comparison.md`
- `mitigation/results/semantic_neighbor_audit/nolan_full_subset_eval/semantic_neighbor_subset_metrics.csv`
- `docs/air_baseline_feasibility.md`
- `docs/tdev_ablation_summary.md`
- `paper/tables/table_semantic_neighbor_fpr.tex`
- `paper/tables/table_region_verifier_pope.tex`
- `paper/tables/table_appendix_caption_proxy.tex`
- `docs/caption_method_route_summary.md`
- `detection/baselines/results/tdev_decode_gate_feasibility/decode_gate_feasibility_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_smoke/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_smoke/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_smoke/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_soft_p1_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_candidate_smoke/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_candidate_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_top30_smoke/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_top30_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_smoke/gated_generation_metrics.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_variant_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v2_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_root_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_single_token_first_open_vocab_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_open_vocab_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v2_open_vocab_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_caption_closed_loop_variant_alias_v3_open_vocab_audit/closed_loop_example_audit.json`
- `detection/baselines/results/tdev_decode_gate_open_vocab_route_summary/open_vocab_leak_summary.md`
- `detection/baselines/results/tdev_decode_gate_open_vocab_auto_map_summary/open_vocab_leak_summary.md`
- `detection/src/sinkdetect/open_vocab_claims.py`
- `detection/baselines/results/tdev_decode_gate_open_vocab_mapping_thresholds/mapping_threshold_audit.md`
- `detection/baselines/results/open_vocab_mapper_caption_cache_audit/open_vocab_mapper_caption_cache_audit.md`
- `detection/baselines/results/tdev_decode_gate_multi_image_prefilter/multi_image_prefilter_summary.md`
- `detection/baselines/results/tdev_decode_gate_prefilter_smoke_5_comparison/prefilter_smoke_5_comparison.md`
