# Current Result and Baseline Comparison

Date: 2026-06-19

This note is the current one-page answer to: *what are the results, and how do
they compare with baselines?* It should be read as a status summary, not as a
new source of numbers. The authoritative artifacts are listed in the last
section.

## Bottom Line

The current evidence supports a **diagnostic plus target-verification paper**.
It does not yet support a strong standalone caption-mitigation paper.

- The reproduced attention/decoding baselines do not close the semantic-neighbor
  false-positive gap.
- Raw object-region evidence is not enough: it improves aggregate MCC but
  over-fires when related objects are present.
- Target-vs-neighbor verification gives the best current POPE tradeoff and is
  much stronger on controlled CHAIR object-mention detection.
- Caption-side correction is now beyond pure text-edit proxy, but still only at
  smoke-test scale. The next required method step is a softer/dynamic
  closed-loop object-claim gate, not more post-hoc deletion or generic noun
  replacement.

## POPE Mitigation and Semantic-Neighbor Controls

Main comparison source:

```text
mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md
```

| Method | Family | Macro MCC | TPR | FPR | Related FPR | Plain FPR | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Vanilla | base | 0.730 | 0.813 | 0.087 | 0.114 | 0.028 | 0.086 |
| PAI attention-only | attention intervention | 0.728 | 0.808 | 0.084 | 0.110 | 0.028 | 0.082 |
| ClearSight VAF | attention intervention | 0.723 | 0.848 | 0.125 | 0.160 | 0.047 | 0.113 |
| VisAttnSink | attention intervention | 0.722 | 0.815 | 0.096 | 0.123 | 0.038 | 0.085 |
| VCD-greedy | contrastive decoding | 0.719 | 0.816 | 0.100 | 0.127 | 0.039 | 0.088 |
| OWLv2 target direct | region evidence | 0.777 | 0.911 | 0.134 | 0.184 | 0.025 | 0.159 |
| OWLv2 margin direct | target-vs-neighbor verifier | 0.445 | 0.359 | 0.013 | 0.010 | 0.019 | -0.009 |
| Two-stage gate | target-vs-neighbor verifier | 0.751 | 0.793 | 0.051 | 0.069 | 0.012 | 0.056 |
| Hybrid gate+rescue | target-vs-neighbor verifier | 0.763 | 0.806 | 0.051 | 0.069 | 0.012 | 0.057 |

Reading:

- Attention and contrastive-decoding baselines stay close to vanilla or worsen
  the related-present false-positive rate.
- Raw OWLv2 target evidence has the highest direct aggregate MCC, but it is not
  target-discriminative enough: related FPR rises to `0.184`.
- The strict margin rule shows the desired behavior on related negatives
  (`0.010` related FPR) but loses too much recall (`0.359` TPR).
- The hybrid gate+rescue rule is the current best tradeoff: vanilla `0.730`
  macro MCC becomes `0.763`, overall FPR drops from `0.087` to `0.051`, and
  related FPR drops from `0.114` to `0.069`.

This is a modest but aligned improvement: it specifically targets the original
failure where related visual evidence makes the model answer `yes` for an absent
target.

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

## Paper-Safe Claim

The strongest safe claim is:

> Existing attention/decoding baselines do not reliably distinguish target
> evidence from related-object evidence. TDEV operationalizes this missing
> target-vs-neighbor verification criterion and gives a modest but more aligned
> reduction in semantic-neighbor false positives, with strong controlled
> object-mention detection evidence. Caption mitigation remains a prototype until
> the closed-loop object-claim gate is scaled and softened enough to preserve
> useful detail.

## Authoritative Artifacts

- `mitigation/results/semantic_neighbor_audit/paper_control_table/semantic_neighbor_control_table.md`
- `docs/tdev_ablation_summary.md`
- `paper/tables/table_semantic_neighbor_fpr.tex`
- `paper/tables/table_region_verifier_pope.tex`
- `paper/tables/table_appendix_caption_proxy.tex`
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
