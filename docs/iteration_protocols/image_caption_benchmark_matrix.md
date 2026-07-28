# Registered protocol: open image-caption mitigation matrix

**Registration date:** 2026-07-28  
**Status:** frozen before benchmark generation  
**Hypothesis:** H1 in `research-state.yaml`

## Question

Does detector-gated rollback deletion reduce open-ended object hallucination
across COCO and two independent open-caption benchmarks, and how does it
compare with published attention-intervention baselines when every method uses
the same base model, input, prompt, and decoding budget?

## Matrix

### Models

1. `llava-hf/llama3-llava-next-8b-hf` (reported as **LLaVA-NeXT-8B**).
2. `/home/chenguanxu/models/Qwen2.5-VL-7B-Instruct`
   (**Qwen2.5-VL-7B-Instruct**).
3. `/home/chenguanxu/models/Qwen3-VL-8B-Instruct`
   (**Qwen3-VL-8B-Instruct**).

The exact local snapshot/revision, `transformers` version, dtype, attention
backend, and image preprocessing resolution are recorded with every result.

### Benchmarks

1. **CHAIR (COCO 2014):** the official deterministic 500-image split used by
   the AllPath CHAIR launcher. Prompt:
   `Please describe this image in detail.` Primary metrics: CHAIRs and CHAIRi.
2. **THRONE-Objects365:** all 5,110 image IDs in the official
   `objects365_subset.txt`, using the corrected Objects365 validation
   annotations. Prompt: `Describe this image in detail.` Primary metric:
   classwise F2; overall precision, recall, F1, F2 and parser valid-vote rate
   are also reported.
3. **AMBER-generative:** all 1,004 official images and annotations. Prompt:
   `Describe this image.` Primary metrics: CHAIR and Hal; Cover and Cog are
   also reported.

All data and manifests must reside under
`/home/chenguanxu/common_dataset`. File counts and hashes of the split
manifests/annotations are recorded before generation.

### Methods

Each cell includes an unmodified **vanilla** output, generated once and reused
as that cell's paired control, plus the following methods:

1. **PAI:** formula-faithful port of official PAI, including both visual
   attention amplification and the unconditional-image contrastive branch.
   Freeze `alpha=0.5`, `gamma=2`, plausibility cutoff `0.1`, and intervention
   layers `[2, L)`. Dynamic visual-token boundaries replace hard-coded
   LLaVA offsets.
2. **ClearSight:** formula-faithful VAF port with enhancement `1.15`,
   suppression `0.95`, and the relative layer interval corresponding to the
   official LLaVA interval `[9,15)` in a 32-layer decoder:
   `[floor(9L/32), ceil(15L/32))`. Dynamic image-token boundaries are used.
3. **DIML:** formula-faithful heads-guided attention intervention with
   `alpha=0.5`; the official LLaVA middle-layer interval `[5,18)` is mapped
   without tuning to `[floor(5L/32), ceil(18L/32))`. At every selected layer,
   absolute visual-attention logits are averaged over heads and added to each
   head's visual logits exactly as disclosed by DIML.
4. **AllPath:** the official four-path output-projection intervention. Because
   the released head lists cover LLaVA-1.5 and legacy Qwen-VL only, head
   categories are re-identified separately for each target architecture on a
   disjoint COCO-resampled development set, using the official identification
   rule. The proportions corresponding to 40 good-format, 40 bad-format,
   50 good-image, and 0 bad-image heads out of 1,024 are used, rounded once per
   architecture; `in_scale=2`, `de_scale=0`. No benchmark examples or labels
   enter head selection.
5. **ours:** model-specific post-row, No-RoPE visual-allocation sensor followed
   by detector-gated rollback-ban deletion. The sensor is trained on disjoint
   COCO caption generations and calibrated on a held-out image split. Object
   candidates are first noun/proper-noun mentions, so application is not
   restricted to the COCO-80 vocabulary. The frozen threshold targets a 2%
   false-trigger rate on grounded calibration mentions. A fired token span is
   rolled back and banned once; repeated rollback at the same prefix is
   forbidden.

The published PAI/ClearSight/DIML/AllPath repositories do not provide native
implementations for all three target architectures. Therefore every such
cross-architecture result is labelled **formula-faithful port**, never
“official reproduction.” The AllPath development cost and selected heads are
reported separately. A failed or technically impossible port remains an
explicit missing cell with the failure reason; it is not replaced by a number
from another model.

## Frozen generation controls

- Images, prompts, system messages, chat templates, processor revisions, dtype,
  and maximum generated tokens are identical across methods within a
  model/benchmark block.
- `max_new_tokens=512`; batch size may vary only for memory.
- Vanilla, ClearSight, DIML, AllPath, and ours use greedy decoding. PAI uses
  the deterministic contrastive logits defined by PAI. No beam search.
- Random seeds: generation `927`; data development split `17`; bootstrap
  `20260728`.
- COCO data used to train/calibrate ours or identify AllPath heads is grouped
  by image and disjoint from the CHAIR 500 evaluation images.
- No threshold, layer interval, head count, or decoding parameter is selected
  on CHAIR, THRONE, or AMBER test labels.
- Methods may change token probabilities, but not the prompt or stopping rule.
  Premature-EOS frequency and length are reported to expose truncation.

## Metrics and audits

For every cell report:

- all benchmark-native hallucination and coverage metrics;
- generated-token count, word count, unique-noun count, repetition rate,
  premature-EOS rate, and wall time;
- paired change from vanilla and a 10,000-resample image-cluster bootstrap
  95% confidence interval;
- for ours: candidate mentions, triggers, rollbacks, completed deletions,
  evaluator-matched removed hallucinations, grounded removals, removal
  precision, trigger rate, and unresolved-trigger rate;
- for attention baselines: modified layers/heads, actual visual-token span,
  intervention-call count, and a zero-intervention equivalence test;
- exact evaluator version and, for THRONE, per-parser agreement/valid-vote
  coverage.

Caption quality is not inferred from hallucination rate alone. At minimum,
object recall/coverage and caption length are paired with every primary metric.
If evaluator dependencies permit, a reference-caption quality metric is
reported only as secondary evidence.

## Validity gates

1. Vanilla regeneration must match a saved vanilla file byte-for-byte on a
   20-image deterministic smoke subset.
2. With every intervention coefficient set to its identity value, generated
   tokens must match vanilla on that subset.
3. Each method must alter at least one intended internal tensor on an
   instrumented example; tensor shapes and token bounds are logged.
4. A result is valid only when the evaluator covers the registered full split,
   with any missing sample IDs listed.
5. H1 is supported only when ours improves the benchmark-native hallucination
   metric with a paired CI excluding zero on at least two benchmarks and does
   not produce a statistically clear loss in the associated coverage/recall
   metric. Otherwise the conclusion is mixed or unsupported.

## Execution order

1. Materialize and checksum datasets/manifests.
2. Implement architecture adapters and pass the identity/smoke tests.
3. Generate vanilla once per model/benchmark.
4. Run PAI, ClearSight, DIML, AllPath, and ours in the registered order, while
   assigning model/benchmark blocks across available GPUs.
5. Run official evaluators, paired audits, and then append results to
   `docs/experiment_results.md`.

