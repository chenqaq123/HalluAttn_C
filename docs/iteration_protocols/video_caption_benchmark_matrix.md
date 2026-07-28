# Registered protocol: Vript-HAL video-caption mitigation matrix

**Registration date:** 2026-07-28  
**Status:** frozen before benchmark generation  
**Hypothesis:** H2 in `research-state.yaml`

## Question

On the same Vript-HAL clips, can detector-gated rollback deletion improve
object-hallucination precision without sacrificing F1, and how does it compare
with VCD and the disclosed SmartSight procedure across three video-capable
model families?

## Matrix

### Benchmark

Use all 122 clips and official annotations in Vript-HAL. The dataset is copied
under `/home/chenguanxu/common_dataset/VRIPT-HAL`; the exact annotation hash,
video count, readable-video count, and missing IDs are recorded. Use the
official caption prompt and evaluation implementation.

Primary metrics are the official object precision, recall, and F1. Report
caption length, candidate-object count, runtime, and intervention actions as
mandatory companion metrics.

### Models

1. `llava-hf/LLaVA-NeXT-Video-7B-hf`
   (**LLaVA-NeXT-Video-7B**).
2. `/home/chenguanxu/models/Qwen2.5-VL-7B-Instruct`
   (**Qwen2.5-VL-7B-Instruct**).
3. `/home/chenguanxu/models/Qwen3-VL-8B-Instruct`
   (**Qwen3-VL-8B-Instruct**).

The exact model revision, processor, chat template, video backend, dtype, and
attention implementation are stored with the output.

### Methods

Each model includes a shared **vanilla** run and three interventions:

1. **+VCD:** video-adapted Visual Contrastive Decoding with the official
   defaults `alpha=0.5`, `beta=0.1`, diffusion/noising step `500`.
   Perturbation is applied independently to every sampled frame while
   preserving the temporal order.
2. **+SmartSight:** disclosed-core SmartSight reimplementation. Sample ten
   i.i.d. captions at temperature `0.7`, compute frame- and segment-level
   Temporal Attention Collapse from attention averaged over generated steps,
   visual patches, heads, and layers, and select the lowest-collapse candidate;
   apply the disclosed sliding visual-attention variation with `alpha=1.2` and
   window `10`. SmartSight does not release code and its paper does not specify
   the segment-boundary threshold. The missing value is completed without
   labels as the within-clip median of adjacent-frame Hungarian matching costs.
   Results are therefore labelled **SmartSight disclosed-core
   reimplementation**, not official reproduction. The selected-candidate
   index, frame entropy, segment entropy, and number of segments are logged.
3. **ours:** model-specific video-native post-row, No-RoPE
   visual-allocation sensor plus detector-gated rollback-ban deletion.
   Training/evaluation uses fixed image-grouped/clip-grouped folds. For the
   122-clip benchmark, every intervention prediction is out-of-fold: the sensor
   and 2%-grounded-false-trigger calibration threshold used on a clip never see
   that clip's labels. Candidate spans are first noun/proper-noun mentions.

No `VCD+ours`, `SmartSight+ours`, or other combined intervention is included;
the rows are mutually exclusive.

## Frozen generation controls

- Uniformly sample 32 chronological frames per clip and resize/process at the
  model's path corresponding to 392-pixel SmartSight input. If a processor
  forces a different native crop, log the realized tensor size rather than
  silently resampling twice.
- Identical clip, frame indices, prompt, system message, processor revision,
  and `max_new_tokens=512` within a model block.
- Vanilla, VCD, and ours use greedy decoding. SmartSight uses its required
  ten draws at temperature `0.7`, with no top-k/top-p restriction beyond the
  model default.
- Seeds: primary generation/sampling `927`; OOF folds `17`; bootstrap
  `20260728`. If compute permits after the complete primary matrix,
  SmartSight-only sampling robustness uses seeds `1701` and `2027`; these
  secondary seeds cannot replace the primary result.
- No method parameter is tuned with Vript-HAL test labels.

## Metrics and audits

For each model/method report:

- official Vript-HAL precision, recall, and F1;
- paired differences from the shared vanilla output with 10,000 clip-cluster
  bootstrap confidence intervals;
- generated tokens/words, object mentions, repetition, premature EOS, wall
  time, and peak memory;
- VCD perturbed-branch calls and a zero-coefficient equivalence check;
- SmartSight candidate count, collapse components, selected index, segment
  count, and selection-time overhead;
- ours candidate mentions, triggers, rollbacks, completed deletions, matched
  hallucinated removals, grounded removals, removal precision, and unresolved
  triggers.

Published SmartSight paper numbers may be listed as external reference values
but never substituted for a missing local cell.

## Validity and conclusion gates

1. All registered video IDs must decode and produce one output; otherwise list
   missing/corrupt IDs and report both attempted and scored counts.
2. Frame indices must match across the four rows within a model.
3. Identity/no-op settings for VCD and ours must reproduce vanilla on a
   deterministic five-clip smoke set.
4. Attention hooks must show the intended visual positions and layers on one
   instrumented clip per architecture.
5. H2 is supported for a model only if ours increases precision with a paired
   CI excluding zero and its paired F1 CI does not exclude zero in the negative
   direction. The cross-model hypothesis is supported only if this holds for
   all three models; otherwise report mixed or unsupported evidence.

## Execution order

1. Copy and checksum Vript-HAL under the required common-dataset root.
2. Freeze frame manifests and validate all videos.
3. Pass adapter and no-op smoke tests.
4. Generate vanilla, VCD, SmartSight, and ours per model.
5. Evaluate, audit paired deletions, and append full results to
   `docs/experiment_results.md`.

