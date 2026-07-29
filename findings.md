# Research Findings

## Research Question

Does detector-gated rollback deletion reduce open-ended image/video
hallucination across modern LVLMs, and is its detector signal specifically
carried by cross-layer and cross-head allocation differences?

## Current Understanding

The durable negative result is two-pronged: coarse attention is not reliable
target-object verification when read as a detector after position control, and
pushing attention can add answer bias or destabilize generation. A supervised
post-row No-RoPE allocation probe nevertheless detects emitted hallucinated
mentions. Its useful action is deliberately narrow: trigger rollback and ban
the detected object's first subtoken, so generation can omit it. Existing COCO
pilots support high-precision deletion on LLaVA-1.5 and Qwen3-VL; an existing
video-native Qwen2.5-VL run supports a precision gain on Vript-HAL.

The current expansion asks whether that constructive corollary survives
independent open-caption benchmarks and modern model families, under matched
baselines. It also consolidates the detector mechanism claim into a direct
single-layer/multi-layer/all-layer by head-mean/per-head comparison.

## Key Results

- LLaVA-1.5 COCO rollback pilot: CHAIRs 0.491 to 0.431, CHAIRi 0.1397 to
  0.1194, removal precision 0.870.
- Qwen3-VL COCO rollback pilot: CHAIRs 0.451 to 0.255, CHAIRi 0.0981 to
  0.0529, removal precision 0.893.
- Qwen2.5-VL video-native Vript-HAL: detector OOF within-bin AUROC 0.7475,
  removal audit 0.735, caption precision +1.65 points with a positive
  clip-bootstrap confidence interval, and approximately flat F1.
- Existing LLaVA detector mechanism audit: all-layer/per-head within-bin AUROC
  0.8812; head-mean 0.7966; head deviations 0.8805; best single layer 0.7354;
  no fixed eight-layer band is sufficient.
- Confirmatory capacity-controlled factorial (16,426 mentions):
  all-layer/per-head 0.8768 versus nested selected single-layer 0.8404,
  fixed eight-layer bands 0.8062--0.8682, halves 0.8679/0.8697, and
  all-layer head-mean 0.8030. Both main-effect gates pass; head-deviation
  reaches 0.8797. The positive interaction gate fails (-0.1824).
- Registered LLaVA-NeXT-Video shared Vript-HAL control (122/122 clips):
  official precision 70.12, recall 23.91, F1 35.19, and mean caption length
  193.69 words. This is a control row, not a mitigation result.
- Registered Qwen2.5-VL shared CHAIR control (500/500 images): CHAIRs 28.20,
  CHAIRi 8.72, recall 58.27, precision 84.06, F1 68.83, and mean caption
  length 156.48 words. This is a control row, not a mitigation result.
- Registered Qwen3-VL AllPath development audit (500/500 images): 1,684
  mentions identify 101 unique good heads and 45 unique hallucination heads
  with no overlap; all 36×32 bounds and finite-score checks pass.
- Registered Qwen2.5-VL CHAIR ClearSight (500/500): CHAIRs 28.20→26.00
  and CHAIRi 8.72→7.59 with recall 58.27→58.21, but all paired intervals
  include zero; captions are 11.04 words longer.
- Registered Qwen2.5-VL CHAIR DIML (500/500): CHAIRs 28.20→15.60 and
  precision 84.06→89.43 significantly improve, while recall 58.27→50.29
  and F1 68.83→64.38 significantly decline. This is a coverage trade-off,
  not a Pareto improvement.
- Registered LLaVA-NeXT CHAIR control (500/500): CHAIRs 34.40, CHAIRi 9.75,
  recall 65.62, and F1 72.91.
- LLaVA-NeXT CHAIR ClearSight and DIML reach CHAIRs/F1
  33.40/73.87 and 31.20/74.32, respectively, but every paired interval
  includes zero. DIML shortens captions substantially without a significant
  recall/F1 loss, unlike its Qwen2.5 coverage trade-off.
- Qwen3-VL video-native sensor (122/122 clips, 6,191 mentions): OOF
  overall/within-bin/matched AUROC 0.7564/0.7519/0.7568; the frozen
  threshold realizes a 2.020% grounded trigger rate.
- Registered Qwen3-VL shared Vript-HAL control (122/122 clips): released
  precision 71.62, recall 39.84, F1 50.49, and mean caption length 271.12
  words. Independent replay passes 20/20 and rollback/shared-control identity
  passes 5/5. This is a validated control, not a mitigation result.
- Corrected Qwen2.5 two-branch decoding passes VCD no-op identity 5/5,
  rollback/shared-control identity 5/5, and frozen vanilla replay 20/20.
  The independent LLaVA decoder gate and committed validation manifest also
  pass, so corrected formal PAI/VCD execution is released.
- Corrected LLaVA-NeXT active PAI changes 20/20 outputs and exercises every
  frozen layer 2--31; mean content tokens change 198.85→176.00 with no
  premature EOS in either branch. This validates activity, not efficacy.
- Registered LLaVA-NeXT image sensor source (2,967 disjoint COCO images,
  7,382 mentions): all four feature shards pass exact image/order,
  per-image mention, finite tensor, unique object-ID, and zero-CHAIR-overlap
  audits. Five folds use 2,318 training and 580 calibration images; the
  independently recomputed threshold is 2.53055 and realizes a 2.020%
  grounded trigger rate.
- LLaVA-NeXT-Video vanilla replay passes 20/20 and rollback
  token/caption/shared-control identity passes 5/5, releasing its formal
  video ours row.
- LLaVA-NeXT image vanilla replay passes 20/20 and rollback
  token/caption/shared-control identity passes 5/5 with the audited sensor,
  releasing formal CHAIR/AMBER ours.
- Registered Qwen3-VL Vript-HAL ours (122/122): precision/recall/F1
  71.62/39.84/50.49→71.92/40.38/51.01. Paired deltas are
  +0.30/+0.54/+0.52 points, but all 95% intervals include zero, so the
  model-level H2 gate fails. Removal precision is 65.09% over 318 completed
  rollbacks; captions remain the same length and have no premature EOS.
- Registered LLaVA-NeXT CHAIR full PAI (500/500): CHAIRs is unchanged at
  34.40, while CHAIRi significantly worsens 9.75→14.54, recall significantly
  falls 65.62→62.43, and F1 significantly falls 72.91→70.57. Captions are
  45.29 words shorter. This is a valid active full-PAI negative result, not
  the earlier POPE attention-only no-op.
- Registered LLaVA-NeXT CHAIR AllPath (500/500): CHAIRs/CHAIRi point
  estimates improve 34.40/9.75→31.80/8.43 but both intervals cross zero;
  recall and F1 significantly rise. Captions expand 164.14→268.45 words,
  repetition rises 0.435→0.643, and 162/500 hit the token cap. The coverage
  gain is therefore entangled with severe verbosity/repetition.
- Registered Qwen2.5-VL CHAIR AllPath (500/500): CHAIRs/CHAIRi significantly
  improve 28.20/8.72→18.20/4.87 and precision rises
  84.06→89.31, while recall and F1 intervals include zero. Captions expand
  156.48→372.81 words, repetition rises 0.481→0.797, and 378/500 hit the
  token cap. This is a significant native-score gain under an extreme
  verbosity/repetition confound, not a clean quality improvement.
- Registered Qwen2.5-VL CHAIR ours (500/500): CHAIRs/CHAIRi move
  28.20/8.72→26.20/7.98, but both paired intervals include zero; recall,
  precision, and F1 are likewise inconclusive. All 500 embedded source
  captions exactly match the frozen vanilla row. Of 1,953 completed
  rollbacks, CHAIR can score 55 removals: 34 hallucinated and 21 grounded,
  for 61.82% removal precision. The model-level CHAIR gate fails.
- Registered Qwen2.5-VL AMBER-generative control (1,004/1,004): released
  CHAIR 4.30, Cover 52.92, Hal 21.51, Cog 1.04, 83.33 mean words, and
  6.56 selected object nouns. This is the frozen paired control, not a
  mitigation result.
- Registered LLaVA-NeXT AMBER-generative control (1,004/1,004): released
  CHAIR 9.79, Cover 62.10, Hal 55.58, Cog 4.16, 144.11 mean words, and
  10.52 selected object nouns. This is the frozen paired control, not a
  mitigation result.

## Patterns and Insights

The allocation fingerprint is distributed across layers and dozens of heads,
with head-relative differences carrying much more signal than the layer-wise
mean trajectory. The new factorial confirms both broad-layer and relative-head
main effects under equal capacity, but finds that they are partly redundant
rather than positively synergistic. Detection strength does not imply
candidate-ranking ability:
same-prefix counterfactual audits show native likelihood is stronger for
choosing alternatives. The intervention therefore uses the detector only
after a mention is emitted and obtains value primarily through deletion, not
replacement.

## Lessons and Constraints

- Keep model-native baseline reproductions separate from new
  cross-architecture ports.
- Do not report full PAI when only its attention component was run.
- DIML's published LLaVA layer range and AllPath's identified heads cannot be
  transferred silently to Qwen architectures.
- SmartSight's disclosed core uses ten temperature-0.7 candidates, TAC
  selection, 32 frames at resolution 392, alpha 1.2, and window 10. Its
  segment-boundary threshold is not disclosed in the paper and official code
  is unavailable, so any local implementation requires a frozen,
  clearly-labelled completion of that detail.
- Generated artifacts remain under ignored experiment directories; benchmark
  data belongs under `/home/chenguanxu/common_dataset`.
- The configured common-dataset CIFS mount is currently read-only. The
  checksum-verified AMBER, THRONE, and Vript-HAL payloads are staged under
  `/tmp/sinkdetect_datasets` and cannot be copied to the requested destination
  until the mount is restored read-write.
- A direct two-branch decoder that bypasses
  `prepare_inputs_for_generation` is invalid for Qwen-VL because it omits
  multimodal RoPE/cache-position state. The earlier Qwen2.5 PAI pilot is
  invalidated. Branch initialization must also mirror GenerationMixin's
  `logits_to_keep=1` support path: LLaVA-NeXT-Video otherwise returns rank-4
  logits through its `None` default. The final repair passes fresh 5/5 gates
  on both Qwen2.5 and LLaVA-NeXT-Video, including 512-token continuations,
  without changing Qwen's audited output.
- Feature extraction checkpoints are a synchronized JSONL/NPZ pair, not two
  independently resumable files. An orphan peer, duplicate image completion,
  per-image mention-count mismatch, duplicate object ID, or partial source
  coverage must fail closed. Readiness is established by full registered-set
  and tensor audit, never by JSONL line count alone.

## Open Questions

- Does rollback deletion preserve gains on THRONE-Objects365 and
  AMBER-generative, where COCO-80 labels are not the sole ontology?
- Can one probe vocabulary/labeling recipe be made fair across all three image
  benchmarks without benchmark test labels leaking into decoding?
- Does video-native sensing transfer to LLaVA-NeXT-Video and Qwen3-VL, or must
  it be retrained per architecture?
- Why are the confirmed cross-layer and cross-head components negatively
  interactive: shared post-row evidence, redundant layer bands, or fold-local
  compression?

## Optimization Trajectory

The detector factorial is complete: cross-layer and cross-head component gates
pass, while the positive interaction and full-mechanism gates fail. Dataset,
formula-port, evaluator, and shard-merging audits are complete. AllPath head
identification is frozen for all three image architectures, and Qwen3's
video-native OOF sensor is complete. Formal image generation is running; the
first mitigation row, Qwen2.5 CHAIR ClearSight, has nominally favorable but
inconclusive paired hallucination deltas. The corrected PAI/VCD decoder is
committed and its hash-bearing dual-model validation manifest now releases
the active and formal queues.
