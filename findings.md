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
- Qwen3-VL video-native sensor (122/122 clips, 6,191 mentions): OOF
  overall/within-bin/matched AUROC 0.7564/0.7519/0.7568; the frozen
  threshold realizes a 2.020% grounded trigger rate.
- Corrected Qwen2.5 two-branch decoding passes VCD no-op identity 5/5,
  rollback/shared-control identity 5/5, and frozen vanilla replay 20/20.
  Formal PAI/VCD still waits for the independent LLaVA decoder gate.

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
inconclusive paired hallucination deltas. Corrected PAI/VCD rows remain gated
only by recording the audited decoder commit and validation manifest; the
fresh dual-model identities and 37-test suite pass.
