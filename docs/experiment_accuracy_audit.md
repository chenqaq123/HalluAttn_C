# Experiment Accuracy Audit

This audit records the current evidence that the main experimental conclusions
are supported by saved artifacts. It should be updated whenever results are
regenerated.

## Detection Baselines

Result root:

```text
detection/baselines/results/coco_llava_7b_baselines/
```

Object cache and label counts match `metrics.json`:

| Quantity | Value |
|---|---:|
| object mentions | 16,426 |
| hallucinated | 4,009 |
| grounded | 12,417 |
| position-only AUROC | 0.830414 |

Independent recomputation from `baseline_scores.npz` matches the saved
`metrics.json` values exactly for the core table:

| Method | Overall | Within-bin | Residual | Interpretation |
|---|---:|---:|---:|---|
| PAS | 0.834859 | 0.592847 | 0.569433 | high global score is mostly position-correlated |
| SVAR | 0.833643 | 0.574802 | 0.556970 | same position-confound pattern as PAS |
| IC | 0.776101 | 0.685611 | 0.633147 | strongest controlled baseline among current methods |
| GLSim-local | 0.772392 | 0.563597 | 0.600581 | some residual signal, weak within-bin |
| Entropy | 0.720592 | 0.636642 | 0.633548 | stable uncertainty baseline |
| NLL | 0.711388 | 0.635994 | 0.629718 | stable uncertainty baseline |
| Beyond-ADS | 0.503325 | 0.510928 | 0.501202 | near random |
| Beyond-CGC | 0.691734 | 0.520838 | 0.518110 | mostly position/weak controlled signal |
| Beyond-ADS+CGC | 0.692573 | 0.522593 | 0.521225 | combined score remains weak after controls |

The stronger post-hoc controls support the same conclusion. Same-object-word
matched AUROC separates stable non-attention baselines from position-confounded
attention-mass baselines:

| Method | Same-word AUROC | Same-word pairs | Linear residual | Cubic residual |
|---|---:|---:|---:|---:|
| PAS | 0.569094 | 123,375 | 0.580939 | 0.557904 |
| SVAR | 0.565447 | 123,375 | 0.572552 | 0.545911 |
| IC | 0.716166 | 123,375 | 0.636844 | 0.633306 |
| Entropy | 0.711562 | 123,375 | 0.633428 | 0.633959 |
| NLL | 0.697613 | 123,375 | 0.628125 | 0.629749 |
| Beyond-ADS+CGC | 0.494565 | 123,375 | 0.526758 | 0.520328 |

Caveat: IC, GLSim, and Beyond-ADS/CGC are adapted or paper-level
implementations, not bit-level official reruns. Paper text should use that
scope explicitly.

## Detection Reproducibility Guard

A real smoke run exposed a processor-version mismatch: current transformers
expanded LLaVA-1.5 image placeholders to 575 tokens, while the existing
`generation.json` uses 576 image tokens. The loader now fills the missing
processor metadata from the model config and sets `num_additional_image_tokens`
to 1 for this checkpoint. A GPU smoke run with `--limit 2` and
`--alignment_policy error` then completed and wrote metrics successfully.

Smoke output root:

```text
/tmp/sinkdetect_baseline_alignment_smoke
```

## Mitigation POPE

Result root:

```text
mitigation/results/coco_llava_7b_attention_only/
```

Strict yes/no parsing was applied offline to every saved POPE prediction. All
36,000 POPE outputs start with an explicit `yes` or `no`, so the stricter parser
has `invalid=0` for every method/split and leaves all saved metrics unchanged.

Macro pattern from saved metrics:

| Split | Method | Yes rate | MCC | Interpretation |
|---|---|---:|---:|---|
| adversarial | vanilla | 0.479333 | 0.665902 | anchor |
| adversarial | PAI attention-only | 0.474667 | 0.664854 | no reliable gain |
| adversarial | ClearSight | 0.524667 | 0.646120 | yes-rate increases, discrimination drops |
| adversarial | VisAttnSink | 0.486000 | 0.656257 | no reliable gain |
| popular | vanilla | 0.445333 | 0.740439 | anchor |
| popular | PAI attention-only | 0.442000 | 0.738318 | no reliable gain |
| popular | ClearSight | 0.479333 | 0.737964 | yes-rate increases, no MCC gain |
| popular | VisAttnSink | 0.450667 | 0.732910 | no reliable gain |
| random | vanilla | 0.425000 | 0.785554 | anchor |
| random | PAI attention-only | 0.421667 | 0.783677 | no reliable gain |
| random | ClearSight | 0.454667 | 0.789920 | modest MCC gain with yes-rate increase |
| random | VisAttnSink | 0.429667 | 0.779080 | no reliable gain |

A two-sample real POPE smoke run using the strict parser also completed with
`invalid=0`.

### Current Runtime Pilot

After committing the shared mitigation runtime stack, a fresh real-model pilot
was run with the local paths supplied for this workspace:

```text
model: /home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9
COCO:  /home/chenguanxu/common_dataset/coco-2014-dataset
POPE:  /home/chenguanxu/common_dataset/pope
GPUs:  CUDA_VISIBLE_DEVICES=1,5, NUM_SHARDS=2
```

Result root:

```text
mitigation/results/pope_random_limit100_runtime_audit/
```

The run covers 100 POPE-random rows and verifies that the tracked runner,
dataset loader, local checkpoint, strict yes/no parser, shard merge, and method
comparison all execute end-to-end. All methods have `invalid=0`, and
`compare_methods.py` confirms matched sample IDs against vanilla.

| Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.870 | 0.744 | 0.820 | 0.080 | 0.450 | anchor |
| PAI attention-only | 0.870 | 0.744 | 0.820 | 0.080 | 0.450 | 0.000 |
| ClearSight | 0.860 | 0.721 | 0.840 | 0.120 | 0.480 | -0.020 |
| VisAttnSink | 0.890 | 0.781 | 0.860 | 0.080 | 0.470 | +0.040 |

This pilot is not the paper-facing full-data result, but it validates the
current tracked runtime against the local model/data environment. It also
matches the broader conclusion that unselective attention interventions are
not reliable mitigation: PAI is unchanged, ClearSight increases FPR more than
TPR, and the small VisAttnSink gain needs full-split and semantic-neighbor
confirmation before it can be treated as a real effect.

### VCD-Greedy Pilot

A second 100-row POPE-random pilot was run after adding the controlled
VCD-greedy port. It uses the official VCD contrastive logit form and diffusion
noise schedule, but keeps greedy decoding to match the rest of this repository's
POPE/CHAIR generation setup.

Result root:

```text
mitigation/results/pope_random_limit100_vcd_seeded_audit/
```

The vanilla anchor exactly repeats the previous 100-row pilot. VCD-greedy has
`invalid=0` and matched sample IDs, but does not improve this subset:

| Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.870 | 0.744 | 0.820 | 0.080 | 0.450 | anchor |
| VCD-greedy | 0.860 | 0.725 | 0.800 | 0.080 | 0.440 | -0.020 |

This is early negative evidence for deterministic VCD under the current greedy
POPE protocol, not a final statement about the official stochastic VCD setup.
The next check is full-split VCD-greedy plus, if needed, a small official-style
sampling parity check.

## Mitigation CHAIR

CHAIR caption metrics support the same scoped conclusion: current attention
ports do not reliably reduce object hallucination under caption-style controls.

| Method | CHAIRi | CHAIRs | Mean words | Object mentions | Hallucinated mentions | Main delta vs vanilla |
|---|---:|---:|---:|---:|---:|---|
| vanilla | 0.133984 | 0.489800 | 89.4534 | 7.6054 | 1.0190 | anchor |
| PAI attention-only | 0.134189 | 0.500400 | 90.1726 | 7.7458 | 1.0394 | slightly worse CHAIR and more hallucinated mentions |
| ClearSight | 0.135051 | 0.493200 | 87.6646 | 7.4920 | 1.0118 | shorter/sparser captions, no CHAIR improvement |
| VisAttnSink | 0.142621 | 0.522000 | 90.9278 | 7.8502 | 1.1196 | richer captions with more hallucinated mentions |

## Current Supported Conclusions

1. Global attention-mass detector performance is heavily position-confounded.
2. Mean-over-head attention shape and fine-grained attention baselines do not
   recover robust target verification under the current controls.
3. Non-attention uncertainty and representation baselines retain more
   position-controlled signal than PAS/SVAR/Beyond in the current run.
4. Current attention-only mitigation ports do not provide reliable object-level
   mitigation under POPE/CHAIR diagnostics; a new 100-row runtime pilot also
   shows no robust attention-only improvement.
5. Claims must remain scoped: these are controlled ports/adapted baselines, not
   proof that every attention-based method or every official method fails.

## Missing Evidence Before ICML Submission

- Official-code parity checks on small subsets for PAI, ClearSight, and
  VisAttnSink where feasible.
- Multi-model replication beyond LLaVA-1.5-7B.
- Full regeneration of POPE/CHAIR mitigation results with the now-tracked
  runtime stack, or a documented hash-level equivalence check against the
  existing full run.
- Full-split VCD-greedy evaluation and, if paper claims compare to official
  VCD, a stochastic decoding parity check on a small subset.
- Head-selection or head-specific mitigation baselines as positive
  counterexamples to unselective attention amplification.
