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

### VCD-Greedy Full POPE Audit

`vcd` is a controlled greedy port of Visual Contrastive Decoding: it uses the
official original/noisy-image contrastive logit form and diffusion noise
schedule, but keeps greedy decoding to match the rest of this repository's
POPE/CHAIR generation setup.

Result root:

```text
mitigation/results/pope_full_vcd_greedy_audit/
```

The run covers all 9,000 POPE rows across random, popular, and adversarial
splits. Vanilla anchors reproduce the same operating point as the existing full
run, with only one-row-scale differences from regenerated deterministic outputs.
VCD-greedy has `invalid=0` and matched sample IDs, but does not improve POPE; it
slightly raises TPR while raising FPR more, so MCC and accuracy drop on every
split.

| Split | Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| random | vanilla | 0.889 | 0.786 | 0.814 | 0.037 | 0.425 | anchor |
| random | VCD-greedy | 0.886 | 0.780 | 0.817 | 0.045 | 0.431 | -0.005 |
| popular | vanilla | 0.869 | 0.742 | 0.814 | 0.077 | 0.445 | anchor |
| popular | VCD-greedy | 0.862 | 0.728 | 0.817 | 0.092 | 0.454 | -0.013 |
| adversarial | vanilla | 0.834 | 0.668 | 0.813 | 0.145 | 0.479 | anchor |
| adversarial | VCD-greedy | 0.827 | 0.654 | 0.815 | 0.162 | 0.489 | -0.014 |
| macro | vanilla | 0.864 | 0.732 | 0.814 | 0.086 | 0.450 | anchor |
| macro | VCD-greedy | 0.858 | 0.720 | 0.816 | 0.100 | 0.458 | -0.011 |

Semantic-neighbor subset audit shows the same failure mode. VCD-greedy increases
FPR on related-present negatives in all three splits:

| Split | Method | Related-present FPR | Plain-absent FPR | Related-minus-plain gap |
|---|---|---:|---:|---:|
| random | vanilla | 0.055 | 0.014 | 0.041 |
| random | VCD-greedy | 0.061 | 0.024 | 0.036 |
| popular | vanilla | 0.099 | 0.035 | 0.064 |
| popular | VCD-greedy | 0.118 | 0.042 | 0.076 |
| adversarial | vanilla | 0.162 | 0.053 | 0.109 |
| adversarial | VCD-greedy | 0.178 | 0.075 | 0.103 |

This is negative evidence for deterministic VCD under the current greedy POPE
protocol. It strengthens the current paper claim that reducing language-prior
reliance is not enough: without target-discriminative visual evidence, a method
can still amplify yes answers on related but absent targets.

### SPIN Adversarial Subset Audit

`spin` is a controlled HuggingFace port of Image-Guided Head Suppression. The
port has passed an 8-row POPE-random smoke test and a larger POPE-adversarial
120-row subset run, both with strict yes/no parsing and `invalid=0`. The subset
is not paper-facing full-data evidence, but it is useful for deciding whether
SPIN should be prioritized as a full baseline.

Result roots:

```text
mitigation/results/pope_spin_smoke/
mitigation/results/pope_spin_adversarial_120/
mitigation/results/semantic_neighbor_audit/spin_adversarial_120_subset_eval/
```

On the adversarial 120-row subset, SPIN behaves like a strong yes-prior shift:

| Method | Accuracy | MCC | TPR | FPR | Yes rate | Delta TPR - Delta FPR |
|---|---:|---:|---:|---:|---:|---:|
| vanilla | 0.833 | 0.667 | 0.850 | 0.183 | 0.517 | anchor |
| SPIN | 0.508 | 0.092 | 1.000 | 0.983 | 0.992 | -0.650 |

The semantic-neighbor subset makes the failure mode explicit:

| Method | Related-present FPR | Plain-absent FPR | Related-minus-plain gap |
|---|---:|---:|---:|
| vanilla | 0.204 | 0.000 | +0.204 |
| SPIN | 1.000 | 0.833 | +0.167 |

This does not prove official SPIN fails under all settings: it is a controlled
HF port, a single split, and only 120 rows. It does show that this port does not
rescue the current mechanism claim; its recall gain comes from answering yes to
nearly every adversarial query.

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
5. Controlled VCD-greedy also fails the target-verification test on full POPE:
   it raises FPR more than TPR and worsens related-present negative FPR.
6. The controlled SPIN port has an early negative adversarial-subset signal: it
   raises TPR to 1.000 by raising FPR to 0.983 and related-present FPR to 1.000.
7. Claims must remain scoped: these are controlled ports/adapted baselines, not
   proof that every attention-based method or every official method fails.

## Missing Evidence Before ICML Submission

- Official-code parity checks on small subsets for PAI, ClearSight, and
  VisAttnSink where feasible.
- Multi-model replication beyond LLaVA-1.5-7B.
- Full regeneration of POPE/CHAIR mitigation results with the now-tracked
  runtime stack, or a documented hash-level equivalence check against the
  existing full run.
- If paper claims compare directly to official VCD, a stochastic decoding parity
  check on a small subset; the controlled greedy VCD full split is complete.
- Head-selection or head-specific mitigation baselines as positive
  counterexamples to unselective attention amplification.
