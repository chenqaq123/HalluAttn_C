# Multi-Model Replication Audit

This note tracks the practical path for testing whether the current TDEV claims
survive beyond LLaVA-1.5-7B.

## Local Model Inventory

Relevant local checkpoints under `/home/chenguanxu/common_model`:

| Candidate | Local status | Current code compatibility | Notes |
|---|---|---|---|
| LLaVA-1.5-7B HF | complete | native | Main reported model; existing attention/TDEV pipeline supports it. |
| Qwen2.5-VL-7B-Instruct | complete snapshot | vanilla POPE script added | Good second-family candidate for model-output plus OWLv2/TDEV gating. Internal attention metrics need a Qwen-specific adapter. |
| InternVL3-1B-hf | complete snapshot | not yet wired | Smaller candidate, but requires InternVL processor/model path and prompt adapter. |
| LLaVA-NeXT / LLaVA-1.6 variants | present | not yet wired | Needs `LlavaNext*` loading and prompt/image span changes before attention metrics are trustworthy. |

The current SinkDetect attention adapter is LLaMA/LLaVA-specific. It should not
be applied to Qwen2.5-VL or InternVL without a separate attention implementation
for those model classes. For a first multi-model check, use vanilla model output
plus model-independent OWLv2/TDEV verification rather than internal attention
claims.

## Qwen2.5-VL Smoke

Added script:

```bash
python mitigation/scripts/run_qwen25vl_pope.py --help
```

Smoke command run on GPU 1:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/run_qwen25vl_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --pope_split adversarial \
  --output_file mitigation/results/qwen25vl_pope_adversarial_smoke/pope/adversarial/vanilla/shard0.jsonl \
  --device cuda:1 \
  --limit 8 \
  --max_new_tokens 16
```

Evaluation command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/merge_evaluate.py \
  --task pope \
  --method_dir mitigation/results/qwen25vl_pope_adversarial_smoke/pope/adversarial/vanilla \
  --num_shards 1 \
  --invalid_policy error
```

Smoke result only, not paper evidence:

| Model | Split | Samples | Invalid | Acc | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | adversarial | 8 | 0 | 0.750 | 0.500 | 0.750 | 0.250 | 0.500 |

This validates the generation/evaluation interface and confirms the model can
run with reduced image resolution (`min_pixels=max_pixels=256*28*28`) on an
RTX 4090 with about 22 GB free. It does not establish multi-model robustness.

## Qwen2.5-VL Adversarial 120

Command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/run_qwen25vl_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --pope_split adversarial \
  --output_file mitigation/results/qwen25vl_pope_adversarial_120/pope/adversarial/vanilla/shard0.jsonl \
  --device cuda:1 \
  --limit 120 \
  --max_new_tokens 16
```

POPE metrics:

| Model | Split | Samples | Invalid | Acc | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | adversarial | 120 | 0 | 0.867 | 0.737 | 0.817 | 0.083 | 0.450 |

Semantic-neighbor subset metrics:

| Subset | Samples | FPR | TNR | Yes rate |
|---|---:|---:|---:|---:|
| all negatives | 60 | 0.083 | 0.917 | 0.083 |
| related-present negatives | 54 | 0.093 | 0.907 | 0.093 |
| plain-absent negatives | 6 | 0.000 | 1.000 | 0.000 |

Interpretation: the semantic-neighbor gap is still present on Qwen2.5-VL, but
it is much smaller than the LLaVA-1.5 adversarial 120 subset used for SPIN/DAMRO
triage. This supports a cross-family failure-mode claim only in scoped form:
related-object negatives remain harder than plain-absent negatives, but model
strength changes the base false-positive rate substantially.

## Fixed-Threshold TDEV on Qwen2.5-VL 120

The LLaVA-selected hybrid operating point was applied without recalibration:

- `low=0.04`
- `high=0.12`
- `margin=-0.20`
- `rescue_high=0.50`
- `rescue_margin=-0.10`

Command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/evaluate_fixed_hybrid_region_rule.py \
  --predictions_csv mitigation/results/semantic_neighbor_audit/owlv2_tdev_zero/owlv2_tdev_predictions.csv \
  --result_root mitigation/results/qwen25vl_pope_adversarial_120 \
  --output_dir mitigation/results/semantic_neighbor_audit/qwen25vl_fixed_hybrid_region_rule_adversarial_120 \
  --splits adversarial \
  --base_method vanilla \
  --low_threshold 0.04 \
  --high_threshold 0.12 \
  --margin_threshold -0.20 \
  --rescue_high_threshold 0.50 \
  --rescue_margin_threshold -0.10
```

| Model / rule | Samples | MCC | TPR | FPR | Related FPR | Plain FPR | Yes rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL vanilla | 120 | 0.737 | 0.817 | 0.083 | 0.093 | 0.000 | 0.450 |
| Qwen2.5-VL + fixed TDEV hybrid | 120 | 0.755 | 0.817 | 0.067 | 0.074 | 0.000 | 0.442 |

This is a useful cross-model subset signal: the same TDEV operating point lowers
false positives without reducing recall on Qwen2.5-VL. It is not yet a final
multi-model paper table because it covers only the first 120 adversarial rows
and reuses OWLv2 scores from the LLaVA audit cache.

## Qwen2.5-VL Full POPE Splits

The 120-row pilot was promoted to the full POPE random, popular, and
adversarial splits. Each split completed with exactly 3,000 predictions and
`invalid=0`. The adversarial run resumed from the existing 120 JSONL rows and
skipped duplicate question IDs; random and popular were run from scratch.

Representative full-run command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/run_qwen25vl_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --pope_split {random,popular,adversarial} \
  --output_file mitigation/results/qwen25vl_pope_<split>_full/pope/<split>/vanilla/shard0.jsonl \
  --device cuda:1 \
  --limit 0 \
  --max_new_tokens 16
```

POPE metrics:

| Model | Split | Samples | Invalid | Acc | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | random | 3000 | 0 | 0.887 | 0.791 | 0.785 | 0.011 | 0.398 |
| Qwen2.5-VL-7B-Instruct | popular | 3000 | 0 | 0.878 | 0.769 | 0.785 | 0.030 | 0.408 |
| Qwen2.5-VL-7B-Instruct | adversarial | 3000 | 0 | 0.863 | 0.736 | 0.786 | 0.059 | 0.423 |
| macro | all | 9000 | 0 | 0.876 | 0.765 | 0.786 | 0.033 | 0.409 |

Semantic-neighbor subset metrics:

| Split | All-negative FPR | Related-present FPR | Plain-absent FPR |
|---|---:|---:|---:|
| random | 0.011 | 0.014 | 0.006 |
| popular | 0.030 | 0.042 | 0.008 |
| adversarial | 0.059 | 0.067 | 0.018 |
| macro | 0.033 | 0.041 | 0.010 |

Interpretation: the full POPE run confirms the pilot direction. Qwen2.5-VL is a
stronger base model than the current LLaVA-1.5 runs, especially on random and
popular negatives, but related-present negatives remain harder than plain-absent
negatives across all splits. The macro related-present FPR is 4.1% versus 1.0%
for plain-absent negatives. This supports the semantic-neighbor failure-mode
claim across model families, with the important caveat that the effect size is
model dependent.

## Fixed-Threshold TDEV on Qwen2.5-VL Full POPE

The same LLaVA-selected hybrid operating point was applied to all Qwen splits
without recalibration:

| Rule | Split | MCC | TPR | FPR | Related FPR | Plain FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|---:|
| vanilla | random | 0.791 | 0.785 | 0.011 | 0.014 | 0.006 | 0.398 |
| fixed TDEV hybrid | random | 0.789 | 0.781 | 0.010 | 0.014 | 0.005 | 0.396 |
| vanilla | popular | 0.769 | 0.785 | 0.030 | 0.042 | 0.008 | 0.408 |
| fixed TDEV hybrid | popular | 0.773 | 0.781 | 0.023 | 0.033 | 0.006 | 0.402 |
| vanilla | adversarial | 0.736 | 0.786 | 0.059 | 0.067 | 0.018 | 0.423 |
| fixed TDEV hybrid | adversarial | 0.745 | 0.782 | 0.048 | 0.054 | 0.013 | 0.415 |
| vanilla macro | all | 0.765 | 0.786 | 0.033 | 0.041 | 0.010 | 0.409 |
| fixed TDEV macro | all | 0.769 | 0.782 | 0.027 | 0.034 | 0.008 | 0.404 |

The fixed rule lowers Qwen's macro FPR by 0.6 points and macro related-present
FPR by 0.7 points while reducing macro TPR by 0.4 points. The effect is small on
random, clearer on popular and adversarial, and directionally consistent with
the LLaVA-selected operating point. This is now all-splits cross-model evidence
that TDEV transfers as a verifier, not only as a LLaVA-specific calibration
artifact. The remaining caveat is that the OWLv2 score cache is reused from the
semantic-neighbor audit, so this should be described as model-independent
post-hoc verification rather than Qwen internal-attention evidence.

## Next Action

The Qwen2.5-VL all-splits replication is complete for vanilla output and fixed
TDEV verification. The next multi-model step is either a third model
(InternVL/LLaVA-NeXT) or a method-side improvement that increases recall while
preserving the related-present FPR reduction. Any paper table should state
clearly that Qwen evidence is output-level plus external region verification,
not Qwen internal attention analysis.
