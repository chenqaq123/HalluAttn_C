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

## Qwen2.5-VL Full Adversarial Split

The 120-row pilot was promoted to the full 3,000-row POPE adversarial split.
The run resumed from the existing 120 JSONL rows and skipped duplicate
question IDs, then completed with exactly 3,000 predictions:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/run_qwen25vl_pope.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --pope_split adversarial \
  --output_file mitigation/results/qwen25vl_pope_adversarial_full/pope/adversarial/vanilla/shard0.jsonl \
  --device cuda:1 \
  --limit 0 \
  --max_new_tokens 16
```

POPE adversarial metrics:

| Model | Split | Samples | Invalid | Acc | MCC | TPR | FPR | Yes rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B-Instruct | adversarial | 3000 | 0 | 0.863 | 0.736 | 0.786 | 0.059 | 0.423 |

Semantic-neighbor subset metrics:

| Subset | Samples | FPR | TNR | Yes rate |
|---|---:|---:|---:|---:|
| all negatives | 1500 | 0.059 | 0.941 | 0.059 |
| related-present negatives | 1272 | 0.067 | 0.933 | 0.067 |
| plain-absent negatives | 228 | 0.018 | 0.982 | 0.018 |

Interpretation: the full split confirms the pilot direction. Qwen2.5-VL is a
stronger base model than the current LLaVA-1.5 adversarial runs, but
related-present negatives are still harder than plain absent negatives
(6.7% vs. 1.8% FPR). This supports the semantic-neighbor failure-mode claim
across model families, with the important caveat that the effect size is model
dependent.

## Fixed-Threshold TDEV on Qwen2.5-VL Full Adversarial

The same LLaVA-selected hybrid operating point was applied to the full Qwen
adversarial run without recalibration:

| Model / rule | Samples | MCC | TPR | FPR | Related FPR | Plain FPR | Yes rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL vanilla | 3000 | 0.736 | 0.786 | 0.059 | 0.067 | 0.018 | 0.423 |
| Qwen2.5-VL + fixed TDEV hybrid | 3000 | 0.745 | 0.782 | 0.048 | 0.054 | 0.013 | 0.415 |

The fixed rule lowers Qwen's overall FPR by 1.1 points and related-present FPR
by 1.3 points while reducing TPR by only 0.4 points. This is stronger evidence
than the 120-row pilot that the TDEV decision rule transfers as a verifier, not
only as a LLaVA-specific calibration artifact. It is still not a complete
multi-model paper table because only the adversarial split has been run for
Qwen, and the OWLv2 score cache is reused from the semantic-neighbor audit.

## Next Action

If compute budget allows, extend Qwen2.5-VL to the random and popular POPE
splits using the same generation path and fixed TDEV thresholds. The current
paper-safe claim is adversarial-only cross-model replication: the
semantic-neighbor gap and the fixed-threshold TDEV FPR reduction persist on the
full adversarial split.
