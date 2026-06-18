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

## Next Action

Run a 120-row adversarial subset for Qwen2.5-VL vanilla, then attach existing
OWLv2/TDEV scores with `base_method=vanilla` under a Qwen-specific result root.
Only promote to full POPE if the subset shows the same semantic-neighbor pattern
and TDEV improves FPR without collapsing recall.
