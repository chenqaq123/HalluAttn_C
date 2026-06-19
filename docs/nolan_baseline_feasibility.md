# NoLan Baseline Feasibility Note

Date: 2026-06-19

## Why This Matters

NoLan is now the closest public-code decoding baseline for the current ICML plan.
It tests a different hypothesis from TDEV: object hallucination can be reduced by
suppressing language priors when multimodal and text-only next-token
distributions are too similar. This is relevant to our motivation because it may
reduce generic hallucination, but it still does not explicitly verify the queried
target against semantic neighbors.

## Source Status

- Paper: `https://arxiv.org/abs/2602.22144`
- Official repository: `https://github.com/lingfengren/NoLan`
- Repository README says code is released and supports LLaVA-1.5, InstructBLIP,
  and Qwen-VL.
- The repository README describes integration by importing
  `nolan_utils.nolan_sample.evolve_nolan_sampling()` and adding
  `prepare_inputs_for_generation_cd` plus `input_ids_cd`, `cd_alpha`, and
  `cd_beta` plumbing to the model.

## Compatibility Audit

Current local environment:

| Component | Local env | NoLan requirement |
|---|---|---|
| Python env | `/home/chenguanxu/miniconda3/envs/latentGuard` | separate `nolan` env suggested |
| transformers | `4.57.6` | `4.31.0` |
| torch | `2.10.0+cu128` | `2.0.1` |

Important integration details from the official code:

1. NoLan monkey-patches `transformers.generation.utils.GenerationMixin.sample`.
2. The current repository mostly uses deterministic `model.generate(...,
   do_sample=False)` for POPE/CHAIR comparability, while NoLan's README example
   uses `do_sample=True`.
3. The NoLan sampling code expects a contrastive/text-only input path through
   `prepare_inputs_for_generation_cd`. HuggingFace `LlavaForConditionalGeneration`
   in our stack does not expose that method by default.
4. Directly downgrading transformers/torch inside `latentGuard` would risk
   breaking already-audited LLaVA, Qwen2.5-VL, OWLv2, and CHAIR scripts.

## Decision

Treat NoLan as a **P0 runnable candidate**, not as an already integrated
baseline. Do not patch the main environment globally.

The safer path is a guarded local compatibility port for the adversarial
semantic-neighbor subset:

1. Keep the existing `latentGuard` environment unchanged.
2. Implement or isolate a NoLan runner that computes multimodal and text-only
   logits per generation step for the same LLaVA-HF model path.
3. Preserve our normal strict yes/no parsing and semantic-neighbor audit metrics:
   MCC, TPR, FPR, yes rate, related-present FPR, plain FPR, and gap.
4. Report the run as `NoLan-compatible port` unless it uses the official monkey
   patch and model modifications exactly.
5. If the port diverges materially from official code, keep it as a diagnostic
   ablation and do not claim official NoLan reproduction.

## Local Smoke Status

A guarded NoLan-compatible greedy port is now wired into the existing mitigation
runner as `--method nolan`. It keeps the main environment unchanged and computes
multimodal and text-only logits side by side for the same LLaVA-HF checkpoint.

Smoke command:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python mitigation/scripts/run_task.py \
  --task pope \
  --method nolan \
  --model_path /home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --pope_split adversarial \
  --output_file mitigation/results/coco_llava_7b_nolan_smoke/pope/adversarial/nolan/predictions.jsonl \
  --device 5 \
  --limit 2 \
  --max_new_tokens 16 \
  --nolan_alpha_scale 0.8
```

Smoke result:

- Generation succeeded on GPU 5.
- Both outputs were strict yes/no parseable.
- `evaluate_semantic_neighbor_subsets.py` accepted the output format.
- The 2-row metric file is
  `mitigation/results/semantic_neighbor_audit/nolan_smoke_subset_eval/semantic_neighbor_subset_metrics.csv`.

This 2-row run is an execution smoke only, not a performance result. The same
runner has now been evaluated on the first 120 POPE-adversarial rows used for the
SPIN/DAMRO subset checks.

120-row subset result:

| Method | MCC | TPR | FPR | Yes Rate | Related FPR |
|---|---:|---:|---:|---:|---:|
| vanilla | 0.667 | 0.850 | 0.183 | 0.517 | 0.204 |
| NoLan-compatible | 0.700 | 0.833 | 0.133 | 0.483 | 0.148 |

The compatible port is a positive baseline on this subset: it reduces related-present
FPR by 5.6 points while losing 1.7 points of TPR. This is stronger than the
existing SPIN default and DAMRO subset checks, but it is still not a full official
NoLan reproduction or a full all-split result. The comparison summary is stored at
`mitigation/results/semantic_neighbor_audit/nolan_adversarial_120_subset_eval/nolan_adversarial_120_comparison.md`.

## First Success Gate

Run only the POPE-adversarial semantic-neighbor subset first. The result is worth
scaling only if it reduces related-present FPR without a yes-rate shortcut and
without collapsing TPR. This matches the paper's core claim that better decoding
or stronger visual influence is not enough unless the target itself is verified.
