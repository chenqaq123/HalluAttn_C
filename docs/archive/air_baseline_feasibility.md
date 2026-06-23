# AIR Baseline Feasibility Note

Date: 2026-06-19

This note records the June 19 refresh for **AIR: Attention Imbalance
Rectification**. AIR is now the closest newly runnable attention-reallocation
baseline after NoLan. It should be treated as a P0 official-code candidate, but
not as a drop-in method for the current HuggingFace mitigation runner.

## Source Check

- Paper: `https://arxiv.org/abs/2603.24058`.
- Official repository from the arXiv code link: `https://github.com/Ice-wave/AIR`.
- Temporary inspection clone: `/tmp/sinkdetect_air_check`.
- Inspected commit: `cc0e00f1b5d608a011a1c312059be5ccc9d25641`.

The repository README says it is the official CVPR 2026 implementation, supports
LLaVA v1.5 / v1.6, and includes CHAIR, POPE, POPEv2, and MMHal evaluation
scripts. The README reports strong CHAIR reductions for AIR* under both
`max_new_tokens=256` and `max_new_tokens=64`.

## What The Code Provides

Relevant files from the inspection clone:

- `LLaVA/eval_scripts/eval_pope_air.py`: POPE generation entry point.
- `LLaVA/bash_scripts/decoding.sh`: official end-to-end caption/POPE runner.
- `LLaVA/llava/model/language_model/modeling_llama.py`: AIR attention changes.
- `LLaVA/llava/model/air_method.py`: method metadata and module names.
- `LLaVA/requirements.txt` and `LLaVA/pyproject.toml`: forked LLaVA dependency stack.

AIR's official implementation sets config flags and modifies LLaMA attention:
modality rebalancing, cross-head vision lens, conditional AD-HH, and variance
projection. It is not just a logits processor.

## Compatibility Assessment

AIR is runnable in principle, but should be isolated from the current environment.
The README requires commands to run inside the repo's `LLaVA/` tree and notes a
Transformers workaround: uncommenting `_validate_model_kwargs(...)` in the
installed Transformers generation utility. The LLaVA stack pins
`transformers==4.37.2`; the `newer_models/` path pins `transformers==4.45.2`.

This means the safest route is an isolated env or container. Do not patch the
current `latentGuard` environment in place unless the change is reversible and
recorded.

There is also a checkpoint-format risk. The AIR LLaVA loader uses the original
LLaVA model classes, while the locally used main checkpoint is a HuggingFace
`llava-hf` snapshot with `LlavaForConditionalGeneration`. No local original-LLaVA
`mm_projector.bin` checkpoint was found under `/home/chenguanxu/common_model` in
the current audit. The official AIR run may therefore need an original-LLaVA
checkpoint such as `liuhaotian/llava-v1.5-7b`, or an explicit compatibility test
inside the isolated AIR environment.

A dry-run in `latentGuard` confirms the mismatch:

- Current env: Python 3.10.20, torch 2.10.0+cu128, transformers 4.57.6,
  tokenizers 0.22.2.
- AIR LLaVA requirements: torch 2.1.2, transformers 4.37.2, tokenizers 0.15.1.
- `PYTHONPATH=/tmp/sinkdetect_air_check/LLaVA python -m eval_scripts.eval_pope_air --help`
  fails before model loading with `ModuleNotFoundError: No module named 'shortuuid'`.

This is not just a missing package; installing AIR requirements into
`latentGuard` would downgrade core libraries used by the rest of the project.

## POPE Semantic-Neighbor Audit Plan

AIR's `eval_pope_air.py` expects LLaVA-style question JSONL rows with:

- `question_id`
- `image`
- `text`

The current local POPE dataset under `/home/chenguanxu/common_dataset/pope` is a
HuggingFace/parquet layout, not the exact `dataset/pope/llava_pope_test.jsonl`
layout expected by AIR's shell script. The project already has robust POPE
record readers in `mitigation/scripts/build_semantic_neighbor_audit.py`; the
next step is to export a LLaVA-style adversarial JSONL subset from the existing
POPE rows, run AIR official generation against the COCO val2014 image folder,
and evaluate the output with the existing semantic-neighbor audit script.

Current export status:

- Export script: `mitigation/scripts/export_air_pope_subset.py`. Regenerate with:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/export_air_pope_subset.py \
  --pope_dir /home/chenguanxu/common_dataset/pope \
  --coco_path /home/chenguanxu/common_dataset/coco-2014-dataset \
  --split adversarial \
  --limit 120 \
  --output_file mitigation/results/air_official_inputs/pope/adversarial/air_adversarial_120_questions.jsonl
```

- Exported input: `mitigation/results/air_official_inputs/pope/adversarial/air_adversarial_120_questions.jsonl`.
- Manifest: `mitigation/results/air_official_inputs/pope/adversarial/air_adversarial_120_questions.jsonl.manifest.json`.
- The exported rows are the same `question_id` sequence as the NoLan adversarial
  120-row run: `1..120`.
- Counts: 120 rows, 60 yes / 60 no; negative rows contain 54 related-present and
  6 plain-absent examples. All image names are normalized to
  `COCO_val2014_*.jpg` and passed existence checks against COCO val2014.

Recommended first run:

1. Use the exported 120-row adversarial input above inside the isolated AIR/LLaVA
   environment.
2. Run official AIR with `max_new_tokens=16` or `32`, greedy decoding, batch size
   1 or 2, and LLaVA-1.5-7B.
3. Prefer the project wrapper once `AIR_LLAVA_ROOT` and `AIR_MODEL_PATH` point to
   a working official AIR/LLaVA environment:

```bash
AIR_LLAVA_ROOT=/path/to/AIR/LLaVA \
AIR_MODEL_PATH=/path/to/original-llava-v1.5-7b \
CUDA_VISIBLE_DEVICES=0 \
mitigation/scripts/run_air_official_adversarial_120.sh
```

4. Or convert AIR `answers.jsonl` to the existing prediction schema manually:

```bash
/home/chenguanxu/miniconda3/envs/latentGuard/bin/python \
  mitigation/scripts/convert_air_answers.py \
  --answers_file <AIR_LLaVA>/results/.../pope/answers.jsonl \
  --questions_file mitigation/results/air_official_inputs/pope/adversarial/air_adversarial_120_questions.jsonl \
  --output_file mitigation/results/air_official_adversarial_120/pope/adversarial/air/predictions.jsonl \
  --method air \
  --strict
```

5. Evaluate with `mitigation/scripts/evaluate_semantic_neighbor_subsets.py` using
   the same audit CSV and strict invalid handling.
6. Promote to full adversarial or all-split only if AIR lowers related-present
   FPR without merely collapsing TPR/yes rate.

The conversion path has been structure-tested with a 120-row dummy AIR answers
file: `convert_air_answers.py` writes the expected prediction schema, and
`evaluate_semantic_neighbor_subsets.py --invalid_policy error` consumes it with
`invalid=0` and the expected subset counts. In `--strict` mode the converter now
fails on unknown AIR answer ids, duplicate answer ids, or missing answers for the
exported questions, so an incomplete official run cannot silently enter the
baseline table.

## Paper Positioning

AIR is a stronger positive counterexample than the previous speculative
attention-reallocation entries because official code is accessible. It increases
pressure on the paper's attention-intervention discussion, but it does not change
the core claim unless it passes the semantic-neighbor audit. The required
comparison is still target verification:

- If AIR lowers related-present FPR while preserving TPR, report it as a strong
  attention-reallocation baseline and sharpen TDEV as target-vs-neighbor rather
  than attention balancing.
- If AIR improves aggregate POPE/CHAIR but leaves related-present FPR high, it
  directly supports the `looking is not verifying` claim.
- If AIR behaves like NoLan by lowering FPR with a lower yes rate, report it as a
  conservative intervention rather than target-discriminative verification.

Do not implement an unofficial AIR surrogate. Use the official fork or report it
as related work until the isolated official-code run succeeds.
