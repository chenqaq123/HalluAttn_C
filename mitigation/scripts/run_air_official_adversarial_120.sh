#!/usr/bin/env bash
# Run official AIR on the prepared POPE-adversarial 120-row subset, then convert
# and evaluate with this repository's semantic-neighbor audit.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
AIR_LLAVA_ROOT="${AIR_LLAVA_ROOT:?Set AIR_LLAVA_ROOT to the official AIR/LLaVA directory, e.g. /path/to/AIR/LLaVA}"
AIR_MODEL_PATH="${AIR_MODEL_PATH:?Set AIR_MODEL_PATH to an AIR-compatible LLaVA model, e.g. liuhaotian/llava-v1.5-7b or a local original-LLaVA checkpoint}"
PROJECT_PYTHON="${PROJECT_PYTHON:-/home/chenguanxu/miniconda3/envs/latentGuard/bin/python}"
COCO_PATH="${COCO_PATH:-/home/chenguanxu/common_dataset/coco-2014-dataset}"
QUESTION_FILE="${QUESTION_FILE:-$PROJECT_ROOT/mitigation/results/air_official_inputs/pope/adversarial/air_adversarial_120_questions.jsonl}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/mitigation/results/air_official_adversarial_120}"
AIR_ANSWERS_FILE="${AIR_ANSWERS_FILE:-$RESULT_ROOT/raw/answers.jsonl}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-$RESULT_ROOT/pope/adversarial/air/predictions.jsonl}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_ROOT/mitigation/results/semantic_neighbor_audit/air_adversarial_120_subset_eval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$(dirname "$AIR_ANSWERS_FILE")" "$(dirname "$PREDICTIONS_FILE")" "$METRICS_DIR"

(
  cd "$AIR_LLAVA_ROOT"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m eval_scripts.eval_pope_air \
    --model-path "$AIR_MODEL_PATH" \
    --image-folder "$COCO_PATH/val2014" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$AIR_ANSWERS_FILE" \
    --conv-mode vicuna_v1 \
    --temperature 0 \
    --batch-size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --air
)

"$PROJECT_PYTHON" "$PROJECT_ROOT/mitigation/scripts/convert_air_answers.py" \
  --answers_file "$AIR_ANSWERS_FILE" \
  --questions_file "$QUESTION_FILE" \
  --output_file "$PREDICTIONS_FILE" \
  --method air \
  --strict

"$PROJECT_PYTHON" "$PROJECT_ROOT/mitigation/scripts/evaluate_semantic_neighbor_subsets.py" \
  --result_root "$RESULT_ROOT" \
  --audit_csv "$PROJECT_ROOT/mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv" \
  --output_dir "$METRICS_DIR" \
  --splits adversarial \
  --methods air \
  --invalid_policy error
