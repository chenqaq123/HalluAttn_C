#!/usr/bin/env bash
# Multi-GPU parallel runner for unified baseline reproduction.
#
# This assumes the current COCO/LLaVA generation and row-cache files exist:
#   experiments/coco_llava_7b/generation.json
#   experiments/coco_llava_7b_rows/attention_row_cache.npz
#   experiments/coco_llava_7b_rows/row_cache_scores.npz
#
# Usage:
#   bash detection/baselines/run_parallel_baselines.sh
#   LIMIT=200 bash detection/baselines/run_parallel_baselines.sh
#   SKIP_MODEL_BASELINES=1 bash detection/baselines/run_parallel_baselines.sh
#
# Env knobs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NUM_SHARDS=4
#   EXP_NAME=coco_llava_7b
#   ROW_EXP_NAME=${EXP_NAME}_rows
#   BASELINE_EXP_NAME=${EXP_NAME}_baselines
#   SHARD_BY=image
#   MODEL_PATH=llava-hf/llava-1.5-7b-hf
#   COCO_PATH=/data/common_dataset/coco-2014-dataset/
#   CHAIR_PKL=../pas/data/chair_coco.pkl
#   LIMIT=0
#   BIN_WIDTH=10
#   MATCHED_DELTA=5
#   GLSIM_TOP_K=32
#   GLSIM_W=0.6
#   TEXT_LAYER=31
#   IMAGE_LAYER=32
#   BEYOND_LAYER=1

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DETECTION_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJ_ROOT="$( cd "$DETECTION_ROOT/.." && pwd )"
cd "$PROJ_ROOT"

if [[ -f "$PROJ_ROOT/.env" ]]; then
    set -a
    source "$PROJ_ROOT/.env"
    set +a
    echo "[env] sourced $PROJ_ROOT/.env"
else
    echo "[env] WARNING: $PROJ_ROOT/.env not found; using built-in defaults." >&2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
EXP_NAME="${EXP_NAME:-coco_llava_7b}"
ROW_EXP_NAME="${ROW_EXP_NAME:-${EXP_NAME}_rows}"
BASELINE_EXP_NAME="${BASELINE_EXP_NAME:-${EXP_NAME}_baselines}"
SHARD_BY="${SHARD_BY:-image}"
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
COCO_PATH="${COCO_PATH:-/data/common_dataset/coco-2014-dataset/}"
CHAIR_PKL="${CHAIR_PKL:-$PROJ_ROOT/../pas/data/chair_coco.pkl}"
LIMIT="${LIMIT:-0}"
BIN_WIDTH="${BIN_WIDTH:-10}"
MATCHED_DELTA="${MATCHED_DELTA:-5}"
GLSIM_TOP_K="${GLSIM_TOP_K:-32}"
GLSIM_W="${GLSIM_W:-0.6}"
TEXT_LAYER="${TEXT_LAYER:-31}"
IMAGE_LAYER="${IMAGE_LAYER:-32}"
BEYOND_LAYER="${BEYOND_LAYER:-1}"
SKIP_MODEL_BASELINES="${SKIP_MODEL_BASELINES:-0}"

if [[ "$CHAIR_PKL" != /* ]]; then
    CHAIR_PKL="$PROJ_ROOT/$CHAIR_PKL"
fi

if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[err] NUM_SHARDS=$NUM_SHARDS must be a positive integer." >&2
    exit 1
fi

IFS=',' read -ra VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_VISIBLE_GPUS="${#VISIBLE_GPU_LIST[@]}"
if (( NUM_SHARDS > NUM_VISIBLE_GPUS )); then
    echo "[err] NUM_SHARDS=$NUM_SHARDS exceeds visible GPU count=$NUM_VISIBLE_GPUS from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES." >&2
    echo "[err] Use NUM_SHARDS <= $NUM_VISIBLE_GPUS, or expose more GPUs." >&2
    exit 1
fi

EXP_DIR="$PROJ_ROOT/experiments/$EXP_NAME"
ROW_EXP_DIR="$PROJ_ROOT/experiments/$ROW_EXP_NAME"
GEN_JSON="$EXP_DIR/generation.json"
ROW_CACHE="$ROW_EXP_DIR/attention_row_cache.npz"
ROW_SCORES="$ROW_EXP_DIR/row_cache_scores.npz"
OUT_DIR="$PROJ_ROOT/detection/baselines/results/$BASELINE_EXP_NAME"
LOG_DIR="$OUT_DIR/logs"

if [[ ! -f "$GEN_JSON" ]]; then
    echo "[err] generation file not found: $GEN_JSON" >&2
    exit 1
fi
if [[ ! -f "$ROW_CACHE" ]]; then
    echo "[err] row cache not found: $ROW_CACHE" >&2
    echo "[err] Build it first with detection/scripts/run_row_cache_parallel.sh, or set ROW_EXP_NAME." >&2
    exit 1
fi
if [[ ! -f "$ROW_SCORES" ]]; then
    echo "[err] row-cache scores not found: $ROW_SCORES" >&2
    echo "[err] Recompute them with detection/scripts/recompute_from_row_cache.py, or set ROW_EXP_NAME." >&2
    exit 1
fi
if [[ ! -f "$CHAIR_PKL" ]]; then
    echo "[err] CHAIR pickle not found: $CHAIR_PKL" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[baseline] generation_json=$GEN_JSON"
echo "[baseline] row_cache=$ROW_CACHE"
echo "[baseline] row_scores=$ROW_SCORES"
echo "[baseline] output_dir=$OUT_DIR"
echo "[baseline] num_shards=$NUM_SHARDS shard_by=$SHARD_BY limit=$LIMIT skip_model=$SKIP_MODEL_BASELINES"

extra_args=()
if [[ "$SKIP_MODEL_BASELINES" == "1" ]]; then
    extra_args+=(--skip_model_baselines)
fi

pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
    shard_dir="$OUT_DIR/shard${i}"
    log="$LOG_DIR/baseline_shard${i}.log"
    mkdir -p "$shard_dir"
    echo "[baseline] shard $i -> $log"
    (
        python "$SCRIPT_DIR/run_all_baselines.py" \
            --generation_json "$GEN_JSON" \
            --row_cache "$ROW_CACHE" \
            --row_scores "$ROW_SCORES" \
            --output_dir "$shard_dir" \
            --model_path "$MODEL_PATH" \
            --coco_path "$COCO_PATH" \
            --chair_pkl "$CHAIR_PKL" \
            --device "$i" \
            --limit "$LIMIT" \
            --shard_idx "$i" \
            --num_shards "$NUM_SHARDS" \
            --shard_by "$SHARD_BY" \
            --bin_width "$BIN_WIDTH" \
            --matched_delta "$MATCHED_DELTA" \
            --glsim_top_k "$GLSIM_TOP_K" \
            --glsim_w "$GLSIM_W" \
            --text_layer "$TEXT_LAYER" \
            --image_layer "$IMAGE_LAYER" \
            --beyond_layer "$BEYOND_LAYER" \
            "${extra_args[@]}"
    ) > "$log" 2>&1 &
    pids+=($!)
done

fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        fail=1
        log="$LOG_DIR/baseline_shard${idx}.log"
        echo "[baseline] shard $idx failed; see $log" >&2
        tail -n 40 "$log" >&2 || true
    fi
done

if [[ "$fail" -ne 0 ]]; then
    echo "[baseline] one or more shards failed; aborting before merge." >&2
    exit 1
fi

echo "[baseline] merging shards"
python "$SCRIPT_DIR/merge_baseline_shards.py" \
    --output_dir "$OUT_DIR" \
    --num_shards "$NUM_SHARDS" \
    --bin_width "$BIN_WIDTH" \
    --matched_delta "$MATCHED_DELTA"

echo
echo "Done. Outputs:"
echo "  $OUT_DIR/object_cache.jsonl"
echo "  $OUT_DIR/baseline_scores.npz"
echo "  $OUT_DIR/baseline_scores.csv"
echo "  $OUT_DIR/metrics.json"
echo "  $OUT_DIR/run_config.json"
echo "  $LOG_DIR/"
