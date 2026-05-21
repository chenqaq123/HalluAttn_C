#!/usr/bin/env bash
# 4-way parallel runner for the attention-row cache path.
#
# This assumes Stage 1 captions already exist, usually at:
#   experiments/coco_llava_7b/generation.json
#
# Usage:
#   bash scripts/run_row_cache_parallel.sh
#   LIMIT=200 bash scripts/run_row_cache_parallel.sh
#   CACHE_LAYERS=0,1,2,3,4 RATIO=0.3 bash scripts/run_row_cache_parallel.sh
#
# Env knobs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   NUM_SHARDS=4
#   EXP_NAME=coco_llava_7b
#   ROW_EXP_NAME=${EXP_NAME}_rows
#   MODEL_PATH=llava-hf/llava-1.5-7b-hf
#   COCO_PATH=/data/common_dataset/coco-2014-dataset/
#   CHAIR_PKL=../pas/data/chair_coco.pkl
#   CACHE_LAYERS=0,1,2,3,4
#   RATIO=0.5
#   LIMIT=0
#   LOCAL_WINDOW=16
#   LOCAL_MAX_TOKENS=8

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
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
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
COCO_PATH="${COCO_PATH:-/data/common_dataset/coco-2014-dataset/}"
CHAIR_PKL="${CHAIR_PKL:-$PROJ_ROOT/../pas/data/chair_coco.pkl}"
CACHE_LAYERS="${CACHE_LAYERS:-0,1,2,3,4}"
RATIO="${RATIO:-0.5}"
LIMIT="${LIMIT:-0}"
LOCAL_WINDOW="${LOCAL_WINDOW:-16}"
LOCAL_MAX_TOKENS="${LOCAL_MAX_TOKENS:-8}"

if [[ "$CHAIR_PKL" != /* ]]; then
    CHAIR_PKL="$PROJ_ROOT/$CHAIR_PKL"
fi

if ! [[ "$NUM_SHARDS" =~ ^[1-4]$ ]]; then
    echo "[err] NUM_SHARDS=$NUM_SHARDS must be in 1..4." >&2
    exit 1
fi
if [[ ! -f "$CHAIR_PKL" ]]; then
    echo "[err] CHAIR pickle not found: $CHAIR_PKL" >&2
    exit 1
fi

EXP_DIR="$PROJ_ROOT/experiments/$EXP_NAME"
ROW_EXP_DIR="$PROJ_ROOT/experiments/$ROW_EXP_NAME"
GEN_JSON="$EXP_DIR/generation.json"
LOG_DIR="$ROW_EXP_DIR/logs"

if [[ ! -f "$GEN_JSON" ]]; then
    echo "[err] generation file not found: $GEN_JSON" >&2
    echo "[err] Run Stage 1 first with scripts/run_parallel.sh, or set EXP_NAME to an experiment with generation.json." >&2
    exit 1
fi

mkdir -p "$ROW_EXP_DIR" "$LOG_DIR"

echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[row-cache] generation_json=$GEN_JSON"
echo "[row-cache] output_dir=$ROW_EXP_DIR"
echo "[row-cache] cache_layers=$CACHE_LAYERS ratio=$RATIO limit=$LIMIT"

pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
    log="$LOG_DIR/row_cache_shard${i}.log"
    echo "[row-cache] shard $i -> $log"
    (
        python "$SCRIPT_DIR/cache_attention_rows.py" \
            --model_path "$MODEL_PATH" \
            --coco_path "$COCO_PATH" \
            --generation_json "$GEN_JSON" \
            --output_dir "$ROW_EXP_DIR" \
            --chair_pkl "$CHAIR_PKL" \
            --ratio "$RATIO" \
            --cache_layers "$CACHE_LAYERS" \
            --device "$i" \
            --limit "$LIMIT" \
            --shard_idx "$i" \
            --num_shards "$NUM_SHARDS" \
            --local_window "$LOCAL_WINDOW" \
            --local_max_tokens "$LOCAL_MAX_TOKENS"
    ) > "$log" 2>&1 &
    pids+=($!)
done

fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        fail=1
        log="$LOG_DIR/row_cache_shard${idx}.log"
        echo "[row-cache] shard $idx failed; see $log" >&2
        tail -n 40 "$log" >&2 || true
    fi
done

if [[ "$fail" -ne 0 ]]; then
    echo "[row-cache] one or more shards failed; aborting." >&2
    exit 1
fi

echo "[row-cache] merging shards"
python "$SCRIPT_DIR/merge_shards.py" \
    --mode row_cache \
    --output_dir "$ROW_EXP_DIR" \
    --num_shards "$NUM_SHARDS"

echo "[row-cache] recomputing metrics"
python "$SCRIPT_DIR/recompute_from_row_cache.py" \
    --cache "$ROW_EXP_DIR/attention_row_cache.npz" \
    --output_dir "$ROW_EXP_DIR" \
    --ratio "$RATIO"

echo
echo "Done. Outputs:"
echo "  $ROW_EXP_DIR/attention_row_cache.npz"
echo "  $ROW_EXP_DIR/row_cache_scores.npz"
echo "  $ROW_EXP_DIR/row_cache_metrics.json"
echo "  $LOG_DIR/"
