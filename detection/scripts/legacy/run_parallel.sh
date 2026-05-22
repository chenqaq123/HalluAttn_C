#!/usr/bin/env bash
# 4-way parallel runner for SinkDetect on GPUs 0,1,2,3.
#
# Sharding is purely data-parallel: each GPU loads its own copy of LLaVA-1.5-7B
# and processes a deterministic stride of the input list. After all shards
# finish, detection/scripts/merge_shards.py concatenates results and recomputes AUROC on
# the pooled set.
#
# Usage:
#   bash detection/scripts/legacy/run_parallel.sh                  # full run, all 4 GPUs
#   NUM_SAMPLES=200 bash detection/scripts/legacy/run_parallel.sh  # smoke test
#   NUM_SHARDS=2 bash detection/scripts/legacy/run_parallel.sh     # only use 2 of the 4 GPUs
#
# Env knobs (defaults shown):
#   CUDA_VISIBLE_DEVICES=0,1,2,3   restrict to these physical GPUs
#   NUM_SHARDS=4                   data-parallel degree
#   EXP_NAME=coco_llava_7b
#   MODEL_PATH=...                 LLaVA-1.5-7B HF dir
#   COCO_PATH=...
#   CHAIR_PKL=../pas/data/chair_coco.pkl
#   NUM_SAMPLES=5000               images for stage 1 (global, before sharding)
#   MAX_NEW_TOKENS=512
#   RATIO=0.5
#   START_LAYER=0  END_LAYER=32
#   SEED=42
#   SAVE_SHAPE_CACHE=1             save shape_cache.npz for fast metric recompute
#   COMPUTE_NO_ROPE_ATTENTION=1    cache/score no-RoPE attention branches
#   FORCE_REGEN=1                  rerun stage 1 even if generation.json exists

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DETECTION_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
ACTIVE_SCRIPT_DIR="$DETECTION_ROOT/scripts"
PROJ_ROOT="$( cd "$DETECTION_ROOT/.." && pwd )"
cd "$PROJ_ROOT"

# ── Source .env if present (exports all KEY=VALUE lines) ────────────────────
if [[ -f "$PROJ_ROOT/.env" ]]; then
    set -a; source "$PROJ_ROOT/.env"; set +a
    echo "[env] sourced $PROJ_ROOT/.env"
else
    echo "[env] WARNING: $PROJ_ROOT/.env not found — using built-in defaults." >&2
fi

# ── Restrict visible GPUs to 0-3 ────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ── Defaults ────────────────────────────────────────────────────────────────
NUM_SHARDS="${NUM_SHARDS:-4}"
EXP_NAME="${EXP_NAME:-coco_llava_7b}"
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
COCO_PATH="${COCO_PATH:-/data/common_dataset/coco-2014-dataset/}"
CHAIR_PKL="${CHAIR_PKL:-$PROJ_ROOT/../pas/data/chair_coco.pkl}"
# CHAIR_PKL may be relative in .env — resolve against project root
if [[ "$CHAIR_PKL" != /* ]]; then
    CHAIR_PKL="$PROJ_ROOT/$CHAIR_PKL"
fi
echo "[env] MODEL_PATH=$MODEL_PATH"
echo "[env] HF_HOME=${HF_HOME:-<unset>}"

NUM_SAMPLES="${NUM_SAMPLES:-5000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RATIO="${RATIO:-0.5}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-32}"
SEED="${SEED:-42}"
SAVE_SHAPE_CACHE="${SAVE_SHAPE_CACHE:-0}"
COMPUTE_NO_ROPE_ATTENTION="${COMPUTE_NO_ROPE_ATTENTION:-0}"

# ── Validation ──────────────────────────────────────────────────────────────
# NUM_SHARDS must be in 1..4 (we only have 4 GPUs visible).
if ! [[ "$NUM_SHARDS" =~ ^[1-4]$ ]]; then
    echo "[err] NUM_SHARDS=$NUM_SHARDS must be in 1..4." >&2
    exit 1
fi
if [[ ! -f "$CHAIR_PKL" ]]; then
    echo "[err] CHAIR pickle not found: $CHAIR_PKL" >&2
    exit 1
fi

EXP_DIR="$PROJ_ROOT/experiments/$EXP_NAME"
mkdir -p "$EXP_DIR"

LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Helper: launch N workers, one per logical GPU, in parallel ──────────────
# Each worker writes its full stdout/stderr to a per-shard log file.
launch_shards() {
    local stage_name="$1"; shift
    local pids=()
    local logs=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        local log="$LOG_DIR/${stage_name}_shard${i}.log"
        logs+=("$log")
        echo "[${stage_name}] shard $i → log: $log"
        (
            CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
            "$@" --shard_idx "$i" --num_shards "$NUM_SHARDS" --device "$i"
        ) > "$log" 2>&1 &
        pids+=($!)
    done
    local fail=0
    for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
            fail=1
            echo "[${stage_name}] shard $idx (pid ${pids[$idx]}) FAILED — see ${logs[$idx]}" >&2
            tail -n 30 "${logs[$idx]}" >&2 || true
        fi
    done
    if [[ "$fail" -ne 0 ]]; then
        echo "[${stage_name}] one or more shards failed — aborting." >&2
        exit 1
    fi
}

# ── Stage 1: parallel captioning ────────────────────────────────────────────
GEN_JSON="$EXP_DIR/generation.json"
if [[ -f "$GEN_JSON" && -z "${FORCE_REGEN:-}" ]]; then
    echo "[stage1] $GEN_JSON exists — skipping (FORCE_REGEN=1 to redo)."
else
    echo "[stage1] Launching $NUM_SHARDS caption workers..."

    # Worker output paths are per-shard; merge_shards.py picks them up.
    # Build a tiny launcher that includes the per-shard --output_path.
    pids=()
    for ((i=0; i<NUM_SHARDS; i++)); do
        log="$LOG_DIR/stage1_shard${i}.log"
        echo "[stage1] shard $i → $log"
        (
            python "$ACTIVE_SCRIPT_DIR/caption.py" \
                --model_path     "$MODEL_PATH" \
                --coco_path      "$COCO_PATH" \
                --output_path    "$EXP_DIR/generation_shard${i}.json" \
                --num_samples    "$NUM_SAMPLES" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --seed           "$SEED" \
                --device         "$i" \
                --shard_idx      "$i" \
                --num_shards     "$NUM_SHARDS"
        ) > "$log" 2>&1 &
        pids+=($!)
    done

    fail=0
    for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
            fail=1
            echo "[stage1] shard $idx (pid ${pids[$idx]}) FAILED — log: $LOG_DIR/stage1_shard${idx}.log" >&2
            tail -n 30 "$LOG_DIR/stage1_shard${idx}.log" >&2 || true
        fi
    done
    [[ "$fail" -eq 0 ]] || { echo "[stage1] aborting." >&2; exit 1; }

    echo "[stage1] Merging shard captions → generation.json"
    python "$ACTIVE_SCRIPT_DIR/merge_shards.py" \
        --mode caption \
        --output_dir "$EXP_DIR" \
        --num_shards "$NUM_SHARDS"
fi

# ── Stage 2: parallel detection ─────────────────────────────────────────────
echo "[stage2] Launching $NUM_SHARDS detect workers..."
shape_cache_arg=()
if [[ "$SAVE_SHAPE_CACHE" == "1" ]]; then
    shape_cache_arg=(--save_shape_cache)
fi
no_rope_arg=()
if [[ "$COMPUTE_NO_ROPE_ATTENTION" == "1" ]]; then
    no_rope_arg=(--compute_no_rope_attention)
fi
pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
    log="$LOG_DIR/stage2_shard${i}.log"
    echo "[stage2] shard $i → $log"
    (
        python "$SCRIPT_DIR/detect.py" \
            --model_path      "$MODEL_PATH" \
            --coco_path       "$COCO_PATH" \
            --generation_json "$GEN_JSON" \
            --output_dir      "$EXP_DIR" \
            --chair_pkl       "$CHAIR_PKL" \
            --ratio           "$RATIO" \
            --start_layer     "$START_LAYER" \
            --end_layer       "$END_LAYER" \
            --device          "$i" \
            --shard_idx       "$i" \
            --num_shards      "$NUM_SHARDS" \
            "${shape_cache_arg[@]}" \
            "${no_rope_arg[@]}"
    ) > "$log" 2>&1 &
    pids+=($!)
done

fail=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        fail=1
        echo "[stage2] shard $idx (pid ${pids[$idx]}) FAILED — log: $LOG_DIR/stage2_shard${idx}.log" >&2
        tail -n 30 "$LOG_DIR/stage2_shard${idx}.log" >&2 || true
    fi
done
[[ "$fail" -eq 0 ]] || { echo "[stage2] aborting." >&2; exit 1; }

echo "[stage2] Merging shard scores → metrics.json + raw_scores.npz"
python "$ACTIVE_SCRIPT_DIR/merge_shards.py" \
    --mode detect \
    --output_dir "$EXP_DIR" \
    --num_shards "$NUM_SHARDS"

echo
echo "Done. Outputs:"
echo "  $EXP_DIR/generation.json"
echo "  $EXP_DIR/metrics.json"
echo "  $EXP_DIR/raw_scores.npz"
if [[ "$SAVE_SHAPE_CACHE" == "1" ]]; then
    echo "  $EXP_DIR/shape_cache.npz"
fi
echo "  $LOG_DIR/                  (per-shard logs)"
