#!/usr/bin/env bash
# One-click runner for SinkDetect (stage 1 caption + stage 2 detect).
#
# Usage:
#   bash detection/scripts/legacy/run.sh                       # full run with defaults
#   NUM_SAMPLES=200 LIMIT=200 bash detection/scripts/legacy/run.sh   # small smoke test
#
# Override any variable below via the environment. Examples:
#   EXP_NAME=coco_llava_7b_r03 RATIO=0.3 bash detection/scripts/legacy/run.sh
#   MODEL_PATH=/path/to/llava-1.5-7b-hf bash detection/scripts/legacy/run.sh
set -euo pipefail

# ── Resolve project root (the parent of this script) ────────────────────────
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

# ── Restrict to GPUs 0-3 (override only if you really need to) ──────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ── Defaults (override via env or .env) ─────────────────────────────────────
EXP_NAME="${EXP_NAME:-coco_llava_7b}"
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
COCO_PATH="${COCO_PATH:-/data/common_dataset/coco-2014-dataset/}"
CHAIR_PKL="${CHAIR_PKL:-$PROJ_ROOT/../pas/data/chair_coco.pkl}"
# CHAIR_PKL may be relative in .env — resolve against project root
if [[ "$CHAIR_PKL" != /* ]]; then
    CHAIR_PKL="$PROJ_ROOT/$CHAIR_PKL"
fi

NUM_SAMPLES="${NUM_SAMPLES:-5000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RATIO="${RATIO:-0.5}"
START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-32}"
LIMIT="${LIMIT:-0}"               # 0 = use all generated captions in detect
DEVICE="${DEVICE:-0}"
SEED="${SEED:-42}"
SAVE_SHAPE_CACHE="${SAVE_SHAPE_CACHE:-0}"
COMPUTE_NO_ROPE_ATTENTION="${COMPUTE_NO_ROPE_ATTENTION:-0}"

# DEVICE is a *logical* index into CUDA_VISIBLE_DEVICES, so 0..3 is the valid range.
if ! [[ "$DEVICE" =~ ^[0-3]$ ]]; then
    echo "[err] DEVICE=$DEVICE is outside the allowed range 0-3."
    echo "      Only GPUs 0-3 are usable. Pick DEVICE from {0,1,2,3}."
    exit 1
fi

EXP_DIR="$PROJ_ROOT/experiments/$EXP_NAME"
GEN_JSON="$EXP_DIR/generation.json"
mkdir -p "$EXP_DIR"

# ── Sanity checks ───────────────────────────────────────────────────────────
# MODEL_PATH may be a HF repo id (resolved via $HF_HOME) or a local dir; only
# warn when it looks like a path but isn't there.
if [[ "$MODEL_PATH" == /* && ! -d "$MODEL_PATH" ]]; then
    echo "[warn] MODEL_PATH looks like a local path but doesn't exist: $MODEL_PATH" >&2
fi
if [[ ! -d "$COCO_PATH" ]]; then
    echo "[warn] COCO_PATH does not exist: $COCO_PATH" >&2
fi
if [[ ! -f "$CHAIR_PKL" ]]; then
    echo "[err] CHAIR pickle not found: $CHAIR_PKL" >&2
    echo "      Run the PAS pipeline once to build it, or point CHAIR_PKL elsewhere." >&2
    exit 1
fi
echo "[env] MODEL_PATH=$MODEL_PATH"
echo "[env] HF_HOME=${HF_HOME:-<unset>}"

# ── Stage 1: caption generation ─────────────────────────────────────────────
if [[ -f "$GEN_JSON" && -z "${FORCE_REGEN:-}" ]]; then
    echo "[stage1] $GEN_JSON exists — skipping (set FORCE_REGEN=1 to redo)."
else
    echo "[stage1] Generating captions → $GEN_JSON"
    python "$ACTIVE_SCRIPT_DIR/caption.py" \
        --model_path     "$MODEL_PATH" \
        --coco_path      "$COCO_PATH" \
        --output_path    "$GEN_JSON" \
        --num_samples    "$NUM_SAMPLES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --seed           "$SEED" \
        --device         "$DEVICE"
fi

# ── Stage 2: purified attention + AUROC ─────────────────────────────────────
echo "[stage2] Running sink/purified detection → $EXP_DIR"
shape_cache_arg=()
if [[ "$SAVE_SHAPE_CACHE" == "1" ]]; then
    shape_cache_arg=(--save_shape_cache)
fi
no_rope_arg=()
if [[ "$COMPUTE_NO_ROPE_ATTENTION" == "1" ]]; then
    no_rope_arg=(--compute_no_rope_attention)
fi
python "$SCRIPT_DIR/detect.py" \
    --model_path      "$MODEL_PATH" \
    --coco_path       "$COCO_PATH" \
    --generation_json "$GEN_JSON" \
    --output_dir      "$EXP_DIR" \
    --chair_pkl       "$CHAIR_PKL" \
    --ratio           "$RATIO" \
    --start_layer     "$START_LAYER" \
    --end_layer       "$END_LAYER" \
    --device          "$DEVICE" \
    --limit           "$LIMIT" \
    "${shape_cache_arg[@]}" \
    "${no_rope_arg[@]}"

echo
echo "Done. Outputs:"
echo "  $EXP_DIR/generation.json"
echo "  $EXP_DIR/metrics.json"
echo "  $EXP_DIR/raw_scores.npz"
if [[ "$SAVE_SHAPE_CACHE" == "1" ]]; then
    echo "  $EXP_DIR/shape_cache.npz"
fi
