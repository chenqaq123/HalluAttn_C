#!/usr/bin/env bash
# Run a small POPE attention-routing audit for mitigation interventions.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MITIGATION_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$MITIGATION_ROOT/.." && pwd )"
cd "$PROJECT_ROOT"

ENV_OVERRIDE_KEYS=(
    CUDA_VISIBLE_DEVICES NUM_SHARDS METHODS POPE_SPLITS ATTN_AUDIT_EXP_NAME
    MODEL_PATH CACHE_DIR HF_HUB_CACHE HF_HOME COCO_PATH POPE_DIR POPE_PATH
    LIMIT SEED MAX_NEW_TOKENS AUDIT_START_LAYER AUDIT_END_LAYER PAI_ALPHA
    VAF_ENHANCE VAF_SUPPRESS VAS_TAU VAS_RHO VAS_VISUAL_MASS VAS_KEEP
)
declare -A CALLER_ENV=()
for key in "${ENV_OVERRIDE_KEYS[@]}"; do
    if [[ -v "$key" ]]; then
        CALLER_ENV["$key"]="${!key}"
    fi
done
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    for key in "${!CALLER_ENV[@]}"; do
        export "$key=${CALLER_ENV[$key]}"
    done
    set +a
    echo "[env] sourced $PROJECT_ROOT/.env"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
METHODS="${METHODS:-vanilla,pai,clearsight,visattnsink}"
POPE_SPLITS="${POPE_SPLITS:-random,popular,adversarial}"
EXP_NAME="${ATTN_AUDIT_EXP_NAME:-coco_llava_7b_attention_audit}"
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
CACHE_DIR="${CACHE_DIR:-${HF_HUB_CACHE:-}}"
if [[ -z "$CACHE_DIR" && -n "${HF_HOME:-}" && -d "$HF_HOME/hub" ]]; then
    CACHE_DIR="$HF_HOME/hub"
fi
COCO_PATH="${COCO_PATH:-$HOME/common_dataset/coco-2014-dataset}"
POPE_DIR="${POPE_DIR:-${POPE_PATH:-$HOME/common_dataset/pope}}"
LIMIT="${LIMIT:-120}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
AUDIT_START_LAYER="${AUDIT_START_LAYER:-2}"
AUDIT_END_LAYER="${AUDIT_END_LAYER:-32}"

PAI_ALPHA="${PAI_ALPHA:-0.2}"
VAF_ENHANCE="${VAF_ENHANCE:-1.15}"
VAF_SUPPRESS="${VAF_SUPPRESS:-0.95}"
VAS_TAU="${VAS_TAU:-20}"
VAS_RHO="${VAS_RHO:-0.5}"
VAS_VISUAL_MASS="${VAS_VISUAL_MASS:-0.2}"
VAS_KEEP="${VAS_KEEP:-0.6}"

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if (( NUM_SHARDS < 1 || NUM_SHARDS > ${#GPU_LIST[@]} )); then
    echo "[err] NUM_SHARDS=$NUM_SHARDS must be within visible GPU count=${#GPU_LIST[@]}." >&2
    exit 1
fi
if [[ "$MODEL_PATH" == /* && ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[err] MODEL_PATH does not look like a local HF model directory: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -d "$COCO_PATH/val2014" ]]; then
    echo "[err] COCO val2014 images not found under $COCO_PATH." >&2
    exit 1
fi
if [[ ! -d "$POPE_DIR" ]]; then
    echo "[err] POPE question directory not found: $POPE_DIR" >&2
    exit 1
fi

OUT_ROOT="$MITIGATION_ROOT/results/$EXP_NAME"
mkdir -p "$OUT_ROOT"
echo "[attention-audit] output=$OUT_ROOT methods=$METHODS splits=$POPE_SPLITS limit=$LIMIT shards=$NUM_SHARDS"
echo "[attention-audit] model=$MODEL_PATH"
echo "[attention-audit] cache_dir=${CACHE_DIR:-<transformers-default>}"

run_method_split() {
    local split="$1"
    local method="$2"
    local out_dir="$OUT_ROOT/pope/$split/$method"
    local log_dir="$out_dir/logs"
    mkdir -p "$log_dir"
    local pids=()

    for ((i=0; i<NUM_SHARDS; i++)); do
        echo "[attention-audit:$split:$method] shard $i"
        (
            python "$SCRIPT_DIR/audit_attention_shift.py" \
                --mode collect \
                --method "$method" \
                --model_path "$MODEL_PATH" \
                --cache_dir "$CACHE_DIR" \
                --coco_path "$COCO_PATH" \
                --pope_dir "$POPE_DIR" \
                --pope_split "$split" \
                --output_file "$out_dir/shard${i}.jsonl" \
                --device "$i" \
                --shard_idx "$i" \
                --num_shards "$NUM_SHARDS" \
                --limit "$LIMIT" \
                --seed "$SEED" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --audit_start_layer "$AUDIT_START_LAYER" \
                --audit_end_layer "$AUDIT_END_LAYER" \
                --pai_alpha "$PAI_ALPHA" \
                --vaf_enhance "$VAF_ENHANCE" \
                --vaf_suppress "$VAF_SUPPRESS" \
                --vas_tau "$VAS_TAU" \
                --vas_rho "$VAS_RHO" \
                --vas_visual_mass "$VAS_VISUAL_MASS" \
                --vas_keep "$VAS_KEEP"
        ) > "$log_dir/shard${i}.log" 2>&1 &
        pids+=($!)
    done

    local failed=0
    for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
            failed=1
            echo "[err] attention audit $split/$method shard $idx failed." >&2
            tail -n 40 "$log_dir/shard${idx}.log" >&2 || true
        fi
    done
    [[ "$failed" -eq 0 ]] || exit 1

    local merged="$out_dir/attention_audit.jsonl"
    : > "$merged"
    for ((i=0; i<NUM_SHARDS; i++)); do
        cat "$out_dir/shard${i}.jsonl" >> "$merged"
    done
}

IFS=',' read -ra METHOD_LIST <<< "$METHODS"
IFS=',' read -ra SPLIT_LIST <<< "$POPE_SPLITS"
for split in "${SPLIT_LIST[@]}"; do
    for method in "${METHOD_LIST[@]}"; do
        run_method_split "$split" "$method"
    done
    python "$SCRIPT_DIR/audit_attention_shift.py" \
        --mode summarize \
        --audit_dir "$OUT_ROOT/pope/$split" \
        --methods "$METHODS"
done

echo "[attention-audit] completed: $OUT_ROOT"
