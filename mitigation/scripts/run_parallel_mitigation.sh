#!/usr/bin/env bash
# Run attention-only mitigation baselines on POPE and/or CHAIR with one model per GPU shard.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MITIGATION_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$MITIGATION_ROOT/.." && pwd )"
cd "$PROJECT_ROOT"

ENV_OVERRIDE_KEYS=(
    CUDA_VISIBLE_DEVICES NUM_SHARDS METHODS RUN_POPE RUN_CHAIR POPE_SPLITS
    MITIGATION_EXP_NAME MODEL_PATH CACHE_DIR HF_HUB_CACHE HF_HOME COCO_PATH
    CHAIR_PKL INVALID_POLICY CHAIR_MANIFEST POPE_DIR POPE_PATH LIMIT SEED
    POPE_MAX_NEW_TOKENS CHAIR_MAX_NEW_TOKENS PAI_ALPHA VAF_ENHANCE
    VAF_SUPPRESS VAS_TAU VAS_RHO VAS_VISUAL_MASS VAS_KEEP VCD_ALPHA
    VCD_BETA VCD_NOISE_STEP
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
RUN_POPE="${RUN_POPE:-1}"
RUN_CHAIR="${RUN_CHAIR:-1}"
POPE_SPLITS="${POPE_SPLITS:-random,popular,adversarial}"
EXP_NAME="${MITIGATION_EXP_NAME:-coco_llava_7b_attention_only}"
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
CACHE_DIR="${CACHE_DIR:-${HF_HUB_CACHE:-}}"
if [[ -z "$CACHE_DIR" && -n "${HF_HOME:-}" && -d "$HF_HOME/hub" ]]; then
    CACHE_DIR="$HF_HOME/hub"
fi
COCO_PATH="${COCO_PATH:-$HOME/common_dataset/coco-2014-dataset}"
CHAIR_PKL="${CHAIR_PKL:-$PROJECT_ROOT/../pas/data/chair_coco.pkl}"
INVALID_POLICY="${INVALID_POLICY:-error}"
CHAIR_MANIFEST="${CHAIR_MANIFEST:-$PROJECT_ROOT/experiments/coco_llava_7b/generation.json}"
POPE_DIR="${POPE_DIR:-${POPE_PATH:-$HOME/common_dataset/pope}}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-42}"
POPE_MAX_NEW_TOKENS="${POPE_MAX_NEW_TOKENS:-16}"
CHAIR_MAX_NEW_TOKENS="${CHAIR_MAX_NEW_TOKENS:-512}"

# Official attention-only defaults.
PAI_ALPHA="${PAI_ALPHA:-0.2}"
VAF_ENHANCE="${VAF_ENHANCE:-1.15}"
VAF_SUPPRESS="${VAF_SUPPRESS:-0.95}"
VAS_TAU="${VAS_TAU:-20}"
VAS_RHO="${VAS_RHO:-0.5}"
VAS_VISUAL_MASS="${VAS_VISUAL_MASS:-0.2}"
VAS_KEEP="${VAS_KEEP:-0.6}"
VCD_ALPHA="${VCD_ALPHA:-0.5}"
VCD_BETA="${VCD_BETA:-0.1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"

if [[ "$CHAIR_PKL" != /* ]]; then
    CHAIR_PKL="$PROJECT_ROOT/$CHAIR_PKL"
fi
IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if (( NUM_SHARDS < 1 || NUM_SHARDS > ${#GPU_LIST[@]} )); then
    echo "[err] NUM_SHARDS=$NUM_SHARDS must be within visible GPU count=${#GPU_LIST[@]}." >&2
    exit 1
fi
if [[ ! -d "$COCO_PATH/val2014" ]]; then
    echo "[err] COCO val2014 images not found under $COCO_PATH." >&2
    exit 1
fi
if [[ "$RUN_CHAIR" == "1" && ! -f "$CHAIR_MANIFEST" ]]; then
    echo "[err] CHAIR manifest not found: $CHAIR_MANIFEST" >&2
    exit 1
fi
if [[ "$RUN_CHAIR" == "1" && ! -f "$CHAIR_PKL" ]]; then
    echo "[err] CHAIR evaluator not found: $CHAIR_PKL" >&2
    exit 1
fi
if [[ "$RUN_POPE" == "1" && ! -d "$POPE_DIR" ]]; then
    echo "[err] POPE question directory not found: $POPE_DIR" >&2
    exit 1
fi

OUT_ROOT="$MITIGATION_ROOT/results/$EXP_NAME"
mkdir -p "$OUT_ROOT"
echo "[mitigation] output=$OUT_ROOT methods=$METHODS shards=$NUM_SHARDS"
echo "[mitigation] model=$MODEL_PATH"
echo "[mitigation] cache_dir=${CACHE_DIR:-<transformers-default>}"

run_method_task() {
    local task="$1"
    local label="$2"
    local method="$3"
    local max_tokens="$4"
    local out_dir="$OUT_ROOT/$task/$label/$method"
    local log_dir="$out_dir/logs"
    mkdir -p "$log_dir"
    local pids=()

    for ((i=0; i<NUM_SHARDS; i++)); do
        local extra_args=()
        if [[ "$task" == "pope" ]]; then
            extra_args=(--pope_dir "$POPE_DIR" --pope_split "$label")
        else
            extra_args=(--chair_manifest "$CHAIR_MANIFEST")
        fi
        echo "[$task:$label:$method] shard $i"
        (
            python "$SCRIPT_DIR/run_task.py" \
                --task "$task" \
                --method "$method" \
                --model_path "$MODEL_PATH" \
                --cache_dir "$CACHE_DIR" \
                --coco_path "$COCO_PATH" \
                --output_file "$out_dir/shard${i}.jsonl" \
                --device "$i" \
                --shard_idx "$i" \
                --num_shards "$NUM_SHARDS" \
                --limit "$LIMIT" \
                --seed "$SEED" \
                --max_new_tokens "$max_tokens" \
                --pai_alpha "$PAI_ALPHA" \
                --vaf_enhance "$VAF_ENHANCE" \
                --vaf_suppress "$VAF_SUPPRESS" \
                --vas_tau "$VAS_TAU" \
                --vas_rho "$VAS_RHO" \
                --vas_visual_mass "$VAS_VISUAL_MASS" \
                --vas_keep "$VAS_KEEP" \
                --vcd_alpha "$VCD_ALPHA" \
                --vcd_beta "$VCD_BETA" \
                --vcd_noise_step "$VCD_NOISE_STEP" \
                "${extra_args[@]}"
        ) > "$log_dir/shard${i}.log" 2>&1 &
        pids+=($!)
    done

    local failed=0
    for idx in "${!pids[@]}"; do
        if ! wait "${pids[$idx]}"; then
            failed=1
            echo "[err] $task/$label/$method shard $idx failed." >&2
            tail -n 40 "$log_dir/shard${idx}.log" >&2 || true
        fi
    done
    [[ "$failed" -eq 0 ]] || exit 1

    local eval_args=(--task "$task" --method_dir "$out_dir" --num_shards "$NUM_SHARDS")
    if [[ "$task" == "chair" ]]; then
        eval_args+=(--chair_pkl "$CHAIR_PKL")
    else
        eval_args+=(--invalid_policy "$INVALID_POLICY")
    fi
    python "$SCRIPT_DIR/merge_evaluate.py" "${eval_args[@]}"
}

IFS=',' read -ra METHOD_LIST <<< "$METHODS"
if [[ "$RUN_POPE" == "1" ]]; then
    IFS=',' read -ra SPLIT_LIST <<< "$POPE_SPLITS"
    for split in "${SPLIT_LIST[@]}"; do
        for method in "${METHOD_LIST[@]}"; do
            run_method_task pope "$split" "$method" "$POPE_MAX_NEW_TOKENS"
        done
        python "$SCRIPT_DIR/compare_methods.py" \
            --task_dir "$OUT_ROOT/pope/$split" \
            --methods "$METHODS"
    done
fi

if [[ "$RUN_CHAIR" == "1" ]]; then
    for method in "${METHOD_LIST[@]}"; do
        run_method_task chair caption "$method" "$CHAIR_MAX_NEW_TOKENS"
    done
    python "$SCRIPT_DIR/compare_methods.py" \
        --task_dir "$OUT_ROOT/chair/caption" \
        --methods "$METHODS"
fi

echo "[mitigation] completed: $OUT_ROOT"
