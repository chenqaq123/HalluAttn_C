#!/usr/bin/env bash
# Run Experiment B: supervised hidden-contrast probe ceiling on full POPE (9000 rows).
#
# Phase 1: Feature extraction in NUM_SHARDS parallel GPU shards.
# Phase 2: Merge shards + run OOF logistic regression + gate on vanilla.
#
# Usage:
#   bash mitigation/scripts/run_hidden_contrast_probe_full.sh
#
# Override via env:
#   CUDA_VISIBLE_DEVICES=0,1,3,4,5 NUM_SHARDS=5 \
#   bash mitigation/scripts/run_hidden_contrast_probe_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
    echo "[env] sourced $PROJECT_ROOT/.env"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4,5}"
NUM_SHARDS="${NUM_SHARDS:-5}"
LAYERS="${LAYERS:-22,31}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9}"
COCO_PATH="${COCO_PATH:-/data/common_dataset/coco-2014-dataset}"
POPE_DIR="${POPE_DIR:-${POPE_PATH:-$HOME/common_dataset/pope}}"

PYTHON="${PYTHON:-/home/chenguanxu/miniconda3/envs/latentGuard/bin/python}"

AUDIT_CSV="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv"
NEIGHBORS_JSON="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json"
RESULT_ROOT="mitigation/results/coco_llava_7b_attention_only"
OUT_ROOT="mitigation/results/semantic_neighbor_audit/hidden_contrast_probe_full"

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"

echo "[run] Experiment B: supervised hidden-contrast probe ceiling (full POPE)"
echo "[run] LAYERS=$LAYERS  NUM_SHARDS=$NUM_SHARDS  GPUs=${GPU_LIST[*]}"
echo "[run] Output: $OUT_ROOT"

mkdir -p "$OUT_ROOT"

# ---- Phase 1: Feature extraction ----
echo "[phase1] Extracting features ($NUM_SHARDS shards)..."
PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_IDX="${GPU_LIST[$i]}"
    SHARD_DIR="$OUT_ROOT/shard${i}"
    mkdir -p "$SHARD_DIR"
    echo "[phase1] shard $i → GPU $GPU_IDX"
    (
        CUDA_VISIBLE_DEVICES="$GPU_IDX" "$PYTHON" "$SCRIPT_DIR/evaluate_hidden_contrast_probe_extract.py" \
            --model_path "$MODEL_SNAPSHOT" \
            --coco_path "$COCO_PATH" \
            --pope_dir "$POPE_DIR" \
            --audit_csv "$AUDIT_CSV" \
            --neighbors_json "$NEIGHBORS_JSON" \
            --output_dir "$SHARD_DIR" \
            --layers "$LAYERS" \
            --device 0 \
            --shard_idx "$i" \
            --num_shards "$NUM_SHARDS"
    ) > "$SHARD_DIR/shard.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for idx in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$idx]}"; then
        echo "[err] Shard $idx failed" >&2
        tail -20 "$OUT_ROOT/shard${idx}/shard.log" >&2 || true
        FAILED=1
    fi
done
[[ "$FAILED" -eq 0 ]] || { echo "[err] Some shards failed; aborting."; exit 1; }
echo "[phase1] Feature extraction complete."

# ---- Phase 2: Merge + OOF evaluation ----
SHARD_DIRS=$(for ((i=0; i<NUM_SHARDS; i++)); do echo -n "$OUT_ROOT/shard${i},"; done | sed 's/,$//')

echo "[phase2] Merging shards + OOF evaluation..."
"$PYTHON" "$SCRIPT_DIR/merge_evaluate_hidden_contrast_probe.py" \
    --shard_dirs "$SHARD_DIRS" \
    --result_root "$RESULT_ROOT" \
    --output_dir "$OUT_ROOT"

echo ""
echo "=== Experiment B: Supervised ceiling summary ==="
"$PYTHON" - <<'PYEOF'
import json, pathlib
p = pathlib.Path("mitigation/results/semantic_neighbor_audit/hidden_contrast_probe_full/hidden_contrast_probe_full_summary.json")
if p.exists():
    d = json.loads(p.read_text())
    print(f"OOF Absent AUROC : {d.get('oof_absent_auroc', 'N/A'):.4f}")
    print(f"Macro MCC        : {d.get('macro_mcc', 'N/A'):.3f}")
    print(f"TPR              : {d.get('macro_tpr', 'N/A'):.3f}")
    print(f"FPR              : {d.get('macro_fpr', 'N/A'):.3f}")
    print(f"Related FPR      : {d.get('macro_related_fpr', 'N/A'):.3f}")
    print(f"Adv. Related FPR : {d.get('adv_related_fpr', 'N/A'):.3f}")
    print(f"Feature dim      : {d.get('feature_dim', 'N/A')}")
else:
    print("Summary file not found.")
PYEOF

echo "[done] Experiment B complete → $OUT_ROOT"
