#!/usr/bin/env bash
# Run Experiment A: per-layer hidden-margin v2 + answer features on full POPE.
#
# Phase 1: 5 GPU shards (feature extraction)
# Phase 2: merge shards into one CSV
# Phase 3: cross-split logreg verifier evaluation
#
# Usage:
#   bash mitigation/scripts/run_hidden_margin_v2_full.sh
#
# Override via env:
#   CUDA_VISIBLE_DEVICES=0,1,3,4,5 \
#   NUM_SHARDS=5 \
#   LAYERS=16,22,27,31 \
#   bash mitigation/scripts/run_hidden_margin_v2_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Source .env (same pattern as other run scripts) ----
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
    echo "[env] sourced $PROJECT_ROOT/.env"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3,4,5}"
NUM_SHARDS="${NUM_SHARDS:-5}"
LAYERS="${LAYERS:-16,22,27,31}"
TOP_NEIGHBORS="${TOP_NEIGHBORS:-1}"
MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
COCO_PATH="${COCO_PATH:-/data/common_dataset/coco-2014-dataset}"
POPE_DIR="${POPE_DIR:-${POPE_PATH:-$HOME/common_dataset/pope}}"
HF_HOME="${HF_HOME:-/data/common_model/huggingface}"
CACHE_DIR="${CACHE_DIR:-$HF_HOME/hub}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/home/chenguanxu/common_model/huggingface/models--llava-hf--llava-1.5-7b-hf/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9}"

PYTHON="${PYTHON:-/home/chenguanxu/miniconda3/envs/latentGuard/bin/python}"

AUDIT_CSV="mitigation/results/semantic_neighbor_audit/semantic_neighbor_rows.csv"
NEIGHBORS_JSON="mitigation/results/semantic_neighbor_audit/cooccurrence_neighbors.json"
RESULT_ROOT="mitigation/results/coco_llava_7b_attention_only"
OUT_ROOT="mitigation/results/semantic_neighbor_audit/hidden_margin_v2_full"

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"

echo "[run] Experiment A: per-layer hidden-margin v2 + answer features"
echo "[run] LAYERS=$LAYERS  NUM_SHARDS=$NUM_SHARDS  GPUs=${GPU_LIST[*]}"
echo "[run] Output: $OUT_ROOT"

mkdir -p "$OUT_ROOT"

# ---- Phase 1: Feature extraction (parallel shards) ----
echo "[phase1] Starting $NUM_SHARDS shard(s)..."
PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU_IDX="${GPU_LIST[$i]}"
    SHARD_DIR="$OUT_ROOT/shard${i}"
    mkdir -p "$SHARD_DIR"
    echo "[phase1] shard $i → GPU $GPU_IDX → $SHARD_DIR"
    (
        CUDA_VISIBLE_DEVICES="$GPU_IDX" "$PYTHON" "$SCRIPT_DIR/evaluate_hidden_margin_tdev_v2_pope.py" \
            --model_path "$MODEL_SNAPSHOT" \
            --coco_path "$COCO_PATH" \
            --pope_dir "$POPE_DIR" \
            --audit_csv "$AUDIT_CSV" \
            --neighbors_json "$NEIGHBORS_JSON" \
            --output_dir "$SHARD_DIR" \
            --layers "$LAYERS" \
            --top_neighbors "$TOP_NEIGHBORS" \
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
echo "[phase1] All shards complete."

# ---- Phase 2: Merge shard CSVs ----
echo "[phase2] Merging shards..."
"$PYTHON" - <<'PYEOF'
import csv, pathlib, sys

project_root = pathlib.Path(".").resolve()
out_root = project_root / "mitigation/results/semantic_neighbor_audit/hidden_margin_v2_full"
shard_dirs = sorted(out_root.glob("shard*"))
all_rows = []
for sd in shard_dirs:
    p = sd / "hidden_margin_v2_predictions.csv"
    if not p.exists():
        print(f"[warn] missing {p}", file=sys.stderr)
        continue
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    all_rows.extend(rows)
    print(f"  {sd.name}: {len(rows)} rows")

if not all_rows:
    raise SystemExit("No rows found — check shard logs.")

all_rows.sort(key=lambda r: (r["split"], int(r["question_id"])))
out_path = out_root / "hidden_margin_v2_predictions.csv"
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Merged {len(all_rows)} rows → {out_path}")
PYEOF
echo "[phase2] Merge complete."

# ---- Phase 3: Cross-split logreg evaluation ----
echo "[phase3] Running cross-split logreg verifier..."

SCORE_CSV="$OUT_ROOT/hidden_margin_v2_predictions.csv"

# Feature sets to evaluate
# Full per-layer set (4 layers × 3 key margins = 12 + 2 answer = 14 features)
FEATURES_LAYER_MARGINS="l16_align_margin,l16_cross_margin,l16_obj_separation,l16_vis_separation,l22_align_margin,l22_cross_margin,l22_obj_separation,l22_vis_separation,l27_align_margin,l27_cross_margin,l27_obj_separation,l27_vis_separation,l31_align_margin,l31_cross_margin,l31_obj_separation,l31_vis_separation"
FEATURES_ANSWER="answer_support_score,answer_contrast_margin"
FEATURES_ALL="${FEATURES_LAYER_MARGINS},${FEATURES_ANSWER}"

# Ablation: answer only
"$PYTHON" "$SCRIPT_DIR/cross_split_logreg_verifier.py" \
    --score_csv "$SCORE_CSV" \
    --feature_columns "$FEATURES_ANSWER" \
    --result_root "$RESULT_ROOT" \
    --output_dir "$OUT_ROOT/logreg_answer_only" \
    --objectives "best_mcc,min_fpr_tpr0.80"

# Ablation: per-layer hidden only
"$PYTHON" "$SCRIPT_DIR/cross_split_logreg_verifier.py" \
    --score_csv "$SCORE_CSV" \
    --feature_columns "$FEATURES_LAYER_MARGINS" \
    --result_root "$RESULT_ROOT" \
    --output_dir "$OUT_ROOT/logreg_hidden_only" \
    --objectives "best_mcc,min_fpr_tpr0.80"

# Main: all features
"$PYTHON" "$SCRIPT_DIR/cross_split_logreg_verifier.py" \
    --score_csv "$SCORE_CSV" \
    --feature_columns "$FEATURES_ALL" \
    --result_root "$RESULT_ROOT" \
    --output_dir "$OUT_ROOT/logreg_all" \
    --objectives "best_mcc,min_fpr_tpr0.80"

echo "[phase3] Logreg evaluation complete."
echo ""
echo "=== Summary (best_mcc objective) ==="
"$PYTHON" - <<'PYEOF'
import json, pathlib

base = pathlib.Path("mitigation/results/semantic_neighbor_audit/hidden_margin_v2_full")
for name in ["logreg_answer_only", "logreg_hidden_only", "logreg_all"]:
    p = base / name / "logreg_summary.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    for row in d.get("macro", []):
        if row["objective"] == "best_mcc":
            print(f"{name:30s}  MCC={row['mcc']:.3f}  TPR={row['tpr']:.3f}  FPR={row['fpr']:.3f}  Related-FPR={row['related_fpr']:.3f}  Adv-Rel-FPR={row.get('adv_related_fpr', float('nan')):.3f}")
PYEOF

echo "[done] Experiment A complete → $OUT_ROOT"
