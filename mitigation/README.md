# Mitigation Track

This directory evaluates attention- and decoding-based hallucination mitigation
methods under controlled behavioral diagnostics.

## Implemented Scope

The current runnable scope for LLaVA-1.5-7B includes four attention/head
interventions and two decoding-time baselines:

| Key | Method | Scope |
|---|---|---|
| `pai` | Paying More Attention to Image | attention manipulation only, excluding CFG/logit refinement |
| `clearsight` | Visual Amplification Fusion | official visual/system attention scaling |
| `visattnsink` | Visual Attention Sink | official sink-based attention redistribution |
| `vcd` | Visual Contrastive Decoding | greedy original/noisy-image logit contrast |
| `damro` | DAMRO | greedy original/outlier-token image-feature logit contrast |
| `spin` | Image-Guided Head Suppression | visual-attention top-head routing with non-routed head suppression |

`vanilla` is always run as the comparison anchor. VCD and DAMRO are available by setting `METHODS=vanilla,vcd` or
`METHODS=vanilla,damro`; they are decoding-time baselines and are therefore
excluded from the attention-shift audit.

## Layout

```text
mitigation/
├── baselines/                  # implementation notes for each paper
├── docs/evaluation_protocol.md # POPE and CHAIR interpretation protocol
├── scripts/
│   ├── run_task.py             # one task, method, and GPU shard
│   ├── merge_evaluate.py       # merge shards and calculate metrics
│   ├── compare_methods.py      # deltas against vanilla
│   ├── run_qwen3vl_xev_pope.py # Qwen3-VL XEV extraction/pilot runner
│   ├── analyze_qwen3vl_xev_pope.py # offline XEV score and behavior analysis
│   └── run_parallel_mitigation.sh
└── src/
    ├── data.py                 # POPE/CHAIR input handling
    ├── decoding.py             # VCD/DAMRO-style decoding utilities
    ├── evaluation.py           # yes-shift and caption analyses
    ├── interventions.py        # attention/head method ports
    └── xev.py                  # region/candidate/relative-evidence primitives
```

## Run

Four-GPU full evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
POPE_DIR=$HOME/common_dataset/pope \
COCO_PATH=$HOME/common_dataset/coco-2014-dataset \
MITIGATION_EXP_NAME=coco_llava_7b_attention_only \
bash mitigation/scripts/run_parallel_mitigation.sh
```

Useful smoke run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
POPE_DIR=$HOME/common_dataset/pope \
COCO_PATH=$HOME/common_dataset/coco-2014-dataset \
POPE_SPLITS=random \
LIMIT=20 \
CHAIR_MAX_NEW_TOKENS=64 \
bash mitigation/scripts/run_parallel_mitigation.sh
```

Set `RUN_POPE=0` or `RUN_CHAIR=0` to run only one task. Outputs are written
under `mitigation/results/<experiment>/`, with per-sample predictions retained
for every method and split. POPE evaluation uses strict yes/no parsing by
default and aborts on invalid answers; set `INVALID_POLICY=as_wrong` or
`INVALID_POLICY=drop` only for explicitly labeled sensitivity analyses.

## Post-hoc Audit

After POPE/CHAIR runs finish, summarize the behavioral effect of every
intervention:

```bash
python mitigation/scripts/audit_results.py \
  --result_root mitigation/results/coco_llava_7b_attention_only
```

This writes:

- `audit.json`
- `audit_pope_by_split.csv`
- `audit_pope_macro.csv`
- `audit_chair.csv`

The audit keeps the paper-facing diagnostics separate from generation:

- POPE yes-rate, TPR, FPR, MCC, and answer-length deltas by split;
- macro POPE deltas and an interpretation flag for answer-prior shifts;
- CHAIR deltas for CHAIRi, CHAIRs, caption length, object mentions, and
  hallucinated object mentions.

The key quantity for POPE is `delta_tpr_minus_delta_fpr`. A positive recall
delta is not evidence of mitigation if FPR rises by the same amount or more.

## Qwen3-VL XEV Pilot

The constructive XEV pilot uses Qwen3-VL and a newer Transformers version than
the LLaVA reproduction environment. Keep it isolated; the first run used Python
3.10 with `transformers==4.57.6`, `pyarrow`, `scipy`, and `scikit-learn`. The
checkpoint is expected at `/home/chenguanxu/models/Qwen3-VL-8B-Instruct` by
default.

One-shard smoke test:

```bash
python mitigation/scripts/run_qwen3vl_xev_pope.py \
  --output_file experiments/qwen3vl_xev/smoke.jsonl \
  --limit 1 \
  --modes target,closed,open \
  --pairwise_top_k 3 \
  --global_support_guard \
  --local_candidate_agreement closed \
  --local_pairwise_gate directional \
  --global_support_matcher aliases \
  --global_inventory_repeats 2 \
  --attention_map no_rope \
  --device cuda:0
```

The 120-row iteration-1 pilot used two processes with `--num_shards 2`, matching
`--limit 120`, and `--shard_idx 0/1` on `cuda:0/1`. Use `--start N` before
sharding to select a disjoint contiguous evaluation window. Analyze any
completed shards without loading the model:

```bash
python mitigation/scripts/analyze_qwen3vl_xev_pope.py \
  experiments/qwen3vl_xev/round1_exact_shard0.jsonl \
  experiments/qwen3vl_xev/round1_exact_shard1.jsonl \
  --output_json experiments/qwen3vl_xev/round1_exact_summary.json
```

`--pairwise_top_k 3` enables the iteration-3 counterbalanced pairwise verifier.
`--attention_map no_rope` is the current default: it recomputes the localization
row from pre-RoPE Q/K without changing the model forward or Yes/No logits. Use
`--attention_map raw` only for the post-RoPE ablation and exact XEV-v1
reproduction.
The iteration-5 image-level certificate additionally uses
`--global_support_guard --local_candidate_agreement closed
--local_pairwise_gate directional --global_support_matcher aliases
--global_inventory_repeats 2`; full-image inventories are generated lazily only
when a strict local proposal exists. `--question_ids 42,50` selects exact cases
for a mechanism audit.
Iteration 10 adds diagnostic-only alternatives:
`--local_relation_gate non_support` admits C/N local identities, while
`--global_support_sources inventories|complement|either` selects the positive
support source. The `either` development variant failed its pre-registered gate
and is not an adopted method. Defaults remain `competing` and `inventories` for
XEV-v1-compatible behavior.
The analyzer defaults to development status; pass
`--pairwise_evaluation confirmatory` only for a rule and window frozen before
inference.

The runner is resumable by `question_id`. Outputs and the derived COCO knowledge
cache remain under git-ignored `experiments/qwen3vl_xev/`. Iteration 1 rejected
the symmetric keep-minus-drop score, and iteration 2 rejected exact open/closed
candidate agreement as a zero-coverage veto certificate; see `docs/proposal.md`
and `docs/experiment_results.md` before changing these formulas. Iteration 3's
pairwise certificate evolved into XEV v1: iteration 8 passed its disjoint
960-row gate with 3 FP corrections and 1 TP veto. Keep the directional/alias
configuration above frozen when running external validations.

## Attention-Shift Audit

The behavioral audit above shows whether outputs improve. The attention-shift
audit checks whether interventions actually change the attention routing they
claim to manipulate.

Run a small four-GPU POPE audit:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_SHARDS=4 \
CACHE_DIR=$HOME/common_model/huggingface/hub \
POPE_DIR=$HOME/common_dataset/pope \
COCO_PATH=$HOME/common_dataset/coco-2014-dataset \
ATTN_AUDIT_EXP_NAME=coco_llava_7b_attention_audit \
LIMIT=120 \
bash mitigation/scripts/run_parallel_attention_audit.sh
```

The runner keeps `MODEL_PATH` as the HuggingFace model id and passes
`CACHE_DIR` explicitly to `from_pretrained(cache_dir=...)`. When `CACHE_DIR` is
not set, it uses `HF_HUB_CACHE`, or `$HF_HOME/hub` when that directory exists.

Outputs are written under:

```text
mitigation/results/<ATTN_AUDIT_EXP_NAME>/pope/<split>/
```

For each split, the runner writes one `attention_audit.jsonl` per method and a
merged `attention_audit_summary.csv/json`. The summary reports:

- pre/post visual attention mass;
- pre/post prefix attention mass;
- pre/post sink attention mass;
- active-layer deltas against matched vanilla rows;
- POPE yes/no metrics and TP/FP visual-mass diagnostics.

Use this audit to support the mechanism claim: an intervention may increase or
redistribute visual attention while still behaving like an answer-prior or
caption-style shift rather than an object-presence verifier.
