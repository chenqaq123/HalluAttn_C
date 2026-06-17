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
│   ├── audit_two_stage_calibration.py # OWLv2 TDEV calibration sensitivity
│   ├── evaluate_hybrid_region_rule.py # asymmetric OWLv2 TDEV gate/rescue rule
│   └── run_parallel_mitigation.sh
└── src/
    ├── data.py                 # POPE/CHAIR input handling
    ├── decoding.py             # VCD/DAMRO-style decoding utilities
    ├── evaluation.py           # yes-shift and caption analyses
    └── interventions.py        # attention/head method ports
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
