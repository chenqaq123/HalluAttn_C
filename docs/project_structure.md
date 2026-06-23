# Project Structure

This repository is the outer code repository for SinkDetect. It intentionally
keeps the paper repository separate: `paper_repo/` is ignored by this Git repo
and should be managed from inside `paper_repo/LookingIsNotGrounding/`.

## Active Layout

```text
SinkDetect/
├── detection/
│   ├── baselines/      # detection baseline reproduction and controls
│   ├── docs/           # detection pipeline, scores, result summaries
│   ├── scripts/        # captioning, row-cache, diagnostics, legacy runners
│   └── src/            # SinkDetect detection package
├── mitigation/
│   ├── baselines/      # intervention scope notes
│   ├── docs/           # POPE/CHAIR interpretation protocol
│   ├── scripts/        # generation, merge, comparison, audits
│   └── src/            # intervention and evaluation code
├── docs/               # project-level design and paper planning
├── paper/              # local paper files/build outputs for this repo
├── paper_repo/         # separate paper Git repo; ignored here
├── ref/                # local reference PDFs
└── reviews/            # local review material
```

Generated artifacts should remain untracked. Important ignored locations include
`experiments/`, `detection/baselines/results/`, `mitigation/results/`, Python
caches, and local tool settings.

## Main Entrypoints

### Detection

| File | Role |
|---|---|
| `detection/scripts/caption.py` | COCO caption generation |
| `detection/scripts/cache_attention_rows.py` | cache object/null visual attention rows |
| `detection/scripts/recompute_from_row_cache.py` | recompute shape scores from row cache |
| `detection/scripts/run_row_cache_parallel.sh` | multi-GPU row-cache runner |
| `detection/scripts/merge_shards.py` | merge caption, detection, and row-cache shards |
| `detection/scripts/analyze_position_effect.py` | position-confound diagnostic |
| `detection/scripts/cache_per_head_rows.py` | per-head attention-shape cache |
| `detection/scripts/diagnose_per_head.py` | supervised per-head diagnostic probe |
| `detection/scripts/legacy/` | retained for reproducing older full-attention runs |

### Detection Baselines

| File | Role |
|---|---|
| `detection/baselines/run_all_baselines.py` | unified baseline runner |
| `detection/baselines/run_parallel_baselines.sh` | multi-GPU baseline runner |
| `detection/baselines/merge_baseline_shards.py` | baseline shard merge |
| `detection/baselines/analyze_controls.py` | nonlinear, category, and image controls |
| `detection/baselines/methods/extract_model_signals.py` | NLL, Entropy, PAS, SVAR, IC, GLSim, Beyond signals |
| `detection/baselines/our_method/collect_sinkdetect_scores.py` | aligns SinkDetect row-cache scores |

### Detection Source

| File | Role |
|---|---|
| `detection/src/sinkdetect/adapter.py` | legacy attention adapter for full matrix extraction |
| `detection/src/sinkdetect/chair.py` | CHAIR evaluation and token labels |
| `detection/src/sinkdetect/grounding.py` | CVG, concentration, and CLC primitives |
| `detection/src/sinkdetect/scoring.py` | score aggregation and AUROC utilities |
| `detection/src/sinkdetect/sink_utils.py` | sink detection, top-mass masking, visual bounds |
| `detection/src/sinkdetect/utils.py` | model loading, prompt, token masks |

### Mitigation

| File | Role |
|---|---|
| `mitigation/src/interventions.py` | PAI attention-only, ClearSight VAF, Visual Attention Sink ports |
| `mitigation/src/data.py` | POPE inputs and fixed CHAIR image manifest |
| `mitigation/src/evaluation.py` | answer-prior and caption-style metrics |
| `mitigation/scripts/run_task.py` | one method/task/GPU shard generation |
| `mitigation/scripts/merge_evaluate.py` | shard merge and POPE/CHAIR evaluation |
| `mitigation/scripts/compare_methods.py` | matched deltas against vanilla |
| `mitigation/scripts/audit_results.py` | POPE yes-shift and CHAIR caption-style audit |
| `mitigation/scripts/audit_attention_shift.py` | pre/post attention-routing audit |
| `mitigation/scripts/run_parallel_mitigation.sh` | four-GPU mitigation runner |
| `mitigation/scripts/run_parallel_attention_audit.sh` | four-GPU attention-audit runner |

## Documentation Ownership

| Doc | Owner |
|---|---|
| `README.md` | project overview and quick starts |
| `docs/design.md` | research argument and method design |
| `docs/aaai2027_paper_plan.md` | paper plan and experiment checklist |
| `detection/docs/pipeline.md` | detection execution contract |
| `detection/docs/scores.md` | score formulas and key names |
| `detection/docs/results_summary.md` | compact baseline results |
| `detection/baselines/README.md` | baseline set and runner usage |
| `mitigation/README.md` | mitigation runner and audit usage |
| `mitigation/docs/evaluation_protocol.md` | mitigation interpretation protocol |

Avoid duplicating long command explanations across docs. Put project-level
entry commands in `README.md`, detection execution details in
`detection/docs/pipeline.md`, score definitions in `detection/docs/scores.md`,
and mitigation interpretation rules in `mitigation/docs/evaluation_protocol.md`.

## Maintenance Rules

- Keep detection implementation under `detection/`; do not recreate top-level
  `baselines/`, `scripts/`, or `src/sinkdetect/` wrappers.
- Keep mitigation outputs under `mitigation/results/` and leave them untracked.
- Keep external paper management inside `paper_repo/LookingIsNotGrounding/`.
- Treat `detection/scripts/legacy/` as reproduction-only unless an old result
  must be regenerated exactly.
- Add ignore rules for generated files instead of committing local artifacts.
