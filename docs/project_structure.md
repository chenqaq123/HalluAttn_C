# Project Structure Plan

This repo is moving from a single SinkDetect prototype into a detection-first
hallucination study. The conceptual roadmap has two tracks:

1. **Detection**: Do attention-based detection scores really measure visual
   grounding, or do they exploit generation-position confounds?
2. **Mitigation**: Do attention/visual-enhancement interventions really reduce
   hallucination, or do they shift the yes/no answer prior on benchmarks such
   as POPE?

The detection track already has complete experiments and its executable code
has been moved under `detection/`. The mitigation track has only
documentation/scaffold directories so far; it should not contain paper-facing
claims until methods are reproduced and evaluated under controlled metrics.

---

## Current Target Layout

```text
SinkDetect/
├── detection/
│   ├── baselines/                # detection baseline reproduction code + docs
│   ├── docs/                     # detection design, score reference, pipeline, results
│   ├── scripts/                  # row-cache, diagnostics, captioning, legacy runners
│   └── src/                      # SinkDetect detection source package
├── mitigation/
│   ├── baselines/                # methods to reproduce and answer-prior diagnostics
│   ├── docs/                     # mitigation design notes once experiments start
│   ├── scripts/                  # future POPE runners and analysis scripts
│   └── src/                      # future reusable mitigation code
├── experiments/                  # ignored generated artifacts
├── docs/                         # project-level docs
└── paper/
```

The old top-level `baselines/`, `scripts/`, and `src/sinkdetect/` directories
have been removed. Detection implementation work should happen under
`detection/`.

The `mitigation/` directory is scaffold-only. It records methods and metrics to
test, but contains no reproduced method implementation yet.

---

## Current Code Map

### Shared Data / Labels

| Current file | Role | Target |
|---|---|---|
| Current file | Role |
|---|---|
| `detection/scripts/caption.py` | COCO caption generation |
| `detection/baselines/data/build_object_cache.py` | object-level cache from generation + CHAIR / row cache |
| `detection/src/sinkdetect/chair.py` | CHAIR evaluation and token labels |
| `detection/src/sinkdetect/utils.py` | model loading, prompt, token masks |

### Detection Baselines

| Current file | Role |
|---|---|
| `detection/baselines/run_all_baselines.py` | unified detection baseline runner |
| `detection/baselines/run_parallel_baselines.sh` | multi-GPU detection baseline runner |
| `detection/baselines/merge_baseline_shards.py` | merge detection shards |
| `detection/baselines/methods/extract_model_signals.py` | NLL/Entropy/PAS/SVAR/IC/GLSim/Beyond computation |
| `detection/baselines/metrics.py` | AUROC and position-control metrics |
| `detection/baselines/our_method/collect_sinkdetect_scores.py` | aligns existing SinkDetect row-cache scores |

### SinkDetect / Attention Shape

| Current file | Role |
|---|---|
| `detection/scripts/cache_attention_rows.py` | cache object/null visual rows |
| `detection/scripts/recompute_from_row_cache.py` | recompute shape metrics from row cache |
| `detection/scripts/run_row_cache_parallel.sh` | multi-GPU row-cache runner |
| `detection/src/sinkdetect/adapter.py` | legacy attention adapter for full matrix extraction |
| `detection/src/sinkdetect/grounding.py` | CVG/concentration/CLC primitives |
| `detection/src/sinkdetect/scoring.py` | score aggregation + AUROC |
| `detection/src/sinkdetect/sink_utils.py` | sink detection, top-mass, visual bounds |

### Diagnostics

| Current file | Role |
|---|---|
| `detection/scripts/analyze_position_effect.py` | position confound analysis |
| `detection/scripts/cache_per_head_rows.py` | per-head shape feature cache |
| `detection/scripts/diagnose_per_head.py` | supervised per-head diagnostic probe |
| `detection/scripts/legacy/fusion.py` | exploratory score fusion retained for older analysis |

### Mitigation Track

No stable implementation exists yet. Current files are README scaffolds. The
eventual target is:

| Target file | Role |
|---|---|
| `mitigation/baselines/README.md` | methods to reproduce and citation notes |
| `mitigation/scripts/run_pope.py` | common POPE runner / parser |
| `mitigation/scripts/evaluate_answer_prior.py` | yes-rate, TPR/FPR, balanced accuracy, MCC, `ΔTPR−ΔFPR` |
| `mitigation/src/` | wrappers for ClearSight, PAI, VCD, head-selection methods after reproduction |

Until reproduced, mitigation should be documented as a suspicion and diagnostic
framework, not as an empirical conclusion.

---

## Cleanup Policy

### Keep As Active

- `detection/baselines/run_all_baselines.py`
- `detection/baselines/run_parallel_baselines.sh`
- `detection/scripts/cache_attention_rows.py`
- `detection/scripts/recompute_from_row_cache.py`
- `detection/scripts/cache_per_head_rows.py`
- `detection/scripts/diagnose_per_head.py`
- `detection/scripts/analyze_position_effect.py`
- `detection/src/sinkdetect/*.py`
- `docs/design.md`
- `detection/baselines/README.md`
- `detection/docs/results_summary.md`
- `paper/acl_latex.tex`

### Freeze / Archive After Replacement

These are still useful for reproducibility but should stop receiving new
features once the new detection entrypoints exist:

- `detection/scripts/legacy/detect.py`
- `detection/scripts/legacy/recompute_shape_from_cache.py`
- `detection/scripts/legacy/recompute_rope_debiased_from_cache.py`
- `detection/scripts/legacy/run.sh`
- `detection/scripts/legacy/run_parallel.sh`
- `detection/scripts/legacy/fusion.py`

Suggested archive path:

```text
detection/scripts/legacy/
```

Each archived script should keep a short header:

```text
Legacy script retained for reproducing pre-row-cache SinkDetect experiments.
Prefer <new command> for current runs.
```

### Remove From Git / Ignore

These should never be tracked:

- `experiments/`
- `detection/baselines/results/`
- `__pycache__/`
- `*.pyc`
- paper build artifacts (`*.aux`, `*.log`, `*.bbl`, `*.pdf`, etc.)
- local tool config such as `.claude/`

Most are already ignored. If a generated file appears in `git status`, add an
ignore rule rather than committing it.

---

## Remaining Cleanup

1. Keep detection implementation under `detection/`; do not recreate top-level
   `baselines/`, `scripts/`, or `src/sinkdetect/`.
2. Split `detection/src/sinkdetect/scoring.py` into shape-score primitives and
   evaluation utilities only if the next implementation step needs it.
3. Keep mitigation scaffold documentation-only until reproducing a specific
   method such as ClearSight, PAI, VCD, or a head-selection intervention.

---

## Current Paper Framing

The safest paper story today:

1. **Detection**: aggregate attention detection is position-confounded.
2. **Diagnostics**: mean-over-head attention fails, but late-layer per-head
   probes recover signal, so attention is not empty.
3. **Mitigation**: aggregate POPE metrics may hide answer-prior shifts; this is
   a future controlled-evaluation track, not yet a reproduced result.

Avoid claiming that mitigation methods are flawed until the mitigation baseline
track is implemented and analyzed.
