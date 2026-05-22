# Detection Track

This directory contains the hallucination **detection** side of the project.

Current focus:

1. Reproduce detection baselines on the same object-level cache.
2. Evaluate all scores under position-controlled metrics.
3. Diagnose whether attention still contains signal after mean-over-head
   aggregation fails.

## Layout

```text
detection/
├── baselines/       # detection baseline reproduction code and docs
├── docs/            # design, score reference, pipeline, result summaries
├── scripts/         # row-cache, diagnostics, captioning, legacy runners
└── src/             # SinkDetect detection source modules
```

The executable detection code now lives here. The old top-level `baselines/`,
`scripts/`, and `src/sinkdetect/` directories have been removed.

## Current Entrypoints

- Detection baselines:
  `bash detection/baselines/run_parallel_baselines.sh`
- Row-cache attention shape:
  `bash detection/scripts/run_row_cache_parallel.sh`
- Per-head diagnostic:
  `python detection/scripts/diagnose_per_head.py --cache_glob 'experiments/coco_llava_7b_rows/per_head_row_cache_shard*.npz'`

See [docs/project_structure.md](../docs/project_structure.md) for the current
layout and remaining cleanup policy.
