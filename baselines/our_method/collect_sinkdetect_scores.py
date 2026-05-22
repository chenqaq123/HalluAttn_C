"""Align existing SinkDetect row-cache scores with baseline object records."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_SINKDETECT_KEYS = {
    "sinkdetect_best_global_hallu_score": "sink_only_conc_top10_mass_layer_0",
    "sinkdetect_best_within_bin_hallu_score": "no_rope_topmass_only_conc_top1_mass_layer_1",
    "sinkdetect_cvg_local_hallu_score": "no_rope_topmass_only_cvg_jsd_local_nonobj_layer_2",
    "sinkdetect_clc_hallu_score": "clc_gen_jsd",
}


def collect_sinkdetect_scores(
    row_score_path: str | Path,
    num_objects: int,
    key_map: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    path = Path(row_score_path)
    if not path.exists():
        return {}
    data = np.load(path)
    key_map = key_map or DEFAULT_SINKDETECT_KEYS
    out: dict[str, np.ndarray] = {}
    for output_key, source_key in key_map.items():
        if source_key not in data.files:
            continue
        values = data[source_key].astype(np.float32)
        if values.shape[0] != num_objects:
            trimmed = np.full(num_objects, np.nan, dtype=np.float32)
            n = min(num_objects, values.shape[0])
            trimmed[:n] = values[:n]
            values = trimmed
        out[output_key] = values
    return out

