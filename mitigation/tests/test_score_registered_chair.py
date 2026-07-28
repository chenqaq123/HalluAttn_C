import numpy as np

from mitigation.scripts.score_registered_chair import (
    metric_vector,
    paired_bootstrap,
)


def test_metric_vector_uses_official_pooled_ratios():
    components = np.asarray(
        [
            [1, 1, 2, 4, 2, 5, 2, 4],
            [0, 1, 0, 1, 1, 3, 1, 1],
        ],
        dtype=np.float64,
    )
    values = metric_vector(components)
    assert np.allclose(
        values,
        [
            0.5,
            2 / 5,
            3 / 8,
            3 / 5,
            2 * (3 / 8) * (3 / 5) / ((3 / 8) + (3 / 5)),
        ],
    )


def test_chair_bootstrap_is_paired_and_deterministic():
    vanilla = np.asarray(
        [
            [1, 1, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 2, 1, 2, 1, 2],
            [1, 1, 1, 2, 1, 2, 1, 2],
        ],
        dtype=np.float64,
    )
    candidate = vanilla.copy()
    candidate[:, 0] = 0
    candidate[:, 2] = 0
    first = paired_bootstrap(
        candidate, vanilla, repeats=1000, seed=17
    )
    second = paired_bootstrap(
        candidate, vanilla, repeats=1000, seed=17
    )
    assert first == second
    assert first["CHAIRs"]["delta_points"] == -100.0
    assert first["CHAIRs"]["ci95_high"] < 0
    assert first["Recall"]["delta_points"] == 0
