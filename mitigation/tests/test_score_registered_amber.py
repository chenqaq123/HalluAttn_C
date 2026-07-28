import numpy as np

from mitigation.scripts.score_registered_amber import (
    metric_vector,
    paired_bootstrap,
)


def test_metric_vector_matches_amber_ratio_directions():
    values = np.asarray(
        [
            [1, 2, 2, 4, 1, 5, 0, 1],
            [0, 1, 1, 2, 2, 5, 1, 1],
        ],
        dtype=np.float64,
    )
    chair, cover, hal, cog = metric_vector(values, epsilon=0.0)
    assert np.isclose(chair, 1 / 3)
    assert np.isclose(cover, 3 / 6)
    assert np.isclose(hal, 0.5)
    assert np.isclose(cog, 3 / 10)


def test_amber_bootstrap_is_paired_and_deterministic():
    vanilla = np.asarray(
        [
            [1, 1, 1, 2, 1, 2, 0, 1],
            [1, 1, 1, 2, 1, 2, 0, 1],
            [1, 1, 1, 2, 1, 2, 0, 1],
        ],
        dtype=np.float64,
    )
    candidate = vanilla.copy()
    candidate[:, 0] = 0
    candidate[:, 6] = 1
    first = paired_bootstrap(
        candidate, vanilla, repeats=1000, seed=17
    )
    second = paired_bootstrap(
        candidate, vanilla, repeats=1000, seed=17
    )
    assert first == second
    assert first["CHAIR"]["ci95_high"] < 0
    assert first["Hal"]["ci95_high"] < 0
    assert first["Cover"]["delta_points"] == 0
