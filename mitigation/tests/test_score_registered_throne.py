import numpy as np

from mitigation.scripts.score_registered_throne import (
    metric_vector,
    paired_bootstrap,
)


def test_throne_metric_matches_classwise_and_micro_definitions():
    predicted = np.asarray(
        [[1, 0], [1, 1], [0, 0]], dtype=bool
    )
    gt = np.asarray([[1, 1], [0, 1], [1, 0]], dtype=bool)
    correct = predicted & gt
    values = metric_vector(predicted, correct, gt)
    class_precision = np.asarray([0.5, 1.0])
    class_recall = np.asarray([0.5, 0.5])
    expected_class_f05 = np.mean(
        1.25
        * class_precision
        * class_recall
        / (0.25 * class_precision + class_recall)
    )
    assert np.isclose(values[1], expected_class_f05)
    # Third image has no prediction, so official micro P/R are both zero;
    # its undefined F score is excluded by nanmean.
    assert np.isclose(values[6], np.mean([1.0, 0.5, 0.0]))
    assert np.isclose(values[7], np.mean([0.5, 1.0, 0.0]))


def test_throne_paired_bootstrap_is_deterministic():
    vanilla_pred = np.asarray(
        [[1, 1], [1, 1], [1, 1], [1, 1]], dtype=bool
    )
    gt = np.asarray(
        [[1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool
    )
    candidate_pred = np.asarray(
        [[1, 0], [1, 0], [1, 1], [1, 1]], dtype=bool
    )
    vanilla = (vanilla_pred, vanilla_pred & gt, gt)
    candidate = (candidate_pred, candidate_pred & gt, gt)
    first = paired_bootstrap(
        candidate, vanilla, repeats=200, chunk=20, seed=17
    )
    second = paired_bootstrap(
        candidate, vanilla, repeats=200, chunk=20, seed=17
    )
    assert first == second
    assert first["classwise_F0.5"]["delta_points"] > 0
    assert first["classwise_P"]["delta_points"] > 0
    assert first["classwise_P"]["ci95_high"] > 0
