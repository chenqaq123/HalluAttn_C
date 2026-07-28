import unittest

import numpy as np

from mitigation.src.allpath_identification import (
    _llava_unpadded_shape,
    aggregate_head_scores,
    overlap_grid,
    proportional_head_counts,
    selected_head_payload,
)


class AllPathIdentificationTest(unittest.TestCase):
    def test_overlap_grid_uses_any_instance_overlap(self):
        mask = np.zeros((4, 6), dtype=bool)
        mask[0, 0] = True
        mask[3, 5] = True
        result = overlap_grid(mask, 2, 3)
        np.testing.assert_array_equal(
            result,
            np.asarray(
                [[True, False, False], [False, False, True]]
            ),
        )

    def test_llava_unpadding_mirrors_symmetric_crop(self):
        self.assertEqual(_llava_unpadded_shape(100, 200, 48, 48), (24, 48))
        self.assertEqual(_llava_unpadded_shape(200, 100, 48, 48), (48, 24))

    def test_official_safe_minus_hallucinated_mean(self):
        safe = np.asarray([[4.0, 8.0], [12.0, 16.0]])
        hallu = np.asarray([[3.0, 6.0], [9.0, 12.0]])
        score = aggregate_head_scores(safe, 2, hallu, 3)
        np.testing.assert_allclose(
            score, np.asarray([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_counts_are_rounded_once_per_architecture(self):
        self.assertEqual(
            proportional_head_counts(784),
            {
                "good_format": 31,
                "hallu_format": 31,
                "good_image": 38,
                "hallu_image": 0,
            },
        )
        self.assertEqual(proportional_head_counts(1024)["good_image"], 50)

    def test_selection_uses_low_format_and_high_format_image(self):
        format_scores = np.arange(16, dtype=float).reshape(4, 4)
        image_scores = -format_scores
        payload = selected_head_payload(format_scores, image_scores)
        # A toy 16-head architecture rounds all registered counts to one.
        self.assertEqual(payload["counts"]["hallu_format"], 1)
        self.assertEqual(payload["counts"]["good_format"], 1)
        self.assertEqual(payload["counts"]["good_image"], 1)
        self.assertEqual(
            payload["selections"]["hallu_format"][0]["score"], 0.0
        )
        self.assertEqual(
            payload["selections"]["good_format"][0]["score"], 15.0
        )
        self.assertEqual(
            payload["selections"]["good_image"][0]["score"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
