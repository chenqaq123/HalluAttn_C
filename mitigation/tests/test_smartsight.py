from __future__ import annotations

import unittest

import numpy as np

from mitigation.src.smartsight import TemporalTrace, score_trace


class SmartSightTests(unittest.TestCase):
    def test_uniform_attention_has_higher_tac_than_collapsed(self):
        segments = [[0, 1], [2, 3]]
        text = np.ones(12)
        visual = np.ones(12)
        uniform = TemporalTrace(
            frame_attention=np.ones((12, 4)),
            visual_attention=visual,
            generated_text_attention=text,
        )
        collapsed_values = np.zeros((12, 4))
        collapsed_values[:, 0] = 1
        collapsed = TemporalTrace(
            frame_attention=collapsed_values,
            visual_attention=visual,
            generated_text_attention=text,
        )
        self.assertGreater(
            score_trace(uniform, segments)["tac"],
            score_trace(collapsed, segments)["tac"],
        )

    def test_vav_requires_consecutive_window(self):
        trace = TemporalTrace(
            frame_attention=np.ones((6, 2)),
            visual_attention=np.asarray([2.0, 0.5, 0.5, 2.0, 0.5, 0.5]),
            generated_text_attention=np.ones(6),
        )
        result = score_trace(
            trace, [[0], [1]], alpha=1.2, window=2
        )
        self.assertEqual(result["vav_index"], 2)


if __name__ == "__main__":
    unittest.main()
