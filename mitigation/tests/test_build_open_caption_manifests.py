from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mitigation.scripts.build_open_caption_manifests import _chair_ids


class ManifestTests(unittest.TestCase):
    def test_extracts_registered_chair_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmarks.py"
            path.write_text(
                "class Other:\n    fixed_500 = []\n\n"
                "class CHAIR:\n    fixed_500 = list(range(500))\n"
            )
            # ast.literal_eval intentionally rejects executable list(range());
            # use a literal to exercise the source-audit path.
            path.write_text(
                "class CHAIR:\n    fixed_500 = "
                + repr(list(range(500)))
                + "\n"
            )
            self.assertEqual(_chair_ids(path), list(range(500)))

    def test_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmarks.py"
            path.write_text("class CHAIR:\n    fixed_500 = [1, 2]\n")
            with self.assertRaises(ValueError):
                _chair_ids(path)


if __name__ == "__main__":
    unittest.main()
