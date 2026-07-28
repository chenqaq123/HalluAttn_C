from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mitigation.scripts.prepare_open_caption_datasets import (  # noqa: E402
    load_fix_names,
    make_fixname_annotation,
    patch_url,
)


class PrepareOpenCaptionDatasetsTest(unittest.TestCase):
    def test_fix_names_validates_old_name_and_replaces_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "names.csv"
            path.write_text("1,Moniter/TV,Monitor/TV\n")
            mapping = load_fix_names(path)
            fixed = make_fixname_annotation(
                {"images": [], "annotations": [], "categories": [
                    {"id": 1, "name": "Moniter/TV"}
                ]},
                mapping,
            )
            self.assertEqual(fixed["categories"][0]["name"], "Monitor/TV")

    def test_fix_names_rejects_unexpected_annotation(self) -> None:
        with self.assertRaises(ValueError):
            make_fixname_annotation(
                {"categories": [{"id": 1, "name": "Different"}]},
                {1: ("Expected", "Corrected")},
            )

    def test_patch_url_accepts_only_official_layout(self) -> None:
        self.assertTrue(
            patch_url("images/v1/patch0").endswith(
                "/images/v1/patch0.tar.gz"
            )
        )
        with self.assertRaises(ValueError):
            patch_url("../../patch0")


if __name__ == "__main__":
    unittest.main()

