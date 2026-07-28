from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from detection.scripts.evaluate_visual_value_evidence import (
    _Probe,
    _image_folds,
)
from mitigation.scripts.audit_disjoint_caption_sensor import audit_sensor


class DisjointCaptionSensorAuditTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> tuple[str, Path, Path]:
        features_path = root / "features.npz"
        image_ids = np.arange(1, 7, dtype=np.int64)
        labels = np.zeros(6, dtype=np.int32)
        no_rope = np.zeros((6, 1, 2, 1, 1), dtype=np.float32)
        np.savez_compressed(
            features_path,
            object_id=image_ids * 128,
            image_id=image_ids,
            label=labels,
            gen_pos=np.arange(6, dtype=np.int32),
            no_rope=no_rope,
        )
        exclude = root / "exclude.jsonl"
        exclude.write_text('{"sample_id": 99}\n')

        seed = 17
        permuted = image_ids[
            np.random.default_rng(seed).permutation(len(image_ids))
        ]
        calibration_images = permuted[:2]
        training_images = permuted[2:]
        folds = _image_folds(training_images, 2, seed)
        records = []
        for heldout in folds:
            model = _Probe(1, 2)
            for parameter in model.parameters():
                parameter.data.zero_()
            records.append(
                {
                    "state_dict": model.state_dict(),
                    "mean": np.zeros(1, dtype=np.float32),
                    "std": np.ones(1, dtype=np.float32),
                    "test_images": heldout,
                }
            )
        sensor = root / "sensor.pt"
        torch.save(
            {
                "protocol": "test",
                "feature": "no_rope__mass_post",
                "input_size": 1,
                "hidden_size": 2,
                "folds": records,
                "config": {
                    "seed": seed,
                    "calibration_fraction": 1 / 3,
                    "folds": 2,
                    "epochs": 1,
                    "exclude_manifest": str(exclude),
                    "features_glob": str(features_path),
                    "hidden_size": 2,
                    "grounded_trigger_rate": 0.02,
                },
            },
            sensor,
        )
        thresholds = root / "thresholds.json"
        thresholds.write_text(
            json.dumps(
                {
                    "tau_risk": 0.0,
                    "target_grounded_trigger_rate": 0.02,
                    "realized_grounded_trigger_rate": 1.0,
                    "all_candidate_trigger_rate": 1.0,
                    "source_mentions": 6,
                    "source_images": 6,
                    "training_images": 4,
                    "training_mentions": 4,
                    "calibration_images": 2,
                    "calibration_mentions": 2,
                    "calibration_grounded_mentions": 2,
                    "excluded_eval_images_with_features": 0,
                    "excluded_eval_ids": [],
                    "seed": seed,
                }
            )
        )
        return str(features_path), exclude, sensor, thresholds

    def test_recomputed_threshold_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, exclude, sensor, thresholds = self._artifacts(root)
            result = audit_sensor(
                features, exclude, sensor, thresholds
            )
            self.assertTrue(result["valid"])
            self.assertTrue(
                result[
                    "recomputed_threshold_matches_within_tolerance"
                ]
            )
            self.assertEqual(result["folds"], 2)

    def test_threshold_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features, exclude, sensor, thresholds = self._artifacts(root)
            payload = json.loads(thresholds.read_text())
            payload["tau_risk"] = 0.6
            thresholds.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "tau_risk"):
                audit_sensor(features, exclude, sensor, thresholds)


if __name__ == "__main__":
    unittest.main()
