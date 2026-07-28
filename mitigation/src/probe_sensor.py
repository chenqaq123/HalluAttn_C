"""Serializable detector ensembles used by registered rollback experiments."""

from __future__ import annotations

import numpy as np
import torch


class MeanProbeSensor:
    def __init__(self, path: str, device: torch.device):
        from detection.scripts.evaluate_visual_value_evidence import _Probe

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.models = []
        self.moments = []
        self.device = device
        for record in payload["folds"]:
            model = _Probe(
                payload["input_size"], payload["hidden_size"]
            ).to(device)
            model.load_state_dict(record["state_dict"])
            model.eval()
            self.models.append(model)
            self.moments.append(
                (
                    torch.from_numpy(record["mean"]).to(device),
                    torch.from_numpy(record["std"]).to(device),
                )
            )

    def score(self, features: torch.Tensor) -> float:
        features = features.to(self.device)
        with torch.inference_mode():
            values = [
                float(model((features - mean) / std).item())
                for model, (mean, std) in zip(
                    self.models, self.moments
                )
            ]
        return float(np.mean(values))


class OOFProbeSensor:
    """Route a registered sample index to the fold that held it out."""

    def __init__(self, path: str, device: torch.device):
        from detection.scripts.evaluate_visual_value_evidence import _Probe

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.by_sample: dict[int, tuple] = {}
        self.device = device
        for record in payload["folds"]:
            model = _Probe(
                payload["input_size"], payload["hidden_size"]
            ).to(device)
            model.load_state_dict(record["state_dict"])
            model.eval()
            mean = torch.from_numpy(record["mean"]).to(device)
            std = torch.from_numpy(record["std"]).to(device)
            for sample in record["test_images"].tolist():
                self.by_sample[int(sample)] = (model, mean, std)

    def score(self, sample_index: int, features: torch.Tensor) -> float:
        model, mean, std = self.by_sample[int(sample_index)]
        features = features.to(self.device)
        with torch.inference_mode():
            return float(model((features - mean) / std).item())
