"""Structural MLP and frozen checkpoint inference."""
import torch
import numpy as np
from torch import nn

class SmallSMLP(nn.Module):
    def __init__(self, features: int, hidden=(128, 32), dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, values):
        return self.net(values).squeeze(-1)

def image_folds(image_ids, folds, seed):
    images = np.unique(image_ids)
    if folds < 3 or len(images) < folds:
        raise ValueError("at least three nonempty image folds are required")
    np.random.default_rng(seed).shuffle(images)
    assignment = {image: index % folds for index, image in enumerate(images)}
    return np.asarray([assignment[image] for image in image_ids], dtype=np.int16)

class FrozenSDetector:
    def __init__(self, payload, device):
        self.mean = payload["mean"].to(device=device, dtype=torch.float32)
        self.scale = payload["scale"].to(device=device, dtype=torch.float32)
        width = int(payload["selected_edges"])
        if self.mean.shape != (width,) or self.scale.shape != (width,):
            raise ValueError("detector normalization width mismatch")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("invalid detector normalization")
        kind = payload.get("classifier_type", "logistic")
        self.model = None
        if kind == "s_mlp_v1":
            self.model = SmallSMLP(width, tuple(payload["hidden"]), float(payload["dropout"])).to(device)
            self.model.load_state_dict(payload["state_dict"], strict=True)
            self.model.eval()
        elif kind == "logistic":
            self.weight = payload["weight"].to(device=device, dtype=torch.float32)
            self.bias = payload["bias"].to(device=device, dtype=torch.float32)
            if self.weight.shape != (width,):
                raise ValueError("detector weight width mismatch")
        else:
            raise ValueError(f"unknown S classifier type: {kind}")

    @torch.inference_mode()
    def __call__(self, s_values):
        values = s_values.half().float()
        values.sub_(self.mean).div_(self.scale)
        logits = self.model(values) if self.model is not None else values @ self.weight + self.bias
        return torch.sigmoid(logits)
