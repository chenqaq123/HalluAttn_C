"""Shared runtime helpers for cached decoding."""
import torch
from .model_adapter import set_attention_implementation

class RuntimeBase:
    def _move(self, inputs):
        return {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
                if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def _set_attention(self, name):
        set_attention_implementation(self.model, name)

    @staticmethod
    def _vision_kwargs(inputs):
        return {k: v for k, v in inputs.items() if k not in {'input_ids', 'attention_mask'}}
