import gc
import weakref
from types import SimpleNamespace

import torch

from mitigation.src.registered_llava_probe import RegisteredLlavaCaptioner


class _Output:
    def __init__(self):
        self.logits = torch.zeros(1, 4, 8)


class _LifetimeModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.output_ref = None

    def __call__(self, **_kwargs):
        output = _Output()
        self.output_ref = weakref.ref(output)
        return output

    def generate(self, *, input_ids, **_kwargs):
        gc.collect()
        assert self.output_ref is not None
        assert self.output_ref() is None
        return torch.cat([input_ids, torch.tensor([[2]])], dim=-1)


def test_llava_rollback_releases_prefill_output_before_generate():
    captioner = RegisteredLlavaCaptioner.__new__(
        RegisteredLlavaCaptioner
    )
    captioner.model = _LifetimeModel()
    captioner.max_new_tokens = 8

    result = captioner.rollback_once(
        {},
        torch.tensor([[10, 11, 12]]),
        {0},
        prompt_length=2,
    )

    assert result.tolist() == [[10, 11, 12, 1, 2]]
