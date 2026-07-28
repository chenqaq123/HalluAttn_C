import torch

from mitigation.src.generation_audit import continuation_audit


class Tokenizer:
    eos_token_id = 2


def test_normal_eos_is_not_premature():
    ids = torch.tensor([[10, 11] + list(range(20, 40)) + [2]])
    audit = continuation_audit(
        ids, 2, Tokenizer(), max_new_tokens=512
    )
    assert audit == {
        "generated_tokens": 21,
        "content_tokens": 20,
        "terminated_by_eos": True,
        "hit_max_new_tokens": False,
        "premature_eos": False,
    }


def test_short_eos_is_premature():
    audit = continuation_audit(
        torch.tensor([[10, 11, 20, 21, 2]]),
        2,
        Tokenizer(),
        max_new_tokens=512,
    )
    assert audit["content_tokens"] == 2
    assert audit["premature_eos"]


def test_length_stop_is_not_eos():
    audit = continuation_audit(
        torch.tensor([[10, 11, 20, 21, 22]]),
        2,
        Tokenizer(),
        max_new_tokens=3,
    )
    assert audit["hit_max_new_tokens"]
    assert not audit["terminated_by_eos"]
    assert not audit["premature_eos"]
