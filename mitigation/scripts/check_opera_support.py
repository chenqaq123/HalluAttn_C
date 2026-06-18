#!/usr/bin/env python3
"""Check whether the active transformers install contains OPERA hooks."""

from __future__ import annotations

import inspect
import json

import transformers
from transformers.generation.utils import GenerationMixin


def main() -> None:
    generate_source = inspect.getsource(GenerationMixin.generate)
    has_generate_flag = "opera_decoding" in generate_source and "key_position" in generate_source
    has_beam_method = hasattr(GenerationMixin, "opera_beam_search")
    beam_source_has_penalty = False
    if has_beam_method:
        beam_source = inspect.getsource(GenerationMixin.opera_beam_search)
        beam_source_has_penalty = "penalty_weights" in beam_source and "num_attn_candidates" in beam_source

    result = {
        "transformers_version": transformers.__version__,
        "generation_mixin_module": GenerationMixin.__module__,
        "supports_opera_generate_flag": has_generate_flag,
        "has_opera_beam_search": has_beam_method,
        "opera_beam_search_has_expected_args": beam_source_has_penalty,
        "ready_for_official_opera": bool(has_generate_flag and has_beam_method and beam_source_has_penalty),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
