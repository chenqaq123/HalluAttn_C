from __future__ import annotations

import numpy as np
from PIL import Image

from mitigation.src.xev import (
    extract_target,
    make_counterfactual_views,
    parse_candidate_list,
    parse_relation_labels,
    propose_regions,
    random_bbox_like,
    relative_verification_score,
)


def test_extract_target_handles_pope_typo_and_plural() -> None:
    assert extract_target("Is there a skis in the image?") == "skis"
    assert extract_target("Is there an umbrella in the imange?") == "umbrella"


def test_candidate_and_relation_parsers() -> None:
    assert parse_candidate_list("1. Car, 2. traffic light; a person") == [
        "car",
        "traffic light",
        "person",
    ]
    assert parse_relation_labels("1: C\n2: N\n3: S", 3) == ["C", "N", "S"]


def test_region_and_control_keep_canvas_stable() -> None:
    attention = np.zeros((4, 4), dtype=np.float32)
    attention[1, 2] = 1.0
    proposal = propose_regions(attention, (80, 40), mass_fraction=0.3, max_regions=1)[0]
    assert proposal.bbox[0] < proposal.bbox[2]
    assert proposal.bbox[1] < proposal.bbox[3]

    image = Image.new("RGB", (80, 40), "white")
    keep, drop = make_counterfactual_views(image, proposal.bbox)
    assert keep.size == drop.size == image.size
    assert random_bbox_like(proposal.bbox, image.size, seed=7) == random_bbox_like(
        proposal.bbox, image.size, seed=7
    )


def test_relative_score_only_penalizes_positive_alternative_evidence() -> None:
    score, support, penalty = relative_verification_score(1.0, [1.2], [-0.5, 0.4])
    assert abs(score - 0.8) < 1e-9
    assert (support, penalty) == (1.2, 0.4)
