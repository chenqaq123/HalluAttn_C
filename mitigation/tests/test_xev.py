from __future__ import annotations

import numpy as np
from PIL import Image

from mitigation.src.xev import (
    combine_global_target_support,
    counterbalanced_pairwise_score,
    extract_target,
    identity_switch_certificate,
    inventory_lexically_supports_target,
    inventory_supports_target,
    make_counterfactual_views,
    pairwise_identity_certificate,
    pairwise_question,
    parse_candidate_list,
    parse_grounding_boxes,
    parse_relation_labels,
    propose_diverse_regions,
    propose_regions,
    random_bbox_like,
    relative_verification_score,
    robust_local_identity_proposal,
    robust_local_target_support,
    select_pairwise_candidates,
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


def test_identity_switch_certificate_requires_open_closed_agreement() -> None:
    certificate = identity_switch_certificate(
        target="handbag",
        open_candidates=["backpack", "person"],
        closed_candidates=["backpack", "suitcase"],
        relations={"backpack": "C", "person": "N"},
        keep_margins={"handbag": -0.75, "backpack": 11.25},
        baseline_margin=3.375,
    )
    assert certificate is not None
    assert certificate["alternative"] == "backpack"

    assert (
        identity_switch_certificate(
            target="handbag",
            open_candidates=["handbag", "backpack"],
            closed_candidates=["backpack"],
            relations={"backpack": "C"},
            keep_margins={"handbag": -0.75, "backpack": 11.25},
            baseline_margin=3.375,
        )
        is None
    )


def test_counterbalanced_pairwise_score_cancels_shared_yes_bias() -> None:
    preference, common_bias = counterbalanced_pairwise_score(5.0, 9.0)
    assert (preference, common_bias) == (-2.0, 7.0)
    assert "handbag" in pairwise_question("Handbag", "Backpack")


def test_pairwise_candidate_selection_is_target_free_and_model_ranked() -> None:
    selected = select_pairwise_candidates(
        target="handbag",
        candidate_sources={
            "open": ["backpack", "person"],
            "closed": ["suitcase", "backpack"],
            "oracle": ["car"],
        },
        relations={"backpack": "C", "person": "N", "suitcase": "C"},
        keep_margins={"backpack": 8.0, "person": 3.0, "suitcase": -1.0},
        limit=2,
    )
    assert [row["candidate"] for row in selected] == ["backpack", "person"]
    assert selected[0]["sources"] == ["closed", "open"]

    guarded = select_pairwise_candidates(
        target="handbag",
        candidate_sources={"open": ["handbags", "backpack"], "closed": []},
        relations={"backpack": "C"},
        keep_margins={"backpack": 8.0},
    )
    assert guarded == []


def test_pairwise_identity_certificate_uses_natural_sign_gates() -> None:
    comparisons = [
        {"candidate": "paper", "relation": "N", "target_preference": -5.0},
        {"candidate": "backpack", "relation": "C", "target_preference": -2.0},
    ]
    certificate = pairwise_identity_certificate(3.0, -0.75, comparisons)
    assert certificate is not None
    assert certificate["candidate"] == "backpack"
    assert pairwise_identity_certificate(3.0, 0.25, comparisons) is None


def test_global_guard_separates_local_proposal_from_target_support() -> None:
    comparisons = [
        {
            "candidate": "backpack",
            "relation": "C",
            "sources": ["closed", "open"],
            "target_preference": -2.0,
        },
        {
            "candidate": "person",
            "relation": "N",
            "sources": ["open"],
            "target_preference": -5.0,
        },
    ]
    assert robust_local_identity_proposal(comparisons)["candidate"] == "backpack"
    assert inventory_supports_target("handbag", ["purse"], {"purse": "S"})
    assert not inventory_supports_target(
        "handbag", ["backpack"], {"backpack": "C"}
    )
    closed_only = robust_local_identity_proposal(
        [
            {
                "candidate": "chair",
                "relation": "C",
                "sources": ["closed"],
                "target_preference": -1.0,
            }
        ],
        required_sources=("closed",),
    )
    assert closed_only["candidate"] == "chair"


def test_directional_switch_and_alias_support_are_strict() -> None:
    weak = {
        "candidate": "suitcase",
        "relation": "C",
        "sources": ["closed", "open"],
        "target_first_margin": 0.0,
        "alternative_first_margin": 0.75,
        "target_preference": -0.375,
    }
    strong = {
        **weak,
        "target_first_margin": -2.0,
        "alternative_first_margin": 3.0,
        "target_preference": -2.5,
    }
    assert robust_local_identity_proposal(
        [weak], require_directional_switch=True
    ) is None
    assert robust_local_identity_proposal(
        [strong], require_directional_switch=True
    ) == strong
    assert inventory_lexically_supports_target("dining table", ["table"])
    assert not inventory_lexically_supports_target("handbag", ["bag"])


def test_local_identity_proposal_can_accept_unrelated_region_identity() -> None:
    comparison = {
        "candidate": "person",
        "relation": "N",
        "sources": ["open"],
        "target_preference": -4.0,
        "target_first_margin": -5.0,
        "alternative_first_margin": 3.0,
    }
    assert robust_local_identity_proposal([comparison], required_sources=()) is None
    proposal = robust_local_identity_proposal(
        [comparison],
        required_sources=(),
        require_directional_switch=True,
        allowed_relations=("C", "N"),
    )
    assert proposal == comparison


def test_global_target_support_can_union_complement_and_inventories() -> None:
    assert combine_global_target_support(False, 0.25, "either") == (True, True)
    assert combine_global_target_support(True, -0.25, "either") == (True, False)
    assert combine_global_target_support(False, -0.25, "either") == (False, False)


def test_diverse_regions_suppress_previous_attention_box() -> None:
    attention = np.zeros((8, 8), dtype=np.float32)
    attention[1, 1] = 0.6
    attention[6, 6] = 0.3
    attention[1, 6] = 0.1
    proposals = propose_diverse_regions(
        attention,
        (80, 80),
        mass_fraction=0.3,
        max_regions=3,
        dilation=0,
    )
    assert [proposal.grid_bbox for proposal in proposals] == [
        (1, 1, 2, 2),
        (6, 6, 7, 7),
        (6, 1, 7, 2),
    ]
    assert abs(sum(proposal.attention_mass for proposal in proposals) - 1.0) < 1e-6


def test_regional_target_support_mirrors_directional_conflict() -> None:
    target_support = {
        "candidate": "person",
        "relation": "N",
        "sources": ["open"],
        "target_preference": 4.0,
        "target_first_margin": 5.0,
        "alternative_first_margin": -3.0,
    }
    ambiguous = {
        **target_support,
        "target_first_margin": 5.0,
        "alternative_first_margin": 1.0,
    }
    assert robust_local_target_support([target_support]) == target_support
    assert robust_local_target_support([ambiguous]) is None


def test_grounding_box_parser_uses_normalized_json_and_discards_labels() -> None:
    raw = '''```json
    [
      {"bbox_2d": [100, 200, 600, 800], "label": "wrong label"},
      {"bbox_2d": [-1, 0, 50, 50], "label": "invalid"}
    ]
    ```'''
    assert parse_grounding_boxes(raw, (200, 100)) == [(20, 20, 120, 80)]
    assert parse_grounding_boxes("no box found", (200, 100)) == []
