from mitigation.scripts.analyze_qwen3vl_grounded_absence import (
    target_negative_certificates,
)


def region(
    proposal: int,
    scale: int,
    bbox: list[int],
    target_margin: float,
    open_alternative: bool = False,
) -> dict:
    comparisons = []
    if open_alternative:
        comparisons.append(
            {
                "candidate": "truck",
                "relation": "C",
                "sources": ["open"],
                "keep_margin": 2.0,
                "target_first_margin": -1.0,
                "alternative_first_margin": 1.0,
                "target_preference": -1.0,
            }
        )
    return {
        "proposal_index": proposal,
        "crop_scale": scale,
        "bbox": bbox,
        "target_margin": target_margin,
        "pairwise_comparisons": comparisons,
    }


def test_target_negative_certificate_joins_replicated_target_and_one_alternative() -> None:
    row = {
        "regions": [
            region(0, 1, [0, 0, 10, 10], -1.0),
            region(0, 2, [0, 0, 20, 20], -2.0),
            region(0, 4, [0, 0, 40, 40], 1.0, open_alternative=True),
        ]
    }
    certificates = target_negative_certificates(row, minimum_views=2)
    assert len(certificates) == 1
    assert len(certificates[0]["negative_views"]) == 2
    assert certificates[0]["alternatives"][0]["candidate"] == "truck"


def test_target_negative_certificate_cannot_join_evidence_across_boxes() -> None:
    row = {
        "regions": [
            region(0, 1, [0, 0, 10, 10], -1.0),
            region(0, 2, [0, 0, 20, 20], -2.0),
            region(1, 4, [0, 0, 40, 40], 1.0, open_alternative=True),
        ]
    }
    assert target_negative_certificates(row, minimum_views=2) == []
