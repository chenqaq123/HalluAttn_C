from mitigation.scripts.analyze_qwen3vl_grounded_pyramid import (
    multiscale_certificates,
)


def comparison(candidate: str, sources: list[str] | None = None) -> dict:
    return {
        "candidate": candidate,
        "relation": "C",
        "sources": sources or ["open"],
        "keep_margin": 2.0,
        "target_first_margin": -1.0,
        "alternative_first_margin": 1.0,
        "target_preference": -1.0,
    }


def region(proposal: int, scale: int, bbox: list[int], candidate: str) -> dict:
    return {
        "proposal_index": proposal,
        "crop_scale": scale,
        "bbox": bbox,
        "pairwise_comparisons": [comparison(candidate)],
    }


def test_certificate_requires_alias_matched_views_from_same_proposal() -> None:
    row = {
        "regions": [
            region(0, 1, [0, 0, 10, 10], "skis"),
            region(0, 2, [0, 0, 20, 20], "ski"),
            region(1, 4, [0, 0, 40, 40], "ski"),
        ]
    }
    certificates = multiscale_certificates(row, minimum_views=2)
    assert len(certificates) == 1
    assert certificates[0]["candidate"] == "skis"
    assert len(certificates[0]["views"]) == 2


def test_certificate_rejects_closed_only_or_duplicate_pixel_views() -> None:
    duplicate = region(0, 2, [0, 0, 10, 10], "bench")
    closed_only = region(0, 4, [0, 0, 40, 40], "bench")
    closed_only["pairwise_comparisons"] = [comparison("bench", ["closed"])]
    row = {
        "regions": [
            region(0, 1, [0, 0, 10, 10], "bench"),
            duplicate,
            closed_only,
        ]
    }
    assert multiscale_certificates(row, minimum_views=2) == []
