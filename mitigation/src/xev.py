"""Core utilities for region-grounded target-vs-alternative verification."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


_TARGET_RE = re.compile(
    r"^\s*is there\s+(?:a|an|any)\s+(.+?)\s+in\s+the\s+ima(?:ge|nge)\s*\?\s*$",
    re.IGNORECASE,
)
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_TARGET_ALIASES = {
    "airplane": {"aeroplane", "plane"},
    "backpack": {"rucksack"},
    "bicycle": {"bike"},
    "cell phone": {"mobile phone", "phone"},
    "couch": {"sofa"},
    "dining table": {"table"},
    "fire hydrant": {"hydrant"},
    "hair drier": {"hair dryer"},
    "handbag": {"purse"},
    "motorcycle": {"motorbike"},
    "potted plant": {"plant"},
    "refrigerator": {"fridge"},
    "remote": {"remote control"},
    "sports ball": {"ball"},
    "teddy bear": {"teddy"},
    "traffic light": {"signal light", "stoplight"},
    "tv": {"television"},
    "wine glass": {"goblet"},
}


@dataclass(frozen=True)
class RegionProposal:
    """A rectangular proposal expressed in image pixels and attention-grid cells."""

    bbox: tuple[int, int, int, int]
    grid_bbox: tuple[int, int, int, int]
    attention_mass: float


def extract_target(question: str) -> str:
    """Extract the queried object phrase from a POPE existence question."""

    match = _TARGET_RE.match(question)
    if not match:
        raise ValueError(f"Unsupported POPE question format: {question!r}")
    return normalize_phrase(match.group(1))


def normalize_phrase(phrase: str) -> str:
    phrase = phrase.strip().lower().replace("_", " ")
    phrase = _LEADING_LIST_MARKER_RE.sub("", phrase)
    phrase = _LEADING_ARTICLE_RE.sub("", phrase)
    phrase = re.sub(r"[^a-z0-9' -]+", " ", phrase)
    return re.sub(r"\s+", " ", phrase).strip(" -")


def lexical_equivalent(left: str, right: str) -> bool:
    """Conservative equivalence used before model-based relation filtering."""

    left = normalize_phrase(left)
    right = normalize_phrase(right)
    if left == right:
        return True

    def singularize(value: str) -> str:
        words = value.split()
        if not words:
            return value
        last = words[-1]
        if last.endswith("ies") and len(last) > 3:
            last = last[:-3] + "y"
        elif last.endswith("ses") and len(last) > 3:
            last = last[:-2]
        elif last.endswith("s") and not last.endswith("ss") and len(last) > 2:
            last = last[:-1]
        return " ".join([*words[:-1], last])

    return singularize(left) == singularize(right)


def parse_candidate_list(text: str, limit: int = 8) -> list[str]:
    """Parse a short model-generated object inventory without guessing relations."""

    text = re.sub(r"```(?:json|text)?", "", text, flags=re.IGNORECASE).replace(
        "```", ""
    )
    pieces = re.split(r"[,;\n|]+", text)
    candidates: list[str] = []
    for piece in pieces:
        piece = (
            re.sub(r"^.*?:", "", piece) if ":" in piece and len(pieces) == 1 else piece
        )
        phrase = normalize_phrase(piece)
        phrase = re.sub(r"\b(?:visible|object|objects|none)\b", "", phrase).strip()
        if not phrase or phrase in {"n a", "no clear object", "no identifiable object"}:
            continue
        if len(phrase.split()) > 5:
            continue
        if phrase not in candidates:
            candidates.append(phrase)
        if len(candidates) >= limit:
            break
    return candidates


def parse_relation_labels(text: str, count: int) -> list[str] | None:
    """Parse indexed S/C/N labels returned by the semantic relation prompt."""

    indexed: dict[int, str] = {}
    for index, label in re.findall(r"(?im)^\s*(\d+)\s*[.):=-]\s*([SCN])\b", text):
        indexed[int(index)] = label.upper()
    if all(index in indexed for index in range(1, count + 1)):
        return [indexed[index] for index in range(1, count + 1)]

    labels = [
        label.upper() for label in re.findall(r"\b([SCN])\b", text, re.IGNORECASE)
    ]
    return labels[:count] if len(labels) >= count else None


def propose_regions(
    attention: np.ndarray,
    image_size: tuple[int, int],
    mass_fraction: float = 0.30,
    max_regions: int = 2,
    dilation: int = 1,
) -> list[RegionProposal]:
    """Turn an attention map into a few connected, rectangular region proposals."""

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Expected a non-empty 2D attention map, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0:
        values = np.ones_like(values)
        total = float(values.sum())
    values /= total

    flat_order = np.argsort(values.ravel())[::-1]
    cumulative = np.cumsum(values.ravel()[flat_order])
    selected_count = int(np.searchsorted(cumulative, mass_fraction, side="left")) + 1
    selected = np.zeros(values.size, dtype=bool)
    selected[flat_order[:selected_count]] = True
    selected = selected.reshape(values.shape)
    if dilation > 0:
        selected = ndimage.binary_dilation(selected, iterations=dilation)

    labeled, component_count = ndimage.label(selected)
    components: list[tuple[float, tuple[int, int, int, int]]] = []
    for component_id in range(1, component_count + 1):
        ys, xs = np.where(labeled == component_id)
        if not len(xs):
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        original_mass = float(values[labeled == component_id].sum())
        components.append((original_mass, bbox))

    if not components:
        peak_y, peak_x = np.unravel_index(int(values.argmax()), values.shape)
        components = [
            (float(values[peak_y, peak_x]), (peak_x, peak_y, peak_x + 1, peak_y + 1))
        ]
    components.sort(reverse=True)

    image_width, image_height = image_size
    grid_height, grid_width = values.shape
    proposals = []
    for mass, (x0, y0, x1, y1) in components[:max_regions]:
        pixel_bbox = (
            max(0, math.floor(x0 * image_width / grid_width)),
            max(0, math.floor(y0 * image_height / grid_height)),
            min(image_width, math.ceil(x1 * image_width / grid_width)),
            min(image_height, math.ceil(y1 * image_height / grid_height)),
        )
        proposals.append(RegionProposal(pixel_bbox, (x0, y0, x1, y1), mass))
    return proposals


def propose_diverse_regions(
    attention: np.ndarray,
    image_size: tuple[int, int],
    mass_fraction: float = 0.30,
    max_regions: int = 3,
    dilation: int = 1,
) -> list[RegionProposal]:
    """Extract spatially distinct proposals by suppressing each selected box."""

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"Expected a non-empty 2D attention map, got {values.shape}")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if float(values.sum()) <= 0:
        values = np.ones_like(values)
    normalized = values / float(values.sum())
    residual = normalized.copy()
    proposals: list[RegionProposal] = []

    for _ in range(max(0, max_regions)):
        if float(residual.sum()) <= 0:
            break
        proposal = propose_regions(
            residual,
            image_size,
            mass_fraction=mass_fraction,
            max_regions=1,
            dilation=dilation,
        )[0]
        x0, y0, x1, y1 = proposal.grid_bbox
        original_mass = float(normalized[y0:y1, x0:x1].sum())
        proposals.append(
            RegionProposal(
                bbox=proposal.bbox,
                grid_bbox=proposal.grid_bbox,
                attention_mass=original_mass,
            )
        )

        padding = max(0, dilation)
        sx0 = max(0, x0 - padding)
        sy0 = max(0, y0 - padding)
        sx1 = min(residual.shape[1], x1 + padding)
        sy1 = min(residual.shape[0], y1 + padding)
        residual[sy0:sy1, sx0:sx1] = 0.0

    return proposals


def make_counterfactual_views(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    blur_fraction: float = 0.04,
    feather_fraction: float = 0.012,
) -> tuple[Image.Image, Image.Image]:
    """Create matched keep-region and drop-region views on an unchanged canvas."""

    image = image.convert("RGB")
    width, height = image.size
    blur_radius = max(4.0, min(width, height) * blur_fraction)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_array = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    mask_array[y0:y1, x0:x1] = 255
    mask = Image.fromarray(mask_array, mode="L")
    feather_radius = max(1.0, min(width, height) * feather_fraction)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    keep = Image.composite(image, blurred, mask)
    drop = Image.composite(blurred, image, mask)
    return keep, drop


def random_bbox_like(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int], seed: int
) -> tuple[int, int, int, int]:
    """Sample a deterministic control box with the same width and height."""

    width, height = image_size
    box_width = min(width, max(1, bbox[2] - bbox[0]))
    box_height = min(height, max(1, bbox[3] - bbox[1]))
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, width - box_width + 1))
    y0 = int(rng.integers(0, height - box_height + 1))
    return x0, y0, x0 + box_width, y0 + box_height


def evidence_margin(keep_margin: float, drop_margin: float) -> float:
    return float(keep_margin - drop_margin)


def relative_verification_score(
    target_evidence: float,
    support_evidence: list[float],
    alternative_evidence: list[float],
) -> tuple[float, float, float]:
    """Return score, support term, and non-negative alternative penalty."""

    support = max([float(target_evidence), *map(float, support_evidence)])
    penalty = max([0.0, *map(float, alternative_evidence)])
    return support - penalty, support, penalty


def pairwise_question(left: str, right: str) -> str:
    """Build one direction of a region-local identity comparison."""

    left = normalize_phrase(left)
    right = normalize_phrase(right)
    return (
        "The image has one clear region and blurred context. Focus only on the "
        "main object identity in the clear region. Is that object better identified "
        f"as {left} than as {right}? Please answer Yes or No only."
    )


def select_pairwise_candidates(
    target: str,
    candidate_sources: dict[str, list[str]],
    relations: dict[str, str],
    keep_margins: dict[str, float],
    limit: int = 3,
    block_on_open_target: bool = True,
) -> list[dict]:
    """Rank target-free open/closed proposals for explicit visual comparison."""

    if limit <= 0:
        return []
    open_candidates = candidate_sources.get("open", [])
    if block_on_open_target and any(
        lexical_equivalent(target, candidate) for candidate in open_candidates
    ):
        return []

    source_names: dict[str, set[str]] = {}
    for source in ("open", "closed"):
        for candidate in candidate_sources.get(source, []):
            candidate = normalize_phrase(candidate)
            if not candidate or lexical_equivalent(target, candidate):
                continue
            source_names.setdefault(candidate, set()).add(source)

    ranked = []
    for candidate, sources in source_names.items():
        if relations.get(candidate) == "S" or candidate not in keep_margins:
            continue
        keep_margin = float(keep_margins[candidate])
        if keep_margin <= 0:
            continue
        ranked.append(
            {
                "candidate": candidate,
                "keep_margin": keep_margin,
                "sources": sorted(sources),
                "relation": relations.get(candidate, "N"),
            }
        )
    ranked.sort(key=lambda item: (-item["keep_margin"], item["candidate"]))
    return ranked[:limit]


def counterbalanced_pairwise_score(
    target_first_margin: float, alternative_first_margin: float
) -> tuple[float, float]:
    """Return target preference and shared yes-bias from reversed pair prompts."""

    target_first_margin = float(target_first_margin)
    alternative_first_margin = float(alternative_first_margin)
    preference = 0.5 * (target_first_margin - alternative_first_margin)
    common_bias = 0.5 * (target_first_margin + alternative_first_margin)
    return preference, common_bias


def pairwise_identity_certificate(
    baseline_margin: float,
    target_keep_margin: float,
    comparisons: list[dict],
) -> dict | None:
    """Certify a local identity switch using three zero-threshold checks."""

    if baseline_margin < 0 or target_keep_margin >= 0:
        return None
    alternatives = [
        comparison
        for comparison in comparisons
        if comparison.get("relation") == "C"
        and float(comparison["target_preference"]) < 0
    ]
    if not alternatives:
        return None
    best = min(alternatives, key=lambda item: float(item["target_preference"]))
    return {
        **best,
        "target_keep_margin": float(target_keep_margin),
    }


def robust_local_identity_proposal(
    comparisons: list[dict],
    required_sources: tuple[str, ...] = ("open", "closed"),
    require_directional_switch: bool = False,
    allowed_relations: tuple[str, ...] = ("C",),
) -> dict | None:
    """Return a local alternative meeting the requested provenance constraint."""

    required = set(required_sources)
    allowed = set(allowed_relations)
    alternatives = [
        comparison
        for comparison in comparisons
        if comparison.get("relation") in allowed
        and required.issubset(comparison.get("sources", []))
        and float(comparison["target_preference"]) < 0
        and (
            not require_directional_switch
            or (
                float(comparison["target_first_margin"]) < 0
                and float(comparison["alternative_first_margin"]) > 0
            )
        )
    ]
    if not alternatives:
        return None
    return min(alternatives, key=lambda item: float(item["target_preference"]))


def robust_local_target_support(
    comparisons: list[dict],
    required_sources: tuple[str, ...] = (),
    allowed_relations: tuple[str, ...] = ("C", "N"),
) -> dict | None:
    """Return a region-level target identity supported in both prompt orders."""

    required = set(required_sources)
    allowed = set(allowed_relations)
    supports = [
        comparison
        for comparison in comparisons
        if comparison.get("relation") in allowed
        and required.issubset(comparison.get("sources", []))
        and float(comparison["target_first_margin"]) > 0
        and float(comparison["alternative_first_margin"]) < 0
    ]
    if not supports:
        return None
    return max(supports, key=lambda item: float(item["target_preference"]))


def combine_global_target_support(
    inventory_support: bool,
    complement_margin: float,
    source: str = "inventories",
) -> tuple[bool, bool]:
    """Combine independent full-image and region-complement target support."""

    complement_support = float(complement_margin) >= 0
    if source == "inventories":
        support = bool(inventory_support)
    elif source == "complement":
        support = complement_support
    elif source == "either":
        support = bool(inventory_support) or complement_support
    else:
        raise ValueError(f"Unknown global support source: {source}")
    return support, complement_support


def inventory_supports_target(
    target: str, candidates: list[str], relations: dict[str, str]
) -> bool:
    """Check a target-free inventory for exact or semantic target support."""

    return any(
        lexical_equivalent(target, candidate) or relations.get(candidate) == "S"
        for candidate in candidates
    )


def inventory_lexically_supports_target(target: str, candidates: list[str]) -> bool:
    """Match target-free inventory phrases using an auditable alias table."""

    target = normalize_phrase(target)
    aliases = {target, *_TARGET_ALIASES.get(target, set())}
    return any(
        any(lexical_equivalent(alias, candidate) for alias in aliases)
        for candidate in candidates
    )


def identity_switch_certificate(
    target: str,
    open_candidates: list[str],
    closed_candidates: list[str],
    relations: dict[str, str],
    keep_margins: dict[str, float],
    baseline_margin: float,
    min_excess: float = 6.0,
) -> dict | None:
    """Return a high-precision certificate that a visible peer outranks target."""

    if baseline_margin < 0:
        return None
    if any(lexical_equivalent(target, candidate) for candidate in open_candidates):
        return None

    closed = set(closed_candidates)
    alternatives = [
        candidate
        for candidate in open_candidates
        if candidate in closed
        and relations.get(candidate) == "C"
        and candidate in keep_margins
    ]
    if not alternatives or target not in keep_margins:
        return None

    alternative = max(alternatives, key=lambda candidate: keep_margins[candidate])
    alternative_margin = float(keep_margins[alternative])
    target_margin = float(keep_margins[target])
    excess = alternative_margin - target_margin
    if alternative_margin <= 0 or excess < min_excess:
        return None
    return {
        "alternative": alternative,
        "alternative_margin": alternative_margin,
        "target_margin": target_margin,
        "excess": excess,
        "min_excess": float(min_excess),
    }


def load_json(path: str | Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
