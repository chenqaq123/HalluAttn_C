"""Offline tests for the paired absent-object attention experiment.

Covers surface forms, the CLIP resize/crop inverse transform, region token
mapping, pair derivation rules, and first-subtoken identification with a
stub tokenizer. No GPU, model weights, or COCO download is required.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

PIPELINE = Path(__file__).resolve().parents[2] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import build_paired_absent_attention_samples_20261113 as builder


# --- surface forms and caption matching ------------------------------------


def test_surface_forms_basic_and_plural():
    assert builder.surface_forms('cat') == {'cat', 'cats'}
    assert builder.surface_forms('traffic light') == {'traffic light', 'traffic lights'}
    assert builder.surface_forms('skis') == {'skis', 'skiss', 'ski'}


@pytest.mark.parametrize("text,forms,expected", [
    ("a cat on the couch", {'cat', 'cats'}, True),
    ("concatenated strings", {'cat', 'cats'}, False),
    ("two traffic lights", {'traffic light', 'traffic lights'}, True),
    ("a skateboarding dog", {'skateboard', 'skateboards'}, False),
    ("hot dogs and buns", {'hot dog', 'hot dogs'}, True),
])
def test_mentioned_word_boundaries(text, forms, expected):
    assert builder.mentioned(text.lower(), forms) is expected


# --- geometry ---------------------------------------------------------------


def test_resize_output_size_matches_clip_shortest_edge():
    assert builder.resize_output_size(640, 427) == (503, 336)
    assert builder.resize_output_size(427, 640) == (336, 503)
    assert builder.resize_output_size(336, 336) == (336, 336)
    assert builder.resize_output_size(200, 100) == (672, 336)


def test_crop_offsets_center_the_crop():
    assert builder.crop_offsets(503, 336) == (0, (503 - 336) // 2)


def test_patch_center_roundtrip_through_forward_transform():
    width, height = 640, 427
    width_r, height_r = builder.resize_output_size(width, height)
    top, left = builder.crop_offsets(width_r, height_r)
    for index in (0, 137, 575):
        x_original, y_original = builder.patch_center_original(
            index, width, height, width_r, height_r, top, left)
        # Forward transform: resize then center crop.
        x_crop = x_original * width_r / width - left
        y_crop = y_original * height_r / height - top
        row, col = divmod(index, builder.GRID)
        assert col * 14 < x_crop < (col + 1) * 14
        assert row * 14 < y_crop < (row + 1) * 14


def test_instance_mask_polygon_and_bbox_fallback():
    height = width = 100
    polygon = dict(segmentation=[[10, 10, 50, 10, 50, 40, 10, 40]], iscrowd=0)
    mask, fallback = builder.instance_mask(polygon, width, height)
    assert not fallback and mask.shape == (100, 100)
    assert mask[20, 30] and not mask[5, 5] and not mask[60, 60]

    rectangle = dict(bbox=[20, 25, 10, 8])
    mask, fallback = builder.instance_mask(rectangle, width, height)
    assert fallback and mask[30, 25] and not mask[10, 10]

    # Uncompressed column-major RLE: five ones at flat indices 10..14 (col 0).
    rle_like = dict(segmentation=dict(size=[100, 100], counts=[10, 5, 9985]), iscrowd=0)
    mask, fallback = builder.instance_mask(rle_like, width, height)
    assert not fallback and mask.dtype == bool and mask[12, 0] and not mask[9, 0]


def test_region_tokens_center_membership_square_identity_transform():
    # 672x672 resizes to 336x336 (factor 0.5, no crop offsets): the transform
    # is a pure half scale, so a [0,336)^2 rectangle covers columns/rows 0..11.
    width = height = 672
    width_r = height_r = 336
    top = left = 0
    mask = np.zeros((height, width), dtype=bool)
    mask[:336, :336] = True
    indices = builder.region_tokens([mask], width, height, width_r, height_r, top, left)
    assert len(indices) == 144
    for index in indices:
        row, col = divmod(index, 24)
        assert row <= 11 and col <= 11
    assert 0 in indices and all(index not in indices for index in (12, 143, 288, 575))


def test_region_tokens_merges_multiple_masks():
    width = height = 672
    mask_a = np.zeros((height, width), dtype=bool)
    mask_b = np.zeros((height, width), dtype=bool)
    mask_a[:336, :336] = True
    mask_b[336:, 336:] = True
    indices = builder.region_tokens([mask_a, mask_b], width, height, 336, 336, 0, 0)
    assert 0 in indices and 575 in indices and 286 not in indices


# --- pair derivation --------------------------------------------------------


@pytest.fixture()
def synthetic_figure4():
    reasons = {
        # Qualifying contextual co-occurrence (support*lift ranking).
        'traffic light|car': [dict(source='train20k co-occurrence', support=422,
                                   conditional=0.62, lift=5.8)],
        'keyboard|mouse': [dict(source='train20k co-occurrence', support=226,
                                conditional=0.61, lift=38.0)],
        'keyboard|tv': [dict(source='train20k co-occurrence', support=200,
                             conditional=0.54, lift=14.0)],
        # Below thresholds: support, conditional, lift respectively.
        'weak|pair': [dict(source='train20k co-occurrence', support=10,
                           conditional=0.5, lift=9.0)],
        'lowcond|pair': [dict(source='train20k co-occurrence', support=100,
                              conditional=0.1, lift=9.0)],
        'lowlift|pair': [dict(source='train20k co-occurrence', support=100,
                              conditional=0.5, lift=1.5)],
        # Family-only same-supercategory pairs (category similarity).
        'cat|dog': ['fixed semantic/context family'],
        'bird|cat': ['fixed semantic/context family'],
        'horse|zebra': ['fixed semantic/context family'],
        # Family-only but different supercategory or non-COCO target.
        'cat|teddy bear': ['fixed semantic/context family'],
        'book|chair': ['fixed semantic/context family'],
        # Mixed evidence and reverse-direction pairs.
        'bus|car': ['fixed semantic/context family',
                    dict(source='train20k co-occurrence', support=368,
                         conditional=0.55, lift=5.2)],
        'dog|cat': ['fixed semantic/context family'],
    }
    graph = {key.split('|')[0]: [key.split('|')[1]] for key in reasons}
    return dict(relation_reasons=reasons, related_graph=graph)


SUPER = {'traffic light': 'outdoor', 'car': 'vehicle', 'keyboard': 'electronic',
         'mouse': 'electronic', 'tv': 'electronic', 'weak': 'x', 'lowcond': 'x',
         'lowlift': 'x', 'cat': 'animal', 'dog': 'animal', 'bird': 'animal',
         'horse': 'animal', 'zebra': 'animal', 'teddy bear': 'animal',
         'book': 'indoor', 'chair': 'furniture', 'bus': 'vehicle'}


def test_cooc_pairs_filter_and_dedup(synthetic_figure4):
    cooc, _ = builder.derive_pairs(synthetic_figure4, SUPER)
    keys = {(p['T'], p['R']) for p in cooc}
    assert ('keyboard', 'mouse') in keys
    assert ('traffic light', 'car') in keys  # flagship is force-included.
    assert ('weak', 'pair') not in keys
    assert ('lowcond', 'pair') not in keys
    assert ('lowlift', 'pair') not in keys
    assert ('bus', 'car') in keys  # cooc evidence wins over family-only.
    unordered = [tuple(sorted((p['T'], p['R']))) for p in cooc]
    assert len(unordered) == len(set(unordered))
    ranks = [p['rank'] for p in cooc]
    assert ranks == sorted(ranks, reverse=True)


def test_similarity_pairs_same_supercategory_only(synthetic_figure4):
    _, sim = builder.derive_pairs(synthetic_figure4, SUPER)
    keys = {(p['T'], p['R']) for p in sim}
    allowed = {('cat', 'dog'), ('dog', 'cat'), ('bird', 'cat'), ('horse', 'zebra')}
    assert keys - {('bicycle', 'motorcycle')} <= allowed
    assert len(keys - {('bicycle', 'motorcycle')}) == 2  # one stratum, cap two.
    assert ('bicycle', 'motorcycle') in keys  # flagship is force-included.
    assert ('cat', 'teddy bear') not in keys  # teddy bear is not a COCO category.
    assert ('book', 'chair') not in keys  # different supercategory.
    assert ('bus', 'car') not in keys  # has co-occurrence evidence.


def test_control_target_switches_for_animal_domain(synthetic_figure4):
    assert builder.control_target({'T': 'cat'}, SUPER) == 'traffic light'
    assert builder.control_target({'T': 'car'}, SUPER) == 'teddy bear'


# --- first subtoken ---------------------------------------------------------


class StubTokenizer:
    """SentencePiece-like offsets for the fixed question template."""

    def __call__(self, question, add_special_tokens=False, return_offsets_mapping=False):
        if 'traffic' in question:
            keys = [('Is', 0, 2), (' there', 2, 8), (' a', 8, 10),
                    (' traffic', 10, 18), (' light', 18, 24), (' in', 24, 27),
                    (' the', 27, 31), (' image', 31, 37), ('?', 37, 38)]
        else:
            keys = [('Is', 0, 2), (' there', 2, 8), (' a', 8, 10),
                    (' teddy', 10, 17), (' bear', 17, 22), (' in', 24, 27),
                    (' the', 27, 31), (' image', 31, 37), ('?', 37, 38)]
        return dict(input_ids=[100 + i for i in range(len(keys))],
                    offset_mapping=[(a, b) for _, a, b in keys])


def test_first_subtoken_finds_first_word_piece():
    tokenizer = StubTokenizer()
    entry = builder.first_subtoken(tokenizer, 'Is there a traffic light in the image?',
                                   'traffic light')
    assert entry == dict(token_id=103, index_in_question=3)
    entry = builder.first_subtoken(tokenizer, 'Is there a teddy bear in the image?',
                                   'teddy bear')
    assert entry == dict(token_id=103, index_in_question=3)


# --- aggregation statistics -------------------------------------------------


def test_fractions_primary_variant_and_per_layer():
    import aggregate_paired_absent_attention_20261113 as agg
    # Two samples, 2 layers x 3 heads. Sample 0: region=half the visual mass in
    # every (layer, head); sample 1: region=quarter.
    total = np.full((2, 2, 3), 4.0)
    region = np.zeros((2, 2, 3))
    region[0] = 2.0
    region[1] = 1.0
    primary, variant, per_layer = agg.fractions(dict(total=total, region=region))
    assert np.allclose(primary, [0.5, 0.25])
    assert np.allclose(variant, [0.5, 0.25])
    assert per_layer.shape == (2, 2) and np.allclose(per_layer[:, 0], [0.5, 0.25])

    # Fraction-of-averaged-map differs from mean-of-fractions when totals vary.
    total = np.array([[[1.0, 3.0]]])  # one sample, one layer, two heads
    region = np.array([[[1.0, 0.0]]])
    primary, variant, _ = agg.fractions(dict(total=total, region=region))
    assert np.allclose(primary, [1.0 / 4.0])       # averaged map: 1/(1+3)
    assert np.allclose(variant, [(1.0 + 0.0) / 2])  # mean of head fractions = 0.5


def test_paired_bootstrap_ci_and_sign_flip():
    import aggregate_paired_absent_attention_20261113 as agg
    delta = [1.0, 2.0, 3.0, 4.0]
    groups = [10, 10, 20, 20]
    low, high = agg.paired_bootstrap(delta, groups, draws=2000, seed=7)
    assert low < np.mean(delta) < high
    low, high = agg.paired_bootstrap([0.0] * 6, [1] * 6, draws=200, seed=7)
    assert low == 0.0 and high == 0.0
    # Clustered resampling keeps images intact: resampled means are cluster
    # means (5 or -5) or their average (0), never within-cluster values.
    low, high = agg.paired_bootstrap([5.0, 5.0, -5.0, -5.0], [1, 1, 2, 2],
                                     draws=4000, seed=3)
    assert low == -5.0 and high == 5.0

    # n=6 all-positive reaches the minimal two-sided sign-flip p of 2/64 = 0.031.
    assert agg.sign_flip_test([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], draws=500, seed=1) <= 0.05
    assert agg.sign_flip_test([5.0, -5.0, 5.0, -5.0], draws=500, seed=1) > 0.5
