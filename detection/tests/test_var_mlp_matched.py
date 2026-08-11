import numpy as np

from detection.scripts.compare_var_mlp_within_bin import (
    build_representations,
    fixed_image_split,
)


def test_build_representations_uses_post_row_and_registered_layers() -> None:
    raw = np.arange(2 * 32 * 2 * 32, dtype=np.float32).reshape(
        2, 32, 2, 32
    )
    no_rope = raw + 100_000.0
    views = build_representations(raw, no_rope)

    assert views["native_l05_18"].shape == (2, 14 * 32)
    assert views["native_l00_31"].shape == (2, 32 * 32)
    assert views["no_rope_l05_18"].shape == (2, 14 * 32)
    assert views["no_rope_l00_31"].shape == (2, 32 * 32)

    np.testing.assert_array_equal(
        views["native_l05_18"], raw[:, 5:19, 1, :].reshape(2, -1)
    )
    np.testing.assert_array_equal(
        views["no_rope_l00_31"], no_rope[:, :, 1, :].reshape(2, -1)
    )


def test_build_representations_rejects_wrong_shapes() -> None:
    raw = np.zeros((2, 31, 2, 32), dtype=np.float32)
    try:
        build_representations(raw, raw)
    except ValueError as error:
        assert "Unexpected raw mass shape" in str(error)
    else:
        raise AssertionError("wrong layer count must fail")


def test_fixed_image_split_is_shared_deterministic_and_disjoint() -> None:
    image_ids = np.arange(5000, dtype=np.int64)
    train_a, test_a = fixed_image_split(image_ids, 4000, 17)
    train_b, test_b = fixed_image_split(image_ids[::-1], 4000, 17)

    assert train_a.size == 4000
    assert test_a.size == 1000
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
    assert np.intersect1d(train_a, test_a).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((train_a, test_a))), image_ids
    )


def test_fixed_image_split_rejects_duplicate_manifest_ids() -> None:
    try:
        fixed_image_split(np.asarray([1, 1, 2]), 2, 17)
    except ValueError as error:
        assert "must be unique" in str(error)
    else:
        raise AssertionError("duplicate image IDs must fail")
