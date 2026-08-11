import numpy as np

from detection.scripts.compare_var_mlp_within_bin import (
    build_representations,
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
