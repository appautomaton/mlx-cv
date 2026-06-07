import numpy as np
import pytest

from mlx_cv.parity import ParityCase, allclose_tree, assert_parity, bisect


def test_allclose_tree():
    a = {"x": np.zeros(3), "y": [1, 2, 3]}
    assert allclose_tree(a, {"x": np.zeros(3), "y": [1, 2, 3]})
    assert not allclose_tree(a, {"x": np.ones(3), "y": [1, 2, 3]})


def test_assert_parity_raises():
    with pytest.raises(AssertionError):
        assert_parity({"x": np.ones(2)}, {"x": np.zeros(2)}, name="x")


def test_bisect_finds_first_divergence():
    ref = {"a": np.zeros(2), "b": np.zeros(2), "c": np.zeros(2)}
    got = {"a": np.zeros(2), "b": np.ones(2), "c": np.zeros(2)}
    assert bisect(ref, got) == "b"
    assert bisect(ref, ref) is None


def test_paritycase():
    pc = ParityCase(name="t", inputs=1, expected=2)
    assert pc.name == "t" and pc.taps is None
