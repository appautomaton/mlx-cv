import numpy as np
import pytest
import mlx.core as mx

from mlx_cv.ops import bilinear_grid_sample_nchw


def test_bilinear_grid_sample_matches_rfdetr_reference_semantics():
    x = mx.array(np.array([[[[1, 2, 3], [4, 5, 6]]]], dtype=np.float32))
    grid = mx.array(np.array([[[[-1, -1], [0, 0], [1, 1], [1.5, 0]]]], dtype=np.float32))
    got = bilinear_grid_sample_nchw(x, grid, padding_mode="zeros", align_corners=False)
    expected = np.array([[[[0.25, 3.5, 1.5, 0.0]]]], dtype=np.float32)
    assert np.allclose(np.array(got), expected, atol=1e-6)


def test_bilinear_grid_sample_border_padding_clamps():
    x = mx.array(np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
    grid = mx.array(np.array([[[[1.5, 1.5]]]], dtype=np.float32))
    got = bilinear_grid_sample_nchw(x, grid, padding_mode="border", align_corners=False)
    assert np.allclose(np.array(got), np.array([[[[4.0]]]], dtype=np.float32), atol=1e-6)


def test_bilinear_grid_sample_shape_errors():
    x = mx.zeros((1, 1, 2, 2))
    with pytest.raises(ValueError, match="grid must have shape"):
        bilinear_grid_sample_nchw(x, mx.zeros((1, 2, 2, 3)))
    with pytest.raises(ValueError, match="input must have shape"):
        bilinear_grid_sample_nchw(mx.zeros((1, 2, 2)), mx.zeros((1, 2, 2, 2)))
    with pytest.raises(ValueError, match="padding_mode"):
        bilinear_grid_sample_nchw(x, mx.zeros((1, 2, 2, 2)), padding_mode="reflection")
