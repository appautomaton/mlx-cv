import numpy as np

from mlx_cv import SpatialTransform


def test_resize_box_roundtrip():
    t = SpatialTransform.resize((480, 640), (240, 320))
    boxes = np.array([[10, 20, 100, 200], [0, 0, 640, 480]], float)
    assert np.allclose(t.invert_boxes(t.apply_boxes(boxes)), boxes)


def test_letterbox_point_roundtrip():
    t = SpatialTransform.letterbox((480, 640), (512, 512))
    pts = np.array([[0, 0], [640, 480], [123, 456]], float)
    assert np.allclose(t.invert_points(t.apply_points(pts)), pts)


def test_letterbox_pads_short_axis():
    # wide image (H=100, W=200) into a square -> uniform scale, vertical padding
    t = SpatialTransform.letterbox((100, 200), (200, 200))
    assert t.scale == (1.0, 1.0)
    m = t.apply_points([[0, 0]])
    assert m[0, 0] == 0 and m[0, 1] > 0  # x flush, y padded


def test_crop_roundtrip():
    t = SpatialTransform.crop((100, 100), (10, 20, 60, 90))
    boxes = np.array([[0, 0, 50, 70]], float)
    assert np.allclose(t.invert_boxes(t.apply_boxes(boxes)), boxes)


def test_invert_clip():
    t = SpatialTransform.resize((100, 100), (50, 50))
    b = t.invert_boxes([[-10, -10, 200, 200]], clip=True)
    assert b[0, 0] >= 0 and b[0, 1] >= 0 and b[0, 2] <= 100 and b[0, 3] <= 100


# -- dense resampling (Slice 2) --------------------------------------------

def _ramp(h, w):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return (2.0 * xx + 3.0 * yy + 1.0).astype(np.float64)   # linear field


def test_apply_dense_shapes_model_space():
    t = SpatialTransform.resize((100, 80), (50, 40))
    out = t.apply_dense(_ramp(100, 80), mode="bilinear")
    assert out.shape == (50, 40)            # model_size (H, W)


def test_depth_bilinear_roundtrip_interior():
    # bilinear is exact on a linear field -> orig->model->orig matches in-domain
    t = SpatialTransform.resize((100, 100), (50, 50))
    depth = _ramp(100, 100)
    back = t.invert_depth(t.apply_dense(depth, mode="bilinear"))
    # last row/col fall just outside the model domain (xo*0.5 > 49) -> excluded
    assert np.allclose(back[:99, :99], depth[:99, :99], atol=1e-6)


def test_mask_nearest_preserves_labels():
    t = SpatialTransform.resize((100, 100), (50, 50))
    mask = np.zeros((100, 100), np.int64)
    mask[20:60, 30:70] = 7
    model_mask = t.apply_dense(mask, mode="nearest", fill=0)
    assert model_mask.shape == (50, 50)
    assert model_mask.dtype == np.int64                 # labels not floated
    assert set(np.unique(model_mask)).issubset({0, 7})  # no interpolated labels


def test_dense_identity_roundtrip_exact():
    t = SpatialTransform.identity((40, 30))
    mask = (np.arange(40 * 30).reshape(40, 30) % 5).astype(np.int64)
    assert np.array_equal(t.invert_mask(t.apply_dense(mask, mode="nearest")), mask)
    depth = _ramp(40, 30)
    assert np.allclose(t.invert_depth(t.apply_dense(depth)), depth)


def test_letterbox_padding_is_filled():
    # wide image into a square -> vertical padding; padded rows must be fill
    t = SpatialTransform.letterbox((100, 200), (200, 200))
    model = t.apply_dense(_ramp(100, 200), mode="bilinear", fill=-1.0)
    assert model.shape == (200, 200)
    assert np.all(model[0] == -1.0) and np.all(model[-1] == -1.0)   # top/bottom pad
    assert np.any(model[100] != -1.0)                                # image band present


def test_dense_needs_model_size():
    import pytest
    t = SpatialTransform(orig_size=(10, 10))   # no model_size
    with pytest.raises(ValueError):
        t.apply_dense(np.zeros((10, 10)))
