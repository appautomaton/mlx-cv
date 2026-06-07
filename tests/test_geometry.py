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
