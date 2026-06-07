import numpy as np

from mlx_cv.transforms import Letterbox, Resize, normalize, to_chw


def test_resize_shape_and_ctx():
    img = np.zeros((480, 640, 3), np.uint8)
    out, ctx = Resize((240, 320))(img)
    assert out.shape == (240, 320, 3)
    assert ctx.orig_size == (480, 640)
    assert ctx.model_size == (240, 320)


def test_letterbox_shape_and_box_roundtrip():
    img = np.zeros((100, 200, 3), np.uint8)
    out, ctx = Letterbox((256, 256))(img)
    assert out.shape == (256, 256, 3)
    b = np.array([[0, 0, 200, 100]], float)
    assert np.allclose(ctx.invert_boxes(ctx.apply_boxes(b)), b)


def test_normalize_and_chw():
    img = np.full((2, 2, 3), 255, np.uint8)
    n = normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert np.allclose(n, 1.0)
    assert to_chw(n).shape == (3, 2, 2)
