import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.core import CameraGeometry
from mlx_cv.models.depth_anything_v3 import DA3MultiViewContext, DA3Processor, DA3ProcessorConfig


def _processor() -> DA3Processor:
    return DA3Processor(
        DA3ProcessorConfig(
            process_res=14,
            patch_size=14,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
    )


def _image(value: int, shape: tuple[int, int] = (14, 14)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


def test_da3_preprocess_accepts_image_list_and_preserves_view_axis_order():
    tensor, ctx = _processor().preprocess([_image(0), _image(128), _image(255)])

    assert isinstance(ctx, DA3MultiViewContext)
    assert ctx.view_count == 3
    assert ctx.orig_size == (14, 14)
    assert tensor.shape == (1, 3, 3, 14, 14)
    arr = np.array(tensor)
    assert np.allclose(arr[0, :, 0, 0, 0], [0.0, 128.0 / 255.0, 1.0])


def test_da3_preprocess_accepts_input_camera_geometry():
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    _, ctx = _processor().preprocess({
        "images": [_image(16), _image(32)],
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    })

    assert ctx.camera_geometry is not None
    assert ctx.camera_geometry.view_count == 2
    assert ctx.camera_geometry.convention == "w2c"
    assert ctx.camera_geometry.extrinsics.shape == (2, 4, 4)
    assert ctx.camera_geometry.intrinsics.shape == (2, 3, 3)


def test_da3_preprocess_rejects_mixed_size_image_sets():
    with pytest.raises(ValueError, match="same-size still images"):
        _processor().preprocess([_image(0, (14, 14)), _image(1, (10, 14))])


def test_da3_preprocess_rejects_invalid_camera_shapes():
    with pytest.raises(ValueError, match="extrinsics must have shape"):
        _processor().preprocess({
            "images": [_image(16), _image(32)],
            "extrinsics": np.zeros((2, 3, 3), dtype=np.float32),
        })

    with pytest.raises(ValueError, match="intrinsics has 3 views, expected 2"):
        _processor().preprocess({
            "images": [_image(16), _image(32)],
            "intrinsics": np.zeros((3, 3, 3), dtype=np.float32),
        })


def test_da3_postprocess_returns_depth_views_camera_and_first_depth():
    processor = _processor()
    _, ctx = processor.preprocess({
        "images": [_image(0), _image(255)],
        "extrinsics": np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
    })
    raw = {
        "depth": mx.array(np.stack([
            np.full((14, 14), 1.0, dtype=np.float32),
            np.full((14, 14), 2.0, dtype=np.float32),
        ], axis=0)[None]),
        "depth_conf": mx.array(np.stack([
            np.full((14, 14), 0.25, dtype=np.float32),
            np.full((14, 14), 0.75, dtype=np.float32),
        ], axis=0)[None]),
        "extrinsics": np.full((2, 3, 4), 3.0, dtype=np.float32),
        "intrinsics": np.full((2, 3, 3), 4.0, dtype=np.float32),
    }

    result = processor.postprocess(raw, ctx)

    assert result.image_size == (14, 14)
    assert result.depth is result.depth_views[0]
    assert len(result.depth_views) == 2
    assert np.allclose(result.depth_views[0].depth, 1.0)
    assert np.allclose(result.depth_views[1].depth, 2.0)
    assert np.allclose(result.depth_views[0].depth_conf, 0.25)
    assert isinstance(result.camera_geometry, CameraGeometry)
    assert result.camera_geometry.extrinsics.shape == (2, 3, 4)
    assert np.allclose(result.camera_geometry.intrinsics, 4.0)


def test_da3_postprocess_uses_input_camera_geometry_when_output_omits_camera():
    processor = _processor()
    _, ctx = processor.preprocess({
        "images": [_image(0), _image(255)],
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
    })

    result = processor.postprocess(
        {"depth": np.ones((2, 14, 14), dtype=np.float32)},
        ctx,
    )

    assert result.camera_geometry is ctx.camera_geometry
    assert result.camera_geometry.intrinsics.shape == (2, 3, 3)


def test_da3_postprocess_rejects_invalid_view_axis():
    processor = _processor()
    _, ctx = processor.preprocess([_image(0), _image(255)])

    with pytest.raises(ValueError, match="has 3 views, expected 2"):
        processor.postprocess({"depth": np.ones((3, 14, 14), dtype=np.float32)}, ctx)

    with pytest.raises(ValueError, match="extrinsics has 3 views, expected 2"):
        processor.postprocess({
            "depth": np.ones((2, 14, 14), dtype=np.float32),
            "extrinsics": np.ones((3, 3, 4), dtype=np.float32),
        }, ctx)
