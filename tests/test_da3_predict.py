import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.core.types import CameraGeometry, DepthMap, Result  # noqa: E402
from mlx_cv.models.depth_anything_v3 import (  # noqa: E402
    DA3MonocularConfig,
    DA3MultiViewConfig,
    DA3Processor,
    DA3ProcessorConfig,
    DepthAnythingV3Monocular,
    DepthAnythingV3MultiView,
)


def _mono_model() -> DepthAnythingV3Monocular:
    # Tiny random-weight fixture: no real checkpoint, no committed weights.
    return DepthAnythingV3Monocular(DA3MonocularConfig.tiny_fixture())


def _multiview_model() -> DepthAnythingV3MultiView:
    return DepthAnythingV3MultiView(DA3MultiViewConfig.tiny_fixture())


def test_da3_monocular_predict_returns_typed_depth_in_original_coords():
    processor = DA3Processor(DA3ProcessorConfig(process_res=28, patch_size=14))
    image = np.full((10, 20, 3), 128, dtype=np.uint8)

    result = _mono_model().predict(image, processor=processor)

    assert isinstance(result, Result)
    assert isinstance(result.depth, DepthMap)
    assert result.image_size == (10, 20)
    assert result.depth.depth.shape == (10, 20)  # inverted back to original size


def test_da3_monocular_predict_builds_default_processor_from_options():
    result = _mono_model().predict(
        np.full((14, 14, 3), 64, dtype=np.uint8), process_res=28, patch_size=14
    )

    assert isinstance(result.depth, DepthMap)
    assert result.image_size == (14, 14)


def test_da3_monocular_predict_rejects_processor_and_options_together():
    processor = DA3Processor(DA3ProcessorConfig(process_res=28, patch_size=14))
    with pytest.raises(ValueError, match="processor is not provided"):
        _mono_model().predict(
            np.zeros((14, 14, 3), dtype=np.uint8), processor=processor, process_res=28
        )


def test_da3_multiview_predict_returns_per_view_depth_and_camera_geometry():
    processor = DA3Processor(DA3ProcessorConfig(process_res=4, patch_size=2))
    images = [
        np.full((4, 4, 3), 96, dtype=np.uint8),
        np.full((4, 4, 3), 160, dtype=np.uint8),
    ]

    result = _multiview_model().predict(images, processor=processor)

    assert isinstance(result, Result)
    assert result.depth_views is not None
    assert len(result.depth_views) == 2
    assert all(isinstance(view, DepthMap) for view in result.depth_views)
    assert isinstance(result.depth, DepthMap)  # first view mirrored into .depth
    assert isinstance(result.camera_geometry, CameraGeometry)
    assert result.camera_geometry.view_count == 2
    assert result.image_size == (4, 4)


def test_da3_multiview_predict_rejects_processor_and_options_together():
    processor = DA3Processor(DA3ProcessorConfig(process_res=4, patch_size=2))
    with pytest.raises(ValueError, match="processor is not provided"):
        _multiview_model().predict(
            [np.zeros((4, 4, 3), dtype=np.uint8)], processor=processor, patch_size=2
        )
