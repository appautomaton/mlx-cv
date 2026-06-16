import numpy as np
import mlx.core as mx

from mlx_cv.backbones.vision.necks import RFDETRMultiScaleProjector
from mlx_cv.core.features import BackboneFeatures, FeatureMap, Layout


def _features():
    first = mx.array(np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4))
    second = mx.array(np.arange(1 * 4 * 4, dtype=np.float32).reshape(1, 4, 4) / 10.0)
    return BackboneFeatures(
        patch_tokens=FeatureMap(second, layout=Layout.BNC, grid=(2, 2), stride=14),
        intermediates=[
            FeatureMap(first, layout=Layout.BNC, grid=(2, 2), stride=14),
            FeatureMap(second, layout=Layout.BNC, grid=(2, 2), stride=14),
        ],
    )


def test_rfdetr_multiscale_projector_outputs_shapes_and_metadata():
    neck = RFDETRMultiScaleProjector(in_channels=(4, 4), out_channels=6, scale_factors=(2.0, 1.0, 0.5))
    mx.eval(neck.parameters())
    pyramid = neck(_features())

    assert len(pyramid.levels) == 3
    assert [level.feature.grid for level in pyramid.levels] == [(4, 4), (2, 2), (1, 1)]
    assert [level.stride for level in pyramid.levels] == [7, 14, 28]
    assert [level.data.shape for level in pyramid.levels] == [(1, 4, 4, 6), (1, 2, 2, 6), (1, 1, 1, 6)]
    assert [level.mask.shape for level in pyramid.levels] == [(1, 4, 4), (1, 2, 2), (1, 1, 1)]
    assert [level.position.shape for level in pyramid.levels] == [(1, 4, 4, 2), (1, 2, 2, 2), (1, 1, 1, 2)]
    assert all(level.feature.layout == Layout.BHWC for level in pyramid.levels)


def test_rfdetr_multiscale_projector_rejects_missing_grid():
    neck = RFDETRMultiScaleProjector(in_channels=(4,), out_channels=6)
    bad = [FeatureMap(mx.zeros((1, 4, 4)), layout=Layout.BNC, grid=None, stride=14)]
    try:
        neck(bad)
    except ValueError as exc:
        assert "grid" in str(exc)
    else:
        raise AssertionError("expected grid validation error")
