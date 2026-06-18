import numpy as np
import mlx.core as mx

from mlx_cv.backbones.vision.necks import RFDETRFeaturePyramid, RFDETRPyramidLevel
from mlx_cv.core.features import FeatureMap, Layout
from mlx_cv.heads.detection import RFDETRDecoderConfig, RFDETRDetectionHead, RFDETRQueryDecoder


def _pyramid(hidden_dim=8):
    levels = []
    for grid, stride in [((2, 2), 14), ((1, 1), 28)]:
        h, w = grid
        data = mx.array(np.arange(1 * h * w * hidden_dim, dtype=np.float32).reshape(1, h, w, hidden_dim) / 100.0)
        levels.append(
            RFDETRPyramidLevel(
                data=data,
                feature=FeatureMap(data, layout=Layout.BHWC, grid=grid, stride=stride),
                mask=mx.zeros((1, h, w), dtype=mx.bool_),
                position=mx.zeros((1, h, w, 2)),
                stride=stride,
            )
        )
    return RFDETRFeaturePyramid(levels)


def _cfg():
    return RFDETRDecoderConfig(
        hidden_dim=8,
        num_queries=3,
        num_heads=2,
        num_layers=2,
        num_points=2,
        num_classes=5,
    )


def test_rfdetr_query_decoder_shapes():
    decoder = RFDETRQueryDecoder(_cfg(), num_levels=2)
    mx.eval(decoder.parameters())
    out = decoder(_pyramid())
    assert out["hidden_states"].shape == (1, 3, 8)
    assert out["decoder_hidden_states"].shape == (2, 1, 3, 8)
    assert out["reference_points"].shape == (1, 3, 2, 2)
    assert out["spatial_shapes"].shape == (2, 2)


def test_rfdetr_detection_head_returns_head_output():
    decoder = RFDETRQueryDecoder(_cfg(), num_levels=2)
    head = RFDETRDetectionHead(_cfg())
    mx.eval(decoder.parameters())
    mx.eval(head.parameters())
    out = head(decoder(_pyramid()))
    assert out["logits"].shape == (1, 3, 5)
    assert out["boxes"].shape == (1, 3, 4)
    boxes = np.array(out["boxes"])
    assert np.all((boxes >= 0.0) & (boxes <= 1.0))


def test_rfdetr_decoder_calls_deformable_attention(monkeypatch):
    import mlx_cv.heads.detection.rfdetr as module

    calls = []

    def fake_core(value, spatial_shapes, sampling_locations, attention_weights):
        calls.append((value.shape, sampling_locations.shape, attention_weights.shape))
        return mx.zeros((value.shape[0], sampling_locations.shape[1], value.shape[1] * value.shape[2]))

    monkeypatch.setattr(module, "ms_deform_attn_core", fake_core)
    decoder = RFDETRQueryDecoder(_cfg(), num_levels=2)
    mx.eval(decoder.parameters())
    out = decoder(_pyramid())
    assert out["hidden_states"].shape == (1, 3, 8)
    assert calls
