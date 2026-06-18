import numpy as np
import mlx.core as mx

from mlx_cv.core.features import Layout
from mlx_cv.models.sam3 import SAM3ImageBackbone, SAM3ImageBackboneConfig


def _cfg():
    return SAM3ImageBackboneConfig(
        image_size=32,
        patch_size=4,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        text_dim=6,
        out_layers=(0, 1),
        neck_channels=4,
        neck_scales=(1.0, 0.5),
    )


def test_sam3_image_backbone_constructs_image_mode_without_tracker_modules():
    cfg = _cfg()
    model = SAM3ImageBackbone(cfg)
    mx.eval(model.parameters())

    features = model(mx.ones((1, 3, 32, 32), dtype=mx.float32))
    mx.eval(features.patch_tokens.data, *[feature.data for feature in features.intermediates])

    assert features.patch_tokens.layout is Layout.BNC
    assert features.patch_tokens.grid == (8, 8)
    assert features.patch_tokens.stride == 4
    assert features.patch_tokens.data.shape == (1, 64, 8)
    assert [feature.data.shape for feature in features.intermediates] == [(1, 64, 8), (1, 64, 8)]
    assert features.extras["text_fused"] is False
    assert not any(name for name in dir(model) if "tracker" in name.lower() or "video" in name.lower())


def test_sam3_image_backbone_represents_text_vl_fusion():
    cfg = _cfg()
    model = SAM3ImageBackbone(cfg)
    image = mx.ones((1, 3, 32, 32), dtype=mx.float32)
    no_text = model(image)
    text_features = mx.ones((5, 1, cfg.text_dim), dtype=mx.float32)
    with_text = model(image, text_features=text_features)
    mx.eval(no_text.patch_tokens.data, with_text.patch_tokens.data)

    assert with_text.extras["text_fused"] is True
    assert with_text.extras["text_summary"].shape == (1, cfg.text_dim)
    assert np.max(np.abs(np.array(with_text.patch_tokens.data - no_text.patch_tokens.data))) > 0


def test_sam3_image_backbone_rejects_mismatched_text_width():
    cfg = _cfg()
    model = SAM3ImageBackbone(cfg)
    try:
        model(mx.ones((1, 3, 32, 32), dtype=mx.float32), text_features=mx.ones((5, 1, cfg.text_dim + 1)))
    except ValueError as exc:
        assert "text feature width" in str(exc)
    else:
        raise AssertionError("expected text width validation error")
