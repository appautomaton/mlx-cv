import mlx.core as mx

from mlx_cv.core.features import Layout
from mlx_cv.models.sam3 import SAM3FeatureNeck, SAM3ImageBackbone, SAM3ImageBackboneConfig


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


def _features(text=False):
    cfg = _cfg()
    backbone = SAM3ImageBackbone(cfg)
    image = mx.ones((1, 3, 32, 32), dtype=mx.float32)
    text_features = mx.ones((4, 1, cfg.text_dim), dtype=mx.float32) if text else None
    return cfg, backbone(image, text_features=text_features)


def test_sam3_feature_neck_outputs_reference_level_shapes_and_metadata():
    cfg, features = _features()
    neck = SAM3FeatureNeck(
        in_channels=(cfg.embed_dim, cfg.embed_dim),
        out_channels=cfg.neck_channels,
        scale_factors=cfg.neck_scales,
    )
    mx.eval(neck.parameters())
    pyramid = neck(features)
    mx.eval(*[level.data for level in pyramid.levels])

    assert len(pyramid.levels) == 2
    assert [level.feature.grid for level in pyramid.levels] == [(8, 8), (4, 4)]
    assert [level.stride for level in pyramid.levels] == [4, 8]
    assert [level.data.shape for level in pyramid.levels] == [(1, 8, 8, 4), (1, 4, 4, 4)]
    assert [level.mask.shape for level in pyramid.levels] == [(1, 8, 8), (1, 4, 4)]
    assert [level.position.shape for level in pyramid.levels] == [(1, 8, 8, 2), (1, 4, 4, 2)]
    assert all(level.feature.layout is Layout.BHWC for level in pyramid.levels)
    assert all(str(level.data.dtype).endswith("float32") for level in pyramid.levels)


def test_sam3_feature_neck_preserves_text_fusion_path():
    cfg, features = _features(text=True)
    assert features.extras["text_fused"] is True
    neck = SAM3FeatureNeck(in_channels=(cfg.embed_dim, cfg.embed_dim), out_channels=4, scale_factors=(1.0,))
    pyramid = neck(features)
    mx.eval(pyramid.levels[0].data)

    assert pyramid.levels[0].data.shape == (1, 8, 8, 4)


def test_sam3_feature_neck_rejects_missing_grid():
    neck = SAM3FeatureNeck(in_channels=(4,), out_channels=4)
    bad = _features()[1].patch_tokens
    bad.grid = None
    try:
        neck([bad])
    except ValueError as exc:
        assert "grid" in str(exc)
    else:
        raise AssertionError("expected grid validation error")
