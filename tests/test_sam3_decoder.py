import mlx.core as mx

from mlx_cv.heads.segmentation import SAM3DecoderConfig, SAM3ImageDecoder, SAM3MaskDecoder
from mlx_cv.models.sam3 import SAM3FeatureNeck, SAM3ImageBackbone, SAM3ImageBackboneConfig, prepare_sam3_prompt
from mlx_cv.core.geometry import SpatialTransform
from mlx_cv.prompts import BoxPrompt


def _pyramid():
    cfg = SAM3ImageBackboneConfig(
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
    backbone = SAM3ImageBackbone(cfg)
    features = backbone(mx.ones((1, 3, 32, 32), dtype=mx.float32))
    neck = SAM3FeatureNeck(in_channels=(8, 8), out_channels=4, scale_factors=(1.0, 0.5))
    return neck(features)


def test_sam3_decoder_and_mask_head_return_image_mode_logits_and_metadata():
    decoder_cfg = SAM3DecoderConfig(hidden_dim=4, num_queries=3, num_layers=2, num_heads=1, num_classes=2, text_dim=6)
    decoder = SAM3ImageDecoder(decoder_cfg)
    mask_decoder = SAM3MaskDecoder(decoder_cfg)
    prompt = prepare_sam3_prompt(
        BoxPrompt([[4, 4, 20, 24]]),
        transform=SpatialTransform.resize((32, 32), (32, 32)),
        model_size=(32, 32),
    )
    text_features = mx.ones((5, 1, 6), dtype=mx.float32)

    pyramid = _pyramid()
    decoder_out = decoder(pyramid, prompt=prompt.geometry, text_output=text_features, capture_taps=True)
    out = mask_decoder(decoder_out, pyramid)
    mx.eval(out["mask_logits"], out["object_scores"], out["labels"], out["boxes"])

    assert out["mask_logits"].shape == (1, 3, 8, 8)
    assert out["object_scores"].shape == (1, 3)
    assert out["class_logits"].shape == (1, 3, 2)
    assert out["labels"].shape == (1, 3)
    assert out["boxes"].shape == (1, 3, 4)
    assert out["decoder_hidden_states"].shape == (2, 1, 3, 4)
    assert decoder_out["memory"].shape[0] == 1


def test_sam3_decoder_rejects_pyramid_channel_mismatch():
    decoder = SAM3ImageDecoder(SAM3DecoderConfig(hidden_dim=5, num_queries=2, num_layers=1, num_heads=1, text_dim=6))
    try:
        decoder(_pyramid())
    except ValueError as exc:
        assert "hidden_dim" in str(exc)
    else:
        raise AssertionError("expected channel validation error")
