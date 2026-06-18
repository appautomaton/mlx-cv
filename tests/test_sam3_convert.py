import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.heads.segmentation import SAM3DecoderConfig
from mlx_cv.models.sam3 import (
    SAM3Config,
    SAM3ImageBackboneConfig,
    SAM3Model,
    SAM3TextConfig,
    SAM3Tokenizer,
    convert_sam3_state_dict,
    load_sam3_weights,
    remap_sam3_key,
)


def _cfg():
    tokenizer = SAM3Tokenizer(context_length=8)
    return SAM3Config(
        image=SAM3ImageBackboneConfig(
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
        ),
        text=SAM3TextConfig(
            d_model=6,
            context_length=8,
            vocab_size=tokenizer.vocab_size,
            width=8,
            heads=2,
            layers=1,
            mlp_ratio=2.0,
        ),
        decoder=SAM3DecoderConfig(hidden_dim=4, num_queries=3, num_layers=1, num_heads=1, num_classes=2, text_dim=6),
    )


def test_remap_sam3_reference_image_mode_keys():
    assert remap_sam3_key("image_encoder.patch_embed.proj.weight") == (
        "feature_extractor.backbone.vision.patch_embed.proj.weight",
        True,
    )
    assert remap_sam3_key("neck.projections.0.weight") == ("feature_extractor.neck.projections.0.weight", True)
    assert remap_sam3_key("mask_decoder.score_embed.bias") == ("mask_decoder.score_embed.bias", False)
    assert remap_sam3_key("model.mask_decoder.mask_feature_proj.weight") == (
        "mask_decoder.mask_feature_proj.weight",
        True,
    )
    assert remap_sam3_key("text_encoder.token_embedding.weight") == ("text_encoder.token_embedding.weight", False)
    assert remap_sam3_key("decoder.query_embed") == ("decoder.query_embed", False)


def test_convert_sam3_state_dict_maps_reference_and_local_keys():
    out = dict(
        convert_sam3_state_dict(
            {
                "image_encoder.patch_embed.proj.bias": np.arange(8, dtype=np.float32),
                "neck.projections.0.bias": np.ones((4,), dtype=np.float32),
                "decoder.query_embed": np.ones((3, 4), dtype=np.float32) * 2,
            }
        )
    )

    assert sorted(out) == [
        "decoder.query_embed",
        "feature_extractor.backbone.vision.patch_embed.proj.bias",
        "feature_extractor.neck.projections.0.bias",
    ]
    np.testing.assert_array_equal(out["feature_extractor.backbone.vision.patch_embed.proj.bias"], np.arange(8))


def test_convert_sam3_state_dict_rejects_video_tracker_variants():
    with pytest.raises(ValueError, match="video/tracker"):
        convert_sam3_state_dict({"video_memory_encoder.weight": np.zeros((1,), dtype=np.float32)})

    with pytest.raises(ValueError, match="video/tracker"):
        convert_sam3_state_dict(
            {
                "__config_json__": np.array('{"model": {"video": true}}'),
                "decoder.query_embed": np.zeros((3, 4), dtype=np.float32),
            }
        )


def test_convert_sam3_state_dict_rejects_unknown_tensor_keys():
    with pytest.raises(ValueError, match="unsupported SAM3 checkpoint keys"):
        convert_sam3_state_dict({"unexpected.weight": np.zeros((1,), dtype=np.float32)})


def test_load_sam3_weights_populates_tiny_model(tmp_path):
    tokenizer = SAM3Tokenizer(context_length=8)
    model = SAM3Model(_cfg(), tokenizer=tokenizer)
    weights_path = tmp_path / "sam3_tiny.npz"
    np.savez(
        weights_path,
        **{
            "decoder.query_embed": np.ones((3, 4), dtype=np.float32) * 0.25,
            "mask_decoder.score_embed.bias": np.array([0.5], dtype=np.float32),
        },
    )

    loaded = load_sam3_weights(model, weights_path)
    params = dict(tree_flatten(loaded.parameters()))
    np.testing.assert_allclose(np.array(params["decoder.query_embed"]), np.ones((3, 4)) * 0.25)
    np.testing.assert_allclose(np.array(params["mask_decoder.score_embed.bias"]), [0.5])


def test_load_sam3_weights_rejects_shape_mismatch(tmp_path):
    model = SAM3Model(_cfg(), tokenizer=SAM3Tokenizer(context_length=8))
    weights_path = tmp_path / "bad.npz"
    np.savez(weights_path, **{"decoder.query_embed": np.zeros((4, 4), dtype=np.float32)})

    with pytest.raises(ValueError, match="expected \\(3, 4\\)"):
        load_sam3_weights(model, weights_path)
