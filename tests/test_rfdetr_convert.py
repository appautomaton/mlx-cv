import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_cv.backbones.vision.dinov2 import DINOv2Config
from mlx_cv.heads.detection import RFDETRDecoderConfig
from mlx_cv.models.rfdetr import (
    RFDETRConfig,
    RFDETRModel,
    convert_rfdetr_state_dict,
    load_rfdetr_weights,
    remap_rfdetr_key,
)


def _cfg():
    return RFDETRConfig(
        backbone=DINOv2Config(
            embed_dim=16,
            depth=2,
            num_heads=2,
            patch_size=14,
            n_register_tokens=2,
            pretrain_grid=2,
        ),
        out_layers=(0, 1),
        projector_out_channels=8,
        projector_scale_factors=(2.0, 1.0),
        decoder=RFDETRDecoderConfig(
            hidden_dim=8,
            num_queries=4,
            num_heads=2,
            num_layers=1,
            num_points=2,
            num_classes=3,
        ),
    )


def test_remap_rfdetr_reference_detection_keys():
    assert remap_rfdetr_key("class_embed.weight") == ("head.class_embed.weight", True)
    assert remap_rfdetr_key("model.bbox_embed.layers.2.bias") == ("head.bbox_embed.layers.2.bias", True)
    assert remap_rfdetr_key("query_feat.weight") == ("decoder.query_embed", True)
    assert remap_rfdetr_key("refpoint_embed.weight") == ("decoder.reference_embed", True)
    assert remap_rfdetr_key("head.class_embed.bias") == ("head.class_embed.bias", False)
    assert remap_rfdetr_key("backbone.0.projector.stages.0.0.cv1.conv.weight") == (
        "feature_extractor.projector.stages.0.0.cv1.conv.weight",
        True,
    )


def test_convert_rfdetr_state_dict_maps_reference_head_and_query_weights():
    out = dict(
        convert_rfdetr_state_dict(
            {
                "class_embed.bias": np.arange(3, dtype=np.float32),
                "bbox_embed.layers.2.weight": np.ones((4, 8), dtype=np.float32),
                "query_feat.weight": np.ones((4, 8), dtype=np.float32) * 2,
                "refpoint_embed.weight": np.ones((4, 4), dtype=np.float32) * 3,
            }
        )
    )

    assert sorted(out) == [
        "decoder.query_embed",
        "decoder.reference_embed",
        "head.bbox_embed.layers.2.weight",
        "head.class_embed.bias",
    ]
    assert out["decoder.reference_embed"].shape == (4, 2)
    np.testing.assert_array_equal(out["head.class_embed.bias"], np.arange(3, dtype=np.float32))


def test_convert_rfdetr_state_dict_maps_projector_stage_weights_to_projector():
    weight = np.arange(2 * 3 * 1 * 1, dtype=np.float32).reshape(2, 3, 1, 1)

    out = dict(
        convert_rfdetr_state_dict(
            {
                "backbone.0.projector.stages.0.0.cv1.conv.weight": weight,
            }
        )
    )

    assert sorted(out) == ["feature_extractor.projector.stages.0.0.cv1.conv.weight"]
    np.testing.assert_array_equal(
        out["feature_extractor.projector.stages.0.0.cv1.conv.weight"],
        np.transpose(weight, (0, 2, 3, 1)),
    )


def test_convert_rfdetr_state_dict_rejects_segmentation_variants():
    with pytest.raises(ValueError, match="segmentation checkpoints"):
        convert_rfdetr_state_dict({"segmentation_head.blocks.0.weight": np.zeros((1,), dtype=np.float32)})

    with pytest.raises(ValueError, match="segmentation_head=True"):
        convert_rfdetr_state_dict(
            {
                "__config_json__": np.array('{"model": {"segmentation_head": true}}'),
                "class_embed.bias": np.zeros((3,), dtype=np.float32),
            }
        )


def test_convert_rfdetr_state_dict_rejects_unknown_tensor_keys():
    with pytest.raises(ValueError, match="unsupported RF-DETR checkpoint keys"):
        convert_rfdetr_state_dict({"unexpected.weight": np.zeros((1,), dtype=np.float32)})


def test_load_rfdetr_weights_populates_tiny_model(tmp_path):
    model = RFDETRModel(_cfg())

    weights_path = tmp_path / "rfdetr_tiny.npz"
    np.savez(
        weights_path,
        **{
            "head.class_embed.bias": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "decoder.query_embed": np.ones((4, 8), dtype=np.float32) * 0.25,
        },
    )

    loaded = load_rfdetr_weights(model, weights_path)
    loaded_params = dict(tree_flatten(loaded.parameters()))
    np.testing.assert_allclose(np.array(loaded_params["head.class_embed.bias"]), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.array(loaded_params["decoder.query_embed"]), np.ones((4, 8)) * 0.25)


def test_load_rfdetr_weights_rejects_shape_mismatch(tmp_path):
    model = RFDETRModel(_cfg())
    weights_path = tmp_path / "bad.npz"
    np.savez(weights_path, **{"head.class_embed.bias": np.zeros((4,), dtype=np.float32)})

    with pytest.raises(ValueError, match="expected \\(3,\\)"):
        load_rfdetr_weights(model, weights_path)
