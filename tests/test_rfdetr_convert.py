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
from mlx_cv.models.rfdetr.convert import RFDETR_INFERENCE_ONLY_EXCLUSIONS


def _labelled_query_tensor(num_queries: int, group_detr: int, dim: int) -> np.ndarray:
    rows = []
    for group in range(group_detr):
        for query in range(num_queries):
            row = np.zeros((dim,), dtype=np.float32)
            row[0] = group
            row[1] = query
            if dim > 2:
                row[2:] = 10 * group + query
            rows.append(row)
    return np.stack(rows, axis=0)


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
    assert remap_rfdetr_key("transformer.decoder.norm.weight") == ("decoder.norm.weight", True)
    assert remap_rfdetr_key("transformer.decoder.ref_point_head.layers.1.weight") == (
        "decoder.ref_point_head.layers.1.weight",
        True,
    )
    assert remap_rfdetr_key("transformer.decoder.layers.0.norm2.weight") == (
        "decoder.layers.0.norm2.weight",
        True,
    )
    assert remap_rfdetr_key("transformer.enc_out_class_embed.12.weight") == (
        "decoder.enc_out_class_embed.12.weight",
        True,
    )
    assert remap_rfdetr_key("backbone.0.encoder.encoder.embeddings.cls_token") == (
        "feature_extractor.backbone.backbone.cls_token",
        True,
    )
    assert remap_rfdetr_key("backbone.0.encoder.encoder.embeddings.position_embeddings") == (
        "feature_extractor.backbone.backbone.pos_embed.table",
        True,
    )
    assert remap_rfdetr_key("backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight") == (
        "feature_extractor.backbone.backbone.patch_embed.proj.weight",
        True,
    )
    assert remap_rfdetr_key("backbone.0.encoder.encoder.encoder.layer.0.layer_scale1.lambda1") == (
        "feature_extractor.backbone.backbone.blocks.0.ls1.gamma",
        True,
    )
    assert remap_rfdetr_key("backbone.0.encoder.encoder.layernorm.weight") == (
        "feature_extractor.backbone.backbone.norm.weight",
        True,
    )
    assert remap_rfdetr_key("backbone.0.encoder.encoder.embeddings.mask_token") == (None, True)


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


def test_convert_rfdetr_state_dict_splits_decoder_self_attention_in_proj():
    weight = np.arange(24 * 8, dtype=np.float32).reshape(24, 8)
    bias = np.arange(24, dtype=np.float32)

    out = dict(
        convert_rfdetr_state_dict(
            {
                "transformer.decoder.layers.0.self_attn.in_proj_weight": weight,
                "transformer.decoder.layers.0.self_attn.in_proj_bias": bias,
            }
        )
    )

    assert sorted(out) == [
        "decoder.layers.0.self_attn.key_proj.bias",
        "decoder.layers.0.self_attn.key_proj.weight",
        "decoder.layers.0.self_attn.query_proj.bias",
        "decoder.layers.0.self_attn.query_proj.weight",
        "decoder.layers.0.self_attn.value_proj.bias",
        "decoder.layers.0.self_attn.value_proj.weight",
    ]
    np.testing.assert_array_equal(out["decoder.layers.0.self_attn.query_proj.weight"], weight[:8])
    np.testing.assert_array_equal(out["decoder.layers.0.self_attn.key_proj.weight"], weight[8:16])
    np.testing.assert_array_equal(out["decoder.layers.0.self_attn.value_proj.bias"], bias[16:24])


def test_convert_rfdetr_state_dict_maps_hf_dinov2_backbone_and_packs_qkv():
    conv = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
    q = np.ones((4, 4), dtype=np.float32)
    k = np.ones((4, 4), dtype=np.float32) * 2
    v = np.ones((4, 4), dtype=np.float32) * 3
    q_bias = np.ones((4,), dtype=np.float32) * 4
    k_bias = np.ones((4,), dtype=np.float32) * 5
    v_bias = np.ones((4,), dtype=np.float32) * 6

    out = dict(
        convert_rfdetr_state_dict(
            {
                "backbone.0.encoder.encoder.embeddings.cls_token": np.ones((1, 1, 4), dtype=np.float32),
                "backbone.0.encoder.encoder.embeddings.mask_token": np.ones((1, 4), dtype=np.float32),
                "backbone.0.encoder.encoder.embeddings.position_embeddings": np.ones(
                    (1, 5, 4),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": conv,
                "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias": np.ones(
                    (4,),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.norm1.weight": np.ones((4,), dtype=np.float32),
                "backbone.0.encoder.encoder.encoder.layer.0.norm1.bias": np.ones((4,), dtype=np.float32),
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.query.weight": q,
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.key.weight": k,
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.value.weight": v,
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.query.bias": q_bias,
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.key.bias": k_bias,
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.value.bias": v_bias,
                "backbone.0.encoder.encoder.encoder.layer.0.attention.output.dense.weight": np.ones(
                    (4, 4),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.attention.output.dense.bias": np.ones(
                    (4,),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.layer_scale1.lambda1": np.ones(
                    (4,),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.norm2.weight": np.ones((4,), dtype=np.float32),
                "backbone.0.encoder.encoder.encoder.layer.0.norm2.bias": np.ones((4,), dtype=np.float32),
                "backbone.0.encoder.encoder.encoder.layer.0.mlp.fc1.weight": np.ones(
                    (8, 4),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.mlp.fc1.bias": np.ones((8,), dtype=np.float32),
                "backbone.0.encoder.encoder.encoder.layer.0.mlp.fc2.weight": np.ones(
                    (4, 8),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.mlp.fc2.bias": np.ones((4,), dtype=np.float32),
                "backbone.0.encoder.encoder.encoder.layer.0.layer_scale2.lambda1": np.ones(
                    (4,),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.layernorm.weight": np.ones((4,), dtype=np.float32),
                "backbone.0.encoder.encoder.layernorm.bias": np.ones((4,), dtype=np.float32),
            }
        )
    )

    assert "feature_extractor.backbone.backbone.embeddings.mask_token" not in out
    np.testing.assert_array_equal(
        out["feature_extractor.backbone.backbone.patch_embed.proj.weight"],
        np.transpose(conv, (0, 2, 3, 1)),
    )
    np.testing.assert_array_equal(
        out["feature_extractor.backbone.backbone.blocks.0.attn.qkv.weight"],
        np.concatenate([q, k, v], axis=0),
    )
    np.testing.assert_array_equal(
        out["feature_extractor.backbone.backbone.blocks.0.attn.qkv.bias"],
        np.concatenate([q_bias, k_bias, v_bias], axis=0),
    )
    assert "feature_extractor.backbone.backbone.blocks.0.ls1.gamma" in out
    assert "feature_extractor.backbone.backbone.blocks.0.ls2.gamma" in out
    assert "feature_extractor.backbone.backbone.norm.weight" in out


def test_convert_rfdetr_state_dict_rejects_incomplete_hf_dinov2_qkv_pack():
    with pytest.raises(ValueError, match="value.weight"):
        convert_rfdetr_state_dict(
            {
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.query.weight": np.ones(
                    (4, 4),
                    dtype=np.float32,
                ),
                "backbone.0.encoder.encoder.encoder.layer.0.attention.attention.key.weight": np.ones(
                    (4, 4),
                    dtype=np.float32,
                ),
            }
        )


def test_convert_rfdetr_state_dict_only_drops_explicit_hf_dinov2_mask_token_exclusion():
    mask_key = "backbone.0.encoder.encoder.embeddings.mask_token"
    assert mask_key in RFDETR_INFERENCE_ONLY_EXCLUSIONS
    assert dict(convert_rfdetr_state_dict({mask_key: np.ones((1, 4), dtype=np.float32)})) == {}

    with pytest.raises(ValueError, match="unexpected_token"):
        convert_rfdetr_state_dict(
            {
                "backbone.0.encoder.encoder.embeddings.unexpected_token": np.ones(
                    (1, 4),
                    dtype=np.float32,
                ),
            }
        )


def test_convert_rfdetr_state_dict_slices_grouped_queries_per_group():
    out = dict(
        convert_rfdetr_state_dict(
            {
                "__args_json__": np.array('{"num_queries": 4, "group_detr": 3}'),
                "query_feat.weight": _labelled_query_tensor(num_queries=4, group_detr=3, dim=2),
                "refpoint_embed.weight": _labelled_query_tensor(num_queries=4, group_detr=3, dim=4),
            },
            target_num_queries=2,
            target_group_detr=3,
            target_query_dim=4,
        )
    )

    expected_first_columns = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(out["decoder.query_embed"], expected_first_columns)
    np.testing.assert_array_equal(out["decoder.reference_embed"][:, :2], expected_first_columns)
    assert out["decoder.reference_embed"].shape == (6, 4)


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


def test_load_rfdetr_weights_strict_rejects_missing_model_params(tmp_path):
    model = RFDETRModel(_cfg())
    weights_path = tmp_path / "partial.npz"
    np.savez(weights_path, **{"head.class_embed.bias": np.zeros((3,), dtype=np.float32)})

    with pytest.raises(ValueError, match="missing RF-DETR inference weights"):
        load_rfdetr_weights(model, weights_path, strict=True)
