import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.models.eomt_dinov3 import convert_eomt_dinov3_state_dict


def test_official_transformers_keys_pack_qkv_and_convert_convolutions():
    state = {
        "attn_mask_probs": np.ones((2,), dtype=np.float32),
        "criterion.empty_weight": np.ones((5,), dtype=np.float32),
        "embeddings.cls_token": np.ones((1, 1, 8), dtype=np.float32),
        "embeddings.register_tokens": np.ones((1, 2, 8), dtype=np.float32),
        "embeddings.patch_embeddings.weight": np.arange(8 * 3 * 2 * 2, dtype=np.float32).reshape(8, 3, 2, 2),
        "embeddings.patch_embeddings.bias": np.ones((8,), dtype=np.float32),
        "layers.0.attention.q_proj.weight": np.full((8, 8), 1.0, dtype=np.float32),
        "layers.0.attention.k_proj.weight": np.full((8, 8), 2.0, dtype=np.float32),
        "layers.0.attention.v_proj.weight": np.full((8, 8), 3.0, dtype=np.float32),
        "layers.0.attention.q_proj.bias": np.full((8,), 4.0, dtype=np.float32),
        "layers.0.attention.v_proj.bias": np.full((8,), 6.0, dtype=np.float32),
        "layers.0.attention.o_proj.weight": np.ones((8, 8), dtype=np.float32),
        "layers.0.layer_scale1.lambda1": np.ones((8,), dtype=np.float32),
        "layers.0.layer_scale2.lambda1": np.ones((8,), dtype=np.float32),
        "layers.0.norm1.weight": np.ones((8,), dtype=np.float32),
        "layers.0.norm2.weight": np.ones((8,), dtype=np.float32),
        "layers.0.mlp.up_proj.weight": np.ones((16, 8), dtype=np.float32),
        "layers.0.mlp.down_proj.weight": np.ones((8, 16), dtype=np.float32),
        "layernorm.weight": np.ones((8,), dtype=np.float32),
        "query.weight": np.ones((3, 8), dtype=np.float32),
        "mask_head.fc1.weight": np.ones((8, 8), dtype=np.float32),
        "class_predictor.weight": np.ones((5, 8), dtype=np.float32),
        "upscale_block.block.0.conv1.weight": np.ones((8, 8, 2, 2), dtype=np.float32),
        "upscale_block.block.0.conv2.weight": np.ones((8, 1, 3, 3), dtype=np.float32),
        "upscale_block.block.0.layernorm2d.weight": np.ones((8,), dtype=np.float32),
    }

    converted = dict(convert_eomt_dinov3_state_dict(state))

    np.testing.assert_array_equal(converted["attn_mask_probs"], 1.0)
    assert converted["backbone.patch_embed.proj.weight"].shape == (8, 2, 2, 3)
    qkv_weight = np.asarray(converted["backbone.blocks.0.attn.qkv.weight"])
    qkv_bias = np.asarray(converted["backbone.blocks.0.attn.qkv.bias"])
    np.testing.assert_array_equal(qkv_weight[:8], 1.0)
    np.testing.assert_array_equal(qkv_weight[8:16], 2.0)
    np.testing.assert_array_equal(qkv_weight[16:], 3.0)
    np.testing.assert_array_equal(qkv_bias[:8], 4.0)
    np.testing.assert_array_equal(qkv_bias[8:16], 0.0)
    np.testing.assert_array_equal(qkv_bias[16:], 6.0)
    assert converted["upscale_block.block.0.conv1.weight"].shape == (8, 2, 2, 8)
    assert converted["upscale_block.block.0.conv2.weight"].shape == (8, 3, 3, 1)


def test_converter_rejects_unknown_checkpoint_branches():
    with pytest.raises(ValueError, match="unsupported EoMT-DINOv3 checkpoint keys"):
        convert_eomt_dinov3_state_dict({"unknown.weight": np.ones((1,), dtype=np.float32)})
