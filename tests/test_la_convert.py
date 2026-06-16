import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.models.locateanything import convert_state_dict, load_locateanything_weights, remap_key
from mlx_cv.models.locateanything.config import LocateAnythingConfig
from mlx_cv.models.locateanything.modeling import LocateAnythingModel


def test_drop_tied_lm_head():
    assert remap_key("language_model.lm_head.weight") is None


def test_vision_rename():
    assert remap_key("vision_model.encoder.blocks.0.self_attn.wqkv.weight") == \
        "vision_tower.blocks.0.self_attn.wqkv.weight"
    assert remap_key("vision_model.patch_embed.proj.weight") == \
        "vision_tower.patch_embed.proj.weight"


def test_projector_rename():
    assert remap_key("mlp1.0.weight") == "multi_modal_projector.layer_norm.weight"
    assert remap_key("mlp1.1.bias") == "multi_modal_projector.linear_1.bias"
    assert remap_key("mlp1.3.weight") == "multi_modal_projector.linear_2.weight"


def test_language_unchanged():
    k = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert remap_key(k) == k


def test_convert_state_dict():
    embed = np.ones((4, 8), dtype=np.float32)
    w = {
        "language_model.lm_head.weight": embed,
        "language_model.model.embed_tokens.weight": embed,
        "vision_model.encoder.blocks.0.x": np.asarray(2),
        "mlp1.0.weight": np.asarray(3),
        "language_model.model.y": np.asarray(4),
    }
    out = dict(convert_state_dict(w))
    assert "language_model.lm_head.weight" not in out
    assert out["vision_tower.blocks.0.x"] == 2
    assert out["multi_modal_projector.layer_norm.weight"] == 3
    assert out["language_model.model.y"] == 4


def test_convert_state_dict_delegates_moonvit_conv_transpose_and_qwen2_tied_drop():
    embed = np.arange(32, dtype=np.float32).reshape(4, 8)
    state = {
        "vision_model.patch_embed.proj.weight": np.ones((8, 3, 2, 2), dtype=np.float32),
        "language_model.model.embed_tokens.weight": embed,
        "language_model.lm_head.weight": embed.copy(),
    }
    out = dict(convert_state_dict(state))

    assert out["vision_tower.patch_embed.proj.weight"].shape == (8, 2, 2, 3)
    assert "language_model.lm_head.weight" not in out
    assert "language_model.model.embed_tokens.weight" in out


def test_convert_state_dict_rejects_untied_full_lm_head():
    state = {
        "language_model.model.embed_tokens.weight": np.zeros((4, 8), dtype=np.float32),
        "language_model.lm_head.weight": np.ones((4, 8), dtype=np.float32),
    }
    try:
        convert_state_dict(state)
    except ValueError as exc:
        assert "language_model.lm_head.weight" in str(exc)
    else:
        raise AssertionError("expected untied language_model.lm_head.weight to fail")


def _tiny_config():
    text = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        text_mask_token_id=7,
    )
    vision = MoonViTConfig(
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        intermediate_size=16,
        patch_size=2,
        num_channels=1,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        merge_kernel_size=(2, 2),
    )
    return LocateAnythingConfig(
        vision_config=vision,
        text_config=text,
        vocab_size=16,
        image_token_index=5,
        text_mask_token_id=7,
    )


def test_load_locateanything_weights_populates_projector_and_scatter(tmp_path):
    model = LocateAnythingModel(_tiny_config())
    params = dict(tree_flatten(model.parameters()))
    embed = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 100.0
    state = {
        "language_model.model.embed_tokens.weight": embed,
        "language_model.lm_head.weight": embed.copy(),
        "mlp1.0.weight": np.ones(params["multi_modal_projector.layer_norm.weight"].shape, dtype=np.float32),
        "mlp1.0.bias": np.zeros(params["multi_modal_projector.layer_norm.bias"].shape, dtype=np.float32),
        "mlp1.1.weight": np.ones(params["multi_modal_projector.linear_1.weight"].shape, dtype=np.float32),
        "mlp1.1.bias": np.zeros(params["multi_modal_projector.linear_1.bias"].shape, dtype=np.float32),
        "mlp1.3.weight": np.ones(params["multi_modal_projector.linear_2.weight"].shape, dtype=np.float32),
        "mlp1.3.bias": np.zeros(params["multi_modal_projector.linear_2.bias"].shape, dtype=np.float32),
    }
    weights_path = tmp_path / "la_weights.npz"
    np.savez(weights_path, **state)

    loaded = load_locateanything_weights(model, weights_path)
    input_ids = mx.array([[1, 5, 2]], dtype=mx.int32)
    cached = [mx.ones((1, 32), dtype=mx.float32)]
    with mx.stream(mx.cpu):
        embeds = loaded.get_input_embeddings(input_ids, cached_image_features=cached)
        mx.eval(embeds)

    assert embeds.shape == (1, 3, 8)
