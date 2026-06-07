from mlx_cv.models.locateanything import convert_state_dict, remap_key


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
    w = {
        "language_model.lm_head.weight": 1,
        "vision_model.encoder.blocks.0.x": 2,
        "mlp1.0.weight": 3,
        "language_model.model.y": 4,
    }
    out = convert_state_dict(w)
    assert "language_model.lm_head.weight" not in out
    assert out["vision_tower.blocks.0.x"] == 2
    assert out["multi_modal_projector.layer_norm.weight"] == 3
    assert out["language_model.model.y"] == 4
