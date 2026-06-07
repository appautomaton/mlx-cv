from mlx_cv.models.locateanything import (
    LocateAnythingConfig,
    MoonViTConfig,
    Qwen2Config,
)


def test_moonvit_defaults():
    v = MoonViTConfig()
    assert (v.hidden_size, v.num_hidden_layers, v.num_attention_heads) == (1152, 27, 16)
    assert v.patch_size == 14 and v.merge_kernel_size == (2, 2)
    assert v.spatial_merge_size == 2


def test_qwen2_defaults():
    t = Qwen2Config()
    assert (t.hidden_size, t.num_hidden_layers) == (2048, 36)
    assert (t.num_attention_heads, t.num_key_value_heads) == (16, 2)
    assert t.vocab_size == 152681 and t.tie_word_embeddings
    assert t.block_size == 6 and t.causal_attn is False


def test_locateanything_tokens_and_grid():
    c = LocateAnythingConfig()
    assert isinstance(c.vision_config, MoonViTConfig)
    assert isinstance(c.text_config, Qwen2Config)
    assert c.image_token_index == 151665
    assert c.box_start_token_id == 151668 and c.box_end_token_id == 151669
    # the [0, 1000] coordinate grid
    assert c.coord_start_token_id == 151677
    assert c.coord_end_token_id - c.coord_start_token_id == 1000
