"""Unit tests for the shared build-once layer families (`backbones/layers`)."""

import numpy as np

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_cv.backbones.layers import Attention, MlpFFN, PatchEmbed, TransformerBlock
from mlx_cv.backbones.layers.position import (
    LearnedAbsPosEmb,
    apply_rope,
    rope_axial_periods,
    rope_axial_sincos,
    rotate_half,
)


def test_patch_embed_shape_and_grid():
    pe = PatchEmbed(in_chans=3, embed_dim=8, patch_size=16)
    mx.eval(pe.parameters())
    tokens, grid = pe(mx.zeros((1, 3, 32, 32)))
    assert tokens.shape == (1, 4, 8)   # 32/16 = 2 -> 2x2 = 4 patches
    assert grid == (2, 2)


def test_attention_shape_with_and_without_rope():
    attn = Attention(dim=16, num_heads=2)
    mx.eval(attn.parameters())
    x = mx.random.normal((1, 5, 16))   # 1 cls + 4 patch tokens
    assert attn(x, rope=None, n_prefix=1).shape == (1, 5, 16)
    sin, cos = rope_axial_sincos(rope_axial_periods(8, 100.0), 2, 2)
    assert attn(x, rope=(sin, cos), n_prefix=1).shape == (1, 5, 16)


def test_attention_qk_norm_is_opt_in_and_default_tree_is_unchanged():
    attn = Attention(dim=16, num_heads=2)
    keys = [key for key, _ in tree_flatten(attn.parameters())]

    assert keys == ["qkv.weight", "qkv.bias", "proj.weight", "proj.bias"]
    assert not any(key.startswith("q_norm.") or key.startswith("k_norm.") for key in keys)


def test_attention_qk_norm_creates_per_head_norm_params():
    attn = Attention(dim=16, num_heads=2, qk_norm=True)
    params = dict(tree_flatten(attn.parameters()))

    assert params["q_norm.weight"].shape == (8,)
    assert params["q_norm.bias"].shape == (8,)
    assert params["k_norm.weight"].shape == (8,)
    assert params["k_norm.bias"].shape == (8,)
    assert attn(mx.random.normal((1, 5, 16))).shape == (1, 5, 16)


def test_attention_rope_only_touches_suffix():
    # rope=None must equal rope with an all-prefix sequence (n_prefix == N): nothing to rotate.
    attn = Attention(dim=16, num_heads=2)
    mx.eval(attn.parameters())
    x = mx.random.normal((1, 4, 16))
    sin, cos = rope_axial_sincos(rope_axial_periods(8, 100.0), 1, 1)  # 1 patch slot, unused
    out_prefix_all = attn(x, rope=(sin, cos), n_prefix=4)             # all tokens are prefix
    out_none = attn(x, rope=None, n_prefix=4)
    assert mx.allclose(out_prefix_all, out_none, atol=1e-6)


def test_rope_periods_length_is_head_dim_over_4():
    assert rope_axial_periods(8, 100.0).shape == (2,)


def test_learned_abs_pos_offset_matches_torch_bicubic_scale_factor():
    torch = pytest.importorskip("torch")
    torch_f = pytest.importorskip("torch.nn.functional")

    dim = 3
    table = np.linspace(-0.5, 0.5, (1 + 37 * 37) * dim, dtype=np.float32).reshape(1, 1 + 37 * 37, dim)
    emb = LearnedAbsPosEmb(dim, 37, interpolate_offset=0.1)
    emb.table = mx.array(table)

    out = emb((8, 8))
    mx.eval(out)

    ref_table = torch.tensor(table)
    ref_patch = torch_f.interpolate(
        ref_table[:, 1:].reshape(1, 37, 37, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        align_corners=False,
        scale_factor=((8.0 + 0.1) / 37.0, (8.0 + 0.1) / 37.0),
    )
    assert ref_patch.shape[-2:] == (8, 8)
    ref = torch.cat(
        [ref_table[:, :1], ref_patch.permute(0, 2, 3, 1).reshape(1, 8 * 8, dim)],
        dim=1,
    )

    np.testing.assert_allclose(np.array(out), ref.numpy(), rtol=3e-5, atol=3e-5)


def test_apply_rope_identity_when_sin0_cos1():
    x = mx.random.normal((1, 2, 4, 6))
    out = apply_rope(x, sin=mx.zeros((4, 6)), cos=mx.ones((4, 6)))
    assert mx.allclose(out, x, atol=1e-6)


def test_rotate_half_negates_swapped_halves():
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    assert mx.array_equal(rotate_half(x), mx.array([[-3.0, -4.0, 1.0, 2.0]]))


def test_mlp_gelu_shape():
    mlp = MlpFFN(dim=8, hidden=32)
    mx.eval(mlp.parameters())
    assert mlp(mx.random.normal((1, 3, 8))).shape == (1, 3, 8)


def test_mlp_swiglu_slot_raises():
    with pytest.raises(NotImplementedError):
        MlpFFN(dim=8, hidden=32, kind="swiglu")


def test_block_forward_shape():
    blk = TransformerBlock(16, 2, layerscale=False)
    mx.eval(blk.parameters())
    assert blk(mx.random.normal((1, 5, 16)), rope=None, n_prefix=1).shape == (1, 5, 16)


def test_block_layerscale_off_has_no_scale_params():
    blk = TransformerBlock(16, 2, layerscale=False)
    keys = [k for k, _ in tree_flatten(blk.parameters())]
    assert keys and not any(("ls1" in k) or ("ls2" in k) or ("gamma" in k) for k in keys)


def test_block_layerscale_on_zero_init_is_identity():
    # LayerScale(init=0) zeroes both residual branches -> block output == input.
    blk = TransformerBlock(16, 2, layerscale=True, layerscale_init=0.0)
    mx.eval(blk.parameters())
    x = mx.random.normal((1, 5, 16))
    assert mx.allclose(blk(x, rope=None, n_prefix=1), x, atol=1e-6)


def test_block_layerscale_on_creates_gamma_params():
    blk = TransformerBlock(16, 2, layerscale=True)
    keys = [k for k, _ in tree_flatten(blk.parameters())]
    assert any("ls1.gamma" in k for k in keys) and any("ls2.gamma" in k for k in keys)


def test_block_can_enable_shared_attention_qk_norm():
    blk = TransformerBlock(16, 2, qk_norm=True)
    params = dict(tree_flatten(blk.parameters()))

    assert params["attn.q_norm.weight"].shape == (8,)
    assert params["attn.k_norm.weight"].shape == (8,)


def test_block_rmsnorm_slot_raises():
    with pytest.raises(NotImplementedError):
        TransformerBlock(16, 2, norm="rmsnorm")
