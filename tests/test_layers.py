"""Unit tests for the shared build-once layer families (`backbones/layers`)."""

import mlx.core as mx
import pytest

from mlx_cv.backbones.layers import Attention, MlpFFN, PatchEmbed
from mlx_cv.backbones.layers.position import (
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
