import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.backbones.vision.moonvit.modeling import (
    Learnable2DInterpPosEmb,
    MoonViTPatchEmbed,
    Rope2DPosEmb,
    apply_rope,
    cu_seqlens_from_grid_hws,
    make_block_attention_mask,
    patch_merger,
)


def test_cu_seqlens_and_block_attention_mask_isolate_images():
    grid_hws = mx.array([[2, 2], [1, 2]], dtype=mx.int32)
    cu = cu_seqlens_from_grid_hws(grid_hws)
    mask = make_block_attention_mask(cu, 6)

    assert np.array_equal(np.array(cu), np.array([0, 4, 6], dtype=np.int32))
    expected = np.zeros((6, 6), dtype=bool)
    expected[:4, :4] = True
    expected[4:, 4:] = True
    assert np.array_equal(np.array(mask), expected)


def test_patch_embed_accepts_packed_nchw_patches_and_adds_pos_emb():
    cfg = MoonViTConfig(
        hidden_size=1,
        num_attention_heads=1,
        patch_size=2,
        num_channels=1,
        init_pos_emb_height=1,
        init_pos_emb_width=1,
    )
    embed = MoonViTPatchEmbed(cfg)
    embed.proj.weight = mx.ones((1, 2, 2, 1))
    embed.proj.bias = mx.zeros((1,))
    embed.pos_emb.weight = mx.zeros((1, 1, 1))
    x = mx.array(np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2))

    out = embed(x, mx.array([[1, 1]], dtype=mx.int32))
    mx.eval(out)

    assert out.shape == (1, 1)
    assert np.allclose(np.array(out), np.array([[6.0]], dtype=np.float32))


def test_patch_embed_rejects_unpacked_image_shape():
    cfg = MoonViTConfig(hidden_size=1, num_attention_heads=1, patch_size=2, num_channels=1)
    embed = MoonViTPatchEmbed(cfg)
    with pytest.raises(ValueError, match="already-patchified"):
        embed(mx.zeros((1, 1, 4, 4)), mx.array([[2, 2]], dtype=mx.int32))


def test_learnable_2d_pos_emb_same_grid_and_interpolated_grid():
    emb = Learnable2DInterpPosEmb(height=2, width=2, dim=1)
    emb.weight = mx.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    x = mx.zeros((4, 1))
    same = emb(x, mx.array([[2, 2]], dtype=mx.int32))
    assert np.allclose(np.array(same), np.array([[1.0], [2.0], [3.0], [4.0]]))

    scalar = Learnable2DInterpPosEmb(height=1, width=1, dim=2)
    scalar.weight = mx.array([[[5.0, 7.0]]])
    up = scalar(mx.zeros((4, 2)), mx.array([[2, 2]], dtype=mx.int32))
    assert up.shape == (4, 2)
    assert np.allclose(np.array(up), np.array([[5.0, 7.0]] * 4), atol=1e-6)


def test_rope_2d_matches_reference_complex_layout_for_small_grid():
    rope = Rope2DPosEmb(dim=4, max_height=4, max_width=4)
    freqs = rope.get_freqs_cis(mx.array([[2, 2]], dtype=mx.int32))
    got = np.array(freqs)

    assert got.shape == (4, 2)
    assert np.allclose(got[0], np.array([1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64))
    assert np.allclose(got[1], np.array([np.cos(1.0) + 1j * np.sin(1.0), 1.0 + 0.0j], dtype=np.complex64))
    assert np.allclose(got[2], np.array([1.0 + 0.0j, np.cos(1.0) + 1j * np.sin(1.0)], dtype=np.complex64))


def test_apply_rope_identity_and_validation():
    q = mx.array(np.arange(8, dtype=np.float32).reshape(2, 1, 4))
    k = q + 1.0
    freqs = mx.ones((2, 2), dtype=mx.complex64)
    q_out, k_out = apply_rope(q, k, freqs)
    assert np.allclose(np.array(q_out), np.array(q))
    assert np.allclose(np.array(k_out), np.array(k))

    with pytest.raises(ValueError, match="head dim"):
        apply_rope(q, k, mx.ones((2, 1), dtype=mx.complex64))


def test_patch_merger_returns_reference_flattened_window_shape_and_order():
    x = mx.array(np.arange(8, dtype=np.float32).reshape(8, 1))
    merged = patch_merger(x, mx.array([[4, 2]], dtype=mx.int32), merge_kernel_size=(2, 2))

    assert len(merged) == 1
    assert merged[0].shape == (2, 4)
    assert np.array_equal(np.array(merged[0]), np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.float32))


def test_patch_merger_rejects_bad_grid_or_length():
    with pytest.raises(ValueError, match="divisible"):
        patch_merger(mx.zeros((3, 1)), mx.array([[3, 1]], dtype=mx.int32))
    with pytest.raises(ValueError, match="does not match"):
        patch_merger(mx.zeros((3, 1)), mx.array([[2, 2]], dtype=mx.int32))
