import math

import numpy as np
import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.masks import make_causal_mask_4d
from mlx_cv.backbones.llm.qwen2.modeling import Qwen2Attention


def _rotate_half(x: np.ndarray) -> np.ndarray:
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate([-x2, x1], axis=-1)


def _rope_tables(seq_len: int, dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freqs = np.outer(np.arange(seq_len, dtype=np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _make_tiny_attention() -> Qwen2Attention:
    cfg = Qwen2Config(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=8,
        max_position_embeddings=16,
        rope_theta=10000.0,
    )
    attn = Qwen2Attention(cfg, layer_idx=0)
    attn.q_proj.weight = mx.array(
        np.array(
            [
                [0.2, -0.1, 0.3, 0.4],
                [-0.5, 0.6, 0.7, -0.8],
                [0.9, -1.0, 1.1, 0.1],
                [0.2, 0.3, -0.4, 0.5],
            ],
            dtype=np.float32,
        )
    )
    attn.q_proj.bias = mx.array(np.array([0.01, -0.02, 0.03, -0.04], dtype=np.float32))
    attn.k_proj.weight = mx.array(
        np.array(
            [
                [0.3, -0.2, 0.1, 0.5],
                [-0.6, 0.7, -0.8, 0.9],
            ],
            dtype=np.float32,
        )
    )
    attn.k_proj.bias = mx.array(np.array([0.05, -0.07], dtype=np.float32))
    attn.v_proj.weight = mx.array(
        np.array(
            [
                [-0.2, 0.4, 0.6, -0.8],
                [1.0, -1.2, 1.4, -1.6],
            ],
            dtype=np.float32,
        )
    )
    attn.v_proj.bias = mx.array(np.array([0.09, -0.11], dtype=np.float32))
    attn.o_proj.weight = mx.array(
        np.array(
            [
                [0.1, 0.2, -0.3, 0.4],
                [-0.5, 0.6, 0.7, -0.8],
                [0.9, -1.0, 1.1, 1.2],
                [-1.3, 1.4, -1.5, 1.6],
            ],
            dtype=np.float32,
        )
    )
    return attn


def _expected_attention(attn: Qwen2Attention, x: np.ndarray, position_ids: np.ndarray, mask: np.ndarray):
    q_w = np.array(attn.q_proj.weight)
    q_b = np.array(attn.q_proj.bias)
    k_w = np.array(attn.k_proj.weight)
    k_b = np.array(attn.k_proj.bias)
    v_w = np.array(attn.v_proj.weight)
    v_b = np.array(attn.v_proj.bias)
    o_w = np.array(attn.o_proj.weight)

    batch, seq_len, hidden = x.shape
    heads = attn.num_heads
    kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim
    q = (x @ q_w.T + q_b).reshape(batch, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
    k = (x @ k_w.T + k_b).reshape(batch, seq_len, kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = (x @ v_w.T + v_b).reshape(batch, seq_len, kv_heads, head_dim).transpose(0, 2, 1, 3)

    cos, sin = _rope_tables(int(position_ids.max()) + 1, head_dim, attn.rope_theta)
    cos = cos[position_ids][:, None, :, :]
    sin = sin[position_ids][:, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    k = np.repeat(k, repeats=attn.num_key_value_groups, axis=1)
    v = np.repeat(v, repeats=attn.num_key_value_groups, axis=1)

    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    probs = _softmax(scores + mask, axis=-1).astype(np.float32)
    out = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)
    return out @ o_w.T, probs


def test_qwen2_attention_matches_numpy_gqa_additive_mask_and_position_ids():
    attn = _make_tiny_attention()
    x = np.array(
        [
            [
                [0.25, -0.5, 0.75, 1.0],
                [-1.0, 0.5, 1.5, -0.25],
                [0.0, 1.25, -1.5, 0.5],
            ]
        ],
        dtype=np.float32,
    )
    position_ids = np.array([[0, 2, 1]], dtype=np.int32)
    mask = np.array(make_causal_mask_4d(1, 3))
    expected_out, expected_probs = _expected_attention(attn, x, position_ids, mask)

    with mx.stream(mx.cpu):
        out, probs, cache = attn(
            mx.array(x),
            attention_mask=mx.array(mask),
            position_ids=mx.array(position_ids),
            output_attentions=True,
        )
        mx.eval(out, probs)

    assert cache is None
    assert out.shape == (1, 3, 4)
    assert probs.shape == (1, 2, 3, 3)
    assert np.allclose(np.array(out), expected_out, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.array(probs), expected_probs, rtol=1e-6, atol=1e-6)


def test_qwen2_attention_bias_layout_and_mask_shape_validation():
    attn = _make_tiny_attention()
    keys = [key for key, _ in tree_flatten(attn.parameters())]
    assert "q_proj.bias" in keys
    assert "k_proj.bias" in keys
    assert "v_proj.bias" in keys
    assert "o_proj.weight" in keys
    assert "o_proj.bias" not in keys
    assert not any("rotary_emb" in key for key in keys)

    with pytest.raises(ValueError, match="Attention mask"):
        attn(
            mx.zeros((1, 3, 4)),
            attention_mask=mx.zeros((1, 1, 2, 3)),
            position_ids=mx.array([[0, 1, 2]], dtype=mx.int32),
        )
