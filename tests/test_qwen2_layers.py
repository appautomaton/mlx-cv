import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.modeling import (
    Qwen2MLP,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rotate_half_np(x: np.ndarray) -> np.ndarray:
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate([-x2, x1], axis=-1)


def test_qwen2_rmsnorm_matches_reference_formula_and_dtype_path():
    norm = Qwen2RMSNorm(4, eps=1e-6)
    weight = np.array([1.0, 2.0, 0.5, -1.0], dtype=np.float32)
    norm.weight = mx.array(weight)

    x = np.array([[[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]]], dtype=np.float16)
    out = norm(mx.array(x))

    x32 = x.astype(np.float32)
    expected = (x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)).astype(
        np.float16
    )
    expected = expected.astype(np.float32) * weight

    assert np.allclose(np.array(out), expected, rtol=1e-3, atol=1e-3)


def test_qwen2_mlp_matches_swiglu_formula():
    cfg = Qwen2Config(hidden_size=3, intermediate_size=4)
    mlp = Qwen2MLP(cfg)
    gate = np.array(
        [
            [0.1, 0.2, -0.3],
            [-0.4, 0.5, 0.6],
            [0.7, -0.8, 0.9],
            [1.0, -1.1, 1.2],
        ],
        dtype=np.float32,
    )
    up = np.array(
        [
            [-0.2, 0.3, 0.4],
            [0.5, -0.6, 0.7],
            [0.8, 0.9, -1.0],
            [-1.1, 1.2, 1.3],
        ],
        dtype=np.float32,
    )
    down = np.array(
        [
            [0.2, -0.3, 0.4, -0.5],
            [0.6, 0.7, -0.8, 0.9],
            [-1.0, 1.1, 1.2, -1.3],
        ],
        dtype=np.float32,
    )
    mlp.gate_proj.weight = mx.array(gate)
    mlp.up_proj.weight = mx.array(up)
    mlp.down_proj.weight = mx.array(down)

    x = np.array([[[0.25, -0.5, 0.75], [1.0, 0.5, -1.5]]], dtype=np.float32)
    with mx.stream(mx.cpu):
        out = mlp(mx.array(x))
        mx.eval(out)

    gate_out = x @ gate.T
    up_out = x @ up.T
    expected = (_silu(gate_out) * up_out) @ down.T

    assert np.allclose(np.array(out), expected, rtol=1e-6, atol=1e-6)


def test_qwen2_leaf_parameter_layout_matches_reference_bias_rules():
    norm = Qwen2RMSNorm(4)
    mlp = Qwen2MLP(Qwen2Config(hidden_size=4, intermediate_size=8))

    assert [key for key, _ in tree_flatten(norm.parameters())] == ["weight"]
    keys = [key for key, _ in tree_flatten(mlp.parameters())]
    assert "gate_proj.weight" in keys
    assert "up_proj.weight" in keys
    assert "down_proj.weight" in keys
    assert not any(key.endswith(".bias") for key in keys)


def test_qwen2_rotary_embedding_matches_half_split_reference_tables():
    dim = 4
    base = 10000.0
    rope = Qwen2RotaryEmbedding(dim=dim, max_position_embeddings=8, base=base)
    x = mx.zeros((1, 2, 3, dim), dtype=mx.float32)
    cos, sin = rope(x, seq_len=5)

    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    freqs = np.outer(np.arange(5, dtype=np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)

    assert np.allclose(np.array(cos), np.cos(emb), rtol=1e-6, atol=1e-6)
    assert np.allclose(np.array(sin), np.sin(emb), rtol=1e-6, atol=1e-6)


def test_qwen2_apply_rotary_pos_emb_uses_explicit_position_ids():
    rope = Qwen2RotaryEmbedding(dim=4, max_position_embeddings=8, base=10000.0)
    x = mx.zeros((1, 1, 3, 4), dtype=mx.float32)
    cos, sin = rope(x, seq_len=4)
    position_ids = mx.array([[2, 0, 1]])

    q_np = np.array([[[[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 1.5, -2.0], [2.0, 0.0, -1.0, 3.0]]]], dtype=np.float32)
    k_np = q_np * 0.5
    q_out, k_out = apply_rotary_pos_emb(mx.array(q_np), mx.array(k_np), cos, sin, position_ids)

    cos_np = np.array(cos)[np.array(position_ids)][:, None, :, :]
    sin_np = np.array(sin)[np.array(position_ids)][:, None, :, :]
    expected_q = q_np * cos_np + _rotate_half_np(q_np) * sin_np
    expected_k = k_np * cos_np + _rotate_half_np(k_np) * sin_np

    assert np.allclose(np.array(q_out), expected_q, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.array(k_out), expected_k, rtol=1e-6, atol=1e-6)


def test_qwen2_rotate_half_and_repeat_kv():
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.array_equal(np.array(rotate_half(x)), np.array([[-3.0, -4.0, 1.0, 2.0]], dtype=np.float32))

    kv_np = np.arange(1 * 2 * 3 * 2, dtype=np.float32).reshape(1, 2, 3, 2)
    kv = mx.array(kv_np)
    repeated = repeat_kv(kv, n_rep=3)

    assert np.array_equal(np.array(repeat_kv(kv, n_rep=1)), kv_np)
    assert np.array_equal(np.array(repeated), np.repeat(kv_np, repeats=3, axis=1))
