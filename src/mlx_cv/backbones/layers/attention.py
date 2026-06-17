"""Multi-head self-attention (build-once family).

Packed-qkv projection + manual-softmax scaled-dot-product attention, faithful
to the official DINOv3 ``SelfAttention`` (`references/dinov3`): scale
``head_dim**-0.5``, qkv reshaped ``(B, N, 3, h, Dh)``, manual softmax (CPU/Metal
parity). RoPE is an **optional** hook — pass ``rope=(sin, cos)`` to rotate q/k
on the suffix tokens (``n_prefix`` cls/storage tokens skipped); pass ``None``
(DINOv2 and other abs-posenc backbones) to skip it entirely. Q/K LayerNorm is
also optional and parameter-free when disabled, preserving the default parameter
tree.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .position import apply_rope_prefixed

__all__ = ["Attention"]


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(head_dim, eps=norm_eps) if qk_norm else None
        self.k_norm = nn.LayerNorm(head_dim, eps=norm_eps) if qk_norm else None
        self.proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
        n_prefix: int = 0,
    ) -> mx.array:
        b, n, c = x.shape
        h, dh = self.num_heads, c // self.num_heads
        qkv = self.qkv(x).reshape(b, n, 3, h, dh)
        q = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))   # (B, h, N, Dh)
        k = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            sin, cos = rope
            q = apply_rope_prefixed(q, sin, cos, n_prefix)
            k = apply_rope_prefixed(k, sin, cos, n_prefix)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v                                  # (B, h, N, Dh)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, n, c)
        return self.proj(out)
