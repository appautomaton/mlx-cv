"""Parameterized pre-norm Transformer block (build-once family).

Composes the shared `Attention` + `MlpFFN` with selectable axes:

* **norm** — ``"layernorm"`` (DINOv3/DINOv2); ``"rmsnorm"`` reserved (Qwen2 ships its own ``Qwen2RMSNorm``).
* **ffn** — ``"gelu"`` (DINOv3/DINOv2); ``"swiglu"`` reserved (forwarded to `MlpFFN`).
* **qk_norm** — optional per-head LayerNorm on attention Q/K (DA3-style), off
  by default so existing block parameter trees are unchanged.
* **layerscale** — ``off`` (DINOv3) or ``on`` (DINOv2). **When off, no scale
  params are created**, so a DINOv3 block's param tree is byte-identical to the
  pre-extraction inline block (parity-preserving).

RoPE is passed through to attention (``rope=(sin,cos)`` for DINOv3) or omitted
(``rope=None`` for abs-posenc backbones like DINOv2).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .mlp import MlpFFN

__all__ = ["TransformerBlock", "LayerScale"]


class LayerScale(nn.Module):
    """Per-channel learned residual scale (DINOv2-style)."""

    def __init__(self, dim: int, init: float = 1.0) -> None:
        super().__init__()
        self.gamma = mx.full((dim,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


def _make_norm(kind: str, dim: int, eps: float) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if kind == "rmsnorm":
        raise NotImplementedError(
            "norm 'rmsnorm' is a reserved slot with no generic consumer; "
            "Qwen2 ships its own Qwen2RMSNorm."
        )
    raise ValueError(f"unknown norm kind {kind!r}")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm: str = "layernorm",
        norm_eps: float = 1e-6,
        ffn: str = "gelu",
        qk_norm: bool = False,
        layerscale: bool = False,
        layerscale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = _make_norm(norm, dim, norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_eps=norm_eps)
        self.norm2 = _make_norm(norm, dim, norm_eps)
        self.mlp = MlpFFN(dim, int(dim * mlp_ratio), kind=ffn)
        # No scale params when off -> param tree matches a plain DINOv3 block.
        self.ls1 = LayerScale(dim, layerscale_init) if layerscale else None
        self.ls2 = LayerScale(dim, layerscale_init) if layerscale else None

    def __call__(
        self,
        x: mx.array,
        rope: tuple[mx.array, mx.array] | None = None,
        n_prefix: int = 0,
    ) -> mx.array:
        a = self.attn(self.norm1(x), rope=rope, n_prefix=n_prefix)
        x = x + (a if self.ls1 is None else self.ls1(a))
        m = self.mlp(self.norm2(x))
        x = x + (m if self.ls2 is None else self.ls2(m))
        return x
