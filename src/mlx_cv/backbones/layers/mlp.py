"""Feed-forward family (build-once).

``MlpFFN`` is the plain two-layer FFN with exact-erf GELU (matching torch
``nn.GELU()``), as DINOv3 and DINOv2 both use. ``kind`` is the selectable
slot: ``"swiglu"`` stays reserved with no generic consumer — Qwen2 ships its
own fused SwiGLU (``Qwen2MLP``), so this slot raises rather than guess a build.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["MlpFFN"]


class MlpFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, *, kind: str = "gelu") -> None:
        super().__init__()
        if kind != "gelu":
            raise NotImplementedError(
                f"FFN kind {kind!r} is a reserved slot with no generic consumer; "
                "Qwen2 ships its own fused SwiGLU (Qwen2MLP)."
            )
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))           # exact (erf) GELU
