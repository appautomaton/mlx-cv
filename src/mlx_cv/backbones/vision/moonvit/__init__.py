"""MoonViT-SO-400M vision backbone (MLX) — Stage 2 (not yet implemented).

Native/variable-resolution ViT with 2D-RoPE, per-image block attention, conv patch
embed (fused ``wqkv``), and 2×2 token merge. The highest-risk module to match
numerically (§16.7). Will register as ``@register_backbone("moonvit-so400m",
kind="vision")`` and satisfy the vision-backbone contract. Requires the ``mlx`` extra.
"""

from __future__ import annotations

__all__: list[str] = []
