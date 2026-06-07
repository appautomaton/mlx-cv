"""Qwen2.5 language backbone (MLX) — Stage 2 (not yet implemented).

Standard Qwen2.5 causal LM (GQA 16/2, tied embeddings, RoPE theta 1e6) plus the
non-causal "magi" block mask PBD needs for parallel decode (§16.7). Will register as
``@register_backbone("qwen2.5-3b", kind="llm")`` and satisfy the language-backbone
contract. Requires the ``mlx`` extra.
"""

from __future__ import annotations

__all__: list[str] = []
