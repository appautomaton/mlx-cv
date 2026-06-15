"""Shared mlx ``nn.Module`` building blocks — the build-once families.

These are the reusable leaves every ViT-family backbone composes from
(DINOv3, DINOv2, …): patch embedding, packed-qkv attention, the FFN, and
(in `position`) the positional-encoding suite. They are parameterized by
plain dims/flags, never by a model-specific config, so a second backbone
instantiates with no new block code.

mlx lives here, behind the ``[mlx]`` extra; ``core/`` stays mlx-free.
"""

from __future__ import annotations

from .attention import Attention
from .block import LayerScale, TransformerBlock
from .mlp import MlpFFN
from .patch_embed import PatchEmbed

__all__ = ["Attention", "LayerScale", "MlpFFN", "PatchEmbed", "TransformerBlock"]
