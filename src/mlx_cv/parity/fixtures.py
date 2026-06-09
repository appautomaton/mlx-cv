"""Canonical fixed inputs + tap schema for golden-fixture parity (§11).

The fixed input is defined **once** here (pure numpy, mlx-free) and reused by
both the out-of-band mint step (PyTorch oracle, ``tools/``) and the MLX parity
test, so the two sides compare on byte-identical input. The tap schema is the
*ordered* list of intermediate probes ``bisect`` walks to localize drift.

Phase-1 oracle (resolved at the plan decision checkpoint): **fixed-seed → MLX**
structural parity on DINOv3 **ViT-S/16** — the PyTorch reference is seeded, run
through ``forward_features``, and its weights are exported to the MLX port; the
two outputs must match. No HF-gated pretrained weights are involved.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DINOV3_VARIANT", "dinov3_fixed_input", "dinov3_tap_order"]

# Locked DINOv3 variant for the Phase-1 proof (DINOv3 ViT-S/16 defaults).
DINOV3_VARIANT = {
    "name": "vit_small",
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "n_storage_tokens": 4,
    "img_size": 64,          # small but non-degenerate: 64/16 -> 4x4 = 16 patch tokens
}


def dinov3_fixed_input(seed: int = 0) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for the DINOv3 parity fixture."""
    h = w = DINOV3_VARIANT["img_size"]
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


def dinov3_tap_order(depth: int | None = None) -> list[str]:
    """Ordered intermediate taps for ``bisect``.

    Forward order: patch-embed -> RoPE sin/cos -> each transformer block ->
    final norm -> the cls / storage / patch split of the normed sequence.
    """
    depth = DINOV3_VARIANT["depth"] if depth is None else depth
    taps = ["patch_embed", "rope_sincos"]
    taps += [f"block_{i:02d}" for i in range(depth)]
    taps += ["norm", "cls", "storage", "patch"]
    return taps
