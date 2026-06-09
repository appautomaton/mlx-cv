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

__all__ = ["DINOV3_VARIANT", "DINOV3_FIXTURE_CONFIG", "dinov3_fixed_input", "dinov3_tap_order"]

# The real Phase-1 target variant (DINOv3 ViT-S/16 defaults). Used for *shape*
# conformance of the MLX port (Slice 4) at realistic dims.
DINOV3_VARIANT = {
    "name": "vit_small",
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "n_storage_tokens": 4,
    "img_size": 64,          # 64/16 -> 4x4 = 16 patch tokens
}

# The committed *numerical-parity* fixture config. A size-reduced instance of the
# SAME DINOv3 architecture / code path (RoPE, attention, Mlp, LayerNorm, storage
# tokens, cls/storage/patch split, multi-block residuals) — small enough that the
# fixed-seed weights commit (~0.6 MB) instead of ViT-S/16's ~88 MB. Fidelity of the
# proof is unchanged: "our MLX forward == reference PyTorch forward for identical
# weights" does not depend on the width/depth. Full ViT-S/16 parity can be minted
# on demand with `tools/mint_dinov3_fixture.py --variant vit_small`.
DINOV3_FIXTURE_CONFIG = {
    "name": "dinov3_tiny_fixture",
    "patch_size": 16,
    "embed_dim": 64,
    "depth": 2,
    "num_heads": 2,
    "ffn_ratio": 4.0,
    "n_storage_tokens": 2,
    "img_size": 32,          # 32/16 -> 2x2 = 4 patch tokens; tokens = 1 + 2 + 4 = 7
    "pos_embed_rope_base": 100.0,
    "pos_embed_rope_dtype": "fp32",   # fp32 RoPE for clean fp32 parity (vs reference bf16 default)
}


def dinov3_fixed_input(seed: int = 0, img_size: int | None = None) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for a DINOv3 parity fixture."""
    h = w = DINOV3_VARIANT["img_size"] if img_size is None else img_size
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
