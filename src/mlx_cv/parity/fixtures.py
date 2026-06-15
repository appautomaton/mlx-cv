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

__all__ = [
    "DINOV3_VARIANT",
    "DINOV3_FIXTURE_CONFIG",
    "DINOV2_DA3_FIXTURE_CONFIG",
    "DA3_MONOCULAR_FIXTURE_CONFIG",
    "QWEN2_FIXTURE_CONFIG",
    "dinov3_fixed_input",
    "dinov3_tap_order",
    "dinov2_da3_fixed_input",
    "dinov2_da3_tap_order",
    "da3_monocular_tap_order",
    "qwen2_fixed_inputs",
]

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


DINOV2_DA3_FIXTURE_CONFIG = {
    "name": "dinov2_da3_tiny_fixture",
    "patch_size": 14,
    "embed_dim": 32,
    "depth": 4,
    "num_heads": 4,
    "ffn_ratio": 4.0,
    "n_register_tokens": 0,
    "pretrain_grid": 2,
    "img_size": 28,          # 28/14 -> 2x2; avoids pos-emb interpolation drift
    "intermediate_layers": [0, 1, 2, 3],
    "layer_norm_eps": 1e-6,
    "final_norm_eps": 1e-5,
}


DA3_MONOCULAR_FIXTURE_CONFIG = {
    "name": "da3_monocular_tiny_fixture",
    "dinov2": DINOV2_DA3_FIXTURE_CONFIG,
    "dpt": {
        "dim_in": 32,
        "patch_size": 14,
        "output_dim": 2,
        "activation": "exp",
        "conf_activation": "expp1",
        "features": 16,
        "out_channels": [8, 8, 8, 8],
        "pos_embed": False,
        "down_ratio": 1,
        "head_name": "depth",
        "use_sky_head": False,
        "norm_type": "idt",
    },
}


QWEN2_FIXTURE_CONFIG = {
    "name": "qwen2_tiny_fixture",
    "seed": 7,
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 16,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_cache": False,
    "tie_word_embeddings": True,
    "attn_implementation": "sdpa",
    "block_size": 2,
    "causal_attn": False,
    "text_mask_token_id": 7,
}


def dinov3_fixed_input(seed: int = 0, img_size: int | None = None) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for a DINOv3 parity fixture."""
    h = w = DINOV3_VARIANT["img_size"] if img_size is None else img_size
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


def dinov2_da3_fixed_input(seed: int = 0, img_size: int | None = None) -> np.ndarray:
    """Deterministic ``(1, 3, H, W)`` float32 input for the DA3 DINOv2 fixture."""
    h = w = DINOV2_DA3_FIXTURE_CONFIG["img_size"] if img_size is None else img_size
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 3, h, w)).astype(np.float32)


def qwen2_fixed_inputs() -> dict[str, np.ndarray]:
    """Deterministic no-cache SDLM input for the tiny Qwen2 parity fixture."""
    cfg = QWEN2_FIXTURE_CONFIG
    return {
        "input_ids": np.array([[3, 5, cfg["text_mask_token_id"], cfg["text_mask_token_id"]]], dtype=np.int64),
        "position_ids": np.array([[0, 1, 0, 1]], dtype=np.int64),
    }


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


def dinov2_da3_tap_order(
    depth: int | None = None,
    intermediate_layers: list[int] | tuple[int, ...] | None = None,
) -> list[str]:
    """Ordered taps for DA3-style DINOv2 parity."""
    depth = DINOV2_DA3_FIXTURE_CONFIG["depth"] if depth is None else depth
    layers = DINOV2_DA3_FIXTURE_CONFIG["intermediate_layers"] if intermediate_layers is None else intermediate_layers
    taps = ["patch_embed"]
    taps += [f"block_{i:02d}" for i in range(depth)]
    taps += [f"intermediate_{int(i):02d}" for i in layers]
    taps += ["norm", "cls", "patch"]
    return taps


def da3_monocular_tap_order() -> list[str]:
    """Ordered taps for end-to-end DA3 monocular parity."""
    taps = [f"dinov2.{k}" for k in dinov2_da3_tap_order()]
    taps += [f"dpt.projected_{i}" for i in range(4)]
    taps += ["dpt.fusion_4", "dpt.fusion_3", "dpt.fusion_2", "dpt.fusion_1", "dpt.output_logits"]
    return taps
