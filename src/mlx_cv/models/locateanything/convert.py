"""Weight-key remap: HF reference (``nvidia/LocateAnything-3B``) -> mlx-cv tree (§16.4).

Declarative rules, verified line-for-line against the merged MLX reference's
``sanitize()``. No transposes / QKV splits here — MoonViT's fused ``wqkv`` and conv
``patch_embed.proj`` are mirrored inside the module definitions (Stage 2), per §8.
Correctness is proven by parity (§11), not by eye.
"""

from __future__ import annotations

__all__ = ["remap_key", "convert_state_dict"]


def remap_key(key: str) -> str | None:
    """Map one reference key to the mlx-cv key, or ``None`` to drop it."""
    if key == "language_model.lm_head.weight":
        return None  # tied to embed_tokens
    if key.startswith("vision_model."):
        return key.replace("vision_model.encoder.", "vision_tower.").replace(
            "vision_model.", "vision_tower."
        )
    if key.startswith("mlp1."):
        return (
            key.replace("mlp1.0.", "multi_modal_projector.layer_norm.")
            .replace("mlp1.1.", "multi_modal_projector.linear_1.")
            .replace("mlp1.3.", "multi_modal_projector.linear_2.")
        )
    return key  # language_model.model.* and everything else: unchanged


def convert_state_dict(weights: dict) -> dict:
    """Apply :func:`remap_key` across a full state dict, dropping ``None`` targets."""
    out: dict = {}
    for k, v in weights.items():
        nk = remap_key(k)
        if nk is not None:
            out[nk] = v
    return out
