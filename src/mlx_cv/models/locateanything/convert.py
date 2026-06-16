"""Weight conversion for full LocateAnything checkpoints."""

from __future__ import annotations

import numpy as np
from mlx.utils import tree_unflatten
import mlx.core as mx

from ...backbones.llm.qwen2.convert import convert_qwen2_state_dict
from ...backbones.vision.moonvit.convert import convert_moonvit_state_dict
from .modeling import LocateAnythingModel

__all__ = ["remap_key", "convert_state_dict", "load_locateanything_weights"]


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


def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):]


def _assert_full_tied_lm_head_is_lossless(weights: dict[str, np.ndarray]) -> None:
    lm_head = weights.get("language_model.lm_head.weight")
    if lm_head is None:
        return
    embed = weights.get("language_model.model.embed_tokens.weight")
    if embed is None:
        raise ValueError("cannot drop language_model.lm_head.weight without language_model.model.embed_tokens.weight")
    if not np.array_equal(lm_head, embed):
        raise ValueError("cannot drop language_model.lm_head.weight because it is not tied to embed_tokens")


def _convert_vision(weights: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    state = {}
    for k, v in weights.items():
        if k.startswith("vision_model.encoder."):
            state[_strip_prefix(k, "vision_model.")] = v
        elif k.startswith("vision_model."):
            state[_strip_prefix(k, "vision_model.")] = v
    return [(f"vision_tower.{k}", v) for k, v in convert_moonvit_state_dict(state)]


def _convert_language(weights: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    _assert_full_tied_lm_head_is_lossless(weights)
    state = {
        _strip_prefix(k, "language_model."): v
        for k, v in weights.items()
        if k.startswith("language_model.")
    }
    return [(f"language_model.{k}", v) for k, v in convert_qwen2_state_dict(state)]


def _convert_projector(weights: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    out: list[tuple[str, mx.array]] = []
    for key, value in weights.items():
        if not key.startswith("mlp1."):
            continue
        mapped = remap_key(key)
        if mapped is not None:
            out.append((mapped, mx.array(value)))
    return out


def convert_state_dict(weights: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    """Convert full reference LocateAnything weights to local MLX parameter paths."""
    items: list[tuple[str, mx.array]] = []
    items.extend(_convert_vision(weights))
    items.extend(_convert_language(weights))
    items.extend(_convert_projector(weights))
    for key, value in weights.items():
        if key.startswith(("vision_model.", "language_model.", "mlp1.")) or key.startswith("__"):
            continue
        mapped = remap_key(key)
        if mapped is not None:
            items.append((mapped, mx.array(value)))
    return items


def load_locateanything_weights(model: LocateAnythingModel, weights_path) -> LocateAnythingModel:
    """Load full LocateAnything weights from an ``.npz`` file into ``model``."""
    npz = np.load(weights_path, allow_pickle=False)
    state = {k: npz[k] for k in npz.files}
    model.update(tree_unflatten(convert_state_dict(state)))
    mx.eval(model.parameters())
    return model
