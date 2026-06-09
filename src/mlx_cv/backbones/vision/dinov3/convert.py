"""Weight conversion: official DINOv3 ``state_dict`` -> MLX param tree (`sanitize`).

The reference exports a flat ``{name: ndarray}`` (see `tools/mint_dinov3_fixture.py`).
Almost every key maps 1:1 onto our module tree (we named submodules to match:
``blocks.{i}.attn.qkv``, ``mlp.fc1`` …). The only fixes:

* ``patch_embed.proj.weight``: PyTorch conv ``(O, in, kH, kW)`` -> mlx ``(O, kH, kW, in)``.
* ``rope_embed.periods`` -> ``periods`` (we hold it as a top-level buffer, not a submodule).
* ``mask_token`` is dropped (only used on the masked-pretraining path; eval forward
  computes ``cls + 0 * mask_token`` == ``cls``).

Linear weights are ``(out, in)`` in both frameworks, so they pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

from .modeling import DINOv3ViT

__all__ = ["convert_dinov3_state_dict", "load_dinov3_weights"]


def convert_dinov3_state_dict(state: dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
    """Map a reference ``state_dict`` to ``[(mlx_path, array)]`` for ``tree_unflatten``."""
    items: list[tuple[str, mx.array]] = []
    for key, value in state.items():
        if key == "mask_token":
            continue
        if key == "rope_embed.periods":
            items.append(("periods", mx.array(value)))
            continue
        if key == "patch_embed.proj.weight":
            value = np.transpose(value, (0, 2, 3, 1))      # (O,in,kH,kW) -> (O,kH,kW,in)
        items.append((key, mx.array(value)))
    return items


def load_dinov3_weights(model: DINOv3ViT, weights_path) -> DINOv3ViT:
    """Load a minted ``*_weights.npz`` into ``model`` in place; returns it."""
    npz = np.load(weights_path)
    items = convert_dinov3_state_dict({k: npz[k] for k in npz.files})
    model.update(tree_unflatten(items))
    mx.eval(model.parameters())
    return model
