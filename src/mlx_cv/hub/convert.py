"""Declarative weight-convert / ``sanitize`` engine (build-once plumbing).

Every model's load path remaps a reference ``state_dict`` onto our mlx param
tree: rename a few keys, fix a conv/layout axis order, drop unused tensors. Rather
than hand-roll that per model, a model declares a list of rules and this engine
applies them:

* `Drop(key)`        — exclude an exact source key (e.g. a pretrain-only tensor).
* `Rename(src, dst)` — move an exact source key to a new mlx path.
* `Transpose(key, axes)` — reorder a tensor's axes (e.g. PyTorch conv ``(O,in,kH,kW)``
  → mlx ``(O,kH,kW,in)``). Keyed by the **source** key; applied before rename.

Rules match exact keys (not prefixes) — explicit and auditable. Keys with no rule
pass through unchanged. DINOv3 is the first consumer (`backbones/vision/dinov3/
convert.py`); generality across separate-qkv / fused layouts is proven as more
models adopt it (Phase 3+).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

__all__ = ["Drop", "Rename", "Transpose", "convert_state_dict", "load_into"]


@dataclass(frozen=True)
class Drop:
    key: str


@dataclass(frozen=True)
class Rename:
    src: str
    dst: str


@dataclass(frozen=True)
class Transpose:
    key: str
    axes: tuple[int, ...]


def convert_state_dict(
    state: dict[str, np.ndarray], rules: list
) -> list[tuple[str, mx.array]]:
    """Apply ``rules`` to a reference ``state_dict`` → ``[(mlx_path, array)]`` for ``tree_unflatten``."""
    drops = {r.key for r in rules if isinstance(r, Drop)}
    renames = {r.src: r.dst for r in rules if isinstance(r, Rename)}
    transposes = {r.key: r.axes for r in rules if isinstance(r, Transpose)}
    items: list[tuple[str, mx.array]] = []
    for key, value in state.items():
        if key in drops:
            continue
        if key in transposes:
            value = np.transpose(value, transposes[key])     # source-keyed, before rename
        items.append((renames.get(key, key), mx.array(value)))
    return items


def load_into(model, state: dict[str, np.ndarray], rules: list):
    """Convert ``state`` by ``rules`` and load it into ``model`` in place; returns ``model``."""
    model.update(tree_unflatten(convert_state_dict(state, rules)))
    mx.eval(model.parameters())
    return model
