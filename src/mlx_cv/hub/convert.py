"""Declarative weight-convert / ``sanitize`` engine (build-once plumbing).

Every model's load path remaps a reference ``state_dict`` onto our mlx param
tree: rename a few keys, fix a conv/layout axis order, drop unused tensors. Rather
than hand-roll that per model, a model declares a list of rules and this engine
applies them:

* `Drop(key)`        — exclude an exact key (e.g. a pretrain-only tensor).
* `PrefixRename(src, dst)` — normalize a source prefix before exact renames.
* `Rename(src, dst)` — move an exact normalized key to a new mlx path.
* `Transpose(key, axes)` — reorder a tensor's axes (e.g. PyTorch conv ``(O,in,kH,kW)``
  → mlx ``(O,kH,kW,in)``). Keyed by source or normalized key.
* `TransposePattern(pattern, axes)` — reorder by converted path pattern for families
  of modules such as dense-head convs.

Keys with no rule pass through unchanged. DINOv3 is the first consumer
(`backbones/vision/dinov3/convert.py`); generality across dense conv layouts is
proven as more models adopt it (Phase 3+).
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase

import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

__all__ = [
    "Drop",
    "PrefixRename",
    "Rename",
    "Transpose",
    "TransposePattern",
    "convert_state_dict",
    "load_into",
]


@dataclass(frozen=True)
class Drop:
    key: str


@dataclass(frozen=True)
class PrefixRename:
    src: str
    dst: str


@dataclass(frozen=True)
class Rename:
    src: str
    dst: str


@dataclass(frozen=True)
class Transpose:
    key: str
    axes: tuple[int, ...]


@dataclass(frozen=True)
class TransposePattern:
    pattern: str
    axes: tuple[int, ...]
    ndim: int | None = None


def _apply_prefixes(key: str, prefixes: list[PrefixRename]) -> str:
    for rule in prefixes:
        if key.startswith(rule.src):
            return f"{rule.dst}{key[len(rule.src):]}"
    return key


def _transpose_checked(key: str, value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    if value.ndim != len(axes):
        raise ValueError(
            f"transpose rule for {key!r} has {len(axes)} axes but tensor has ndim {value.ndim}"
        )
    return np.transpose(value, axes)


def convert_state_dict(
    state: dict[str, np.ndarray], rules: list
) -> list[tuple[str, mx.array]]:
    """Apply ``rules`` to a reference ``state_dict`` → ``[(mlx_path, array)]`` for ``tree_unflatten``."""
    drops = {r.key for r in rules if isinstance(r, Drop)}
    prefixes = [r for r in rules if isinstance(r, PrefixRename)]
    renames = {r.src: r.dst for r in rules if isinstance(r, Rename)}
    transposes = {r.key: r.axes for r in rules if isinstance(r, Transpose)}
    transpose_patterns = [r for r in rules if isinstance(r, TransposePattern)]
    items: list[tuple[str, mx.array]] = []
    for key, value in state.items():
        normalized = _apply_prefixes(key, prefixes)
        if key in drops or normalized in drops:
            continue
        exact_axes = transposes.get(key, transposes.get(normalized))
        if exact_axes is not None:
            value = _transpose_checked(key, value, exact_axes)
        dst = renames.get(normalized, normalized)
        if exact_axes is None:
            for rule in transpose_patterns:
                if fnmatchcase(dst, rule.pattern):
                    if rule.ndim is not None and value.ndim != rule.ndim:
                        raise ValueError(
                            f"transpose pattern {rule.pattern!r} matched {dst!r} "
                            f"but tensor has ndim {value.ndim}, expected {rule.ndim}"
                        )
                    value = _transpose_checked(dst, value, rule.axes)
                    break
        items.append((dst, mx.array(value)))
    return items


def load_into(model, state: dict[str, np.ndarray], rules: list):
    """Convert ``state`` by ``rules`` and load it into ``model`` in place; returns ``model``."""
    model.update(tree_unflatten(convert_state_dict(state, rules)))
    mx.eval(model.parameters())
    return model
