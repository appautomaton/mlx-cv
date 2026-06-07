"""Name -> builder registries with optional ``kind`` namespacing + plugin discovery.

Adding a model / backbone / head is one registry line, never a spine edit (§10).
Third-party packages extend mlx-cv via the ``mlx_cv.models`` entry-point group.

Backbones come in two *kinds* (§5.3, §16): ``"vision"`` encoders (image -> features)
and ``"llm"`` decoders (embeds -> hidden states). ``BACKBONES.list(kind=...)`` filters.
"""

from __future__ import annotations

import importlib.metadata as _im
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

__all__ = [
    "Registry", "MODELS", "BACKBONES", "HEADS",
    "register_model", "register_backbone", "register_head", "load_plugins",
]


class Registry(Generic[T]):
    """A name -> object map usable as a decorator or a direct call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}
        self._kinds: dict[str, str | None] = {}

    def register(self, key: str, obj: T | None = None, *, kind: str | None = None):
        def deco(o: T) -> T:
            if key in self._items:
                raise KeyError(f"{key!r} already registered in {self.name!r}")
            self._items[key] = o
            self._kinds[key] = kind
            return o

        return deco if obj is None else deco(obj)

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(f"{key!r} not in {self.name!r}; have {self.list()}") from None

    def list(self, *, kind: str | None = None) -> list[str]:
        if kind is None:
            return sorted(self._items)
        return sorted(k for k, v in self._kinds.items() if v == kind)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)


MODELS: Registry = Registry("models")
BACKBONES: Registry = Registry("backbones")
HEADS: Registry = Registry("heads")


def register_model(name: str) -> Callable:
    return MODELS.register(name)


def register_backbone(name: str, *, kind: str = "vision") -> Callable:
    return BACKBONES.register(name, kind=kind)


def register_head(name: str) -> Callable:
    return HEADS.register(name)


def load_plugins(group: str = "mlx_cv.models") -> list[str]:
    """Import third-party plugins advertised under an entry-point ``group``."""
    try:
        eps = _im.entry_points(group=group)
    except TypeError:  # Python < 3.10 API
        eps = _im.entry_points().get(group, [])  # type: ignore[attr-defined]
    loaded: list[str] = []
    for ep in eps:
        ep.load()
        loaded.append(ep.name)
    return loaded
