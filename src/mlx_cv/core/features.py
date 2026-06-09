"""Backbone feature + head I/O contracts — the spine's vision data lingua franca.

A vision backbone emits more than a bare list of tensors: ViTs carry a ``cls``
token, optional ``storage``/``register`` tokens, a patch grid with a stride, and
(for multi-view depth models) a view axis. Heads need that structure *plus* the
spatial context, not just raw features. These typed containers make the §6.1
"richer feature contract" implementable without guessing shapes downstream.

Design rules (ARCHITECTURE §5.4, BUILDING-BLOCKS Part 2):

* **mlx-free.** ``core`` imports no ``mlx``. Tensor payloads are framework-agnostic
  (``Any`` — an ``mlx.array`` in practice); only *metadata* (layout, grid, dtype
  name, offsets) is materialized here, in plain Python / numpy.
* **Token order is ``[cls, storage…, patch…]``** (the DINOv3 convention). A
  ``TokenLayout`` records where each group sits so a combined sequence (e.g.
  ``x_prenorm`` or an intermediate tap) can be sliced consistently.
* **Foundation-forward.** Single-image ViTs (DINOv3) populate the ``B,N,C`` +
  grid + cls/storage subset; ``view_axis`` / packed ``L,C`` / multi-view ``B,S,N,C``
  fields stay defined-but-unused until a multi-view model (Depth Anything 3) needs
  them — present in the contract, not over-fit to DINOv3.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Layout", "TokenLayout", "FeatureMap", "BackboneFeatures", "HeadInput", "HeadOutput"]


class Layout(enum.Enum):
    """How a feature tensor's axes are arranged."""

    BHWC = "B,H,W,C"    # dense spatial grid (conv / un-tokenized maps)
    BNC = "B,N,C"       # token sequence (ViT patch/cls tokens)
    LC = "L,C"          # packed/flattened tokens (variable-length, no batch axis)
    BSNC = "B,S,N,C"    # multi-view token sequence (S = views; Depth Anything 3)


def _dtype_name(data: Any) -> str | None:
    """Best-effort framework-agnostic dtype *name* (no tensor-framework import).

    numpy -> ``dtype.name`` (``"float32"``); mlx/torch -> last component of
    ``str(dtype)`` (``"mlx.core.float32"`` / ``"torch.float32"`` -> ``"float32"``).
    """
    dt = getattr(data, "dtype", None)
    if dt is None:
        return None
    name = getattr(dt, "name", None) or str(dt)
    return name.rsplit(".", 1)[-1]


@dataclass(frozen=True)
class TokenLayout:
    """Index bookkeeping for a ``[cls, storage…, patch…]`` token sequence.

    For DINOv3: ``cls_offset=0``, ``n_storage=R`` (storage tokens at ``1 .. R``),
    ``patch_offset=R+1``. A model with no cls token sets ``cls_offset=None`` and
    ``patch_offset=0``.
    """

    cls_offset: int | None = 0
    n_storage: int = 0
    patch_offset: int = 1

    @property
    def storage_slice(self) -> tuple[int, int]:
        """``(start, stop)`` half-open range of storage tokens in the sequence."""
        start = 0 if self.cls_offset is None else self.cls_offset + 1
        return (start, start + self.n_storage)

    @classmethod
    def vit(cls, n_storage: int = 0, *, has_cls: bool = True) -> "TokenLayout":
        """Standard ViT prefix: ``[cls] + storage*n + patches``."""
        if has_cls:
            return cls(cls_offset=0, n_storage=n_storage, patch_offset=1 + n_storage)
        return cls(cls_offset=None, n_storage=n_storage, patch_offset=n_storage)


@dataclass
class FeatureMap:
    """One feature tensor + its layout metadata (the unit of a backbone output).

    ``grid`` is the ``(Hp, Wp)`` patch grid for tokenized spatial maps (so ``N`` =
    ``Hp*Wp`` patch tokens); ``stride`` is the patch size / total downsample versus
    the model input. ``view_axis`` names the ``S`` axis for ``BSNC`` (else ``None``).
    """

    data: Any
    layout: Layout
    grid: tuple[int, int] | None = None
    stride: int | None = None
    view_axis: int | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        if self.dtype is None:
            self.dtype = _dtype_name(self.data)


@dataclass
class BackboneFeatures:
    """A vision backbone's structured output: patch features + special tokens.

    Mirrors DINOv3 ``forward_features`` — ``patch_tokens`` ≙ ``x_norm_patchtokens``,
    ``cls_token`` ≙ ``x_norm_clstoken``, ``storage_tokens`` ≙ ``x_storage_tokens`` —
    while staying model-agnostic. ``intermediates`` holds per-layer taps (ordered)
    so the parity harness can ``bisect`` drift; ``extras`` carries model-specific
    odds and ends (e.g. ``x_prenorm``). ``valid_mask`` marks valid token/pixel
    positions for padded inputs (``None`` = all valid).
    """

    patch_tokens: FeatureMap
    cls_token: Any | None = None
    storage_tokens: Any | None = None
    token_layout: TokenLayout | None = None
    valid_mask: Any | None = None
    intermediates: list[FeatureMap] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def layout(self) -> Layout:
        return self.patch_tokens.layout

    @property
    def grid(self) -> tuple[int, int] | None:
        return self.patch_tokens.grid

    @property
    def n_storage(self) -> int:
        return 0 if self.token_layout is None else self.token_layout.n_storage

    @property
    def dtype(self) -> str | None:
        return self.patch_tokens.dtype


@dataclass
class HeadInput:
    """What a task head consumes: backbone features + the spatial context it needs.

    ``image_size`` is the model-input ``(H, W)``; ``grid`` defaults to the patch
    grid carried by ``features``. ``prompt`` / ``memory`` stay ``None`` until the
    promptable / video phases need them (kept here so the contract is stable).
    """

    features: BackboneFeatures
    image_size: tuple[int, int] | None = None
    grid: tuple[int, int] | None = None
    prompt: Any | None = None
    memory: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grid is None:
            self.grid = self.features.grid


@dataclass
class HeadOutput:
    """Raw head outputs, still in model space (pre-``postprocess``).

    A task-keyed bag (``logits`` / ``boxes`` / ``depth`` / ``masks`` …) so one
    container serves every head; ``Processor.postprocess`` maps it back to
    original-image coords via the ``SpatialTransform``.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
