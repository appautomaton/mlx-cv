"""Dense prediction heads."""

from __future__ import annotations

from .dpt import DPTConfig, DPTHead, build_dpt
from .dualdpt import DA3DualDPT, DA3DualDPTConfig, build_da3_dualdpt

__all__ = [
    "DA3DualDPT",
    "DA3DualDPTConfig",
    "DPTConfig",
    "DPTHead",
    "build_da3_dualdpt",
    "build_dpt",
]
