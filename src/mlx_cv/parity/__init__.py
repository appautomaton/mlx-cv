"""Parity / trust harness — golden fixtures + bisect (§11)."""

from __future__ import annotations

from .fixtures import (
    DINOV2_DA3_FIXTURE_CONFIG,
    DA3_MONOCULAR_FIXTURE_CONFIG,
    DINOV3_FIXTURE_CONFIG,
    DINOV3_VARIANT,
    da3_monocular_tap_order,
    dinov2_da3_fixed_input,
    dinov2_da3_tap_order,
    dinov3_fixed_input,
    dinov3_tap_order,
)
from .harness import (
    ParityCase,
    allclose_tree,
    assert_parity,
    bisect,
    load_case,
    save_case,
)

__all__ = [
    "ParityCase", "allclose_tree", "assert_parity", "bisect", "save_case", "load_case",
    "DINOV3_VARIANT", "DINOV3_FIXTURE_CONFIG", "dinov3_fixed_input", "dinov3_tap_order",
    "DINOV2_DA3_FIXTURE_CONFIG", "dinov2_da3_fixed_input", "dinov2_da3_tap_order",
    "DA3_MONOCULAR_FIXTURE_CONFIG", "da3_monocular_tap_order",
]
