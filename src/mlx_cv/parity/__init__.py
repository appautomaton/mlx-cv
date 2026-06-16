"""Parity / trust harness — golden fixtures + bisect (§11)."""

from __future__ import annotations

from .fixtures import (
    DINOV2_DA3_FIXTURE_CONFIG,
    DA3_MONOCULAR_FIXTURE_CONFIG,
    DINOV3_FIXTURE_CONFIG,
    DINOV3_VARIANT,
    MOONVIT_FIXTURE_CONFIG,
    LOCATEANYTHING_FIXTURE_CONFIG,
    RFDETR_FIXTURE_CONFIG,
    SAM3_FIXTURE_CONFIG,
    QWEN2_FIXTURE_CONFIG,
    da3_monocular_tap_order,
    dinov2_da3_fixed_input,
    dinov2_da3_tap_order,
    dinov3_fixed_input,
    dinov3_tap_order,
    qwen2_fixed_inputs,
    moonvit_fixed_inputs,
    moonvit_tap_order,
    locateanything_fixed_inputs,
    locateanything_tap_order,
    rfdetr_fixed_image,
    rfdetr_fixed_input,
    rfdetr_tap_order,
    sam3_fixed_image,
    sam3_pcs_prompt,
    sam3_tap_order,
    sam3_text_prompt,
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
    "QWEN2_FIXTURE_CONFIG", "qwen2_fixed_inputs",
    "MOONVIT_FIXTURE_CONFIG", "moonvit_fixed_inputs", "moonvit_tap_order",
    "LOCATEANYTHING_FIXTURE_CONFIG", "locateanything_fixed_inputs", "locateanything_tap_order",
    "RFDETR_FIXTURE_CONFIG", "rfdetr_fixed_input", "rfdetr_fixed_image", "rfdetr_tap_order",
    "SAM3_FIXTURE_CONFIG", "sam3_fixed_image", "sam3_text_prompt", "sam3_pcs_prompt", "sam3_tap_order",
]
