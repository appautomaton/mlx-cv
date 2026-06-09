"""Slice 4: DINOv3 MLX backbone — shape conformance + decorator registration.

Numerical parity (with minted weights) is asserted separately in
`test_dinov3_parity.py`; here we only check the forward produces the
`BackboneFeatures` contract at realistic ViT-S/16 dims, and that importing the
backbone self-registers it with no `core/` edit.
"""

import pytest

from mlx_cv.core import BACKBONES, BackboneFeatures, Layout
from mlx_cv.parity import DINOV3_VARIANT, dinov3_fixed_input

mx = pytest.importorskip("mlx.core")
import mlx_cv.backbones.vision.dinov3 as d3   # noqa: E402  (import self-registers)


def test_dinov3_self_registers():
    # Registration is decorator-driven on import of the backbone module — core/ is
    # never edited to add it (proof the spine generalizes).
    assert "dinov3" in BACKBONES
    assert "dinov3" in BACKBONES.list(kind="vision")


def test_dinov3_forward_shapes_vit_small():
    model = d3.build_dinov3(DINOV3_VARIANT)
    x = mx.array(dinov3_fixed_input(img_size=DINOV3_VARIANT["img_size"]))   # (1,3,64,64)
    feats = model(x)
    assert isinstance(feats, BackboneFeatures)
    assert feats.layout is Layout.BNC
    assert feats.grid == (4, 4)                       # 64/16
    assert feats.n_storage == 4
    assert tuple(feats.patch_tokens.data.shape) == (1, 16, 384)
    assert tuple(feats.cls_token.shape) == (1, 384)
    assert tuple(feats.storage_tokens.shape) == (1, 4, 384)
    assert feats.dtype == "float32"


def test_dinov3_build_from_config_object():
    cfg = d3.DINOv3Config(embed_dim=64, depth=2, num_heads=2, n_storage_tokens=2, patch_size=16)
    model = d3.build_dinov3(cfg)
    feats = model(mx.array(dinov3_fixed_input(img_size=32)))    # (1,3,32,32) -> 2x2 patches
    assert tuple(feats.patch_tokens.data.shape) == (1, 4, 64)
    assert tuple(feats.cls_token.shape) == (1, 64)
