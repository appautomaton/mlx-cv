"""DINOv2 structural second-config tests — the Phase-2 generalization proof.

DINOv2 is built entirely from the shared families (no new block code); these
assert it instantiates + forwards with correct shapes/token order, that registers
get no positional embedding (eng-review B2), and that it is the shared
`ViTBackbone`. No numerical parity here — weights/parity are Phase 3.
"""

import mlx.core as mx
import pytest

from mlx_cv.backbones.vision.vit import ViTBackbone
from mlx_cv.core import BACKBONES, BackboneFeatures

import mlx_cv.backbones.vision.dinov2 as _d2  # noqa: F401  (import self-registers)
from mlx_cv.backbones.vision.dinov2 import DINOv2Config, build_dinov2

# Small structural config; pretrain grid 4 so a 2x2 runtime grid exercises the
# 4->2 cubic interpolation of the abs pos-emb.
SMALL = DINOv2Config(
    embed_dim=32, depth=2, num_heads=2, patch_size=14,
    n_register_tokens=4, pretrain_grid=4,
)


def test_dinov2_registered_as_vision_backbone():
    assert "dinov2" in BACKBONES
    assert "dinov2" in BACKBONES.list(kind="vision")


def test_dinov2_forward_shapes_and_token_order():
    m = build_dinov2(SMALL)
    mx.eval(m.parameters())
    feats = m.forward_features(mx.zeros((1, 3, 28, 28)))   # patch 14 -> 2x2 grid
    assert isinstance(feats, BackboneFeatures)
    assert feats.cls_token.shape == (1, 32)
    assert feats.storage_tokens.shape == (1, 4, 32)        # 4 registers ride storage
    assert feats.patch_tokens.data.shape == (1, 4, 32)     # 2x2 = 4 patches
    assert feats.grid == (2, 2)
    # token order [cls, register*4, patch]
    assert feats.token_layout.n_storage == 4
    assert feats.token_layout.patch_offset == 5


def test_dinov2_registers_receive_no_pos_embed():
    # eng-review B2: pos table covers [cls, patch] only -> width 1 + grid^2, NO register slots.
    m = build_dinov2(SMALL)
    assert m.pos_embed.table.shape == (1, 1 + 4 * 4, 32)


def test_dinov2_is_the_shared_vit_assembly():
    # generalization: DINOv2 reuses ViTBackbone, defines no bespoke body.
    assert isinstance(build_dinov2(SMALL), ViTBackbone)


def test_dinov2_real_with_registers_small_instantiates():
    # with-registers-small dims (shape conformance only; no forward).
    cfg = DINOv2Config(embed_dim=384, depth=12, num_heads=6, patch_size=14,
                       n_register_tokens=4, pretrain_grid=37)
    m = build_dinov2(cfg)
    assert m.pos_embed.table.shape == (1, 1 + 37 * 37, 384)


def test_dinov2_selected_intermediates_are_final_normed_patch_tokens():
    cfg = DINOv2Config(
        embed_dim=32, depth=4, num_heads=4, patch_size=14,
        n_register_tokens=0, pretrain_grid=2,
    )
    m = build_dinov2(cfg)
    mx.eval(m.parameters())
    feats = m.forward_features(
        mx.zeros((1, 3, 28, 28)),
        intermediate_layers=[1, 3],
        capture_taps=True,
    )

    assert len(feats.intermediates) == 2
    assert [fm.layout for fm in feats.intermediates] == [feats.patch_tokens.layout] * 2
    assert [fm.grid for fm in feats.intermediates] == [(2, 2), (2, 2)]
    taps = feats.extras["taps"]
    assert list(k for k in taps if k.startswith("intermediate_")) == [
        "intermediate_01",
        "intermediate_03",
    ]
    assert mx.allclose(feats.intermediates[0].data, taps["intermediate_01"])
    assert mx.allclose(feats.intermediates[1].data, taps["intermediate_03"])


def test_dinov2_intermediate_count_selects_last_n_layers():
    m = build_dinov2(SMALL)
    feats = m.forward_features(mx.zeros((1, 3, 28, 28)), intermediate_layers=1)
    assert len(feats.intermediates) == 1


def test_dinov2_invalid_intermediate_layer_is_named():
    m = build_dinov2(SMALL)
    with pytest.raises(ValueError, match="outside valid range"):
        m.forward_features(mx.zeros((1, 3, 28, 28)), intermediate_layers=[99])
