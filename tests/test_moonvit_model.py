import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.backbones.vision.moonvit.modeling import (
    MoonViTBackbone,
    MoonViTEncoderLayer,
    Rope2DPosEmb,
    build_moonvit_so400m,
    cu_seqlens_from_grid_hws,
    make_block_attention_mask,
)
from mlx_cv.core.registry import BACKBONES


def _tiny_cfg(**overrides) -> MoonViTConfig:
    values = dict(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        patch_size=2,
        num_channels=1,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        merge_kernel_size=(2, 2),
    )
    values.update(overrides)
    return MoonViTConfig(**values)


def test_moonvit_encoder_layer_residual_order_matches_manual_expansion():
    cfg = _tiny_cfg()
    layer = MoonViTEncoderLayer(cfg)
    hidden = mx.array(np.arange(4 * 8, dtype=np.float32).reshape(4, 8) / 10.0)
    grid_hws = mx.array([[2, 2]], dtype=mx.int32)
    cu = cu_seqlens_from_grid_hws(grid_hws)
    rope = Rope2DPosEmb(cfg.head_dim, max_height=2, max_width=2).get_freqs_cis(grid_hws)

    with mx.stream(mx.cpu):
        out = layer(hidden, cu, rope)
        after_attn = hidden + layer.attention_qkvpacked(layer.norm0(hidden), cu, rope)
        manual = after_attn + layer.mlp(layer.norm1(after_attn))
        mx.eval(out, manual)

    assert np.allclose(np.array(out), np.array(manual), rtol=1e-6, atol=1e-6)


def test_mlx_sdpa_bool_mask_matches_isolated_image_attention():
    cfg = _tiny_cfg()
    layer = MoonViTEncoderLayer(cfg)
    hidden = mx.array(np.arange(6 * 8, dtype=np.float32).reshape(6, 8) / 100.0)
    grid_hws = mx.array([[2, 2], [1, 2]], dtype=mx.int32)
    cu = cu_seqlens_from_grid_hws(grid_hws)
    rope = Rope2DPosEmb(cfg.head_dim, max_height=2, max_width=2).get_freqs_cis(grid_hws)
    mask = make_block_attention_mask(cu, 6)

    with mx.stream(mx.cpu):
        packed = layer.attention_qkvpacked(hidden, cu, rope, mask)
        first = layer.attention_qkvpacked(
            hidden[:4],
            cu_seqlens_from_grid_hws(mx.array([[2, 2]], dtype=mx.int32)),
            rope[:4],
            None,
        )
        mx.eval(packed, first)

    assert np.allclose(np.array(packed[:4]), np.array(first), rtol=1e-5, atol=1e-5)


def test_moonvit_backbone_forward_returns_merged_per_image_outputs_and_taps():
    cfg = _tiny_cfg(num_hidden_layers=2)
    model = MoonViTBackbone(cfg)
    pixel_values = mx.zeros((8, 1, 2, 2))
    grid_hws = mx.array([[2, 2], [2, 2]], dtype=mx.int32)

    with mx.stream(mx.cpu):
        merged, taps = model(pixel_values, grid_hws, capture_taps=True)
        mx.eval(model.parameters(), *merged, *taps.values())

    assert len(merged) == 2
    assert [item.shape for item in merged] == [(1, 32), (1, 32)]
    assert list(taps) == [
        "patch_embed",
        "rope_freqs_cis",
        "attention_mask_visible",
        "block_00",
        "block_01",
        "norm",
        "merged_00",
        "merged_01",
    ]
    assert taps["patch_embed"].shape == (8, 8)
    assert taps["rope_freqs_cis"].shape == (8, 2)


def test_moonvit_registered_as_vision_backbone_when_modeling_imported():
    assert "moonvit-so400m" in BACKBONES.list(kind="vision")
    model = build_moonvit_so400m(_tiny_cfg(num_hidden_layers=0))
    assert isinstance(model, MoonViTBackbone)
    model_from_dict = BACKBONES.get("moonvit-so400m")(
        {
            "hidden_size": 8,
            "num_hidden_layers": 0,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "patch_size": 2,
            "num_channels": 1,
            "init_pos_emb_height": 2,
            "init_pos_emb_width": 2,
        }
    )
    assert isinstance(model_from_dict, MoonViTBackbone)


def test_moonvit_package_root_import_is_mlx_free_and_does_not_register():
    code = (
        "import sys\n"
        "import mlx_cv.backbones.vision.moonvit\n"
        "from mlx_cv.core.registry import BACKBONES\n"
        "assert 'moonvit-so400m' not in BACKBONES.list(kind='vision')\n"
        "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)\n"
    )
    subprocess.check_call([sys.executable, "-c", code])
