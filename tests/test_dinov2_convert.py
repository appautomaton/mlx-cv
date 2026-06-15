"""DINOv2 DA3 state-dict conversion tests."""

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_cv.backbones.vision.dinov2 import (
    DINOv2Config,
    DINOv2ViT,
    convert_dinov2_state_dict,
    load_dinov2_weights,
)


def _tiny_model() -> DINOv2ViT:
    return DINOv2ViT(
        DINOv2Config(
            embed_dim=8,
            depth=1,
            num_heads=2,
            patch_size=2,
            n_register_tokens=2,
            pretrain_grid=2,
        )
    )


def _source_state_from_model(model: DINOv2ViT) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source: dict[str, np.ndarray] = {}
    expected: dict[str, np.ndarray] = {}
    cursor = 0
    for key, value in tree_flatten(model.parameters()):
        shape = tuple(value.shape)
        size = int(np.prod(shape))
        if key == "patch_embed.proj.weight":
            o, kh, kw, i = shape
            src = np.arange(cursor, cursor + size, dtype=np.float32).reshape(o, i, kh, kw)
            source["pretrained.patch_embed.proj.weight"] = src
            expected[key] = np.transpose(src, (0, 2, 3, 1))
        else:
            src = np.arange(cursor, cursor + size, dtype=np.float32).reshape(shape)
            if key == "pos_embed.table":
                source["pretrained.pos_embed"] = src
            elif key == "storage_tokens":
                source["pretrained.register_tokens"] = src
            else:
                source[f"pretrained.{key}"] = src
            expected[key] = src
        cursor += size
    source["pretrained.mask_token"] = np.ones((1, 1, 8), dtype=np.float32)
    return source, expected


def test_dinov2_rules_strip_wrapper_and_keep_packed_qkv():
    state = {
        "pretrained.pos_embed": np.ones((1, 5, 8), dtype=np.float32),
        "pretrained.blocks.0.attn.qkv.weight": np.ones((24, 8), dtype=np.float32),
        "pretrained.patch_embed.proj.weight": np.ones((8, 3, 2, 2), dtype=np.float32),
        "pretrained.mask_token": np.ones((1, 1, 8), dtype=np.float32),
    }

    out = dict(convert_dinov2_state_dict(state))

    assert "pos_embed.table" in out
    assert "blocks.0.attn.qkv.weight" in out
    assert "blocks.0.attn.q.weight" not in out
    assert "mask_token" not in out
    assert out["patch_embed.proj.weight"].shape == (8, 2, 2, 3)


def test_load_dinov2_weights_loads_minted_npz(tmp_path):
    model = _tiny_model()
    source, expected = _source_state_from_model(model)
    weights_path = tmp_path / "dinov2_weights.npz"
    np.savez(weights_path, **source)

    loaded = load_dinov2_weights(model, weights_path)
    mx.eval(loaded.parameters())
    params = dict(tree_flatten(loaded.parameters()))

    assert set(expected) <= set(params)
    for key, value in expected.items():
        assert np.array_equal(np.array(params[key]), value), key
