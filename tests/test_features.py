"""Slice 1: backbone feature + head I/O contracts (mlx-free)."""

import subprocess
import sys

import numpy as np

from mlx_cv import (
    BackboneFeatures,
    FeatureMap,
    Head,
    HeadInput,
    HeadOutput,
    Layout,
    TokenLayout,
)


def _dinov3_like() -> BackboneFeatures:
    """A DINOv3-shaped single-image feature bundle: 1 cls + 4 storage + 16 patches."""
    B, C, R = 1, 8, 4
    grid = (4, 4)
    n_patch = grid[0] * grid[1]
    return BackboneFeatures(
        patch_tokens=FeatureMap(
            data=np.zeros((B, n_patch, C), np.float32),
            layout=Layout.BNC, grid=grid, stride=16,
        ),
        cls_token=np.zeros((B, C), np.float32),
        storage_tokens=np.zeros((B, R, C), np.float32),
        token_layout=TokenLayout.vit(n_storage=R),
    )


def test_featuremap_carries_layout_grid_stride_dtype():
    fm = FeatureMap(data=np.zeros((1, 16, 8), np.float32), layout=Layout.BNC,
                    grid=(4, 4), stride=16)
    assert fm.layout is Layout.BNC
    assert fm.grid == (4, 4)
    assert fm.stride == 16
    assert fm.dtype == "float32"            # derived from data, no mlx import
    assert fm.view_axis is None             # single-view: defined-but-unused


def test_token_layout_cls_storage_patch_offsets():
    tl = TokenLayout.vit(n_storage=4)
    assert tl.cls_offset == 0
    assert tl.n_storage == 4
    assert tl.storage_slice == (1, 5)       # storage tokens at indices 1..4
    assert tl.patch_offset == 5             # patches start after [cls, storage*4]


def test_token_layout_no_cls():
    tl = TokenLayout.vit(n_storage=0, has_cls=False)
    assert tl.cls_offset is None
    assert tl.storage_slice == (0, 0)
    assert tl.patch_offset == 0


def test_backbone_features_dinov3_shape():
    bf = _dinov3_like()
    assert bf.layout is Layout.BNC
    assert bf.grid == (4, 4)
    assert bf.n_storage == 4
    assert bf.dtype == "float32"
    assert bf.cls_token.shape == (1, 8)
    assert bf.storage_tokens.shape == (1, 4, 8)
    assert bf.patch_tokens.data.shape == (1, 16, 8)
    # multi-view / packed fields exist on the contract but stay unused for DINOv3
    assert bf.valid_mask is None
    assert bf.intermediates == []


def test_identity_head_consumes_headinput_to_headoutput():
    class IdentityHead:
        def __call__(self, inp: HeadInput) -> HeadOutput:
            return HeadOutput(data={"patch_tokens": inp.features.patch_tokens.data,
                                    "grid": inp.grid})

    bf = _dinov3_like()
    inp = HeadInput(features=bf, image_size=(64, 64))
    assert inp.grid == (4, 4)               # defaulted from features
    head = IdentityHead()
    assert isinstance(head, Head)           # satisfies the runtime_checkable protocol
    out = head(inp)
    assert isinstance(out, HeadOutput)
    assert "patch_tokens" in out
    assert out["grid"] == (4, 4)
    assert out.get("missing") is None


def test_core_import_is_mlx_free():
    """`import mlx_cv.core` must not pull in mlx (the spine stays numpy-only)."""
    code = ("import sys, mlx_cv.core; "
            "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules), "
            "sorted(m for m in sys.modules if m.startswith('mlx') and m != 'mlx_cv')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
