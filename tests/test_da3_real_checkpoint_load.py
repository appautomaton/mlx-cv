from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


da3_checkpoint = _load_tool("da3_checkpoint", "tools/da3_checkpoint.py")
da3_convert_tool = _load_tool("da3_convert_checkpoint", "tools/da3_convert_checkpoint.py")


def _outside_repo(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    root = REPO.resolve()
    return resolved != root and root not in resolved.parents


def _converted_weights_for_real_load(*, environ=None, cache_root=None) -> Path:
    required = da3_checkpoint.required_gate_enabled(environ)
    try:
        converted = da3_convert_tool.resolve_da3_converted_weights(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except da3_convert_tool.DA3ConversionDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    except (da3_checkpoint.DA3CheckpointError, da3_convert_tool.DA3ConversionError) as exc:
        if required:
            pytest.fail(str(exc))
        raise
    if converted is None:
        if required:
            pytest.fail("DA3 real load skipped in required mode")
        pytest.skip("DA3 checkpoint not configured")
    return converted


def _runtime_imports_for_real_load():
    required = da3_checkpoint.required_gate_enabled()
    try:
        from mlx.utils import tree_flatten
        from mlx_cv.models.depth_anything_v3 import (
            DA3MultiViewConfig,
            DepthAnythingV3MultiView,
            load_da3_multiview_weights,
        )
    except Exception as exc:
        if required:
            pytest.fail(f"DA3 real load requires the MLX runtime: {exc}")
        pytest.skip(f"DA3 real load requires the MLX runtime: {exc}")
    return tree_flatten, DA3MultiViewConfig, DepthAnythingV3MultiView, load_da3_multiview_weights


def test_optional_no_checkpoint_real_load_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="DA3 checkpoint not configured"):
        _converted_weights_for_real_load(environ={}, cache_root=tmp_path)


def test_required_no_checkpoint_real_load_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _converted_weights_for_real_load(
            environ={da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_missing_converted_real_load_fails_instead_of_skipping(tmp_path):
    missing = tmp_path / "missing-da3-small.npz"

    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _converted_weights_for_real_load(
            environ={
                da3_checkpoint.DA3_REQUIRED_GATE_ENV: "1",
                da3_convert_tool.DA3_CONVERTED_ENV: str(missing),
            },
            cache_root=tmp_path,
        )


def test_convert_rejects_unsupported_da3_branches():
    from mlx_cv.models.depth_anything_v3 import convert_da3_multiview_state_dict

    with pytest.raises(ValueError, match="unsupported DA3 checkpoint branches"):
        convert_da3_multiview_state_dict({"model.gs_head.weight": object()})


def test_real_da3_small_checkpoint_converts_and_strict_loads():
    converted = _converted_weights_for_real_load()
    assert converted.name.endswith(".npz")
    assert _outside_repo(converted)

    checkpoint = da3_checkpoint.resolve_da3_checkpoint(required=True)
    cfg = da3_convert_tool.config_from_checkpoint(checkpoint)
    assert cfg == cfg.small()
    assert cfg.head.features == 64
    assert cfg.head.out_channels == (48, 96, 192, 384)

    tree_flatten, _, DepthAnythingV3MultiView, load_weights = _runtime_imports_for_real_load()
    model = DepthAnythingV3MultiView(cfg)
    loaded = load_weights(model, converted, strict=True)
    params = dict(tree_flatten(loaded.parameters()))

    assert len(params) == 437
    assert tuple(params["backbone.camera_token"].shape) == (1, 2, 384)
    assert tuple(params["backbone.blocks.4.attn.q_norm.weight"].shape) == (64,)
    assert tuple(params["head.projects.0.weight"].shape) == (48, 1, 1, 768)
    assert "head.scratch.output_conv2_aux.3.2.weight" not in params
    assert tuple(params["cam_enc.pose_branch.fc1.weight"].shape) == (192, 9)
    assert tuple(params["cam_dec.fc_t.weight"].shape) == (3, 768)
