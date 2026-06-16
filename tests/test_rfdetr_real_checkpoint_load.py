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


rfdetr_checkpoint = _load_tool("rfdetr_checkpoint", "tools/rfdetr_checkpoint.py")
rfdetr_convert = _load_tool("rfdetr_convert_checkpoint", "tools/rfdetr_convert_checkpoint.py")


def _outside_repo(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    root = REPO.resolve()
    return resolved != root and root not in resolved.parents


def _converted_weights_for_real_load(*, environ=None, cache_root=None) -> Path:
    required = rfdetr_checkpoint.required_gate_enabled(environ)
    try:
        converted = rfdetr_convert.resolve_rfdetr_nano_converted_weights(
            environ=environ,
            cache_root=cache_root,
            required=required,
        )
    except rfdetr_convert.RFDETRConversionDependencyError as exc:
        if required:
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    except (rfdetr_checkpoint.CheckpointError, rfdetr_convert.RFDETRConversionError) as exc:
        if required:
            pytest.fail(str(exc))
        raise
    if converted is None:
        if required:
            pytest.fail("RF-DETR Nano real load skipped in required mode")
        pytest.skip("RF-DETR Nano checkpoint not configured")
    return converted


def _runtime_imports_for_real_load():
    required = rfdetr_checkpoint.required_gate_enabled()
    try:
        from mlx.utils import tree_flatten
        from mlx_cv.models.rfdetr import RFDETRConfig, RFDETRModel, load_rfdetr_weights
    except Exception as exc:
        if required:
            pytest.fail(f"RF-DETR Nano real load requires the MLX runtime: {exc}")
        pytest.skip(f"RF-DETR Nano real load requires the MLX runtime: {exc}")
    return tree_flatten, RFDETRConfig, RFDETRModel, load_rfdetr_weights


def test_optional_no_checkpoint_real_load_skips_cleanly(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="checkpoint not configured"):
        _converted_weights_for_real_load(environ={}, cache_root=tmp_path)


def test_required_no_checkpoint_real_load_fails_instead_of_skipping(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="checkpoint is required but missing"):
        _converted_weights_for_real_load(
            environ={rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_required_checksum_mismatch_real_load_fails_instead_of_skipping(tmp_path):
    checkpoint = tmp_path / rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"not the verified RF-DETR Nano checkpoint")

    with pytest.raises(pytest.fail.Exception, match="expected"):
        _converted_weights_for_real_load(
            environ={
                rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1",
                rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_ENV: str(checkpoint),
            },
            cache_root=tmp_path,
        )


def test_required_missing_converted_real_load_fails_instead_of_converting(tmp_path):
    missing = tmp_path / "missing-rfdetr-nano.npz"

    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _converted_weights_for_real_load(
            environ={
                rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV: "1",
                rfdetr_convert.RFDETR_NANO_CONVERTED_ENV: str(missing),
            },
            cache_root=tmp_path,
        )


def test_real_rfdetr_nano_checkpoint_converts_and_strict_loads():
    converted = _converted_weights_for_real_load()
    assert converted.name.endswith(".npz")
    assert _outside_repo(converted)

    tree_flatten, RFDETRConfig, RFDETRModel, load_rfdetr_weights = _runtime_imports_for_real_load()
    cfg = RFDETRConfig.rfdetr_nano()
    model = RFDETRModel(cfg)
    loaded = load_rfdetr_weights(model, converted, strict=True)
    params = dict(tree_flatten(loaded.parameters()))

    assert len(params) == 424
    assert tuple(params["feature_extractor.backbone.backbone.blocks.0.attn.qkv.weight"].shape) == (1152, 384)
    assert tuple(params["feature_extractor.backbone.backbone.pos_embed.table"].shape) == (1, 577, 384)
    assert tuple(params["feature_extractor.projector.stages.0.0.cv1.conv.weight"].shape) == (256, 1, 1, 1536)
    assert tuple(params["decoder.query_embed"].shape) == (3900, 256)
    assert tuple(params["decoder.reference_embed"].shape) == (3900, 4)
    assert tuple(params["decoder.enc_out_class_embed.12.weight"].shape) == (91, 256)
    assert tuple(params["head.class_embed.weight"].shape) == (91, 256)
