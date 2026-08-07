"""Opt-in real EoMT-DINOv3 CPU and Metal parity gates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ENV = "MLX_CV_EOMT_DINOV3_PACKAGE"
LEGACY_DELTA_ENV = "MLX_CV_EOMT_DINOV3_CHECKPOINT"
REFERENCE_ENV = "MLX_CV_EOMT_DINOV3_REFERENCE_OUTPUT"
INPUT_ENV = "MLX_CV_EOMT_DINOV3_INPUT"
REQUIRED_ENV = "MLX_CV_REQUIRE_EOMT_DINOV3_GATE"
METAL_REQUIRED_ENV = "MLX_CV_REQUIRE_EOMT_DINOV3_METAL_GATE"

REFERENCE_SCHEMA_VERSION = 1
BLOCK_NAMES = tuple(f"block_{index:02d}" for index in range(12))
CPU_ATOL = 2.0e-3
CPU_RTOL = 2.0e-3
METAL_MASK_SIGN_AGREEMENT = 0.999
METAL_PANOPTIC_MAP_AGREEMENT = 0.9998
COCO_STUFF_CLASSES = list(range(80, 133))


def _enabled(environ: dict[str, str], name: str) -> bool:
    return environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _required(environ: dict[str, str]) -> bool:
    return _enabled(environ, REQUIRED_ENV) or _enabled(environ, METAL_REQUIRED_ENV)


def _resolve_package(environ: dict[str, str]) -> Path:
    package_value = environ.get(PACKAGE_ENV)
    if not package_value:
        legacy = environ.get(LEGACY_DELTA_ENV)
        if legacy:
            pytest.fail(
                "unsupported_checkpoint_format: the legacy pytorch_model.bin is a delta; "
                "use the current full tue-mps/eomt-dinov3-coco-panoptic-small-640 package"
            )
        message = f"external_checkpoint_missing: set {PACKAGE_ENV} to the full model package"
        if _required(environ):
            pytest.fail(message)
        pytest.skip(message)
    package = Path(package_value)
    if not package.is_dir():
        pytest.fail(f"external_checkpoint_missing: EoMT package is not a directory: {package}")
    for filename in ("config.json", "model.safetensors"):
        if not (package / filename).is_file():
            pytest.fail(f"external_checkpoint_missing: EoMT package is missing {filename}: {package}")
    return package


def _load_reference(environ: dict[str, str]) -> dict[str, np.ndarray] | None:
    reference_value = environ.get(REFERENCE_ENV)
    if not reference_value:
        if _required(environ):
            pytest.fail(
                f"comparison_tap_missing: set {REFERENCE_ENV} to a complete EoMT capture"
            )
        return None
    reference_path = Path(reference_value)
    if not reference_path.is_file():
        pytest.fail(f"comparison_tap_missing: EoMT capture does not exist: {reference_path}")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {name: np.asarray(archive[name]) for name in archive.files}

    required_keys = {
        "schema_version",
        "metadata_json",
        "segments_json",
        "image",
        "image_size",
        "pixel_values",
        "masks",
        "classes",
        "panoptic_map",
        "panoptic_segment_ids",
        "panoptic_label_ids",
        *BLOCK_NAMES,
    }
    missing = sorted(required_keys - reference.keys())
    if missing:
        pytest.fail(f"reference_contract_invalid: EoMT capture is missing {missing}")
    if int(reference["schema_version"].item()) != REFERENCE_SCHEMA_VERSION:
        pytest.fail(
            "reference_contract_invalid: unsupported EoMT capture schema "
            f"{reference['schema_version'].item()}"
        )
    try:
        metadata = json.loads(str(reference["metadata_json"].item()))
        json.loads(str(reference["segments_json"].item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        pytest.fail(f"reference_contract_invalid: malformed EoMT metadata: {exc}")
    if metadata.get("block_count") != len(BLOCK_NAMES):
        pytest.fail(
            "reference_contract_invalid: EoMT capture does not describe all 12 blocks"
        )
    expected_postprocess = {
        "panoptic_threshold": 0.8,
        "mask_threshold": 0.5,
        "overlap_mask_area_threshold": 0.8,
        "stuff_classes": COCO_STUFF_CLASSES,
    }
    mismatches = {
        name: (metadata.get(name), expected)
        for name, expected in expected_postprocess.items()
        if metadata.get(name) != expected
    }
    if mismatches:
        pytest.fail(
            f"reference_contract_invalid: EoMT postprocess metadata mismatch: {mismatches}"
        )
    return reference


def _pixel_values(
    environ: dict[str, str],
    reference: dict[str, np.ndarray] | None,
) -> np.ndarray:
    input_path = environ.get(INPUT_ENV)
    if input_path:
        with np.load(input_path, allow_pickle=False) as archive:
            pixel_values = np.asarray(archive["pixel_values"], dtype=np.float32)
        if reference is not None:
            np.testing.assert_array_equal(pixel_values, reference["pixel_values"])
        return pixel_values
    if reference is not None:
        return np.asarray(reference["pixel_values"], dtype=np.float32)
    return np.linspace(-2.0, 2.0, 3 * 640 * 640, dtype=np.float32).reshape(
        1, 3, 640, 640
    )


def _runtime(environ: dict[str, str]):
    try:
        import mlx.core as mx
        from mlx_cv.models.eomt_dinov3 import (
            EoMTDINOv3,
            EoMTDINOv3Processor,
        )
    except Exception as exc:
        if _required(environ):
            pytest.fail(f"reference_runtime_unavailable: MLX EoMT runtime failed to import: {exc}")
        pytest.skip(f"reference_runtime_unavailable: {exc}")
    return mx, EoMTDINOv3, EoMTDINOv3Processor


def test_optional_gate_reports_precise_missing_package():
    with pytest.raises(pytest.skip.Exception, match="external_checkpoint_missing"):
        _resolve_package({})


def test_required_gate_rejects_legacy_delta_checkpoint(tmp_path):
    delta = tmp_path / "pytorch_model.bin"
    delta.touch()
    with pytest.raises(pytest.fail.Exception, match="unsupported_checkpoint_format"):
        _resolve_package({REQUIRED_ENV: "1", LEGACY_DELTA_ENV: str(delta)})


def test_real_eomt_dinov3_cpu_blocks_and_logits():
    environ = dict(os.environ)
    package = _resolve_package(environ)
    reference = _load_reference(environ)
    mx, EoMTDINOv3, _ = _runtime(environ)
    pixel_values = _pixel_values(environ, reference)

    with mx.stream(mx.cpu):
        model = EoMTDINOv3.from_pretrained(package, strict=True)
        output = model(mx.array(pixel_values), capture_taps=True)
        mx.eval(output.data)
    masks = np.asarray(output.data["masks_queries_logits"])
    classes = np.asarray(output.data["class_queries_logits"])
    taps = output.data["taps"]

    assert masks.shape == (1, 200, 160, 160)
    assert classes.shape == (1, 200, 134)
    assert taps["query_insertion"].shape == (1, 1805, 384)
    assert tuple(name for name in BLOCK_NAMES if name in taps) == BLOCK_NAMES
    assert np.all(np.isfinite(masks))
    assert np.all(np.isfinite(classes))

    if reference is None:
        return
    for name in BLOCK_NAMES:
        np.testing.assert_allclose(
            np.asarray(taps[name]),
            reference[name],
            rtol=CPU_RTOL,
            atol=CPU_ATOL,
            err_msg=f"EoMT CPU parity drifted at {name}",
        )
    np.testing.assert_allclose(
        masks, reference["masks"], rtol=CPU_RTOL, atol=CPU_ATOL
    )
    np.testing.assert_allclose(
        classes, reference["classes"], rtol=CPU_RTOL, atol=CPU_ATOL
    )


def test_real_eomt_dinov3_metal_panoptic_output():
    environ = dict(os.environ)
    package = _resolve_package(environ)
    reference = _load_reference(environ)
    if reference is None:
        pytest.skip(f"comparison_tap_missing: set {REFERENCE_ENV} for the Metal gate")
    mx, EoMTDINOv3, EoMTDINOv3Processor = _runtime(environ)
    if not mx.metal.is_available():
        message = "reference_runtime_unavailable: MLX Metal is not available"
        if _enabled(environ, METAL_REQUIRED_ENV):
            pytest.fail(message)
        pytest.skip(message)

    with mx.stream(mx.gpu):
        model = EoMTDINOv3.from_pretrained(package, strict=True)
        output = model(mx.array(reference["pixel_values"]), capture_taps=False)
        mx.eval(output.data)
    masks = np.asarray(output.data["masks_queries_logits"])
    classes = np.asarray(output.data["class_queries_logits"])

    processor = EoMTDINOv3Processor.from_model_config(model.cfg)
    _, context = processor.preprocess(reference["image"])
    assert context.image_size == tuple(int(value) for value in reference["image_size"])
    result = processor.postprocess(output, context)
    assert result.masks is not None
    panoptic_map = np.asarray(result.masks.data)
    segments_info = result.metadata["segments_info"]

    np.testing.assert_array_equal(
        classes.argmax(axis=-1), reference["classes"].argmax(axis=-1)
    )
    mask_sign_agreement = float(np.mean((masks >= 0) == (reference["masks"] >= 0)))
    assert mask_sign_agreement >= METAL_MASK_SIGN_AGREEMENT
    np.testing.assert_array_equal(
        np.asarray([segment["id"] for segment in segments_info], dtype=np.int64),
        reference["panoptic_segment_ids"],
    )
    np.testing.assert_array_equal(
        np.asarray([segment["label_id"] for segment in segments_info], dtype=np.int64),
        reference["panoptic_label_ids"],
    )
    assert panoptic_map.shape == reference["panoptic_map"].shape
    panoptic_agreement = float(np.mean(panoptic_map == reference["panoptic_map"]))
    assert panoptic_agreement >= METAL_PANOPTIC_MAP_AGREEMENT
