"""Depth Anything 3 upstream-vs-local parity and demo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import da3_convert_checkpoint  # noqa: E402
from da3_checkpoint import print_checkpoint_evidence, resolve_da3_checkpoint  # noqa: E402
import da3_upstream  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("/tmp/mlx-cv-da3-demo")
SELECTED_TAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("feat_layer_5", "aux_feat_layer_05"),
    ("feat_layer_7", "aux_feat_layer_07"),
    ("feat_layer_9", "aux_feat_layer_09"),
    ("feat_layer_11", "aux_feat_layer_11"),
)
FIELD_TOLERANCES: dict[str, dict[str, float]] = {
    "depth": {"atol": 5.0e-2, "rtol": 0.0},
    "confidence": {"atol": 5.0e-2, "rtol": 0.0},
    "extrinsics": {"atol": 1.5e-1, "rtol": 0.0},
    "intrinsics": {"atol": 1.2e1, "rtol": 0.0},
    "tap.feat_layer_5": {"atol": 3.0, "rtol": 0.0},
    "tap.feat_layer_7": {"atol": 2.0, "rtol": 0.0},
    "tap.feat_layer_9": {"atol": 6.0, "rtol": 0.0},
    "tap.feat_layer_11": {"atol": 3.0, "rtol": 0.0},
}


class DA3ParityError(AssertionError):
    """Raised when required DA3 upstream-vs-local parity drifts."""


@dataclass(frozen=True)
class FieldComparison:
    name: str
    reference_shape: list[int]
    local_shape: list[int]
    atol: float
    rtol: float
    max_abs_error: float | None
    max_rel_error: float | None
    passed: bool


def _as_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


def _public_depth(local: Any) -> np.ndarray:
    return _as_float_array(local.raw_depth[0])


def _public_confidence(local: Any) -> np.ndarray:
    return _as_float_array(local.raw_confidence[0])


def _public_extrinsics(local: Any) -> np.ndarray:
    return _as_float_array(local.extrinsics[0])


def _public_intrinsics(local: Any) -> np.ndarray:
    return _as_float_array(local.intrinsics[0])


def _max_rel_error(got: np.ndarray, expected: np.ndarray) -> float:
    denom = np.maximum(np.abs(expected), 1.0e-8)
    return float(np.max(np.abs(got - expected) / denom))


def _compare_array(name: str, reference: Any, local: Any, tolerances: Mapping[str, float]) -> FieldComparison:
    ref = _as_float_array(reference)
    got = _as_float_array(local)
    atol = float(tolerances["atol"])
    rtol = float(tolerances["rtol"])
    same_shape = got.shape == ref.shape
    finite = bool(np.all(np.isfinite(got)) and np.all(np.isfinite(ref))) if same_shape else False
    max_abs = None
    max_rel = None
    passed = False
    if same_shape and finite:
        diff = np.abs(got - ref)
        max_abs = float(np.max(diff))
        max_rel = _max_rel_error(got, ref)
        passed = bool(np.all(diff <= (atol + rtol * np.abs(ref))))
    return FieldComparison(
        name=name,
        reference_shape=list(ref.shape),
        local_shape=list(got.shape),
        atol=atol,
        rtol=rtol,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        passed=passed,
    )


def compare_da3_captures(
    reference: Any,
    local: Any,
    *,
    tap_pairs: Sequence[tuple[str, str]] = SELECTED_TAP_PAIRS,
    tolerances: Mapping[str, Mapping[str, float]] = FIELD_TOLERANCES,
) -> dict[str, Any]:
    """Compare required public DA3 outputs and selected diagnostic taps."""

    comparisons = [
        _compare_array("depth", reference.depth, _public_depth(local), tolerances["depth"]),
        _compare_array("confidence", reference.confidence, _public_confidence(local), tolerances["confidence"]),
        _compare_array("extrinsics", reference.extrinsics, _public_extrinsics(local), tolerances["extrinsics"]),
        _compare_array("intrinsics", reference.intrinsics, _public_intrinsics(local), tolerances["intrinsics"]),
    ]
    for reference_key, local_key in tap_pairs:
        if reference_key not in reference.taps:
            raise DA3ParityError(f"DA3 upstream capture missing selected tap {reference_key!r}")
        if local_key not in local.taps:
            raise DA3ParityError(f"DA3 local capture missing selected tap {local_key!r}")
        field_name = f"tap.{reference_key}"
        comparisons.append(
            _compare_array(field_name, reference.taps[reference_key], local.taps[local_key], tolerances[field_name])
        )

    field_dicts = [asdict(item) for item in comparisons]
    return {
        "passed": all(item["passed"] for item in field_dicts),
        "tolerances": {key: dict(value) for key, value in tolerances.items()},
        "selected_tap_pairs": [
            {"reference": reference_key, "local": local_key}
            for reference_key, local_key in tap_pairs
        ],
        "fields": field_dicts,
        "reference_summary": reference.summary(),
        "local_summary": local.summary(),
    }


def raise_for_failed_parity(report: Mapping[str, Any]) -> None:
    failed = [field for field in report["fields"] if not field["passed"]]
    if failed:
        details = ", ".join(
            f"{field['name']} max_abs={field['max_abs_error']} max_rel={field['max_rel_error']} "
            f"tol=({field['atol']},{field['rtol']})"
            for field in failed
        )
        raise DA3ParityError(f"DA3 upstream-vs-local parity drift: {details}")


def load_da3_local_capture(
    converted_weights: str | Path,
    *,
    images: np.ndarray | None = None,
    process_res: int = da3_upstream.DEFAULT_PROCESS_RES,
) -> Any:
    """Strict-load local DA3-SMALL weights and capture the fixed input."""

    from mlx_cv.parity.da3_real import build_da3_small_local_model, capture_da3_small_local

    model = build_da3_small_local_model(converted_weights)
    return capture_da3_small_local(
        model,
        images=images,
        process_res=process_res,
        capture_taps=True,
    )


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    arr = _as_float_array(depth)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = float(np.max(finite))
        low = float(np.min(finite))
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return np.asarray(np.round(scaled * 255.0), dtype=np.uint8)


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    Image.fromarray(_normalize_depth(depth), mode="L").save(path)


def _camera_summary(reference: Any, local: Any) -> dict[str, Any]:
    return {
        "reference": {
            "extrinsics_shape": list(np.asarray(reference.extrinsics).shape),
            "intrinsics_shape": list(np.asarray(reference.intrinsics).shape),
            "extrinsics": np.asarray(reference.extrinsics).tolist(),
            "intrinsics": np.asarray(reference.intrinsics).tolist(),
            "selected_reference_index": reference.selected_reference_index,
            "view_order": list(reference.view_order),
        },
        "local": {
            "extrinsics_shape": list(_public_extrinsics(local).shape),
            "intrinsics_shape": list(_public_intrinsics(local).shape),
            "extrinsics": _public_extrinsics(local).tolist(),
            "intrinsics": _public_intrinsics(local).tolist(),
            "view_order": [0, 1, 2],
        },
    }


def write_da3_demo_artifacts(
    reference: Any,
    local: Any,
    parity_report: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Write visible DA3 demo artifacts for the fixed three-view input."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    ref_depth = _as_float_array(reference.depth)
    local_depth = _public_depth(local)
    view_count = int(ref_depth.shape[0])
    for index in range(view_count):
        ref_path = out / f"view_{index:02d}_upstream_depth.png"
        local_path = out / f"view_{index:02d}_local_depth.png"
        diff_path = out / f"view_{index:02d}_absdiff_depth.png"
        _save_depth_png(ref_path, ref_depth[index])
        _save_depth_png(local_path, local_depth[index])
        _save_depth_png(diff_path, np.abs(local_depth[index] - ref_depth[index]))
        artifacts[f"view_{index:02d}_upstream_depth"] = str(ref_path)
        artifacts[f"view_{index:02d}_local_depth"] = str(local_path)
        artifacts[f"view_{index:02d}_absdiff_depth"] = str(diff_path)

    camera_path = out / "camera_summary.json"
    parity_path = out / "parity_summary.json"
    camera_path.write_text(json.dumps(_camera_summary(reference, local), indent=2, sort_keys=True))
    parity_path.write_text(json.dumps(parity_report, indent=2, sort_keys=True))
    artifacts["camera_summary"] = str(camera_path)
    artifacts["parity_summary"] = str(parity_path)
    return artifacts


def run_da3_demo(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_root: Path | None = None,
    process_res: int = da3_upstream.DEFAULT_PROCESS_RES,
    process_res_method: str = da3_upstream.DEFAULT_PROCESS_RES_METHOD,
    ref_view_strategy: str = da3_upstream.DEFAULT_REF_VIEW_STRATEGY,
) -> dict[str, Any]:
    checkpoint = resolve_da3_checkpoint(cache_root=cache_root, required=True)
    if checkpoint is None:  # pragma: no cover - required=True raises instead.
        raise RuntimeError("required DA3 checkpoint unexpectedly resolved to None")
    print_checkpoint_evidence(checkpoint)
    converted = da3_convert_checkpoint.resolve_da3_converted_weights(
        cache_root=cache_root,
        required=True,
    )
    if converted is None:  # pragma: no cover - required=True raises instead.
        raise RuntimeError("required DA3 converted weights unexpectedly resolved to None")
    reference = da3_upstream.capture_da3_upstream_reference(
        checkpoint,
        process_res=process_res,
        process_res_method=process_res_method,
        ref_view_strategy=ref_view_strategy,
    )
    local = load_da3_local_capture(
        converted,
        images=reference.input_images,
        process_res=process_res,
    )
    report = compare_da3_captures(reference, local)
    artifacts = write_da3_demo_artifacts(reference, local, report, output_dir=output_dir)
    raise_for_failed_parity(report)
    return {
        "checkpoint": checkpoint.evidence(),
        "converted_weights": str(converted),
        "parity": report,
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run DA3 upstream-vs-local parity and write demo artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--process-res", type=int, default=da3_upstream.DEFAULT_PROCESS_RES)
    parser.add_argument("--process-res-method", default=da3_upstream.DEFAULT_PROCESS_RES_METHOD)
    parser.add_argument("--ref-view-strategy", default=da3_upstream.DEFAULT_REF_VIEW_STRATEGY)
    args = parser.parse_args(argv)

    try:
        summary = run_da3_demo(
            output_dir=args.output_dir,
            cache_root=args.cache_root,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            ref_view_strategy=args.ref_view_strategy,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
