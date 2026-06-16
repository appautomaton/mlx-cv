"""Depth Anything 3 upstream multi-view reference capture.

This module may import PyTorch, torchvision, cv2, huggingface_hub, and the DA3
reference checkout at capture time. Keep it under ``tools/``; ``mlx_cv`` runtime
code must not import the upstream DA3 package.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from da3_checkpoint import (  # noqa: E402
    DA3_CHECKPOINT_FILENAME,
    DA3_CONFIG_FILENAME,
    DA3CheckpointInfo,
    print_checkpoint_evidence,
    resolve_da3_checkpoint,
)
from mlx_cv.parity.fixtures import da3_multiview_fixed_images  # noqa: E402

DEFAULT_PROCESS_RES = 112
DEFAULT_PROCESS_RES_METHOD = "upper_bound_resize"
DEFAULT_REF_VIEW_STRATEGY = "middle"
DEFAULT_EXPORT_FEAT_LAYERS = (5, 7, 9, 11)


class DA3ReferenceDependencyError(RuntimeError):
    """Raised when the upstream DA3 reference environment is unavailable."""


class DA3UpstreamCaptureError(RuntimeError):
    """Raised when upstream DA3 returns an unusable capture."""


@dataclass(frozen=True)
class DA3ReferenceCapture:
    checkpoint: DA3CheckpointInfo
    input_images: np.ndarray
    processed_images: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    taps: dict[str, np.ndarray]
    reference_view_strategy: str
    selected_reference_index: int | None
    view_order: tuple[int, ...]
    process_res: int
    process_res_method: str
    autocast_policy: dict[str, Any]

    def as_arrays(self) -> dict[str, np.ndarray]:
        arrays = {
            "input_images": self.input_images,
            "processed_images": self.processed_images,
            "depth": self.depth,
            "confidence": self.confidence,
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics,
            "selected_reference_index": np.asarray(
                -1 if self.selected_reference_index is None else self.selected_reference_index,
                dtype=np.int64,
            ),
            "view_order": np.asarray(self.view_order, dtype=np.int64),
        }
        arrays.update({f"tap.{name}": value for name, value in self.taps.items()})
        return arrays

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint": {
                "model_id": self.checkpoint.model_id,
                "revision": self.checkpoint.revision,
                "config_path": str(self.checkpoint.config_path),
                "config_sha256": self.checkpoint.config_sha256,
                "checkpoint_path": str(self.checkpoint.checkpoint_path),
                "checkpoint_sha256": self.checkpoint.checkpoint_sha256,
                "license": self.checkpoint.license_note,
                "source": self.checkpoint.source,
            },
            "input_shape": list(self.input_images.shape),
            "processed_image_shape": list(self.processed_images.shape),
            "depth_shape": list(self.depth.shape),
            "confidence_shape": list(self.confidence.shape),
            "extrinsics_shape": list(self.extrinsics.shape),
            "intrinsics_shape": list(self.intrinsics.shape),
            "tap_shapes": {name: list(value.shape) for name, value in self.taps.items()},
            "reference_view_strategy": self.reference_view_strategy,
            "selected_reference_index": self.selected_reference_index,
            "view_order": list(self.view_order),
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "autocast_policy": self.autocast_policy,
        }


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return arr


def _validate_fixed_views(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim != 4 or arr.shape[0] != 3 or arr.shape[-1] != 3:
        raise DA3UpstreamCaptureError(
            f"DA3 upstream capture expects exactly three RGB views, got shape {arr.shape}"
        )
    if arr.shape[1] != arr.shape[2]:
        raise DA3UpstreamCaptureError(f"DA3 upstream capture expects same-size square views, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise DA3UpstreamCaptureError(f"DA3 upstream capture expects uint8 views, got {arr.dtype}")
    return arr


def _selected_reference_index(strategy: str, view_count: int) -> int | None:
    if strategy == "first":
        return 0
    if strategy == "middle":
        return view_count // 2
    return None


def _require_recorded_reference_selection(
    *,
    recorded_selections: Sequence[Sequence[int]],
    ref_view_strategy: str,
    view_count: int,
) -> int:
    if ref_view_strategy == DEFAULT_REF_VIEW_STRATEGY and view_count >= 3 and not recorded_selections:
        raise DA3UpstreamCaptureError("DA3 upstream capture did not exercise reference-view selection")
    if recorded_selections and recorded_selections[-1]:
        return int(recorded_selections[-1][0])
    selected = _selected_reference_index(ref_view_strategy, view_count)
    if selected is None:
        raise DA3UpstreamCaptureError(
            f"DA3 upstream capture could not determine selected reference view for {ref_view_strategy!r}"
        )
    return selected


def _install_export_stub() -> None:
    """Avoid importing DA3's unrelated GS/video export dependencies."""
    module_name = "depth_anything_3.utils.export"
    if module_name in sys.modules:
        return

    export_module = types.ModuleType(module_name)

    def export(*_args: Any, **_kwargs: Any) -> None:
        raise DA3ReferenceDependencyError("DA3 upstream capture does not support export_dir.")

    export_module.export = export
    sys.modules[module_name] = export_module


def _install_pose_align_stub() -> None:
    """Avoid importing evo for unposed captures that never align input poses."""
    module_name = "depth_anything_3.utils.pose_align"
    if module_name in sys.modules:
        return

    pose_align_module = types.ModuleType(module_name)

    def align_poses_umeyama(*_args: Any, **_kwargs: Any) -> None:
        raise DA3ReferenceDependencyError("DA3 upstream capture does not support input-pose alignment.")

    pose_align_module.align_poses_umeyama = align_poses_umeyama
    sys.modules[module_name] = pose_align_module


def _import_reference():
    try:
        import torch
        _install_export_stub()
        _install_pose_align_stub()
        from depth_anything_3.api import DepthAnything3
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise DA3ReferenceDependencyError(
            "DA3 upstream capture requires torch, torchvision, cv2, huggingface_hub, "
            "einops, addict, imageio, omegaconf, and PYTHONPATH=references/Depth-Anything-3/src."
        ) from exc
    return torch, DepthAnything3


def _same_resolver_directory(checkpoint: DA3CheckpointInfo) -> Path | None:
    parent = checkpoint.checkpoint_path.parent
    if (
        checkpoint.checkpoint_path.name == DA3_CHECKPOINT_FILENAME
        and checkpoint.config_path.name == DA3_CONFIG_FILENAME
        and checkpoint.config_path.parent == parent
    ):
        return parent
    return None


def _link_or_copy(src: Path, dest: Path) -> None:
    try:
        dest.symlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def _load_reference_model(DepthAnything3: Any, checkpoint: DA3CheckpointInfo) -> Any:
    model_dir = _same_resolver_directory(checkpoint)
    if model_dir is not None:
        return DepthAnything3.from_pretrained(str(model_dir))

    tmp = tempfile.TemporaryDirectory(prefix="mlx-cv-da3-upstream-")
    tmp_dir = Path(tmp.name)
    _link_or_copy(checkpoint.config_path, tmp_dir / DA3_CONFIG_FILENAME)
    _link_or_copy(checkpoint.checkpoint_path, tmp_dir / DA3_CHECKPOINT_FILENAME)
    model = DepthAnything3.from_pretrained(str(tmp_dir))
    setattr(model, "_mlx_cv_da3_upstream_tmpdir", tmp)
    return model


def _model_to_cpu_float32(model: Any) -> Any:
    if hasattr(model, "to"):
        model = model.to("cpu")
    if hasattr(model, "float"):
        model = model.float()
    if hasattr(model, "eval"):
        model.eval()
    return model


@contextmanager
def _record_reference_selection():
    import depth_anything_3.model.dinov2.vision_transformer as vision_transformer
    import depth_anything_3.model.reference_view_selector as reference_view_selector

    calls: list[np.ndarray] = []
    original_selector = reference_view_selector.select_reference_view
    original_vit_selector = vision_transformer.select_reference_view

    def wrapped_select_reference_view(x, strategy=DEFAULT_REF_VIEW_STRATEGY):
        selected = original_selector(x, strategy=strategy)
        calls.append(_np(selected).astype(np.int64, copy=False))
        return selected

    reference_view_selector.select_reference_view = wrapped_select_reference_view
    vision_transformer.select_reference_view = wrapped_select_reference_view
    try:
        yield calls
    finally:
        reference_view_selector.select_reference_view = original_selector
        vision_transformer.select_reference_view = original_vit_selector


def _run_upstream_float32(
    *,
    torch: Any,
    model: Any,
    images: np.ndarray,
    process_res: int,
    process_res_method: str,
    export_feat_layers: Sequence[int],
    ref_view_strategy: str,
) -> tuple[Any, dict[str, Any]]:
    imgs_cpu, extrinsics, intrinsics = model._preprocess_inputs(
        [image for image in images],
        process_res=process_res,
        process_res_method=process_res_method,
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

    inference_mode = torch.inference_mode() if hasattr(torch, "inference_mode") else nullcontext()
    device_type = getattr(getattr(imgs, "device", None), "type", "cpu")
    autocast = torch.autocast(device_type=device_type, enabled=False) if hasattr(torch, "autocast") else nullcontext()
    autocast_policy = {
        "device": device_type,
        "dtype": "float32",
        "depthanything3_forward_autocast": "bypassed",
        "torch_autocast_enabled": False,
    }

    with _record_reference_selection() as reference_selection_calls:
        with inference_mode:
            with autocast:
                raw_output = model.model(
                    imgs,
                    ex_t_norm,
                    in_t,
                    list(export_feat_layers),
                    False,
                    False,
                    ref_view_strategy,
                )
    prediction = model._convert_to_prediction(raw_output)
    prediction = model._align_to_input_extrinsics_intrinsics(extrinsics, intrinsics, prediction)
    prediction = model._add_processed_images(prediction, imgs_cpu)
    autocast_policy["reference_selector_calls"] = [call.tolist() for call in reference_selection_calls]
    return prediction, autocast_policy


def _array_attr(prediction: Any, attr: str) -> np.ndarray:
    value = getattr(prediction, attr, None)
    if value is None:
        raise DA3UpstreamCaptureError(f"DA3 upstream prediction missing {attr!r}")
    return _np(value)


def _capture_taps(prediction: Any) -> dict[str, np.ndarray]:
    aux = getattr(prediction, "aux", None) or {}
    taps = {str(name): _np(value) for name, value in aux.items()}
    if not taps:
        raise DA3UpstreamCaptureError("DA3 upstream prediction did not return requested auxiliary taps")
    return taps


def _validate_capture_schema(
    *,
    input_images: np.ndarray,
    processed_images: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    view_count = input_images.shape[0]
    if processed_images.shape[0] != view_count or processed_images.ndim != 4 or processed_images.shape[-1] != 3:
        raise DA3UpstreamCaptureError(f"processed_images must be (V,H,W,3), got {processed_images.shape}")
    if depth.shape != confidence.shape or depth.ndim != 3 or depth.shape[0] != view_count:
        raise DA3UpstreamCaptureError(
            f"depth/confidence must both be (V,H,W), got {depth.shape} and {confidence.shape}"
        )
    if extrinsics.ndim != 3 or extrinsics.shape[0] != view_count or extrinsics.shape[1:] not in ((3, 4), (4, 4)):
        raise DA3UpstreamCaptureError(f"extrinsics must be (V,3,4) or (V,4,4), got {extrinsics.shape}")
    if intrinsics.shape != (view_count, 3, 3):
        raise DA3UpstreamCaptureError(f"intrinsics must be (V,3,3), got {intrinsics.shape}")


def capture_da3_upstream_reference(
    checkpoint: DA3CheckpointInfo,
    *,
    images: np.ndarray | None = None,
    process_res: int = DEFAULT_PROCESS_RES,
    process_res_method: str = DEFAULT_PROCESS_RES_METHOD,
    export_feat_layers: Sequence[int] = DEFAULT_EXPORT_FEAT_LAYERS,
    ref_view_strategy: str = DEFAULT_REF_VIEW_STRATEGY,
) -> DA3ReferenceCapture:
    """Run upstream DA3 on the fixed three-view capture input."""
    fixed_images = _validate_fixed_views(da3_multiview_fixed_images() if images is None else images)
    torch, DepthAnything3 = _import_reference()
    model = _model_to_cpu_float32(_load_reference_model(DepthAnything3, checkpoint))
    prediction, autocast_policy = _run_upstream_float32(
        torch=torch,
        model=model,
        images=fixed_images,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
        ref_view_strategy=ref_view_strategy,
    )

    processed_images = _array_attr(prediction, "processed_images")
    depth = _array_attr(prediction, "depth")
    confidence = _array_attr(prediction, "conf")
    extrinsics = _array_attr(prediction, "extrinsics")
    intrinsics = _array_attr(prediction, "intrinsics")
    _validate_capture_schema(
        input_images=fixed_images,
        processed_images=processed_images,
        depth=depth,
        confidence=confidence,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

    recorded_selections = autocast_policy.get("reference_selector_calls", [])
    selected = _require_recorded_reference_selection(
        recorded_selections=recorded_selections,
        ref_view_strategy=ref_view_strategy,
        view_count=fixed_images.shape[0],
    )
    return DA3ReferenceCapture(
        checkpoint=checkpoint,
        input_images=fixed_images,
        processed_images=processed_images,
        depth=depth.astype(np.float32, copy=False),
        confidence=confidence.astype(np.float32, copy=False),
        extrinsics=extrinsics.astype(np.float32, copy=False),
        intrinsics=intrinsics.astype(np.float32, copy=False),
        taps=_capture_taps(prediction),
        reference_view_strategy=ref_view_strategy,
        selected_reference_index=selected,
        view_order=tuple(range(fixed_images.shape[0])),
        process_res=process_res,
        process_res_method=process_res_method,
        autocast_policy=autocast_policy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run upstream DA3 three-view reference capture.")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    parser.add_argument("--process-res-method", default=DEFAULT_PROCESS_RES_METHOD)
    parser.add_argument("--ref-view-strategy", default=DEFAULT_REF_VIEW_STRATEGY)
    parser.add_argument("--save-npz", type=Path, default=None)
    args = parser.parse_args(argv)

    checkpoint = resolve_da3_checkpoint(cache_root=args.cache_root, required=True)
    if checkpoint is None:  # pragma: no cover - required=True raises instead.
        raise RuntimeError("required DA3 checkpoint unexpectedly resolved to None")
    print_checkpoint_evidence(checkpoint)
    capture = capture_da3_upstream_reference(
        checkpoint,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        ref_view_strategy=args.ref_view_strategy,
    )
    print(json.dumps(capture.summary(), indent=2, sort_keys=True))
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_npz, **capture.as_arrays())
        print(f"saved {args.save_npz}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
