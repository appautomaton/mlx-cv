"""SAM 3.1 image checkpoint admission and upstream parity gate helpers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SAM3_IMAGE_CHECKPOINT_ENV = "MLX_CV_SAM3_IMAGE_CHECKPOINT"
SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV = "MLX_CV_SAM3_IMAGE_UPSTREAM_CHECKPOINT"
SAM3_IMAGE_LOCAL_CHECKPOINT_ENV = "MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT"
SAM3_IMAGE_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_SAM3_IMAGE_GATE"
SAM3_IMAGE_REFERENCE_PATH = Path("references/sam3")
TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
_SUPPORTED_IMAGE_FORMATS = {".npz", ".safetensors"}
_SUPPORTED_UPSTREAM_FORMATS = {".pt", ".pth"}
_VIDEO_KEY_PARTS = (
    "video",
    "tracker",
    "track",
    "memory_encoder",
    "memory_attention",
    "temporal",
    "maskmem",
    "multiplex",
    "sam2_predictor",
    "obj_ptr",
)

# Public result masks and paired detections should match after both processors
# threshold and select detections. Text token ids are exact; text embeddings
# allow fp32 implementation noise across Torch and MLX.
SAM3_IMAGE_FIELD_TOLERANCES: dict[str, dict[str, float]] = {
    "masks": {"atol": 0.0, "rtol": 0.0},
    "boxes": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "scores": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "class_ids": {"atol": 0.0, "rtol": 0.0},
    "tap.text.token_ids": {"atol": 0.0, "rtol": 0.0},
    "tap.text.language_features": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "tap.text.language_embeds": {"atol": 1.0e-4, "rtol": 1.0e-4},
}
SAM3_IMAGE_SELECTED_TAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("text.token_ids", "text.token_ids"),
    ("text.language_features", "text.language_features"),
    ("text.language_embeds", "text.language_embeds"),
)
SAM3_IMAGE_DETECTION_SCORE_THRESHOLD = 0.0
SAM3_IMAGE_DETECTION_ORDER = "score_desc_stable"


@dataclass(frozen=True)
class SAM3ImageGateResult:
    status: str
    checkpoint_env: str
    required_gate_env: str
    reference_path: str
    upstream_checkpoint_env: str = SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV
    local_checkpoint_env: str = SAM3_IMAGE_LOCAL_CHECKPOINT_ENV
    checkpoint_path: str | None = None
    upstream_checkpoint_path: str | None = None
    local_checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    blocked_reason: str | None = None
    admitted: bool = False
    comparison_report: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


class SAM3ImageReferenceDependencyError(RuntimeError):
    """Raised when the upstream SAM3 image reference runtime is unavailable."""


class SAM3ImageReferenceCaptureError(RuntimeError):
    """Raised when upstream SAM3 image capture cannot run or is malformed."""


class SAM3ImageLocalCaptureError(RuntimeError):
    """Raised when the local MLX SAM3 image capture cannot run."""


class SAM3ImageParityError(AssertionError):
    """Raised when SAM3 image upstream-vs-MLX comparison cannot be evaluated."""


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


@dataclass(frozen=True)
class SAM3ImageCapture:
    source: str
    prompt_kind: str
    image: np.ndarray
    prompt: Any
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    taps: dict[str, np.ndarray]

    def inputs_for_local(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "prompt": self.prompt,
            "prompt_kind": self.prompt_kind,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "prompt_kind": self.prompt_kind,
            "image_shape": list(np.asarray(self.image).shape),
            "masks_shape": list(np.asarray(self.masks).shape),
            "boxes_shape": list(np.asarray(self.boxes).shape),
            "scores_shape": list(np.asarray(self.scores).shape),
            "class_ids_shape": list(np.asarray(self.class_ids).shape),
            "tap_order": list(self.taps),
            "tap_shapes": {name: list(np.asarray(value).shape) for name, value in self.taps.items()},
        }


def required_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SAM3_IMAGE_REQUIRED_GATE_ENV) == "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(reason: str, *, environ: Mapping[str, str]) -> SAM3ImageGateResult:
    return SAM3ImageGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=SAM3_IMAGE_CHECKPOINT_ENV,
        required_gate_env=SAM3_IMAGE_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_IMAGE_REFERENCE_PATH),
        checkpoint_path=environ.get(SAM3_IMAGE_CHECKPOINT_ENV),
        upstream_checkpoint_path=environ.get(SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV),
        local_checkpoint_path=environ.get(SAM3_IMAGE_LOCAL_CHECKPOINT_ENV),
        blocked_reason=reason,
    )


def _admit(path: Path, *, environ: Mapping[str, str]) -> SAM3ImageGateResult:
    return SAM3ImageGateResult(
        status="ADMITTED",
        checkpoint_env=SAM3_IMAGE_CHECKPOINT_ENV,
        required_gate_env=SAM3_IMAGE_REQUIRED_GATE_ENV,
        reference_path=str(SAM3_IMAGE_REFERENCE_PATH),
        checkpoint_path=str(path),
        upstream_checkpoint_path=environ.get(SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV),
        local_checkpoint_path=environ.get(SAM3_IMAGE_LOCAL_CHECKPOINT_ENV),
        checkpoint_sha256=_sha256(path),
        admitted=True,
    )


def _block_from_admission(
    admission: SAM3ImageGateResult,
    reason: str,
    *,
    upstream_checkpoint_path: Path | None = None,
    local_checkpoint_path: Path | None = None,
) -> SAM3ImageGateResult:
    return SAM3ImageGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=admission.checkpoint_env,
        required_gate_env=admission.required_gate_env,
        reference_path=admission.reference_path,
        upstream_checkpoint_env=admission.upstream_checkpoint_env,
        local_checkpoint_env=admission.local_checkpoint_env,
        checkpoint_path=admission.checkpoint_path,
        upstream_checkpoint_path=(
            str(upstream_checkpoint_path) if upstream_checkpoint_path is not None else admission.upstream_checkpoint_path
        ),
        local_checkpoint_path=(
            str(local_checkpoint_path) if local_checkpoint_path is not None else admission.local_checkpoint_path
        ),
        checkpoint_sha256=admission.checkpoint_sha256,
        blocked_reason=reason,
        admitted=admission.admitted,
    )


def _pass_from_admission(
    admission: SAM3ImageGateResult,
    report: dict[str, Any],
    *,
    upstream_checkpoint_path: Path,
    local_checkpoint_path: Path,
) -> SAM3ImageGateResult:
    return SAM3ImageGateResult(
        status="UPSTREAM_PASSED",
        checkpoint_env=admission.checkpoint_env,
        required_gate_env=admission.required_gate_env,
        reference_path=admission.reference_path,
        upstream_checkpoint_env=admission.upstream_checkpoint_env,
        local_checkpoint_env=admission.local_checkpoint_env,
        checkpoint_path=admission.checkpoint_path,
        upstream_checkpoint_path=str(upstream_checkpoint_path),
        local_checkpoint_path=str(local_checkpoint_path),
        checkpoint_sha256=admission.checkpoint_sha256,
        admitted=True,
        comparison_report=report,
    )


def _contains_video_or_tracker_key(keys: list[str]) -> bool:
    return any(any(part in key.lower() for part in _VIDEO_KEY_PARTS) for key in keys)


def _npz_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as npz:
        return list(npz.files)


def evaluate_sam3_image_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
) -> SAM3ImageGateResult:
    env = os.environ if environ is None else environ
    checkpoint = env.get(SAM3_IMAGE_CHECKPOINT_ENV)
    if not checkpoint:
        return _block(f"{SAM3_IMAGE_CHECKPOINT_ENV} is unset", environ=env)

    path = Path(checkpoint)
    if not path.exists():
        return _block(f"{SAM3_IMAGE_CHECKPOINT_ENV} does not point to an existing path: {path}", environ=env)
    if not path.is_file():
        return _block(f"{SAM3_IMAGE_CHECKPOINT_ENV} does not point to a file: {path}", environ=env)
    if path.stat().st_size < min_checkpoint_bytes:
        return _block(f"{path} is not a usable SAM3 image checkpoint", environ=env)
    if path.suffix not in _SUPPORTED_IMAGE_FORMATS:
        return _block(
            f"SAM3 image checkpoint format is not loadable by the local image converter: {path.suffix or path.name}",
            environ=env,
        )
    if path.suffix == ".npz" and _contains_video_or_tracker_key(_npz_keys(path)):
        return _block(f"{path} appears to be a video/tracker checkpoint, not a SAM3 image checkpoint", environ=env)

    return _admit(path, environ=env)


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _normalize_masks(masks: Any) -> np.ndarray:
    arr = np.asarray(masks)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise SAM3ImageParityError(f"SAM3 image masks must have shape (N,H,W), got {arr.shape}")
    return arr.astype(np.bool_, copy=False)


def _normalize_public_outputs(
    capture: Any,
    *,
    score_threshold: float = SAM3_IMAGE_DETECTION_SCORE_THRESHOLD,
) -> dict[str, np.ndarray]:
    masks = _normalize_masks(capture.masks)
    boxes = np.asarray(capture.boxes, dtype=np.float64)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise SAM3ImageParityError(f"SAM3 image boxes must have shape (N,4), got {boxes.shape}")
    scores = np.asarray(capture.scores, dtype=np.float32).reshape(-1)
    class_ids = np.asarray(capture.class_ids, dtype=np.int64).reshape(-1)
    count = len(scores)
    if boxes.shape[0] != count or class_ids.shape[0] != count or masks.shape[0] != count:
        raise SAM3ImageParityError(
            "SAM3 image paired detections require masks, boxes, scores, and class_ids to share N; "
            f"got masks={masks.shape[0]} boxes={boxes.shape[0]} scores={scores.shape[0]} class_ids={class_ids.shape[0]}"
        )

    keep = np.flatnonzero(scores > np.float32(score_threshold))
    order = keep[np.argsort(-scores[keep], kind="mergesort")]
    return {
        "masks": masks[order],
        "boxes": boxes[order],
        "scores": scores[order],
        "class_ids": class_ids[order],
    }


def _ensure_src_on_path() -> None:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _default_capture_inputs() -> dict[str, Any]:
    _ensure_src_on_path()
    from mlx_cv.parity.fixtures import sam3_fixed_image, sam3_text_prompt

    return {
        "image": sam3_fixed_image(),
        "prompt": sam3_text_prompt(),
        "prompt_kind": "text",
    }


def _capture_inputs(inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    out = _default_capture_inputs() if inputs is None else dict(inputs)
    if "image" not in out:
        raise SAM3ImageParityError("SAM3 image comparison inputs are missing 'image'")
    if "prompt" not in out:
        raise SAM3ImageParityError("SAM3 image comparison inputs are missing 'prompt'")
    out.setdefault("prompt_kind", "text" if isinstance(out["prompt"], str) else "pcs")
    return out


def _import_reference(reference_path: Path = SAM3_IMAGE_REFERENCE_PATH):
    if not reference_path.exists():
        raise SAM3ImageReferenceDependencyError(f"SAM3 reference path is missing: {reference_path}")

    try:
        import torch
        from PIL import Image

        __import__("torchvision")
        __import__("huggingface_hub")
        __import__("iopath")
        __import__("ftfy")
        __import__("regex")
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise SAM3ImageReferenceDependencyError(
            "SAM3 image upstream reference capture requires torch, torchvision, PIL, "
            "huggingface_hub, iopath, ftfy, regex, and the local references/sam3 checkout."
        ) from exc

    try:
        package_name = "sam3"
        package_root = reference_path / "sam3"
        if not package_root.exists():
            raise FileNotFoundError(package_root)
        package = sys.modules.get(package_name)
        if package is None or str(package_root.resolve()) not in list(getattr(package, "__path__", [])):
            package = types.ModuleType(package_name)
            package.__path__ = [str(package_root.resolve())]  # type: ignore[attr-defined]
            sys.modules[package_name] = package
        builder = importlib.import_module("sam3.model_builder")
        processor_mod = importlib.import_module("sam3.model.sam3_image_processor")
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise SAM3ImageReferenceDependencyError(
            "SAM3 image upstream reference modules could not be imported from "
            f"{reference_path}; ensure optional reference dependencies are installed."
        ) from exc

    return torch, Image, builder.build_sam3_image_model, processor_mod.Sam3Processor


def _load_reference_model(build_sam3_image_model: Any, checkpoint_path: Path) -> Any:
    try:
        return build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
            device="cpu",
            eval_mode=True,
            enable_segmentation=True,
        )
    except Exception as exc:  # pragma: no cover - requires real upstream checkpoint.
        raise SAM3ImageReferenceCaptureError(
            f"SAM3 image upstream model load failed for {checkpoint_path}: {exc}"
        ) from exc


def capture_sam3_image_upstream_reference(
    checkpoint_path: str | Path,
    *,
    reference_path: Path = SAM3_IMAGE_REFERENCE_PATH,
    inputs: Mapping[str, Any] | None = None,
) -> SAM3ImageCapture:
    """Run the upstream Torch SAM3 image reference on deterministic text input."""

    path = Path(checkpoint_path)
    torch, Image, build_sam3_image_model, RefProcessor = _import_reference(reference_path)
    model = _load_reference_model(build_sam3_image_model, path)
    capture_inputs = _capture_inputs(inputs)
    image = np.asarray(capture_inputs["image"], dtype=np.uint8)
    prompt = capture_inputs["prompt"]
    if not isinstance(prompt, str):
        raise SAM3ImageReferenceCaptureError(
            "SAM3 image upstream comparison currently captures token/text evidence with a text prompt; "
            f"got prompt_kind={capture_inputs.get('prompt_kind')!r}"
        )

    try:
        processor = RefProcessor(model, device="cpu", confidence_threshold=SAM3_IMAGE_DETECTION_SCORE_THRESHOLD)
        with torch.inference_mode():
            state = processor.set_image(Image.fromarray(image))
            output = processor.set_text_prompt(prompt=prompt, state=state)
        token_ids = model.backbone.language_backbone.tokenizer(
            [prompt],
            context_length=model.backbone.language_backbone.context_length,
        )
        backbone_out = output["backbone_out"]
        masks = _normalize_masks(_np(output["masks"]))
        boxes = _np(output["boxes"]).astype(np.float64, copy=False).reshape(-1, 4)
        scores = _np(output["scores"]).astype(np.float32, copy=False).reshape(-1)
        class_ids = np.zeros((len(scores),), dtype=np.int64)
        taps = {
            "text.token_ids": _np(token_ids).astype(np.int64),
            "text.language_features": _np(backbone_out["language_features"]),
            "text.language_embeds": _np(backbone_out["language_embeds"]),
        }
    except Exception as exc:  # pragma: no cover - requires real upstream checkpoint.
        raise SAM3ImageReferenceCaptureError(f"SAM3 image upstream reference capture failed: {exc}") from exc

    return SAM3ImageCapture(
        source="upstream_reference",
        prompt_kind=str(capture_inputs["prompt_kind"]),
        image=image,
        prompt=prompt,
        masks=masks,
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        taps=taps,
    )


def _metadata_config(path: Path) -> dict[str, Any] | None:
    if path.suffix != ".npz":
        return None
    try:
        with np.load(path, allow_pickle=False) as weights:
            if "__config_json__" not in weights.files:
                return None
            raw = np.asarray(weights["__config_json__"]).item()
    except Exception:
        return None
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return None


def _tuple_fields(data: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    out = dict(data)
    for field in fields:
        if field in out and isinstance(out[field], list):
            out[field] = tuple(out[field])
    return out


def _local_model_config(weights_path: Path):
    _ensure_src_on_path()
    from mlx_cv.heads.segmentation import SAM3DecoderConfig
    from mlx_cv.models.sam3 import SAM3Config, SAM3ImageBackboneConfig, SAM3TextConfig, SAM3Tokenizer

    metadata = _metadata_config(weights_path) or {}
    image_data = _tuple_fields(metadata.get("image", {}), ("out_layers", "neck_scales"))
    text_data = dict(metadata.get("text", {}))
    decoder_data = dict(metadata.get("decoder", {}))
    tokenizer = SAM3Tokenizer(context_length=int(text_data.get("context_length", SAM3TextConfig().context_length)))
    text_data.setdefault("vocab_size", tokenizer.vocab_size)
    return SAM3Config(
        image=SAM3ImageBackboneConfig(**image_data),
        text=SAM3TextConfig(**text_data),
        decoder=SAM3DecoderConfig(**decoder_data),
    )


def _local_processor_config(cfg: Any, metadata: Mapping[str, Any]):
    _ensure_src_on_path()
    from mlx_cv.models.sam3 import SAM3ProcessorConfig

    image_size = tuple(metadata.get("image_size", (cfg.image.image_size, cfg.image.image_size)))
    labels = metadata.get("labels")
    if labels is not None:
        labels = tuple(labels)
    return SAM3ProcessorConfig(
        image_size=image_size,
        top_k=int(metadata.get("num_select", cfg.decoder.num_queries)),
        score_threshold=SAM3_IMAGE_DETECTION_SCORE_THRESHOLD,
        labels=labels,
    )


def capture_sam3_image_local(
    local_checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, Any] | None = None,
) -> SAM3ImageCapture:
    """Run the local MLX SAM3 image path from a converted .npz or safetensors checkpoint."""

    path = Path(local_checkpoint_path)
    if not path.is_file() or path.suffix not in _SUPPORTED_IMAGE_FORMATS:
        raise SAM3ImageLocalCaptureError(
            "SAM3 image local MLX capture requires a converted local .npz or safetensors checkpoint; "
            f"got {path}."
        )
    _ensure_src_on_path()
    try:
        import mlx.core as mx

        from mlx_cv.models.sam3 import SAM3Model, SAM3Processor, SAM3Tokenizer, load_sam3_weights
    except Exception as exc:  # pragma: no cover - depends on local MLX runtime.
        raise SAM3ImageLocalCaptureError(f"SAM3 image local MLX capture requires mlx-cv runtime imports: {exc}") from exc

    metadata = _metadata_config(path) or {}
    try:
        cfg = _local_model_config(path)
        tokenizer = SAM3Tokenizer(context_length=int(cfg.text.context_length))
        model = load_sam3_weights(SAM3Model(cfg, tokenizer=tokenizer), path)
        processor = SAM3Processor(_local_processor_config(cfg, metadata))
    except Exception as exc:
        raise SAM3ImageLocalCaptureError(f"SAM3 image local MLX checkpoint load failed: {exc}") from exc

    capture_inputs = _capture_inputs(inputs)
    try:
        image = np.asarray(capture_inputs["image"], dtype=np.uint8)
        prompt = capture_inputs["prompt"]
        with mx.stream(mx.cpu):
            model_inputs, ctx = processor.preprocess({"image": image, "prompt": prompt})
            raw = model(model_inputs["pixel_values"], model_inputs["prompt"], capture_taps=True)
            result = processor.postprocess(raw, ctx)
            mx.eval(raw["mask_logits"], raw["object_scores"], raw["boxes"])

        taps = {key: _np(value) for key, value in raw["taps"].items()}
        masks = _normalize_masks(result.masks.data)
        boxes = np.asarray(result.detections.boxes, dtype=np.float64)
        scores = np.asarray(result.detections.scores, dtype=np.float32)
        class_ids = np.asarray(result.detections.class_ids, dtype=np.int64)
    except Exception as exc:
        raise SAM3ImageLocalCaptureError(f"SAM3 image local MLX capture failed: {exc}") from exc

    return SAM3ImageCapture(
        source="mlx_local",
        prompt_kind=str(capture_inputs["prompt_kind"]),
        image=image,
        prompt=prompt,
        masks=masks,
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        taps=taps,
    )


def _max_rel_error(got: np.ndarray, expected: np.ndarray) -> float:
    denom = np.maximum(np.abs(expected), 1.0e-8)
    if got.size == 0:
        return 0.0
    return float(np.max(np.abs(got - expected) / denom))


def _compare_array(name: str, reference: Any, local: Any, tolerances: Mapping[str, float]) -> FieldComparison:
    ref = _np(reference)
    got = _np(local)
    atol = float(tolerances["atol"])
    rtol = float(tolerances["rtol"])
    same_shape = got.shape == ref.shape
    finite = bool(np.all(np.isfinite(got)) and np.all(np.isfinite(ref))) if same_shape else False
    max_abs = None
    max_rel = None
    passed = False
    if same_shape and finite:
        ref64 = ref.astype(np.float64, copy=False)
        got64 = got.astype(np.float64, copy=False)
        diff = np.abs(got64 - ref64)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        max_rel = _max_rel_error(got64, ref64)
        passed = bool(np.all(diff <= (atol + rtol * np.abs(ref64))))
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


def _capture_summary(capture: Any) -> dict[str, Any]:
    summary = getattr(capture, "summary", None)
    if callable(summary):
        return dict(summary())
    return {
        "source": getattr(capture, "source", type(capture).__name__),
        "masks_shape": list(np.asarray(capture.masks).shape),
        "boxes_shape": list(np.asarray(capture.boxes).shape),
        "scores_shape": list(np.asarray(capture.scores).shape),
        "class_ids_shape": list(np.asarray(capture.class_ids).shape),
        "tap_order": list(getattr(capture, "taps", {})),
    }


def compare_sam3_image_captures(
    reference: Any,
    local: Any,
    *,
    selected_tap_pairs: Sequence[tuple[str, str]] = SAM3_IMAGE_SELECTED_TAP_PAIRS,
    tolerances: Mapping[str, Mapping[str, float]] = SAM3_IMAGE_FIELD_TOLERANCES,
) -> dict[str, Any]:
    """Compare SAM3 image masks, paired detections, and selected token/text taps."""

    reference_public = _normalize_public_outputs(reference)
    local_public = _normalize_public_outputs(local)
    comparisons = [
        _compare_array("masks", reference_public["masks"], local_public["masks"], tolerances["masks"]),
        _compare_array("boxes", reference_public["boxes"], local_public["boxes"], tolerances["boxes"]),
        _compare_array("scores", reference_public["scores"], local_public["scores"], tolerances["scores"]),
        _compare_array("class_ids", reference_public["class_ids"], local_public["class_ids"], tolerances["class_ids"]),
    ]
    for reference_key, local_key in selected_tap_pairs:
        if reference_key not in reference.taps:
            raise SAM3ImageParityError(f"SAM3 image upstream capture missing selected tap {reference_key!r}")
        if local_key not in local.taps:
            raise SAM3ImageParityError(f"SAM3 image local capture missing selected tap {local_key!r}")
        field_name = f"tap.{reference_key}"
        comparisons.append(
            _compare_array(field_name, reference.taps[reference_key], local.taps[local_key], tolerances[field_name])
        )

    fields = [asdict(item) for item in comparisons]
    return {
        "passed": all(item["passed"] for item in fields),
        "tolerances": {key: dict(value) for key, value in tolerances.items()},
        "selected_tap_pairs": [
            {"reference": reference_key, "local": local_key}
            for reference_key, local_key in selected_tap_pairs
        ],
        "detection_selection": {
            "score_threshold": SAM3_IMAGE_DETECTION_SCORE_THRESHOLD,
            "score_threshold_op": ">",
            "order": SAM3_IMAGE_DETECTION_ORDER,
        },
        "fields": fields,
        "reference_summary": _capture_summary(reference),
        "local_summary": _capture_summary(local),
    }


def _failure_summary(report: Mapping[str, Any]) -> str:
    failed = [field for field in report["fields"] if not field["passed"]]
    if not failed:
        return "unknown comparison failure"
    first = failed[0]
    return (
        f"{first['name']} max_abs={first['max_abs_error']} max_rel={first['max_rel_error']} "
        f"tol=({first['atol']},{first['rtol']})"
    )


def _reference_inputs(capture: Any) -> Mapping[str, Any] | None:
    method = getattr(capture, "inputs_for_local", None)
    if callable(method):
        return method()
    if hasattr(capture, "image") and hasattr(capture, "prompt"):
        return {
            "image": np.asarray(capture.image),
            "prompt": capture.prompt,
            "prompt_kind": getattr(capture, "prompt_kind", "text"),
        }
    return None


def _resolve_upstream_checkpoint(env: Mapping[str, str]) -> Path:
    upstream = env.get(SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV)
    if not upstream:
        raise SAM3ImageReferenceCaptureError(
            f"{SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV} is unset; set it to an upstream Torch .pt/.pth "
            "SAM3 image checkpoint for reference capture"
        )
    path = Path(upstream)
    if not path.exists():
        raise SAM3ImageReferenceCaptureError(
            f"{SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV} does not point to an existing path: {path}"
        )
    if not path.is_file():
        raise SAM3ImageReferenceCaptureError(f"{SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV} does not point to a file: {path}")
    if path.suffix not in _SUPPORTED_UPSTREAM_FORMATS:
        raise SAM3ImageReferenceCaptureError(
            f"{SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV} must point to an upstream Torch .pt/.pth checkpoint: {path}"
        )
    return path


def _resolve_local_checkpoint(env: Mapping[str, str], *, min_checkpoint_bytes: int) -> Path:
    local = env.get(SAM3_IMAGE_LOCAL_CHECKPOINT_ENV)
    if not local:
        raise SAM3ImageLocalCaptureError(
            f"{SAM3_IMAGE_LOCAL_CHECKPOINT_ENV} is unset; set it to a converted local MLX .npz or "
            "safetensors SAM3 image checkpoint"
        )
    path = Path(local)
    admission = evaluate_sam3_image_gate(
        environ={SAM3_IMAGE_CHECKPOINT_ENV: str(path)},
        min_checkpoint_bytes=min_checkpoint_bytes,
    )
    if admission.blocked:
        reason = (admission.blocked_reason or admission.status).replace(
            SAM3_IMAGE_CHECKPOINT_ENV,
            SAM3_IMAGE_LOCAL_CHECKPOINT_ENV,
        )
        raise SAM3ImageLocalCaptureError(
            f"{SAM3_IMAGE_LOCAL_CHECKPOINT_ENV} is not a usable converted local SAM3 image checkpoint: {reason}"
        )
    return path


def evaluate_sam3_image_comparison_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_checkpoint_bytes: int = 1_000_000,
    check_reference_dependencies: bool = True,
    reference_capture_func: Callable[..., Any] = capture_sam3_image_upstream_reference,
    local_capture_func: Callable[..., Any] = capture_sam3_image_local,
    compare_func: Callable[..., dict[str, Any]] = compare_sam3_image_captures,
) -> SAM3ImageGateResult:
    """Evaluate the full upstream-vs-MLX SAM3 image comparison gate.

    Admission is checked first, preserving checkpoint blockers such as
    image-vs-video rejection. With the reference checkout, optional Torch
    runtime, and a converted local MLX checkpoint available, this captures a
    deterministic text-prompt image pass on both sides and compares public
    masks, paired detections, and token/text taps with
    ``SAM3_IMAGE_FIELD_TOLERANCES``. Missing prerequisites return precise
    blockers instead of synthetic pass/fail status.
    """

    env = os.environ if environ is None else environ
    admission = evaluate_sam3_image_gate(environ=env, min_checkpoint_bytes=min_checkpoint_bytes)
    if admission.blocked:
        return admission

    if not SAM3_IMAGE_REFERENCE_PATH.exists():
        return _block_from_admission(admission, f"SAM3 reference path is missing: {SAM3_IMAGE_REFERENCE_PATH}")

    try:
        upstream_checkpoint = _resolve_upstream_checkpoint(env)
    except SAM3ImageReferenceCaptureError as exc:
        return _block_from_admission(admission, str(exc))

    try:
        local_checkpoint = _resolve_local_checkpoint(env, min_checkpoint_bytes=min_checkpoint_bytes)
    except SAM3ImageLocalCaptureError as exc:
        return _block_from_admission(admission, str(exc), upstream_checkpoint_path=upstream_checkpoint)

    if check_reference_dependencies:
        try:
            _import_reference(SAM3_IMAGE_REFERENCE_PATH)
        except SAM3ImageReferenceDependencyError as exc:
            return _block_from_admission(
                admission,
                str(exc),
                upstream_checkpoint_path=upstream_checkpoint,
                local_checkpoint_path=local_checkpoint,
            )

    assert admission.checkpoint_path is not None
    try:
        reference = reference_capture_func(upstream_checkpoint, reference_path=SAM3_IMAGE_REFERENCE_PATH)
    except SAM3ImageReferenceDependencyError as exc:
        return _block_from_admission(
            admission,
            str(exc),
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )
    except SAM3ImageReferenceCaptureError as exc:
        return _block_from_admission(
            admission,
            f"SAM3 image upstream reference capture failed: {exc}",
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )
    except Exception as exc:
        return _block_from_admission(
            admission,
            f"SAM3 image upstream reference capture failed: {exc}",
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )

    try:
        local = local_capture_func(local_checkpoint, inputs=_reference_inputs(reference))
    except SAM3ImageLocalCaptureError as exc:
        return _block_from_admission(
            admission,
            str(exc),
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )
    except Exception as exc:
        return _block_from_admission(
            admission,
            f"SAM3 image local MLX capture failed: {exc}",
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )

    try:
        report = compare_func(reference, local)
    except SAM3ImageParityError as exc:
        return _block_from_admission(
            admission,
            f"SAM3 image comparison component unavailable: {exc}",
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )
    except Exception as exc:
        return _block_from_admission(
            admission,
            f"SAM3 image comparison failed: {exc}",
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )

    if not report.get("passed", False):
        return _block_from_admission(
            admission,
            "SAM3 image upstream-vs-MLX parity drift for masks, paired detections, or token/text taps: "
            + _failure_summary(report),
            upstream_checkpoint_path=upstream_checkpoint,
            local_checkpoint_path=local_checkpoint,
        )

    return _pass_from_admission(
        admission,
        report,
        upstream_checkpoint_path=upstream_checkpoint,
        local_checkpoint_path=local_checkpoint,
    )


def status_dict(result: SAM3ImageGateResult) -> dict:
    out = asdict(result)
    out["model"] = "sam3_image"
    out["display_name"] = "SAM 3.1 image-mode"
    if result.blocked:
        claim_level = "external_blocker"
    elif result.status == "UPSTREAM_PASSED" and result.comparison_report is not None:
        claim_level = "upstream_passed"
    elif result.admitted:
        claim_level = "checkpoint_admitted"
    else:
        claim_level = "unknown"
    out["claim_level"] = claim_level
    out["comparison_scope"] = "public masks, paired boxes/scores/class_ids, text token ids and text embeddings"
    out["upstream_checkpoint_env"] = SAM3_IMAGE_UPSTREAM_CHECKPOINT_ENV
    out["local_checkpoint_env"] = SAM3_IMAGE_LOCAL_CHECKPOINT_ENV
    return out
