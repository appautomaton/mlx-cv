"""LocateAnything checkpoint admission and upstream parity gate helpers."""

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


LOCATEANYTHING_CHECKPOINT_ENV = "MLX_CV_LOCATEANYTHING_CHECKPOINT"
LOCATEANYTHING_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_LOCATEANYTHING_GATE"
LOCATEANYTHING_LOCAL_CHECKPOINT_ENV = "MLX_CV_LOCATEANYTHING_LOCAL_CHECKPOINT"
LOCATEANYTHING_REFERENCE_PATH = Path("references/LocateAnything-3B")
TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent

# The real comparison is intentionally narrow and documented: public decoded
# geometry should be exact after deterministic token parsing, while projector
# and embedding taps allow small fp32 implementation noise across Torch/MLX.
LOCATEANYTHING_FIELD_TOLERANCES: dict[str, dict[str, float]] = {
    "boxes": {"atol": 1.0e-6, "rtol": 0.0},
    "points": {"atol": 1.0e-6, "rtol": 0.0},
    "tap.projector": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "tap.inputs_embeds": {"atol": 1.0e-4, "rtol": 1.0e-4},
    "tap.sampled_tokens": {"atol": 0.0, "rtol": 0.0},
}
LOCATEANYTHING_SELECTED_TAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("projector", "projector"),
    ("inputs_embeds", "inputs_embeds"),
    ("sampled_tokens", "sampled_tokens"),
)


@dataclass(frozen=True)
class LocateAnythingGateResult:
    status: str
    checkpoint_env: str
    required_gate_env: str
    reference_path: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    blocked_reason: str | None = None
    admitted: bool = False
    comparison_report: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


class LocateAnythingReferenceDependencyError(RuntimeError):
    """Raised when the upstream LocateAnything reference runtime is unavailable."""


class LocateAnythingReferenceCaptureError(RuntimeError):
    """Raised when upstream LocateAnything capture cannot run or is malformed."""


class LocateAnythingLocalCaptureError(RuntimeError):
    """Raised when the local MLX LocateAnything capture cannot run."""


class LocateAnythingParityError(AssertionError):
    """Raised when LocateAnything upstream-vs-MLX comparison cannot be evaluated."""


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
class LocateAnythingCapture:
    source: str
    input_ids: np.ndarray
    cached_image_features: np.ndarray
    pbd_block_logits: np.ndarray
    generated_ids: np.ndarray
    boxes: np.ndarray
    points: np.ndarray
    taps: dict[str, np.ndarray]

    def inputs_for_local(self) -> dict[str, np.ndarray]:
        return {
            "input_ids": self.input_ids,
            "cached_image_features": self.cached_image_features,
            "pbd_block_logits": self.pbd_block_logits,
            "generated_ids": self.generated_ids,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "input_ids_shape": list(np.asarray(self.input_ids).shape),
            "cached_image_features_shape": list(np.asarray(self.cached_image_features).shape),
            "pbd_block_logits_shape": list(np.asarray(self.pbd_block_logits).shape),
            "generated_ids_shape": list(np.asarray(self.generated_ids).shape),
            "boxes_shape": list(np.asarray(self.boxes).shape),
            "points_shape": list(np.asarray(self.points).shape),
            "tap_order": list(self.taps),
            "tap_shapes": {name: list(np.asarray(value).shape) for name, value in self.taps.items()},
        }


def required_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(LOCATEANYTHING_REQUIRED_GATE_ENV) == "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(reason: str, *, environ: Mapping[str, str]) -> LocateAnythingGateResult:
    return LocateAnythingGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=LOCATEANYTHING_CHECKPOINT_ENV,
        required_gate_env=LOCATEANYTHING_REQUIRED_GATE_ENV,
        reference_path=str(LOCATEANYTHING_REFERENCE_PATH),
        checkpoint_path=environ.get(LOCATEANYTHING_CHECKPOINT_ENV),
        blocked_reason=reason,
    )


def _admit(path: Path, *, environ: Mapping[str, str], sha256: str | None = None) -> LocateAnythingGateResult:
    return LocateAnythingGateResult(
        status="ADMITTED",
        checkpoint_env=LOCATEANYTHING_CHECKPOINT_ENV,
        required_gate_env=LOCATEANYTHING_REQUIRED_GATE_ENV,
        reference_path=str(LOCATEANYTHING_REFERENCE_PATH),
        checkpoint_path=str(path),
        checkpoint_sha256=sha256,
        admitted=True,
    )


def _block_from_admission(admission: LocateAnythingGateResult, reason: str) -> LocateAnythingGateResult:
    return LocateAnythingGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=admission.checkpoint_env,
        required_gate_env=admission.required_gate_env,
        reference_path=admission.reference_path,
        checkpoint_path=admission.checkpoint_path,
        checkpoint_sha256=admission.checkpoint_sha256,
        blocked_reason=reason,
        admitted=admission.admitted,
    )


def _pass_from_admission(admission: LocateAnythingGateResult, report: dict[str, Any]) -> LocateAnythingGateResult:
    return LocateAnythingGateResult(
        status="UPSTREAM_PASSED",
        checkpoint_env=admission.checkpoint_env,
        required_gate_env=admission.required_gate_env,
        reference_path=admission.reference_path,
        checkpoint_path=admission.checkpoint_path,
        checkpoint_sha256=admission.checkpoint_sha256,
        admitted=True,
        comparison_report=report,
    )


def _index_shards(index_path: Path) -> list[str] | None:
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None
    shards = sorted(set(index.get("weight_map", {}).values()))
    return shards or None


def evaluate_locateanything_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
) -> LocateAnythingGateResult:
    env = os.environ if environ is None else environ
    checkpoint = env.get(LOCATEANYTHING_CHECKPOINT_ENV)
    if not checkpoint:
        return _block(f"{LOCATEANYTHING_CHECKPOINT_ENV} is unset", environ=env)

    path = Path(checkpoint)
    if not path.exists():
        return _block(f"{LOCATEANYTHING_CHECKPOINT_ENV} does not point to an existing path: {path}", environ=env)

    if path.is_file():
        if path.suffix not in {".npz", ".safetensors"}:
            return _block(f"unsupported LocateAnything checkpoint format: {path.suffix or path.name}", environ=env)
        if path.stat().st_size < min_shard_bytes:
            return _block(f"{path} is not a usable LocateAnything checkpoint file", environ=env)
        return _admit(path, environ=env, sha256=_sha256(path))

    if not path.is_dir():
        return _block(f"{LOCATEANYTHING_CHECKPOINT_ENV} is neither a file nor a directory: {path}", environ=env)

    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return _block(f"LocateAnything checkpoint directory is missing {index_path.name}: {path}", environ=env)

    shards = _index_shards(index_path)
    if not shards:
        return _block(f"LocateAnything safetensors index has no weight_map entries: {index_path}", environ=env)

    missing = [shard for shard in shards if not (path / shard).exists()]
    if missing:
        return _block(f"LocateAnything checkpoint directory is missing shard(s): {', '.join(missing[:3])}", environ=env)

    stub_shards = [shard for shard in shards if (path / shard).stat().st_size < min_shard_bytes]
    if stub_shards:
        return _block(
            f"LocateAnything checkpoint shard(s) are LFS stubs or incomplete: {', '.join(stub_shards[:3])}",
            environ=env,
        )

    return _admit(path, environ=env)


def status_dict(result: LocateAnythingGateResult) -> dict:
    out = asdict(result)
    out["model"] = "locateanything"
    out["display_name"] = "LocateAnything-3B"
    if result.blocked:
        claim_level = "external_blocker"
    elif result.status == "UPSTREAM_PASSED" and result.comparison_report is not None:
        claim_level = "upstream_passed"
    elif result.admitted:
        claim_level = "checkpoint_admitted"
    else:
        claim_level = "unknown"
    out["claim_level"] = claim_level
    out["comparison_scope"] = "decoded boxes, decoded points, projector/input-embedding taps, sampled PBD tokens"
    out["local_checkpoint_env"] = LOCATEANYTHING_LOCAL_CHECKPOINT_ENV
    return out


def _np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _ensure_src_on_path() -> None:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _import_reference(reference_path: Path = LOCATEANYTHING_REFERENCE_PATH):
    try:
        import torch

        __import__("transformers")
        __import__("peft")
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise LocateAnythingReferenceDependencyError(
            "LocateAnything upstream reference capture requires torch, transformers, peft, "
            "and the local references/LocateAnything-3B checkout."
        ) from exc

    if not reference_path.exists():
        raise LocateAnythingReferenceDependencyError(f"LocateAnything reference path is missing: {reference_path}")

    try:
        package_name = "locateanything_ref"
        package = sys.modules.get(package_name)
        if package is None:
            package = types.ModuleType(package_name)
            package.__path__ = [str(reference_path.resolve())]  # type: ignore[attr-defined]
            sys.modules[package_name] = package
        config_mod = importlib.import_module(f"{package_name}.configuration_locateanything")
        model_mod = importlib.import_module(f"{package_name}.modeling_locateanything")
        generate_mod = importlib.import_module(f"{package_name}.generate_utils")
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise LocateAnythingReferenceDependencyError(
            "LocateAnything upstream reference modules could not be imported from "
            f"{reference_path}; ensure optional reference dependencies are installed."
        ) from exc

    return torch, config_mod.LocateAnythingConfig, model_mod.LocateAnythingForConditionalGeneration, generate_mod


def _load_reference_model(torch: Any, RefConfig: Any, RefModel: Any, checkpoint_path: Path) -> Any:
    if not checkpoint_path.is_dir():
        raise LocateAnythingReferenceCaptureError(
            "LocateAnything upstream reference capture requires a checkpoint directory "
            f"with config.json and safetensors shards; got {checkpoint_path}"
        )
    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise LocateAnythingReferenceCaptureError(f"LocateAnything reference checkpoint is missing config.json: {checkpoint_path}")

    try:
        config = RefConfig.from_pretrained(str(checkpoint_path))
        config._attn_implementation = "sdpa"
        config.vision_config._attn_implementation = "sdpa"
        config.text_config._attn_implementation = "sdpa"
        model = RefModel.from_pretrained(str(checkpoint_path), config=config, torch_dtype=torch.float32)
        model = model.to("cpu").float()
        model.eval()
    except Exception as exc:  # pragma: no cover - requires real upstream checkpoint.
        raise LocateAnythingReferenceCaptureError(f"LocateAnything upstream model load failed: {exc}") from exc
    return model


def _config_int(config: Any, name: str) -> int:
    return int(getattr(config, name))


def _text_config_int(config: Any, name: str) -> int:
    return int(getattr(config.text_config, name))


def _fixed_capture_inputs(config: Any) -> dict[str, np.ndarray]:
    vision = config.vision_config
    merge_h, merge_w = tuple(getattr(vision, "merge_kernel_size", (2, 2)))
    input_dim = int(vision.hidden_size) * int(merge_h) * int(merge_w)
    vocab_size = max(
        _text_config_int(config, "vocab_size"),
        _config_int(config, "coord_end_token_id") + 1,
        _config_int(config, "box_end_token_id") + 1,
        _config_int(config, "ref_end_token_id") + 1,
        _text_config_int(config, "null_token_id") + 1,
        _text_config_int(config, "switch_token_id") + 1,
    )
    image_token = _config_int(config, "image_token_index")
    coord = _config_int(config, "coord_start_token_id")
    box_start = _config_int(config, "box_start_token_id")
    box_end = _config_int(config, "box_end_token_id")

    sampled = np.asarray([box_start, coord + 100, coord + 200, coord + 700, coord + 800, box_end], dtype=np.int64)
    logits = np.full((6, vocab_size), -20.0, dtype=np.float32)
    for row, token in enumerate(sampled):
        logits[row, int(token)] = 20.0

    generated = np.asarray(
        [
            *sampled.tolist(),
            box_start,
            coord + 500,
            coord + 250,
            box_end,
        ],
        dtype=np.int32,
    )
    return {
        "input_ids": np.asarray([[10, image_token, 11]], dtype=np.int64),
        "cached_image_features": np.linspace(-0.25, 0.25, num=input_dim, dtype=np.float32).reshape(1, input_dim),
        "pbd_block_logits": logits,
        "generated_ids": generated,
    }


def _decode_generated_geometry(generated_ids: Sequence[int] | np.ndarray, config: Any) -> tuple[np.ndarray, np.ndarray]:
    _ensure_src_on_path()
    from mlx_cv.core.geometry import SpatialTransform
    from mlx_cv.models.locateanything.decode import TokenScheme, parse_grounding_tokens

    image_size = (10, 20)
    model_size = (20, 40)
    transform = SpatialTransform.resize(image_size, model_size)
    scheme = TokenScheme(
        box_start=_config_int(config, "box_start_token_id"),
        box_end=_config_int(config, "box_end_token_id"),
        coord_start=_config_int(config, "coord_start_token_id"),
        coord_end=_config_int(config, "coord_end_token_id"),
        ref_start=_config_int(config, "ref_start_token_id"),
        ref_end=_config_int(config, "ref_end_token_id"),
        none_id=_config_int(config, "none_token_id"),
    )

    boxes: list[np.ndarray] = []
    points: list[np.ndarray] = []
    model_h, model_w = model_size
    for item in parse_grounding_tokens(generated_ids, scheme):
        if item.kind == "box":
            coords = item.coords[:4]
            model_box = [
                coords[0] / 1000.0 * model_w,
                coords[1] / 1000.0 * model_h,
                coords[2] / 1000.0 * model_w,
                coords[3] / 1000.0 * model_h,
            ]
            boxes.append(transform.invert_boxes([model_box], clip=True)[0])
        elif item.kind == "point":
            coords = item.coords[:2]
            model_point = [
                coords[0] / 1000.0 * model_w,
                coords[1] / 1000.0 * model_h,
            ]
            points.append(transform.invert_points([model_point], clip=True)[0])

    return (
        np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        np.asarray(points, dtype=np.float64).reshape(-1, 2),
    )


def _reference_projector_and_embeddings(torch: Any, model: Any, inputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    input_ids = torch.as_tensor(inputs["input_ids"], dtype=torch.long, device="cpu")
    cached = torch.as_tensor(inputs["cached_image_features"], dtype=torch.float32, device="cpu")
    with torch.no_grad():
        projector = model.mlp1(cached)
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        batch, seq_len, channels = text_embeds.shape
        flat = text_embeds.reshape(batch * seq_len, channels).clone()
        selected = input_ids.reshape(batch * seq_len) == int(model.image_token_index)
        if int(selected.sum().item()) != int(projector.shape[0]):
            raise LocateAnythingReferenceCaptureError(
                "LocateAnything reference capture image token count does not match projected feature count: "
                f"{int(selected.sum().item())} tokens vs {int(projector.shape[0])} features"
            )
        flat[selected] = projector
        inputs_embeds = flat.reshape(batch, seq_len, channels)
    return _np(projector), _np(inputs_embeds)


def _reference_sampled_tokens(torch: Any, generate_mod: Any, model: Any, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    block_logits = torch.as_tensor(inputs["pbd_block_logits"], dtype=torch.float32, device="cpu")
    generated = torch.as_tensor(inputs["input_ids"], dtype=torch.long, device="cpu")
    with torch.no_grad():
        _probs, _confidence, greedy, box_avg = generate_mod.sample_tokens(
            block_logits.unsqueeze(0),
            generated,
            model.token_ids,
            generation_mode="hybrid",
            temperature=0,
            keep_k_avg=5,
        )
        new_tokens = greedy[0] if bool((box_avg[0] == 0).all().item()) else box_avg[0]
        pattern = generate_mod.handle_pattern(new_tokens, model.token_ids, generation_mode="hybrid")
    return np.asarray(pattern["tokens"], dtype=np.int64)


def capture_locateanything_upstream_reference(
    checkpoint_path: str | Path,
    *,
    reference_path: Path = LOCATEANYTHING_REFERENCE_PATH,
    inputs: Mapping[str, np.ndarray] | None = None,
) -> LocateAnythingCapture:
    """Run the upstream Torch reference on deterministic comparable inputs."""

    path = Path(checkpoint_path)
    torch, RefConfig, RefModel, generate_mod = _import_reference(reference_path)
    model = _load_reference_model(torch, RefConfig, RefModel, path)
    capture_inputs = _fixed_capture_inputs(model.config) if inputs is None else {
        key: np.asarray(value) for key, value in inputs.items()
    }
    projector, inputs_embeds = _reference_projector_and_embeddings(torch, model, capture_inputs)
    sampled = _reference_sampled_tokens(torch, generate_mod, model, capture_inputs)
    boxes, points = _decode_generated_geometry(capture_inputs["generated_ids"], model.config)
    return LocateAnythingCapture(
        source="upstream_reference",
        input_ids=np.asarray(capture_inputs["input_ids"]),
        cached_image_features=np.asarray(capture_inputs["cached_image_features"], dtype=np.float32),
        pbd_block_logits=np.asarray(capture_inputs["pbd_block_logits"], dtype=np.float32),
        generated_ids=np.asarray(capture_inputs["generated_ids"], dtype=np.int32),
        boxes=boxes,
        points=points,
        taps={
            "projector": projector,
            "inputs_embeds": inputs_embeds,
            "sampled_tokens": sampled,
        },
    )


class _TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{int(i)}" for i in ids)


def _config_from_json_dict(data: Mapping[str, Any]):
    _ensure_src_on_path()
    from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
    from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
    from mlx_cv.models.locateanything.config import LocateAnythingConfig

    vision_data = data.get("vision_config", data.get("vision", {}))
    text_data = data.get("text_config", data.get("text", {}))
    text_data = dict(text_data)
    if "text_mask_token_id" not in text_data and "text_mask_token_id" in data:
        text_data["text_mask_token_id"] = data["text_mask_token_id"]
    if "null_token_id" not in text_data and "null_token_id" in data:
        text_data["null_token_id"] = data["null_token_id"]
    if "switch_token_id" not in text_data and "switch_token_id" in data:
        text_data["switch_token_id"] = data["switch_token_id"]

    text = Qwen2Config.from_dict(text_data)
    vision = MoonViTConfig.from_dict(vision_data)
    defaults = LocateAnythingConfig()
    return LocateAnythingConfig(
        vision_config=vision,
        text_config=text,
        vocab_size=int(data.get("vocab_size", text.vocab_size)),
        image_token_index=int(data.get("image_token_index", defaults.image_token_index)),
        box_start_token_id=int(data.get("box_start_token_id", defaults.box_start_token_id)),
        box_end_token_id=int(data.get("box_end_token_id", defaults.box_end_token_id)),
        coord_start_token_id=int(data.get("coord_start_token_id", defaults.coord_start_token_id)),
        coord_end_token_id=int(data.get("coord_end_token_id", defaults.coord_end_token_id)),
        ref_start_token_id=int(data.get("ref_start_token_id", defaults.ref_start_token_id)),
        ref_end_token_id=int(data.get("ref_end_token_id", defaults.ref_end_token_id)),
        none_token_id=int(data.get("none_token_id", defaults.none_token_id)),
        null_token_id=int(data.get("null_token_id", text.null_token_id)),
        switch_token_id=int(data.get("switch_token_id", text.switch_token_id)),
        text_mask_token_id=int(data.get("text_mask_token_id", text.text_mask_token_id)),
    )


def _load_local_config(weights: Any):
    if "__config_json__" not in weights.files:
        _ensure_src_on_path()
        from mlx_cv.models.locateanything.config import LocateAnythingConfig

        return LocateAnythingConfig()
    config_json = weights["__config_json__"]
    data = json.loads(str(np.asarray(config_json).item()))
    return _config_from_json_dict(data)


def capture_locateanything_local(
    local_checkpoint_path: str | Path,
    *,
    inputs: Mapping[str, np.ndarray] | None = None,
) -> LocateAnythingCapture:
    """Strictly run the local MLX LocateAnything path from a user-supplied .npz checkpoint."""

    path = Path(local_checkpoint_path)
    if not path.is_file() or path.suffix != ".npz":
        raise LocateAnythingLocalCaptureError(
            "LocateAnything local MLX capture requires a converted local .npz checkpoint; "
            f"got {path}. Set {LOCATEANYTHING_LOCAL_CHECKPOINT_ENV} to a local MLX .npz weights file."
        )
    _ensure_src_on_path()
    try:
        import mlx.core as mx
        from mlx.utils import tree_unflatten

        from mlx_cv.core.geometry import SpatialTransform
        from mlx_cv.models.locateanything.modeling import LocateAnythingModel
        from mlx_cv.models.locateanything.pbd import get_token_ids, sample_block
        from mlx_cv.models.locateanything.processor import LocateAnythingProcessor, LocateAnythingProcessorContext
    except Exception as exc:  # pragma: no cover - depends on local MLX runtime.
        raise LocateAnythingLocalCaptureError(f"LocateAnything local MLX capture requires mlx-cv runtime imports: {exc}") from exc

    try:
        weights = np.load(path, allow_pickle=False)
        config = _load_local_config(weights)
        model = LocateAnythingModel(config)
        params = [(key, mx.array(weights[key])) for key in weights.files if not key.startswith("__")]
        model.update(tree_unflatten(params))
    except Exception as exc:
        raise LocateAnythingLocalCaptureError(f"LocateAnything local MLX checkpoint load failed: {exc}") from exc

    capture_inputs = _fixed_capture_inputs(config) if inputs is None else {key: np.asarray(value) for key, value in inputs.items()}
    try:
        input_ids = mx.array(capture_inputs["input_ids"], dtype=mx.int32)
        cached = mx.array(capture_inputs["cached_image_features"], dtype=mx.float32)
        block_logits = mx.array(capture_inputs["pbd_block_logits"], dtype=mx.float32)
        with mx.stream(mx.cpu):
            projector = model.multi_modal_projector(cached)
            inputs_embeds = model.get_input_embeddings(input_ids, cached_image_features=cached)
            sampled = np.asarray(sample_block(block_logits, get_token_ids(model.config)), dtype=np.int64)
            mx.eval(projector, inputs_embeds)

        processor = LocateAnythingProcessor(model.config, tokenizer=_TinyTokenizer())
        ctx = LocateAnythingProcessorContext(
            transform=SpatialTransform.resize((10, 20), (20, 40)),
            image_size=(10, 20),
            model_size=(20, 40),
            image_grid_hws=((2, 2),),
        )
        result = processor.postprocess(capture_inputs["generated_ids"], ctx)
        boxes = (
            np.asarray(result.detections.boxes, dtype=np.float64)
            if result.detections is not None
            else np.zeros((0, 4), dtype=np.float64)
        )
        points = (
            np.asarray(result.points.points, dtype=np.float64)
            if result.points is not None
            else np.zeros((0, 2), dtype=np.float64)
        )
    except Exception as exc:
        raise LocateAnythingLocalCaptureError(f"LocateAnything local MLX capture failed: {exc}") from exc

    return LocateAnythingCapture(
        source="mlx_local",
        input_ids=np.asarray(capture_inputs["input_ids"]),
        cached_image_features=np.asarray(capture_inputs["cached_image_features"], dtype=np.float32),
        pbd_block_logits=np.asarray(capture_inputs["pbd_block_logits"], dtype=np.float32),
        generated_ids=np.asarray(capture_inputs["generated_ids"], dtype=np.int32),
        boxes=boxes,
        points=points,
        taps={
            "projector": np.asarray(projector),
            "inputs_embeds": np.asarray(inputs_embeds),
            "sampled_tokens": sampled,
        },
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
        diff = np.abs(got.astype(np.float64, copy=False) - ref.astype(np.float64, copy=False))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        max_rel = _max_rel_error(got.astype(np.float64, copy=False), ref.astype(np.float64, copy=False))
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


def _capture_summary(capture: Any) -> dict[str, Any]:
    summary = getattr(capture, "summary", None)
    if callable(summary):
        return dict(summary())
    return {
        "source": getattr(capture, "source", type(capture).__name__),
        "boxes_shape": list(np.asarray(capture.boxes).shape),
        "points_shape": list(np.asarray(capture.points).shape),
        "tap_order": list(getattr(capture, "taps", {})),
    }


def compare_locateanything_captures(
    reference: Any,
    local: Any,
    *,
    selected_tap_pairs: Sequence[tuple[str, str]] = LOCATEANYTHING_SELECTED_TAP_PAIRS,
    tolerances: Mapping[str, Mapping[str, float]] = LOCATEANYTHING_FIELD_TOLERANCES,
) -> dict[str, Any]:
    """Compare decoded LocateAnything boxes/points and selected stable taps."""

    comparisons = [
        _compare_array("boxes", reference.boxes, local.boxes, tolerances["boxes"]),
        _compare_array("points", reference.points, local.points, tolerances["points"]),
    ]
    for reference_key, local_key in selected_tap_pairs:
        if reference_key not in reference.taps:
            raise LocateAnythingParityError(f"LocateAnything upstream capture missing selected tap {reference_key!r}")
        if local_key not in local.taps:
            raise LocateAnythingParityError(f"LocateAnything local capture missing selected tap {local_key!r}")
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


def _reference_inputs(capture: Any) -> Mapping[str, np.ndarray] | None:
    method = getattr(capture, "inputs_for_local", None)
    if callable(method):
        return method()
    required = ("input_ids", "cached_image_features", "pbd_block_logits", "generated_ids")
    if all(hasattr(capture, name) for name in required):
        return {name: np.asarray(getattr(capture, name)) for name in required}
    return None


def _resolve_local_checkpoint(admission: LocateAnythingGateResult, env: Mapping[str, str]) -> Path:
    local = env.get(LOCATEANYTHING_LOCAL_CHECKPOINT_ENV)
    if local:
        path = Path(local)
        if not path.exists():
            raise LocateAnythingLocalCaptureError(
                f"{LOCATEANYTHING_LOCAL_CHECKPOINT_ENV} does not point to an existing path: {path}"
            )
        if not path.is_file() or path.suffix != ".npz":
            raise LocateAnythingLocalCaptureError(
                f"{LOCATEANYTHING_LOCAL_CHECKPOINT_ENV} must point to a local MLX .npz weights file: {path}"
            )
        return path

    if admission.checkpoint_path:
        admitted = Path(admission.checkpoint_path)
        if admitted.is_file() and admitted.suffix == ".npz":
            return admitted

    raise LocateAnythingLocalCaptureError(
        f"{LOCATEANYTHING_LOCAL_CHECKPOINT_ENV} is unset and the admitted checkpoint is not a local MLX .npz file; "
        "upstream safetensors are admitted for reference capture but are not a local MLX capture."
    )


def evaluate_locateanything_comparison_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
    check_reference_dependencies: bool = True,
    reference_capture_func: Callable[..., Any] = capture_locateanything_upstream_reference,
    local_capture_func: Callable[..., Any] = capture_locateanything_local,
    compare_func: Callable[..., dict[str, Any]] = compare_locateanything_captures,
) -> LocateAnythingGateResult:
    """Evaluate the full upstream-vs-MLX comparison gate.

    Admission is checked first. With a real upstream checkpoint directory,
    optional reference runtime, and local MLX ``.npz`` weights available, this
    captures decoded boxes, decoded points, and selected stable taps on both
    sides and compares them with ``LOCATEANYTHING_FIELD_TOLERANCES``. Missing
    prerequisites return precise blockers instead of synthetic success.
    """

    env = os.environ if environ is None else environ
    admission = evaluate_locateanything_gate(environ=env, min_shard_bytes=min_shard_bytes)
    if admission.blocked:
        return admission

    if not LOCATEANYTHING_REFERENCE_PATH.exists():
        return _block_from_admission(admission, f"LocateAnything reference path is missing: {LOCATEANYTHING_REFERENCE_PATH}")

    if check_reference_dependencies:
        try:
            _import_reference(LOCATEANYTHING_REFERENCE_PATH)
        except LocateAnythingReferenceDependencyError as exc:
            return _block_from_admission(admission, str(exc))

    assert admission.checkpoint_path is not None
    try:
        local_checkpoint = _resolve_local_checkpoint(admission, env)
    except LocateAnythingLocalCaptureError as exc:
        return _block_from_admission(admission, str(exc))

    try:
        reference = reference_capture_func(Path(admission.checkpoint_path), reference_path=LOCATEANYTHING_REFERENCE_PATH)
    except LocateAnythingReferenceDependencyError as exc:
        return _block_from_admission(admission, str(exc))
    except LocateAnythingReferenceCaptureError as exc:
        return _block_from_admission(admission, f"LocateAnything upstream reference capture failed: {exc}")
    except Exception as exc:
        return _block_from_admission(admission, f"LocateAnything upstream reference capture failed: {exc}")

    try:
        local = local_capture_func(local_checkpoint, inputs=_reference_inputs(reference))
    except LocateAnythingLocalCaptureError as exc:
        return _block_from_admission(admission, str(exc))
    except Exception as exc:
        return _block_from_admission(admission, f"LocateAnything local MLX capture failed: {exc}")

    try:
        report = compare_func(reference, local)
    except LocateAnythingParityError as exc:
        return _block_from_admission(admission, f"LocateAnything comparison component unavailable: {exc}")
    except Exception as exc:
        return _block_from_admission(admission, f"LocateAnything comparison failed: {exc}")

    if not report.get("passed", False):
        return _block_from_admission(
            admission,
            "LocateAnything upstream-vs-MLX parity drift for decoded boxes/points or stable taps: "
            + _failure_summary(report),
        )

    return _pass_from_admission(admission, report)
