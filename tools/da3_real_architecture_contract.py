"""Depth Anything 3 real-checkpoint architecture contract.

This module inspects DA3 config JSON and safetensors metadata without importing
the upstream Torch reference runtime. Keep it under ``tools/`` so the MLX
runtime package stays independent of the DA3 reference dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from da3_checkpoint import DA3CheckpointInfo, print_checkpoint_evidence, resolve_da3_checkpoint
except ModuleNotFoundError:  # pragma: no cover - used when imported outside tools/.
    _CHECKPOINT_PATH = Path(__file__).with_name("da3_checkpoint.py")
    _SPEC = importlib.util.spec_from_file_location("da3_checkpoint", _CHECKPOINT_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
    DA3CheckpointInfo = _MODULE.DA3CheckpointInfo
    print_checkpoint_evidence = _MODULE.print_checkpoint_evidence
    resolve_da3_checkpoint = _MODULE.resolve_da3_checkpoint


EXPECTED_MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "depth-anything/DA3-SMALL": {
        "model_name": "da3-small",
        "dinov2_variant": "vits",
        "embed_dim": 384,
        "num_heads": 6,
        "dualdpt_features": 64,
        "dualdpt_out_channels": (48, 96, 192, 384),
    },
    "depth-anything/DA3-BASE": {
        "model_name": "da3-base",
        "dinov2_variant": "vitb",
        "embed_dim": 768,
        "num_heads": 12,
        "dualdpt_features": 128,
        "dualdpt_out_channels": (96, 192, 384, 768),
    },
}

EXPECTED_OUT_LAYERS = (5, 7, 9, 11)
EXPECTED_START_LAYER = 4
EXPECTED_DEPTH = 12
EXPECTED_PATCH_SIZE = 14
EXPECTED_DUALDPT_OUTPUT_DIM = 2
EXPECTED_DUALDPT_AUX_OUTPUT_DIM = 7
EXPECTED_CAMERA_POSE_DIM = 9
EXPECTED_CAMERA_ENCODER_TRUNK_DEPTH = 4

TENSOR_GROUP_NAMES = (
    "backbone",
    "dualdpt",
    "camera_encoder",
    "camera_decoder",
    "excluded_branches",
)

UNSUPPORTED_BRANCHES = (
    "NestedDepthAnything3Net metric scaling branch (da3/da3_metric)",
    "Gaussian splatting branch (gs_head/gs_adapter)",
    "metric-only and mono-large presets outside the selected Small/Base any-view contract",
)

UNSUPPORTED_CHECKPOINT_PREFIXES = (
    "da3.",
    "da3_metric.",
    "gs_head.",
    "gs_adapter.",
    "metric.",
    "sky_output_conv2.",
    "gaussian_param_head.",
    "all_heads.",
)

CAMERA_POSE_UTILITY_DEPENDENCIES = (
    "depth_anything_3.model.utils.transform.extri_intri_to_pose_encoding",
    "depth_anything_3.model.utils.transform.pose_encoding_to_extri_intri",
    "depth_anything_3.utils.geometry.affine_inverse",
    "depth_anything_3.utils.geometry.as_homogeneous",
    "depth_anything_3.utils.ray_utils.get_extrinsic_from_camray",
)

DEFAULT_INITIALIZED_LOCAL_TENSORS = tuple(
    key
    for level_index in range(1, 4)
    for key in (
        f"head.scratch.output_conv2_aux.{level_index}.2.weight",
        f"head.scratch.output_conv2_aux.{level_index}.2.bias",
    )
)


class DA3ArchitectureContractError(RuntimeError):
    """Raised when a DA3 checkpoint/config cannot satisfy the real contract."""


class DA3ArchitectureContractDependencyError(DA3ArchitectureContractError):
    """Raised when optional metadata-inspection dependencies are unavailable."""


@dataclass(frozen=True)
class DA3RealArchitectureContract:
    model_id: str
    config_path: str
    checkpoint_path: str
    provenance: dict[str, str]
    architecture: dict[str, Any]
    tensor_groups: dict[str, tuple[str, ...]]
    required_tensor_groups: dict[str, tuple[str, ...]]
    tensor_shapes: dict[str, tuple[int, ...]]
    group_summary: dict[str, Any]
    local_monocular_gaps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _normalize_state_key(key: str) -> str:
    return key[len("model.") :] if key.startswith("model.") else key


def _checkpoint_tensor_shapes(checkpoint_path: Path) -> dict[str, tuple[int, ...]]:
    """Return normalized checkpoint tensor shapes without importing Torch."""

    safetensors_error: Exception | None = None
    try:
        from safetensors import safe_open

        with safe_open(str(checkpoint_path), framework="np") as handle:
            return {
                _normalize_state_key(key): tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                for key in handle.keys()
            }
    except ModuleNotFoundError as exc:
        safetensors_error = exc
    except Exception as exc:
        safetensors_error = exc

    try:
        import mlx.core as mx

        state = mx.load(str(checkpoint_path))
        return {_normalize_state_key(key): _shape_tuple(value.shape) for key, value in state.items()}
    except Exception as exc:  # pragma: no cover - depends on optional local deps and invalid files.
        raise DA3ArchitectureContractDependencyError(
            "DA3 architecture audit requires safetensors or mlx to inspect checkpoint metadata "
            "outside src/mlx_cv."
        ) from (safetensors_error or exc)


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise DA3ArchitectureContractError(f"DA3 config file is missing: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise DA3ArchitectureContractError(f"DA3 config is not valid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise DA3ArchitectureContractError("DA3 config root must be a JSON object")
    return config


def _required_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key)
    if not isinstance(child, dict):
        raise DA3ArchitectureContractError(f"DA3 config missing object {key!r}")
    return child


def _object_name(value: dict[str, Any]) -> tuple[str, str]:
    obj = value.get("__object__")
    if not isinstance(obj, dict):
        raise DA3ArchitectureContractError("DA3 config object is missing __object__ metadata")
    path = obj.get("path")
    name = obj.get("name")
    if not isinstance(path, str) or not isinstance(name, str):
        raise DA3ArchitectureContractError("DA3 config __object__ metadata must include path and name")
    return path, name


def _required_int(value: dict[str, Any], key: str) -> int:
    if key not in value:
        raise DA3ArchitectureContractError(f"DA3 config missing {key!r}")
    return int(value[key])


def _required_bool(value: dict[str, Any], key: str) -> bool:
    if key not in value:
        raise DA3ArchitectureContractError(f"DA3 config missing {key!r}")
    return bool(value[key])


def _required_int_tuple(value: dict[str, Any], key: str) -> tuple[int, ...]:
    if key not in value:
        raise DA3ArchitectureContractError(f"DA3 config missing {key!r}")
    return tuple(int(item) for item in value[key])


def _validate_provenance(checkpoint: DA3CheckpointInfo) -> dict[str, str]:
    fields = {
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "config_sha256": checkpoint.config_sha256,
        "checkpoint_url": checkpoint.checkpoint_url,
        "config_url": checkpoint.config_url,
        "revision": checkpoint.revision,
        "license_note": checkpoint.license_note,
        "source": checkpoint.source,
    }
    missing = tuple(name for name, value in fields.items() if not value)
    if missing:
        raise DA3ArchitectureContractError(
            "DA3 checkpoint provenance is incomplete: " + ", ".join(missing)
        )
    return dict(fields)


def _architecture_from_config(model_id: str, raw_config: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED_MODEL_CONTRACTS.get(model_id)
    if expected is None:
        raise DA3ArchitectureContractError(f"unsupported DA3 model id for architecture contract: {model_id}")

    if raw_config.get("model_name") != expected["model_name"]:
        raise DA3ArchitectureContractError(
            f"DA3 config model_name={raw_config.get('model_name')!r}; expected {expected['model_name']!r}"
        )

    config = _required_mapping(raw_config, "config")
    root_path, root_name = _object_name(config)
    if (root_path, root_name) != ("depth_anything_3.model.da3", "DepthAnything3Net"):
        raise DA3ArchitectureContractError(
            f"DA3 root object {(root_path, root_name)!r}; expected DepthAnything3Net"
        )

    net = _required_mapping(config, "net")
    head = _required_mapping(config, "head")
    cam_enc = _required_mapping(config, "cam_enc")
    cam_dec = _required_mapping(config, "cam_dec")

    net_path, net_name = _object_name(net)
    head_path, head_name = _object_name(head)
    cam_enc_path, cam_enc_name = _object_name(cam_enc)
    cam_dec_path, cam_dec_name = _object_name(cam_dec)
    expected_objects = {
        "net": ("depth_anything_3.model.dinov2.dinov2", "DinoV2", net_path, net_name),
        "head": ("depth_anything_3.model.dualdpt", "DualDPT", head_path, head_name),
        "cam_enc": ("depth_anything_3.model.cam_enc", "CameraEnc", cam_enc_path, cam_enc_name),
        "cam_dec": ("depth_anything_3.model.cam_dec", "CameraDec", cam_dec_path, cam_dec_name),
    }
    for key, (expected_path, expected_name, actual_path, actual_name) in expected_objects.items():
        if (actual_path, actual_name) != (expected_path, expected_name):
            raise DA3ArchitectureContractError(
                f"DA3 config {key} object {(actual_path, actual_name)!r}; "
                f"expected {(expected_path, expected_name)!r}"
            )

    variant = str(net.get("name"))
    embed_dim = int(expected["embed_dim"])
    head_dim_in = _required_int(head, "dim_in")
    cam_enc_dim_out = _required_int(cam_enc, "dim_out")
    cam_dec_dim_in = _required_int(cam_dec, "dim_in")
    if variant != expected["dinov2_variant"]:
        raise DA3ArchitectureContractError(
            f"DA3 DINOv2 variant={variant!r}; expected {expected['dinov2_variant']!r}"
        )
    if _required_int_tuple(net, "out_layers") != EXPECTED_OUT_LAYERS:
        raise DA3ArchitectureContractError("DA3 config must select DINOv2 out_layers [5, 7, 9, 11]")
    for start_name in ("alt_start", "qknorm_start", "rope_start"):
        if _required_int(net, start_name) != EXPECTED_START_LAYER:
            raise DA3ArchitectureContractError(f"DA3 config {start_name} must equal 4")
    if _required_bool(net, "cat_token") is not True:
        raise DA3ArchitectureContractError("DA3 config cat_token must be True")
    if head_dim_in != embed_dim * 2:
        raise DA3ArchitectureContractError(
            f"DualDPT dim_in={head_dim_in}; expected doubled embed_dim*2={embed_dim * 2}"
        )
    if cam_enc_dim_out != embed_dim:
        raise DA3ArchitectureContractError(
            f"CameraEnc dim_out={cam_enc_dim_out}; expected embed_dim={embed_dim}"
        )
    if cam_dec_dim_in != embed_dim * 2:
        raise DA3ArchitectureContractError(
            f"CameraDec dim_in={cam_dec_dim_in}; expected doubled embed_dim*2={embed_dim * 2}"
        )

    dualdpt_out_channels = _required_int_tuple(head, "out_channels")
    if dualdpt_out_channels != tuple(expected["dualdpt_out_channels"]):
        raise DA3ArchitectureContractError(
            f"DualDPT out_channels={dualdpt_out_channels}; expected {expected['dualdpt_out_channels']}"
        )
    if _required_int(head, "features") != int(expected["dualdpt_features"]):
        raise DA3ArchitectureContractError(
            f"DualDPT features={head['features']}; expected {expected['dualdpt_features']}"
        )
    if _required_int(head, "output_dim") != EXPECTED_DUALDPT_OUTPUT_DIM:
        raise DA3ArchitectureContractError("DualDPT output_dim must be 2")

    return {
        "model_id": model_id,
        "model_name": raw_config["model_name"],
        "root_module": root_name,
        "dinov2": {
            "variant": variant,
            "embed_dim": embed_dim,
            "num_heads": int(expected["num_heads"]),
            "depth": EXPECTED_DEPTH,
            "patch_size": EXPECTED_PATCH_SIZE,
            "out_layers": EXPECTED_OUT_LAYERS,
            "alt_start": EXPECTED_START_LAYER,
            "qknorm_start": EXPECTED_START_LAYER,
            "rope_start": EXPECTED_START_LAYER,
            "cat_token": True,
            "head_input_dim": embed_dim * 2,
            "qknorm_layers": tuple(range(EXPECTED_START_LAYER, EXPECTED_DEPTH)),
            "rope_layers": tuple(range(EXPECTED_START_LAYER, EXPECTED_DEPTH)),
            "global_attention_layers": tuple(
                layer for layer in range(EXPECTED_START_LAYER, EXPECTED_DEPTH) if layer % 2 == 1
            ),
            "local_attention_layers": tuple(
                layer for layer in range(EXPECTED_DEPTH) if layer < EXPECTED_START_LAYER or layer % 2 == 0
            ),
        },
        "dualdpt": {
            "dim_in": head_dim_in,
            "dim_in_rule": "embed_dim*2",
            "output_dim": EXPECTED_DUALDPT_OUTPUT_DIM,
            "aux_output_dim": EXPECTED_DUALDPT_AUX_OUTPUT_DIM,
            "features": int(expected["dualdpt_features"]),
            "out_channels": dualdpt_out_channels,
            "head_names": ("depth", "ray"),
            "intermediate_layer_idx": (0, 1, 2, 3),
            "aux_pyramid_levels": 4,
            "default_initialized_aux_layernorm_levels": (1, 2, 3),
            "default_initialized_aux_layernorm_shape": (32,),
        },
        "camera_encoder": {
            "dim_in": EXPECTED_CAMERA_POSE_DIM,
            "dim_out": cam_enc_dim_out,
            "target_dim": EXPECTED_CAMERA_POSE_DIM,
            "trunk_depth": EXPECTED_CAMERA_ENCODER_TRUNK_DEPTH,
            "pose_branch_hidden_dim": cam_enc_dim_out // 2,
        },
        "camera_decoder": {
            "dim_in": cam_dec_dim_in,
            "translation_dim": 3,
            "quaternion_dim": 4,
            "fov_dim": 2,
            "pose_encoding_dim": EXPECTED_CAMERA_POSE_DIM,
        },
        "camera_pose_utility_dependencies": CAMERA_POSE_UTILITY_DEPENDENCIES,
        "unsupported_branches": UNSUPPORTED_BRANCHES,
    }


def _backbone_required_keys(embed_dim: int) -> tuple[str, ...]:
    keys = [
        "backbone.pretrained.cls_token",
        "backbone.pretrained.camera_token",
        "backbone.pretrained.pos_embed",
        "backbone.pretrained.patch_embed.proj.weight",
        "backbone.pretrained.patch_embed.proj.bias",
        "backbone.pretrained.norm.weight",
        "backbone.pretrained.norm.bias",
    ]
    for layer_index in range(EXPECTED_DEPTH):
        base = f"backbone.pretrained.blocks.{layer_index}"
        keys.extend(
            [
                f"{base}.norm1.weight",
                f"{base}.norm1.bias",
                f"{base}.attn.qkv.weight",
                f"{base}.attn.qkv.bias",
                f"{base}.attn.proj.weight",
                f"{base}.attn.proj.bias",
                f"{base}.ls1.gamma",
                f"{base}.norm2.weight",
                f"{base}.norm2.bias",
                f"{base}.mlp.fc1.weight",
                f"{base}.mlp.fc1.bias",
                f"{base}.mlp.fc2.weight",
                f"{base}.mlp.fc2.bias",
                f"{base}.ls2.gamma",
            ]
        )
        if layer_index >= EXPECTED_START_LAYER:
            keys.extend(
                [
                    f"{base}.attn.q_norm.weight",
                    f"{base}.attn.q_norm.bias",
                    f"{base}.attn.k_norm.weight",
                    f"{base}.attn.k_norm.bias",
                ]
            )
    return tuple(keys)


def _dualdpt_required_keys() -> tuple[str, ...]:
    keys = [
        "head.norm.weight",
        "head.norm.bias",
        "head.scratch.output_conv1.weight",
        "head.scratch.output_conv1.bias",
        "head.scratch.output_conv2.0.weight",
        "head.scratch.output_conv2.0.bias",
        "head.scratch.output_conv2.2.weight",
        "head.scratch.output_conv2.2.bias",
    ]
    for stage_index in range(4):
        keys.extend(
            [
                f"head.projects.{stage_index}.weight",
                f"head.projects.{stage_index}.bias",
                f"head.scratch.layer{stage_index + 1}_rn.weight",
            ]
        )
    for resize_index in (0, 1, 3):
        keys.extend([f"head.resize_layers.{resize_index}.weight", f"head.resize_layers.{resize_index}.bias"])
    for refine_index in range(1, 5):
        for suffix in ("", "_aux"):
            base = f"head.scratch.refinenet{refine_index}{suffix}"
            keys.extend([f"{base}.out_conv.weight", f"{base}.out_conv.bias"])
            if refine_index != 4:
                keys.extend(
                    [
                        f"{base}.resConfUnit1.conv1.weight",
                        f"{base}.resConfUnit1.conv1.bias",
                        f"{base}.resConfUnit1.conv2.weight",
                        f"{base}.resConfUnit1.conv2.bias",
                    ]
                )
            keys.extend(
                [
                    f"{base}.resConfUnit2.conv1.weight",
                    f"{base}.resConfUnit2.conv1.bias",
                    f"{base}.resConfUnit2.conv2.weight",
                    f"{base}.resConfUnit2.conv2.bias",
                ]
            )
    for level_index in range(4):
        for conv_index in range(5):
            keys.extend(
                [
                    f"head.scratch.output_conv1_aux.{level_index}.{conv_index}.weight",
                    f"head.scratch.output_conv1_aux.{level_index}.{conv_index}.bias",
                ]
            )
        keys.extend(
            [
                f"head.scratch.output_conv2_aux.{level_index}.0.weight",
                f"head.scratch.output_conv2_aux.{level_index}.0.bias",
                f"head.scratch.output_conv2_aux.{level_index}.5.weight",
                f"head.scratch.output_conv2_aux.{level_index}.5.bias",
            ]
        )
        if level_index == 0:
            keys.extend(
                [
                    f"head.scratch.output_conv2_aux.{level_index}.2.weight",
                    f"head.scratch.output_conv2_aux.{level_index}.2.bias",
                ]
            )
    return tuple(keys)


def _camera_encoder_required_keys() -> tuple[str, ...]:
    keys = [
        "cam_enc.pose_branch.fc1.weight",
        "cam_enc.pose_branch.fc1.bias",
        "cam_enc.pose_branch.fc2.weight",
        "cam_enc.pose_branch.fc2.bias",
        "cam_enc.token_norm.weight",
        "cam_enc.token_norm.bias",
        "cam_enc.trunk_norm.weight",
        "cam_enc.trunk_norm.bias",
    ]
    for layer_index in range(EXPECTED_CAMERA_ENCODER_TRUNK_DEPTH):
        base = f"cam_enc.trunk.{layer_index}"
        keys.extend(
            [
                f"{base}.norm1.weight",
                f"{base}.norm1.bias",
                f"{base}.attn.qkv.weight",
                f"{base}.attn.qkv.bias",
                f"{base}.attn.proj.weight",
                f"{base}.attn.proj.bias",
                f"{base}.ls1.gamma",
                f"{base}.norm2.weight",
                f"{base}.norm2.bias",
                f"{base}.mlp.fc1.weight",
                f"{base}.mlp.fc1.bias",
                f"{base}.mlp.fc2.weight",
                f"{base}.mlp.fc2.bias",
                f"{base}.ls2.gamma",
            ]
        )
    return tuple(keys)


def _camera_decoder_required_keys() -> tuple[str, ...]:
    return (
        "cam_dec.backbone.0.weight",
        "cam_dec.backbone.0.bias",
        "cam_dec.backbone.2.weight",
        "cam_dec.backbone.2.bias",
        "cam_dec.fc_t.weight",
        "cam_dec.fc_t.bias",
        "cam_dec.fc_qvec.weight",
        "cam_dec.fc_qvec.bias",
        "cam_dec.fc_fov.0.weight",
        "cam_dec.fc_fov.0.bias",
    )


def _required_tensor_groups(architecture: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    embed_dim = int(architecture["dinov2"]["embed_dim"])
    return {
        "backbone": _backbone_required_keys(embed_dim),
        "dualdpt": _dualdpt_required_keys(),
        "camera_encoder": _camera_encoder_required_keys(),
        "camera_decoder": _camera_decoder_required_keys(),
        "excluded_branches": (),
    }


def _group_checkpoint_keys(shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {name: [] for name in TENSOR_GROUP_NAMES}
    for key in sorted(shapes):
        if key.startswith("backbone."):
            groups["backbone"].append(key)
        elif key.startswith("head."):
            groups["dualdpt"].append(key)
        elif key.startswith("cam_enc."):
            groups["camera_encoder"].append(key)
        elif key.startswith("cam_dec."):
            groups["camera_decoder"].append(key)
        elif key.startswith(UNSUPPORTED_CHECKPOINT_PREFIXES):
            groups["excluded_branches"].append(key)
        else:
            groups["excluded_branches"].append(key)
    return {name: tuple(keys) for name, keys in groups.items()}


def _require_shape(
    shapes: dict[str, tuple[int, ...]],
    key: str,
    expected: tuple[int, ...],
) -> None:
    actual = shapes.get(key)
    if actual != expected:
        raise DA3ArchitectureContractError(f"checkpoint tensor {key!r} has shape {actual}; expected {expected}")


def _validate_tensor_contract(
    architecture: dict[str, Any],
    shapes: dict[str, tuple[int, ...]],
    required_groups: dict[str, tuple[str, ...]],
    tensor_groups: dict[str, tuple[str, ...]],
) -> None:
    missing = tuple(key for keys in required_groups.values() for key in keys if key not in shapes)
    if missing:
        sample = ", ".join(repr(key) for key in missing[:5])
        more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise DA3ArchitectureContractError(f"DA3 checkpoint missing required tensors: {sample}{more}")
    if tensor_groups["excluded_branches"]:
        sample = ", ".join(repr(key) for key in tensor_groups["excluded_branches"][:5])
        raise DA3ArchitectureContractError(f"DA3 checkpoint contains unsupported branch tensors: {sample}")

    embed_dim = int(architecture["dinov2"]["embed_dim"])
    num_heads = int(architecture["dinov2"]["num_heads"])
    head_dim = embed_dim // num_heads
    head_input_dim = int(architecture["dinov2"]["head_input_dim"])
    features = int(architecture["dualdpt"]["features"])
    out_channels = tuple(int(item) for item in architecture["dualdpt"]["out_channels"])
    half_features = features // 2

    _require_shape(shapes, "backbone.pretrained.cls_token", (1, 1, embed_dim))
    _require_shape(shapes, "backbone.pretrained.camera_token", (1, 2, embed_dim))
    _require_shape(shapes, "backbone.pretrained.patch_embed.proj.weight", (embed_dim, 3, 14, 14))
    _require_shape(shapes, "backbone.pretrained.patch_embed.proj.bias", (embed_dim,))
    _require_shape(shapes, "backbone.pretrained.norm.weight", (embed_dim,))
    _require_shape(shapes, "backbone.pretrained.norm.bias", (embed_dim,))
    for layer_index in range(EXPECTED_DEPTH):
        base = f"backbone.pretrained.blocks.{layer_index}"
        _require_shape(shapes, f"{base}.attn.qkv.weight", (embed_dim * 3, embed_dim))
        _require_shape(shapes, f"{base}.attn.qkv.bias", (embed_dim * 3,))
        _require_shape(shapes, f"{base}.attn.proj.weight", (embed_dim, embed_dim))
        _require_shape(shapes, f"{base}.mlp.fc1.weight", (embed_dim * 4, embed_dim))
        _require_shape(shapes, f"{base}.mlp.fc2.weight", (embed_dim, embed_dim * 4))
        if layer_index < EXPECTED_START_LAYER:
            for norm_name in ("q_norm", "k_norm"):
                early_key = f"{base}.attn.{norm_name}.weight"
                if early_key in shapes:
                    raise DA3ArchitectureContractError(
                        f"q/k norm tensor {early_key!r} appears before qknorm_start=4"
                    )
        else:
            _require_shape(shapes, f"{base}.attn.q_norm.weight", (head_dim,))
            _require_shape(shapes, f"{base}.attn.k_norm.weight", (head_dim,))

    _require_shape(shapes, "head.norm.weight", (head_input_dim,))
    _require_shape(shapes, "head.norm.bias", (head_input_dim,))
    for stage_index, channels in enumerate(out_channels):
        _require_shape(shapes, f"head.projects.{stage_index}.weight", (channels, head_input_dim, 1, 1))
        _require_shape(shapes, f"head.projects.{stage_index}.bias", (channels,))
        _require_shape(shapes, f"head.scratch.layer{stage_index + 1}_rn.weight", (features, channels, 3, 3))
    _require_shape(shapes, "head.scratch.output_conv1.weight", (half_features, features, 3, 3))
    _require_shape(shapes, "head.scratch.output_conv2.0.weight", (32, half_features, 3, 3))
    _require_shape(shapes, "head.scratch.output_conv2.2.weight", (EXPECTED_DUALDPT_OUTPUT_DIM, 32, 1, 1))
    _require_shape(shapes, "head.scratch.output_conv2_aux.0.2.weight", (32,))
    _require_shape(shapes, "head.scratch.output_conv2_aux.0.2.bias", (32,))
    _require_shape(shapes, "head.scratch.output_conv2_aux.3.5.weight", (EXPECTED_DUALDPT_AUX_OUTPUT_DIM, 32, 1, 1))

    cam_dim = int(architecture["camera_encoder"]["dim_out"])
    _require_shape(shapes, "cam_enc.pose_branch.fc1.weight", (cam_dim // 2, EXPECTED_CAMERA_POSE_DIM))
    _require_shape(shapes, "cam_enc.pose_branch.fc2.weight", (cam_dim, cam_dim // 2))
    _require_shape(shapes, "cam_enc.token_norm.weight", (cam_dim,))
    _require_shape(shapes, "cam_enc.trunk_norm.weight", (cam_dim,))
    for layer_index in range(EXPECTED_CAMERA_ENCODER_TRUNK_DEPTH):
        base = f"cam_enc.trunk.{layer_index}"
        _require_shape(shapes, f"{base}.attn.qkv.weight", (cam_dim * 3, cam_dim))
        _require_shape(shapes, f"{base}.mlp.fc1.weight", (cam_dim * 4, cam_dim))
        _require_shape(shapes, f"{base}.mlp.fc2.weight", (cam_dim, cam_dim * 4))

    cam_dec_dim = int(architecture["camera_decoder"]["dim_in"])
    _require_shape(shapes, "cam_dec.backbone.0.weight", (cam_dec_dim, cam_dec_dim))
    _require_shape(shapes, "cam_dec.backbone.2.weight", (cam_dec_dim, cam_dec_dim))
    _require_shape(shapes, "cam_dec.fc_t.weight", (3, cam_dec_dim))
    _require_shape(shapes, "cam_dec.fc_qvec.weight", (4, cam_dec_dim))
    _require_shape(shapes, "cam_dec.fc_fov.0.weight", (2, cam_dec_dim))


def _layer_indexes_with_prefix(shapes: dict[str, tuple[int, ...]], stem: str) -> tuple[int, ...]:
    indexes = set()
    prefix = stem + "."
    for key in shapes:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        index_text = rest.split(".", 1)[0]
        if index_text.isdigit():
            indexes.add(int(index_text))
    return tuple(sorted(indexes))


def _group_summary(
    architecture: dict[str, Any],
    shapes: dict[str, tuple[int, ...]],
    groups: dict[str, tuple[str, ...]],
    required_groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    return {
        "state_key_count": len(shapes),
        "group_counts": {name: len(keys) for name, keys in groups.items()},
        "required_tensor_count": sum(len(keys) for keys in required_groups.values()),
        "local_default_tensor_count": len(DEFAULT_INITIALIZED_LOCAL_TENSORS),
        "local_expected_parameter_count": len(shapes) + len(DEFAULT_INITIALIZED_LOCAL_TENSORS),
        "default_initialized_local_tensors": DEFAULT_INITIALIZED_LOCAL_TENSORS,
        "backbone_block_indexes": _layer_indexes_with_prefix(shapes, "backbone.pretrained.blocks"),
        "selected_out_layers": architecture["dinov2"]["out_layers"],
        "qknorm_layer_indexes": tuple(
            layer
            for layer in range(EXPECTED_DEPTH)
            if f"backbone.pretrained.blocks.{layer}.attn.q_norm.weight" in shapes
        ),
        "rope_layer_indexes": architecture["dinov2"]["rope_layers"],
        "dualdpt_project_channels": tuple(
            shapes[f"head.projects.{stage}.weight"][0] for stage in range(4)
        ),
        "camera_token_shape": shapes["backbone.pretrained.camera_token"],
        "dualdpt_head_input_shape": shapes["head.norm.weight"],
        "camera_encoder_trunk_indexes": _layer_indexes_with_prefix(shapes, "cam_enc.trunk"),
        "camera_decoder_pose_head_shapes": {
            "translation": shapes["cam_dec.fc_t.weight"],
            "quaternion": shapes["cam_dec.fc_qvec.weight"],
            "fov": shapes["cam_dec.fc_fov.0.weight"],
        },
        "unsupported_branch_key_count": len(groups["excluded_branches"]),
    }


def _default_local_fixture_config() -> dict[str, Any]:
    try:
        from mlx_cv.parity.fixtures import DA3_MONOCULAR_FIXTURE_CONFIG

        return DA3_MONOCULAR_FIXTURE_CONFIG
    except ModuleNotFoundError:
        fixtures_path = Path(__file__).resolve().parents[1] / "src" / "mlx_cv" / "parity" / "fixtures.py"
        spec = importlib.util.spec_from_file_location("mlx_cv_parity_fixtures_for_da3_contract", fixtures_path)
        if spec is None or spec.loader is None:
            raise DA3ArchitectureContractError(f"could not load DA3 fixture config from {fixtures_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.DA3_MONOCULAR_FIXTURE_CONFIG


def local_monocular_gaps(
    fixture_config: dict[str, Any],
    contract: DA3RealArchitectureContract,
) -> tuple[str, ...]:
    """Return why the existing local monocular DA3 path cannot load this checkpoint."""

    arch = contract.architecture
    dinov2 = arch["dinov2"]
    dualdpt = arch["dualdpt"]
    cam_enc = arch["camera_encoder"]
    cam_dec = arch["camera_decoder"]
    fixture_dino = fixture_config["dinov2"]
    fixture_dpt = fixture_config["dpt"]
    gaps = [
        "local DepthAnythingV3Monocular accepts a single NCHW image path, not DA3's B,N,3,H,W any-view input",
        "local DINOv2 fixture has no any-view DINOv2 behavior: no alt_start camera-token insertion, "
        "global/local view attention, reference-view selection, qknorm_start, or rope_start",
        "local head is a monocular DPTHead, not DualDPT with depth/ray outputs and independent auxiliary fusion",
        "local model has no camera tokens wired through CameraEnc or CameraDec",
        "local model has no camera geometry modules for extrinsics/intrinsics propagation",
        "local model has no pose conversion utilities for pose_encoding_to_extri_intri or extri_intri_to_pose_encoding",
        "local weight loader consumes only backbone.* and head.* prefixes, so cam_enc.* and cam_dec.* tensors are unmapped",
    ]
    if int(fixture_dino["embed_dim"]) * 2 != int(dualdpt["dim_in"]):
        gaps.append(
            f"local fixture DINOv2 embed_dim*2={int(fixture_dino['embed_dim']) * 2}, "
            f"not real DualDPT dim_in={dualdpt['dim_in']}"
        )
    if tuple(fixture_dino["intermediate_layers"]) != tuple(dinov2["out_layers"]):
        gaps.append(
            f"local fixture out_layers={tuple(fixture_dino['intermediate_layers'])}, "
            f"not real DA3 out_layers={tuple(dinov2['out_layers'])}"
        )
    if int(fixture_dino["depth"]) < int(dinov2["depth"]):
        gaps.append(f"local fixture DINOv2 depth={fixture_dino['depth']}, not real depth {dinov2['depth']}")
    if int(fixture_dpt["dim_in"]) != int(dualdpt["dim_in"]):
        gaps.append(f"local DPT dim_in={fixture_dpt['dim_in']}, not DualDPT dim_in={dualdpt['dim_in']}")
    if int(fixture_dpt["features"]) != int(dualdpt["features"]):
        gaps.append(f"local DPT features={fixture_dpt['features']}, not DualDPT features={dualdpt['features']}")
    if tuple(fixture_dpt["out_channels"]) != tuple(dualdpt["out_channels"]):
        gaps.append(
            f"local DPT out_channels={tuple(fixture_dpt['out_channels'])}, "
            f"not DualDPT out_channels={tuple(dualdpt['out_channels'])}"
        )
    gaps.append(
        f"real camera encoder/decoder dimensions are required: CameraEnc dim_out={cam_enc['dim_out']} "
        f"and CameraDec dim_in={cam_dec['dim_in']}"
    )
    return tuple(gaps)


def audit_da3_real_architecture_contract(
    checkpoint: DA3CheckpointInfo,
    *,
    fixture_config: dict[str, Any] | None = None,
) -> DA3RealArchitectureContract:
    """Audit a resolved DA3 Small/Base config and checkpoint as an executable contract."""

    provenance = _validate_provenance(checkpoint)
    raw_config = _load_config(checkpoint.config_path)
    architecture = _architecture_from_config(checkpoint.model_id, raw_config)
    shapes = _checkpoint_tensor_shapes(checkpoint.checkpoint_path)
    groups = _group_checkpoint_keys(shapes)
    required_groups = _required_tensor_groups(architecture)
    _validate_tensor_contract(architecture, shapes, required_groups, groups)
    contract_without_gaps = DA3RealArchitectureContract(
        model_id=checkpoint.model_id,
        config_path=str(checkpoint.config_path),
        checkpoint_path=str(checkpoint.checkpoint_path),
        provenance=provenance,
        architecture=architecture,
        tensor_groups=groups,
        required_tensor_groups=required_groups,
        tensor_shapes=shapes,
        group_summary=_group_summary(architecture, shapes, groups, required_groups),
        local_monocular_gaps=(),
    )
    fixture_source = _default_local_fixture_config() if fixture_config is None else fixture_config
    gaps = local_monocular_gaps(fixture_source, contract_without_gaps)
    return DA3RealArchitectureContract(
        model_id=contract_without_gaps.model_id,
        config_path=contract_without_gaps.config_path,
        checkpoint_path=contract_without_gaps.checkpoint_path,
        provenance=contract_without_gaps.provenance,
        architecture=contract_without_gaps.architecture,
        tensor_groups=contract_without_gaps.tensor_groups,
        required_tensor_groups=contract_without_gaps.required_tensor_groups,
        tensor_shapes=contract_without_gaps.tensor_shapes,
        group_summary=contract_without_gaps.group_summary,
        local_monocular_gaps=gaps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the real DA3 Small/Base architecture contract.")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args(argv)

    checkpoint = resolve_da3_checkpoint(cache_root=args.cache_root, required=True)
    if checkpoint is None:  # pragma: no cover - required=True raises instead.
        raise RuntimeError("required DA3 checkpoint unexpectedly resolved to None")
    print_checkpoint_evidence(checkpoint)
    contract = audit_da3_real_architecture_contract(checkpoint)
    print(json.dumps(contract.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
