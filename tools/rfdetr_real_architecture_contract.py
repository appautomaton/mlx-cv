"""RF-DETR Nano real-checkpoint architecture contract.

This module may import PyTorch to inspect a ``.pth`` checkpoint. Keep it under
``tools/`` so the ``mlx_cv`` runtime package remains independent of Torch and
the upstream RF-DETR reference package.
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
    from rfdetr_checkpoint import CheckpointInfo, print_checkpoint_evidence, resolve_rfdetr_nano_checkpoint
except ModuleNotFoundError:  # pragma: no cover - used when imported outside tools/.
    _CHECKPOINT_PATH = Path(__file__).with_name("rfdetr_checkpoint.py")
    _SPEC = importlib.util.spec_from_file_location("rfdetr_checkpoint", _CHECKPOINT_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
    CheckpointInfo = _MODULE.CheckpointInfo
    print_checkpoint_evidence = _MODULE.print_checkpoint_evidence
    resolve_rfdetr_nano_checkpoint = _MODULE.resolve_rfdetr_nano_checkpoint


EXPECTED_NANO_CONTRACT = {
    "out_feature_indexes": (3, 6, 9, 12),
    "local_zero_based_layers": (2, 5, 8, 11),
    "projector_scale": ("P4",),
    "dec_layers": 2,
    "group_detr": 13,
    "bbox_reparam": True,
    "lite_refpoint_refine": True,
    "num_feature_levels": 1,
}

REQUIRED_TENSOR_GROUP_NAMES = (
    "windowed_dinov2",
    "p4_c2f_projector",
    "two_stage_encoder_proposal_heads",
    "decoder_self_attention_norm_refpoint_head",
    "grouped_query_slicing",
    "detection_head",
)


class ArchitectureContractError(RuntimeError):
    """Raised when the real RF-DETR checkpoint cannot satisfy the Nano contract."""


class ArchitectureContractDependencyError(ArchitectureContractError):
    """Raised when optional checkpoint-inspection dependencies are unavailable."""


@dataclass(frozen=True)
class RFDETRNanoArchitectureContract:
    checkpoint_path: str
    checkpoint_md5: str
    architecture: dict[str, Any]
    checkpoint_class_head_shape: tuple[int, ...]
    required_tensor_groups: dict[str, tuple[str, ...]]
    tensor_shapes: dict[str, tuple[int, ...]]
    group_summary: dict[str, Any]
    local_fixture_gaps: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shape(value: Any) -> tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise ArchitectureContractError(f"checkpoint value has no tensor shape: {type(value)!r}")
    return tuple(int(dim) for dim in value.shape)


def _args_to_dict(args: Any) -> dict[str, Any]:
    if hasattr(args, "__dict__"):
        source = vars(args)
    elif isinstance(args, dict):
        source = args
    else:
        raise ArchitectureContractError(f"checkpoint args have unsupported type {type(args)!r}")
    return {str(key): _plain_value(value) for key, value in source.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_plain_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return repr(value)


def _required_arg(args: dict[str, Any], name: str) -> Any:
    if name not in args:
        raise ArchitectureContractError(f"checkpoint args missing {name!r}")
    return args[name]


def _default_local_fixture_config() -> dict[str, Any]:
    try:
        from mlx_cv.parity.fixtures import RFDETR_FIXTURE_CONFIG

        return RFDETR_FIXTURE_CONFIG
    except ModuleNotFoundError:
        fixtures_path = Path(__file__).resolve().parents[1] / "src" / "mlx_cv" / "parity" / "fixtures.py"
        spec = importlib.util.spec_from_file_location("mlx_cv_parity_fixtures_for_rfdetr_contract", fixtures_path)
        if spec is None or spec.loader is None:
            raise ArchitectureContractError(f"could not load RF-DETR fixture config from {fixtures_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.RFDETR_FIXTURE_CONFIG


def _dinov2_layer_keys(layer_index: int) -> tuple[str, ...]:
    base = f"backbone.0.encoder.encoder.encoder.layer.{layer_index}"
    return (
        f"{base}.norm1.weight",
        f"{base}.norm1.bias",
        f"{base}.attention.attention.query.weight",
        f"{base}.attention.attention.query.bias",
        f"{base}.attention.attention.key.weight",
        f"{base}.attention.attention.key.bias",
        f"{base}.attention.attention.value.weight",
        f"{base}.attention.attention.value.bias",
        f"{base}.attention.output.dense.weight",
        f"{base}.attention.output.dense.bias",
        f"{base}.layer_scale1.lambda1",
        f"{base}.norm2.weight",
        f"{base}.norm2.bias",
        f"{base}.mlp.fc1.weight",
        f"{base}.mlp.fc1.bias",
        f"{base}.mlp.fc2.weight",
        f"{base}.mlp.fc2.bias",
        f"{base}.layer_scale2.lambda1",
    )


def _windowed_dinov2_keys(depth: int) -> tuple[str, ...]:
    keys = [
        "backbone.0.encoder.encoder.embeddings.cls_token",
        "backbone.0.encoder.encoder.embeddings.mask_token",
        "backbone.0.encoder.encoder.embeddings.position_embeddings",
        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight",
        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.bias",
    ]
    for layer_index in range(depth):
        keys.extend(_dinov2_layer_keys(layer_index))
    keys.extend(
        [
            "backbone.0.encoder.encoder.layernorm.weight",
            "backbone.0.encoder.encoder.layernorm.bias",
        ]
    )
    return tuple(keys)


def _p4_c2f_projector_keys() -> tuple[str, ...]:
    stage = "backbone.0.projector.stages.0"
    keys = [
        f"{stage}.0.cv1.conv.weight",
        f"{stage}.0.cv1.bn.weight",
        f"{stage}.0.cv1.bn.bias",
        f"{stage}.0.cv2.conv.weight",
        f"{stage}.0.cv2.bn.weight",
        f"{stage}.0.cv2.bn.bias",
    ]
    for block_index in range(3):
        for branch in ("cv1", "cv2"):
            base = f"{stage}.0.m.{block_index}.{branch}"
            keys.extend(
                [
                    f"{base}.conv.weight",
                    f"{base}.bn.weight",
                    f"{base}.bn.bias",
                ]
            )
    keys.extend([f"{stage}.1.weight", f"{stage}.1.bias"])
    return tuple(keys)


def _two_stage_encoder_proposal_head_keys(group_count: int) -> tuple[str, ...]:
    keys: list[str] = []
    for group_index in range(group_count):
        keys.extend(
            [
                f"transformer.enc_output.{group_index}.weight",
                f"transformer.enc_output.{group_index}.bias",
                f"transformer.enc_output_norm.{group_index}.weight",
                f"transformer.enc_output_norm.{group_index}.bias",
                f"transformer.enc_out_class_embed.{group_index}.weight",
                f"transformer.enc_out_class_embed.{group_index}.bias",
            ]
        )
        for layer_index in range(3):
            keys.extend(
                [
                    f"transformer.enc_out_bbox_embed.{group_index}.layers.{layer_index}.weight",
                    f"transformer.enc_out_bbox_embed.{group_index}.layers.{layer_index}.bias",
                ]
            )
    return tuple(keys)


def _decoder_keys(dec_layers: int) -> tuple[str, ...]:
    keys: list[str] = []
    for layer_index in range(dec_layers):
        base = f"transformer.decoder.layers.{layer_index}"
        keys.extend(
            [
                f"{base}.self_attn.in_proj_weight",
                f"{base}.self_attn.in_proj_bias",
                f"{base}.self_attn.out_proj.weight",
                f"{base}.self_attn.out_proj.bias",
                f"{base}.norm1.weight",
                f"{base}.norm1.bias",
                f"{base}.norm2.weight",
                f"{base}.norm2.bias",
                f"{base}.norm3.weight",
                f"{base}.norm3.bias",
                f"{base}.cross_attn.sampling_offsets.weight",
                f"{base}.cross_attn.sampling_offsets.bias",
                f"{base}.cross_attn.attention_weights.weight",
                f"{base}.cross_attn.attention_weights.bias",
                f"{base}.cross_attn.value_proj.weight",
                f"{base}.cross_attn.value_proj.bias",
                f"{base}.cross_attn.output_proj.weight",
                f"{base}.cross_attn.output_proj.bias",
            ]
        )
    keys.extend(
        [
            "transformer.decoder.norm.weight",
            "transformer.decoder.norm.bias",
            "transformer.decoder.ref_point_head.layers.0.weight",
            "transformer.decoder.ref_point_head.layers.0.bias",
            "transformer.decoder.ref_point_head.layers.1.weight",
            "transformer.decoder.ref_point_head.layers.1.bias",
        ]
    )
    return tuple(keys)


def _required_tensor_groups(args: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    depth = int(_required_arg(args, "vit_encoder_num_layers"))
    dec_layers = int(_required_arg(args, "dec_layers"))
    group_detr = int(_required_arg(args, "group_detr"))
    return {
        "windowed_dinov2": _windowed_dinov2_keys(depth),
        "p4_c2f_projector": _p4_c2f_projector_keys(),
        "two_stage_encoder_proposal_heads": _two_stage_encoder_proposal_head_keys(group_detr),
        "decoder_self_attention_norm_refpoint_head": _decoder_keys(dec_layers),
        "grouped_query_slicing": ("query_feat.weight", "refpoint_embed.weight"),
        "detection_head": (
            "class_embed.weight",
            "class_embed.bias",
            "bbox_embed.layers.0.weight",
            "bbox_embed.layers.0.bias",
            "bbox_embed.layers.1.weight",
            "bbox_embed.layers.1.bias",
            "bbox_embed.layers.2.weight",
            "bbox_embed.layers.2.bias",
        ),
    }


def _state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ArchitectureContractError(f"checkpoint root has unsupported type {type(checkpoint)!r}")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, dict):
        raise ArchitectureContractError("checkpoint does not contain a model state dict")
    return state


def _architecture(args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    out_feature_indexes = tuple(int(i) for i in _required_arg(args, "out_feature_indexes"))
    local_layers = tuple(i - 1 for i in out_feature_indexes)
    return {
        "encoder": _required_arg(args, "encoder"),
        "dinov2_num_windows": int(_required_arg(args, "dinov2_num_windows")),
        "dinov2_patch_size": int(_required_arg(args, "dinov2_patch_size")),
        "resolution": int(_required_arg(args, "resolution")),
        "vit_encoder_num_layers": int(_required_arg(args, "vit_encoder_num_layers")),
        "out_feature_indexes": out_feature_indexes,
        "local_zero_based_layers": local_layers,
        "projector_scale": tuple(_required_arg(args, "projector_scale")),
        "dec_layers": int(_required_arg(args, "dec_layers")),
        "group_detr": int(_required_arg(args, "group_detr")),
        "bbox_reparam": bool(_required_arg(args, "bbox_reparam")),
        "lite_refpoint_refine": bool(_required_arg(args, "lite_refpoint_refine")),
        "num_feature_levels": int(_required_arg(args, "num_feature_levels")),
        "two_stage": bool(_required_arg(args, "two_stage")),
        "hidden_dim": int(_required_arg(args, "hidden_dim")),
        "num_queries": int(_required_arg(args, "num_queries")),
        "num_select": int(_required_arg(args, "num_select")),
        "checkpoint_class_head_shape": _shape(state["class_embed.weight"]),
    }


def _group_summary(
    architecture: dict[str, Any],
    state: dict[str, Any],
    groups: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    group_detr = int(architecture["group_detr"])
    num_queries = int(architecture["num_queries"])
    return {
        "state_key_count": len(state),
        "required_tensor_count": sum(len(keys) for keys in groups.values()),
        "windowed_dinov2_layer_indexes": tuple(range(int(architecture["vit_encoder_num_layers"]))),
        "selected_windowed_dinov2_layers": tuple(architecture["local_zero_based_layers"]),
        "projector_stage_indexes": (0,),
        "two_stage_group_indexes": tuple(range(group_detr)),
        "decoder_layer_indexes": tuple(range(int(architecture["dec_layers"]))),
        "base_query_count": num_queries,
        "grouped_query_count": num_queries * group_detr,
        "query_feat_shape": _shape(state["query_feat.weight"]),
        "refpoint_embed_shape": _shape(state["refpoint_embed.weight"]),
    }


def audit_rfdetr_nano_checkpoint(
    checkpoint: CheckpointInfo,
    *,
    fixture_config: dict[str, Any] | None = None,
) -> RFDETRNanoArchitectureContract:
    """Load and audit the verified RF-DETR Nano checkpoint architecture contract."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional torch env.
        raise ArchitectureContractDependencyError(
            "RF-DETR Nano architecture audit requires torch outside src/mlx_cv."
        ) from exc

    raw = torch.load(checkpoint.path, map_location="cpu", weights_only=False)
    args = _args_to_dict(_required_arg(raw, "args"))
    state = _state_dict(raw)
    architecture = _architecture(args, state)
    groups = _required_tensor_groups(args)
    missing = tuple(key for keys in groups.values() for key in keys if key not in state)
    if missing:
        sample = ", ".join(repr(key) for key in missing[:5])
        more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
        raise ArchitectureContractError(f"RF-DETR Nano checkpoint missing required tensors: {sample}{more}")

    shapes = {key: _shape(state[key]) for keys in groups.values() for key in keys}
    contract_without_gaps = RFDETRNanoArchitectureContract(
        checkpoint_path=str(checkpoint.path),
        checkpoint_md5=checkpoint.md5,
        architecture=architecture,
        checkpoint_class_head_shape=_shape(state["class_embed.weight"]),
        required_tensor_groups=groups,
        tensor_shapes=shapes,
        group_summary=_group_summary(architecture, state, groups),
    )
    fixture_source = _default_local_fixture_config() if fixture_config is None else fixture_config
    gaps = local_fixture_gaps(fixture_source, contract_without_gaps)
    contract = RFDETRNanoArchitectureContract(
        checkpoint_path=contract_without_gaps.checkpoint_path,
        checkpoint_md5=contract_without_gaps.checkpoint_md5,
        architecture=contract_without_gaps.architecture,
        checkpoint_class_head_shape=contract_without_gaps.checkpoint_class_head_shape,
        required_tensor_groups=contract_without_gaps.required_tensor_groups,
        tensor_shapes=contract_without_gaps.tensor_shapes,
        group_summary=contract_without_gaps.group_summary,
        local_fixture_gaps=gaps,
    )
    _validate_known_nano_contract(contract)
    return contract


def _validate_known_nano_contract(contract: RFDETRNanoArchitectureContract) -> None:
    arch = contract.architecture
    for key, expected in EXPECTED_NANO_CONTRACT.items():
        actual = arch[key]
        if actual != expected:
            raise ArchitectureContractError(f"Nano contract {key}={actual!r}; expected {expected!r}")
    grouped_query_count = contract.group_summary["grouped_query_count"]
    if contract.tensor_shapes["query_feat.weight"] != (grouped_query_count, arch["hidden_dim"]):
        raise ArchitectureContractError("query_feat.weight does not match grouped Nano query contract")
    if contract.tensor_shapes["refpoint_embed.weight"] != (grouped_query_count, 4):
        raise ArchitectureContractError("refpoint_embed.weight does not match grouped Nano refpoint contract")


def local_fixture_gaps(fixture_config: dict[str, Any], contract: RFDETRNanoArchitectureContract) -> tuple[str, ...]:
    """Return why the existing local RF-DETR fixture cannot close the real checkpoint."""

    arch = contract.architecture
    decoder = fixture_config["decoder"]
    gaps: list[str] = []
    if tuple(fixture_config["out_layers"]) != tuple(arch["local_zero_based_layers"]):
        gaps.append(
            "fixture selects out_layers="
            f"{tuple(fixture_config['out_layers'])}, not real local layers {tuple(arch['local_zero_based_layers'])}"
        )
    if int(fixture_config["backbone"]["depth"]) < int(arch["vit_encoder_num_layers"]):
        gaps.append(
            f"fixture DINOv2 depth={fixture_config['backbone']['depth']}, "
            f"not real depth {arch['vit_encoder_num_layers']}"
        )
    if int(fixture_config["backbone"]["patch_size"]) != int(arch["dinov2_patch_size"]):
        gaps.append(
            f"fixture patch_size={fixture_config['backbone']['patch_size']}, "
            f"not real patch_size {arch['dinov2_patch_size']}"
        )
    if len(fixture_config["projector_scale_factors"]) != int(arch["num_feature_levels"]):
        gaps.append(
            f"fixture feature levels={len(fixture_config['projector_scale_factors'])}, "
            f"not real num_feature_levels {arch['num_feature_levels']}"
        )
    if int(fixture_config["projector_out_channels"]) != int(arch["hidden_dim"]):
        gaps.append(
            f"fixture projector_out_channels={fixture_config['projector_out_channels']}, "
            f"not real hidden_dim {arch['hidden_dim']}"
        )
    if int(decoder["num_layers"]) != int(arch["dec_layers"]):
        gaps.append(f"fixture decoder layers={decoder['num_layers']}, not real dec_layers {arch['dec_layers']}")
    if int(decoder["num_queries"]) != int(contract.group_summary["grouped_query_count"]):
        gaps.append(
            f"fixture queries={decoder['num_queries']}, "
            f"not real grouped queries {contract.group_summary['grouped_query_count']}"
        )
    if int(decoder["num_classes"]) != int(contract.checkpoint_class_head_shape[0]):
        gaps.append(
            f"fixture class head rows={decoder['num_classes']}, "
            f"not checkpoint class head rows {contract.checkpoint_class_head_shape[0]}"
        )
    return tuple(gaps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the real RF-DETR Nano checkpoint architecture contract.")
    parser.add_argument("--cache-root", type=Path, default=None)
    args = parser.parse_args(argv)

    checkpoint = resolve_rfdetr_nano_checkpoint(cache_root=args.cache_root, required=True)
    if checkpoint is None:  # pragma: no cover - required=True raises instead.
        raise RuntimeError("required checkpoint unexpectedly resolved to None")
    print_checkpoint_evidence(checkpoint)
    contract = audit_rfdetr_nano_checkpoint(checkpoint)
    print(json.dumps(contract.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
