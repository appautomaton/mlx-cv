"""RF-DETR detection checkpoint conversion."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mlx.core as mx

from .modeling import RFDETRModel

__all__ = [
    "RFDETR_INFERENCE_ONLY_EXCLUSIONS",
    "convert_rfdetr_state_dict",
    "load_rfdetr_weights",
    "remap_rfdetr_key",
]


_LOCAL_PREFIXES = (
    "feature_extractor.",
    "decoder.",
    "head.",
)
_PROJECTOR_PREFIXES = (
    "backbone.0.projector.",
    "backbone.projector.",
    "backbone.body.projector.",
)
_METADATA_KEYS = {"args", "config", "model_config", "__args_json__", "__config_json__", "__metadata_json__"}
_SEGMENTATION_KEY_PARTS = (
    "segmentation_head",
    "mask_head",
    "pred_masks",
    "query_features",
)
_QUERY_PARAM_KEYS = {"decoder.query_embed", "decoder.reference_embed"}
_HF_DINOV2_PREFIX = "backbone.0.encoder.encoder."
_HF_DINOV2_QKV_RE = re.compile(
    r"^backbone\.0\.encoder\.encoder\.encoder\.layer\.(\d+)"
    r"\.attention\.attention\.(query|key|value)\.(weight|bias)$"
)
RFDETR_INFERENCE_ONLY_EXCLUSIONS = {
    "backbone.0.encoder.encoder.embeddings.mask_token": (
        "training-only DINOv2 mask token; RF-DETR inference never constructs masked image tokens"
    ),
}
_HF_DINOV2_INFERENCE_EXCLUSIONS = RFDETR_INFERENCE_ONLY_EXCLUSIONS


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _json_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.dtype.kind in ("U", "S"):
            value = "".join(str(x) for x in value.reshape(-1))
        else:
            return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        return value
    return None


def _metadata_args(state: dict[str, Any]) -> dict[str, Any] | None:
    for key in _METADATA_KEYS:
        if key not in state:
            continue
        metadata = _json_like(state[key])
        if not isinstance(metadata, dict):
            continue
        args = metadata.get("args", metadata)
        if isinstance(args, dict):
            return args
    return None


def _metadata_int(args: dict[str, Any] | None, key: str) -> int | None:
    if args is None or key not in args:
        return None
    try:
        return int(args[key])
    except (TypeError, ValueError):
        return None


def _first_query_rows(state: dict[str, Any]) -> int | None:
    for key in ("query_feat.weight", "refpoint_embed.weight"):
        for candidate in (key, f"model.{key}", f"detr.{key}"):
            if candidate in state:
                return int(_to_numpy(state[candidate]).shape[0])
    return None


def _slice_query_param_per_group(
    tensor: np.ndarray,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> np.ndarray:
    if ckpt_num_queries <= 0 or ckpt_group_detr <= 0 or target_num_queries <= 0 or target_group_detr <= 0:
        raise ValueError("query slicing dimensions must be positive")
    expected_total = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_total:
        return tensor[: target_num_queries * target_group_detr]
    keep_groups = min(target_group_detr, ckpt_group_detr)
    keep_per_group = min(target_num_queries, ckpt_num_queries)
    pieces = [
        tensor[group * ckpt_num_queries : group * ckpt_num_queries + keep_per_group]
        for group in range(keep_groups)
    ]
    return np.concatenate(pieces, axis=0)


def _contains_segmentation_flag(obj: Any) -> bool:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) == "segmentation_head" and bool(value):
                return True
            if _contains_segmentation_flag(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_contains_segmentation_flag(v) for v in obj)
    return False


def _reject_unsupported_variant(state: dict[str, Any]) -> None:
    for key, value in state.items():
        lowered = key.lower()
        if any(part in lowered for part in _SEGMENTATION_KEY_PARTS):
            raise ValueError(
                "RF-DETR segmentation checkpoints are not supported by the detection loader; "
                f"found segmentation key {key!r}"
            )
        if key in _METADATA_KEYS:
            metadata = _json_like(value)
            if _contains_segmentation_flag(metadata):
                raise ValueError(
                    "RF-DETR segmentation checkpoints are not supported by the detection loader; "
                    f"metadata key {key!r} declares segmentation_head=True"
                )


def _strip_reference_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "detr."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _state_value_for_stripped_key(state: dict[str, Any], stripped_key: str) -> Any:
    for prefix in ("", "module.", "model.", "detr."):
        candidate = f"{prefix}{stripped_key}"
        if candidate in state:
            return state[candidate]
    raise KeyError(stripped_key)


def _is_inference_only_exclusion(key: str) -> bool:
    return _strip_reference_prefix(key) in _HF_DINOV2_INFERENCE_EXCLUSIONS


def _hf_dinov2_qkv_component(key: str) -> tuple[int, str, str] | None:
    match = _HF_DINOV2_QKV_RE.match(_strip_reference_prefix(key))
    if match is None:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def _validate_hf_dinov2_qkv_groups(state: dict[str, Any]) -> None:
    groups: dict[tuple[int, str], set[str]] = {}
    for key in state:
        component = _hf_dinov2_qkv_component(key)
        if component is None:
            continue
        layer_index, part, leaf = component
        groups.setdefault((layer_index, leaf), set()).add(part)

    for (layer_index, leaf), parts in sorted(groups.items()):
        missing = [part for part in ("query", "key", "value") if part not in parts]
        if missing:
            base = f"backbone.0.encoder.encoder.encoder.layer.{layer_index}.attention.attention"
            names = ", ".join(f"{base}.{part}.{leaf}" for part in missing)
            raise ValueError(f"RF-DETR DINOv2 qkv pack for layer {layer_index} is missing {names}")


def _pack_hf_dinov2_qkv(key: str, state: dict[str, Any]) -> list[tuple[str, np.ndarray]] | None:
    component = _hf_dinov2_qkv_component(key)
    if component is None:
        return None
    layer_index, part, leaf = component
    if part != "query":
        return []

    base = f"backbone.0.encoder.encoder.encoder.layer.{layer_index}.attention.attention"
    values = [
        _to_numpy(_state_value_for_stripped_key(state, f"{base}.{name}.{leaf}"))
        for name in ("query", "key", "value")
    ]
    shapes = {name: tuple(value.shape) for name, value in zip(("query", "key", "value"), values)}
    if len(set(shapes.values())) != 1:
        raise ValueError(
            f"RF-DETR DINOv2 qkv pack for layer {layer_index} {leaf} has mismatched shapes: {shapes}"
        )
    return [
        (
            f"feature_extractor.backbone.backbone.blocks.{layer_index}.attn.qkv.{leaf}",
            np.concatenate(values, axis=0),
        )
    ]


def _transpose_conv_if_reference(key: str, value: np.ndarray) -> np.ndarray:
    if value.ndim == 4 and key.endswith(".weight"):
        return np.transpose(value, (0, 2, 3, 1))
    return value


def _remap_hf_dinov2_key(key: str) -> str | None:
    if not key.startswith(_HF_DINOV2_PREFIX):
        return None

    rest = key[len(_HF_DINOV2_PREFIX):]
    base = "feature_extractor.backbone.backbone"
    if rest == "embeddings.cls_token":
        return f"{base}.cls_token"
    if rest == "embeddings.mask_token":
        return None
    if rest == "embeddings.position_embeddings":
        return f"{base}.pos_embed.table"
    if rest.startswith("embeddings.patch_embeddings.projection."):
        leaf = rest[len("embeddings.patch_embeddings.projection."):]
        return f"{base}.patch_embed.proj.{leaf}"
    if rest.startswith("encoder.layer."):
        layer_rest = rest[len("encoder.layer."):]
        layer_index, _, suffix = layer_rest.partition(".")
        if not suffix:
            return None
        layer = f"{base}.blocks.{layer_index}"
        replacements = (
            ("attention.output.dense.", "attn.proj."),
            ("layer_scale1.lambda1", "ls1.gamma"),
            ("layer_scale2.lambda1", "ls2.gamma"),
        )
        for source, target in replacements:
            if suffix == source:
                return f"{layer}.{target}"
            if suffix.startswith(source):
                return f"{layer}.{target}{suffix[len(source):]}"
        if suffix.startswith("attention.attention."):
            return None
        return f"{layer}.{suffix}"
    if rest.startswith("layernorm."):
        return f"{base}.norm.{rest[len('layernorm.'):]}"
    return None


def remap_rfdetr_key(key: str) -> tuple[str | None, bool]:
    """Map one reference RF-DETR key to the local MLX key.

    Returns ``(mapped_key, reference_layout)``. ``reference_layout`` is true when
    a PyTorch layout conversion may be needed for the value.
    """
    if key.startswith("__") or key in _METADATA_KEYS:
        return None, False
    if key.startswith(_LOCAL_PREFIXES):
        return key, False

    key = _strip_reference_prefix(key)
    if _is_inference_only_exclusion(key):
        return None, True
    if key.startswith(_LOCAL_PREFIXES):
        return key, False

    if key.startswith("class_embed."):
        return f"head.{key}", True
    if key.startswith("bbox_embed."):
        return f"head.{key}", True
    if key == "query_feat.weight":
        return "decoder.query_embed", True
    if key == "refpoint_embed.weight":
        return "decoder.reference_embed", True

    for prefix in (
        "transformer.enc_output.",
        "transformer.enc_output_norm.",
        "transformer.enc_out_class_embed.",
        "transformer.enc_out_bbox_embed.",
    ):
        if key.startswith(prefix):
            rest = key[len("transformer."):]
            return f"decoder.{rest}", True

    for prefix in _PROJECTOR_PREFIXES:
        if key.startswith(prefix):
            rest = key[len(prefix):]
            return f"feature_extractor.projector.{rest}", True

    hf_dinov2 = _remap_hf_dinov2_key(key)
    if hf_dinov2 is not None:
        return hf_dinov2, True
    if key.startswith(_HF_DINOV2_PREFIX):
        return None, True

    for prefix in ("backbone.", "backbone.0.", "backbone.body."):
        if key.startswith(prefix):
            rest = key[len(prefix):]
            return f"feature_extractor.backbone.backbone.{rest}", True

    if key.startswith("transformer.decoder.norm."):
        rest = key[len("transformer.decoder.norm."):]
        return f"decoder.norm.{rest}", True
    if key.startswith("transformer.decoder.ref_point_head."):
        rest = key[len("transformer.decoder.ref_point_head."):]
        return f"decoder.ref_point_head.{rest}", True

    if key.startswith("transformer.decoder.layers."):
        rest = key[len("transformer.decoder.layers."):]
        layer_ix, _, suffix = rest.partition(".")
        if not suffix:
            return None, True
        if suffix in {"self_attn.in_proj_weight", "self_attn.in_proj_bias"}:
            return None, True
        suffix = suffix.replace("cross_attn.output_proj.", "out_proj.")
        suffix = suffix.replace("cross_attn.", "")
        suffix = suffix.replace("linear1.", "ffn1.")
        suffix = suffix.replace("linear2.", "ffn2.")
        return f"decoder.layers.{layer_ix}.{suffix}", True

    return None, True


def _split_decoder_in_proj(key: str, value: Any) -> list[tuple[str, np.ndarray]] | None:
    stripped = _strip_reference_prefix(key)
    if not stripped.startswith("transformer.decoder.layers."):
        return None
    rest = stripped[len("transformer.decoder.layers."):]
    layer_ix, _, suffix = rest.partition(".")
    if suffix not in {"self_attn.in_proj_weight", "self_attn.in_proj_bias"}:
        return None
    out = _to_numpy(value)
    if out.shape[0] % 3 != 0:
        raise ValueError(f"RF-DETR self_attn in_proj tensor {key!r} has non-triplet shape {out.shape}")
    q, k, v = np.split(out, 3, axis=0)
    leaf = "weight" if suffix.endswith("_weight") else "bias"
    return [
        (f"decoder.layers.{layer_ix}.self_attn.query_proj.{leaf}", np.asarray(q)),
        (f"decoder.layers.{layer_ix}.self_attn.key_proj.{leaf}", np.asarray(k)),
        (f"decoder.layers.{layer_ix}.self_attn.value_proj.{leaf}", np.asarray(v)),
    ]


def _convert_value(
    mapped_key: str,
    value: Any,
    *,
    reference_layout: bool,
    target_num_queries: int | None,
    target_group_detr: int | None,
    target_query_dim: int | None,
    ckpt_num_queries: int | None,
    ckpt_group_detr: int | None,
) -> np.ndarray:
    out = _to_numpy(value)
    if (
        reference_layout
        and mapped_key in _QUERY_PARAM_KEYS
        and target_num_queries is not None
        and target_group_detr is not None
        and ckpt_num_queries is not None
        and ckpt_group_detr is not None
    ):
        out = _slice_query_param_per_group(
            out,
            ckpt_num_queries=ckpt_num_queries,
            ckpt_group_detr=ckpt_group_detr,
            target_num_queries=target_num_queries,
            target_group_detr=target_group_detr,
        )
    if (
        mapped_key == "decoder.reference_embed"
        and out.ndim == 2
        and out.shape[-1] == 4
        and target_query_dim != 4
    ):
        out = out[:, :2]
    if reference_layout:
        out = _transpose_conv_if_reference(mapped_key, out)
    return np.asarray(out)


def convert_rfdetr_state_dict(
    state: dict[str, Any],
    *,
    target_num_queries: int | None = None,
    target_group_detr: int | None = None,
    target_query_dim: int | None = None,
    ckpt_num_queries: int | None = None,
    ckpt_group_detr: int | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Convert RF-DETR detection weights into local MLX parameter paths.

    The converter intentionally rejects unknown tensor keys. That keeps partial
    upstream segmentation or full-architecture checkpoints from silently loading
    into this detection-only path.
    """
    _reject_unsupported_variant(state)
    args = _metadata_args(state)
    ckpt_num_queries = ckpt_num_queries if ckpt_num_queries is not None else _metadata_int(args, "num_queries")
    ckpt_group_detr = ckpt_group_detr if ckpt_group_detr is not None else _metadata_int(args, "group_detr")
    rows = _first_query_rows(state)
    if rows is not None:
        if ckpt_num_queries is not None and ckpt_group_detr is None and rows % ckpt_num_queries == 0:
            ckpt_group_detr = rows // ckpt_num_queries
        if ckpt_group_detr is not None and ckpt_num_queries is None and rows % ckpt_group_detr == 0:
            ckpt_num_queries = rows // ckpt_group_detr
    items: list[tuple[str, np.ndarray]] = []
    unknown: list[str] = []
    _validate_hf_dinov2_qkv_groups(state)
    for key, value in state.items():
        split = _split_decoder_in_proj(key, value)
        if split is not None:
            items.extend(split)
            continue
        packed = _pack_hf_dinov2_qkv(key, state)
        if packed is not None:
            items.extend(packed)
            continue
        if _is_inference_only_exclusion(key):
            continue
        mapped, reference_layout = remap_rfdetr_key(key)
        if mapped is None:
            if key.startswith("__") or key in _METADATA_KEYS or _is_inference_only_exclusion(key):
                continue
            unknown.append(key)
            continue
        items.append(
            (
                mapped,
                _convert_value(
                    mapped,
                    value,
                    reference_layout=reference_layout,
                    target_num_queries=target_num_queries,
                    target_group_detr=target_group_detr,
                    target_query_dim=target_query_dim,
                    ckpt_num_queries=ckpt_num_queries,
                    ckpt_group_detr=ckpt_group_detr,
                ),
            )
        )
    if unknown:
        sample = ", ".join(repr(k) for k in unknown[:5])
        more = "" if len(unknown) <= 5 else f", and {len(unknown) - 5} more"
        raise ValueError(f"unsupported RF-DETR checkpoint keys: {sample}{more}")
    return items


def _load_weight_arrays(weights_path) -> dict[str, np.ndarray]:
    path = Path(weights_path)
    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}
    if path.suffix == ".safetensors":
        return {k: np.array(v) for k, v in mx.load(str(path)).items()}
    raise ValueError(f"unsupported RF-DETR weight format: {path}")


def load_rfdetr_weights(model: RFDETRModel, weights_path, *, strict: bool = False) -> RFDETRModel:
    """Load converted RF-DETR detection weights from ``.npz`` or safetensors."""
    state = _load_weight_arrays(weights_path)
    converted = [
        (k, mx.array(v))
        for k, v in convert_rfdetr_state_dict(
            state,
            target_num_queries=model.cfg.decoder.num_queries,
            target_group_detr=model.cfg.decoder.group_detr,
            target_query_dim=model.cfg.decoder.query_dim,
        )
    ]
    params = dict(tree_flatten(model.parameters()))
    seen: set[str] = set()
    duplicates: list[str] = []
    for key, value in converted:
        if key in seen:
            duplicates.append(key)
            continue
        seen.add(key)
        if key not in params:
            raise ValueError(f"converted RF-DETR key {key!r} is not present in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted RF-DETR key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )
    if duplicates:
        sample = ", ".join(repr(k) for k in duplicates[:5])
        more = "" if len(duplicates) <= 5 else f", and {len(duplicates) - 5} more"
        raise ValueError(f"duplicate converted RF-DETR keys: {sample}{more}")
    if strict:
        missing = sorted(key for key in params if key not in seen)
        if missing:
            sample = ", ".join(repr(k) for k in missing[:5])
            more = "" if len(missing) <= 5 else f", and {len(missing) - 5} more"
            raise ValueError(f"missing RF-DETR inference weights for local model keys: {sample}{more}")
    model.update(tree_unflatten(converted))
    mx.eval(model.parameters())
    return model
