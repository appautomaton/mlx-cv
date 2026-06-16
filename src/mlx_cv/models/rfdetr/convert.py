"""RF-DETR detection checkpoint conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mlx.core as mx

from .modeling import RFDETRModel

__all__ = [
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


def _transpose_conv_if_reference(key: str, value: np.ndarray) -> np.ndarray:
    if value.ndim == 4 and key.endswith(".weight"):
        return np.transpose(value, (0, 2, 3, 1))
    return value


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
    for key, value in state.items():
        split = _split_decoder_in_proj(key, value)
        if split is not None:
            items.extend(split)
            continue
        mapped, reference_layout = remap_rfdetr_key(key)
        if mapped is None:
            if key.startswith("__") or key in _METADATA_KEYS:
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


def load_rfdetr_weights(model: RFDETRModel, weights_path) -> RFDETRModel:
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
    for key, value in converted:
        if key not in params:
            raise ValueError(f"converted RF-DETR key {key!r} is not present in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted RF-DETR key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )
    model.update(tree_unflatten(converted))
    mx.eval(model.parameters())
    return model
