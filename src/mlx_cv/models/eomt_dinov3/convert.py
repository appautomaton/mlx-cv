"""Official Transformers EoMT-DINOv3 weights to the local MLX tree."""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from .modeling import EoMTDINOv3

__all__ = ["convert_eomt_dinov3_state_dict", "load_eomt_dinov3_weights"]

_LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.+)$")
_IGNORED_KEYS = {"criterion.empty_weight"}


def _array(value) -> np.ndarray:
    return np.asarray(value)


def _pack_qkv(state: dict[str, np.ndarray], layer: int, suffix: str) -> np.ndarray:
    prefix = f"layers.{layer}.attention."
    values = []
    reference = None
    for name in ("q_proj", "k_proj", "v_proj"):
        value = state.get(f"{prefix}{name}.{suffix}")
        if value is not None:
            reference = _array(value)
            break
    if reference is None:
        raise ValueError(f"EoMT layer {layer} has no attention {suffix} tensors")
    for name in ("q_proj", "k_proj", "v_proj"):
        value = state.get(f"{prefix}{name}.{suffix}")
        values.append(np.zeros_like(reference) if value is None else _array(value))
    return np.concatenate(values, axis=0)


def _remap_layer_key(layer: int, suffix: str) -> str | None:
    prefix = f"backbone.blocks.{layer}."
    if suffix.startswith("attention."):
        attention_suffix = suffix.removeprefix("attention.")
        if attention_suffix.startswith(("q_proj.", "k_proj.", "v_proj.")):
            return None
        if attention_suffix.startswith("o_proj."):
            return prefix + "attn.proj." + attention_suffix.removeprefix("o_proj.")
    if suffix.startswith("layer_scale1."):
        return prefix + "ls1.gamma"
    if suffix.startswith("layer_scale2."):
        return prefix + "ls2.gamma"
    if suffix.startswith("mlp.up_proj."):
        return prefix + "mlp.fc1." + suffix.removeprefix("mlp.up_proj.")
    if suffix.startswith("mlp.down_proj."):
        return prefix + "mlp.fc2." + suffix.removeprefix("mlp.down_proj.")
    if suffix.startswith("norm1.") or suffix.startswith("norm2."):
        return prefix + suffix
    return None


def convert_eomt_dinov3_state_dict(state: dict[str, np.ndarray]):
    """Convert a full official Transformers checkpoint into MLX parameter paths."""

    if "backbone.patch_embed.proj.weight" in state:
        return [(key, mx.array(value)) for key, value in state.items()]

    normalized = {str(key): _array(value) for key, value in state.items()}
    items: list[tuple[str, mx.array]] = []
    unknown: list[str] = []
    packed_layers: set[int] = set()

    for key, value in normalized.items():
        if key in _IGNORED_KEYS or key.startswith("__"):
            continue
        if key == "attn_mask_probs":
            mapped, converted = key, value
        elif key == "embeddings.cls_token":
            mapped, converted = "backbone.cls_token", value
        elif key == "embeddings.register_tokens":
            mapped, converted = "backbone.storage_tokens", value
        elif key == "embeddings.patch_embeddings.weight":
            mapped = "backbone.patch_embed.proj.weight"
            converted = np.transpose(value, (0, 2, 3, 1))
        elif key == "embeddings.patch_embeddings.bias":
            mapped, converted = "backbone.patch_embed.proj.bias", value
        elif key.startswith("layernorm."):
            mapped, converted = "backbone.norm." + key.removeprefix("layernorm."), value
        else:
            layer_match = _LAYER_KEY.match(key)
            if layer_match:
                layer = int(layer_match.group(1))
                suffix = layer_match.group(2)
                if suffix.startswith(("attention.q_proj.", "attention.k_proj.", "attention.v_proj.")):
                    if layer not in packed_layers:
                        items.append(
                            (f"backbone.blocks.{layer}.attn.qkv.weight", mx.array(_pack_qkv(normalized, layer, "weight")))
                        )
                        if any(
                            f"layers.{layer}.attention.{name}.bias" in normalized
                            for name in ("q_proj", "k_proj", "v_proj")
                        ):
                            items.append(
                                (f"backbone.blocks.{layer}.attn.qkv.bias", mx.array(_pack_qkv(normalized, layer, "bias")))
                            )
                        packed_layers.add(layer)
                    continue
                mapped = _remap_layer_key(layer, suffix)
                if mapped is None:
                    unknown.append(key)
                    continue
                converted = value
            elif key.startswith(("query.", "mask_head.", "class_predictor.")):
                mapped, converted = key, value
            elif key.startswith("upscale_block.block."):
                mapped, converted = key, value
                if key.endswith(".conv1.weight"):
                    converted = np.transpose(value, (1, 2, 3, 0))
                elif key.endswith(".conv2.weight"):
                    converted = np.transpose(value, (0, 2, 3, 1))
            else:
                unknown.append(key)
                continue
        items.append((mapped, mx.array(converted)))

    if unknown:
        sample = ", ".join(repr(key) for key in unknown[:8])
        more = "" if len(unknown) <= 8 else f", and {len(unknown) - 8} more"
        raise ValueError(f"unsupported EoMT-DINOv3 checkpoint keys: {sample}{more}")
    return items


def _load_weight_arrays(weights_path) -> dict[str, np.ndarray]:
    path = Path(weights_path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if path.suffix == ".safetensors":
        return {key: np.asarray(value) for key, value in mx.load(str(path)).items()}
    raise ValueError(f"unsupported EoMT-DINOv3 weight format: {path}")


def load_eomt_dinov3_weights(
    model: EoMTDINOv3,
    weights_path,
    *,
    strict: bool = False,
) -> EoMTDINOv3:
    """Load a full official or already-converted checkpoint into ``model``."""

    converted = convert_eomt_dinov3_state_dict(_load_weight_arrays(weights_path))
    params = dict(tree_flatten(model.parameters()))
    seen: dict[str, mx.array] = {}
    for key, value in converted:
        if key in seen:
            raise ValueError(f"duplicate converted EoMT-DINOv3 key: {key!r}")
        if key not in params:
            raise ValueError(f"converted EoMT-DINOv3 key {key!r} is not in the local model")
        if tuple(params[key].shape) != tuple(value.shape):
            raise ValueError(
                f"converted EoMT-DINOv3 key {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(params[key].shape)}"
            )
        seen[key] = value
    if strict:
        deterministic = {"backbone.periods"}
        missing = sorted(key for key in params if key not in seen and key not in deterministic)
        if missing:
            sample = ", ".join(repr(key) for key in missing[:8])
            more = "" if len(missing) <= 8 else f", and {len(missing) - 8} more"
            raise ValueError(f"missing EoMT-DINOv3 inference weights: {sample}{more}")
    model.update(tree_unflatten(list(seen.items())))
    mx.eval(model.parameters())
    return model
