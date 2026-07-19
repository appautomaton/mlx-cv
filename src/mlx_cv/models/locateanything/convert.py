"""Weight conversion for full LocateAnything checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "LOCATEANYTHING_CHECKPOINT_METADATA",
    "LocateAnythingCheckpointError",
    "remap_key",
    "convert_state_dict",
    "load_locateanything_weights",
]


LOCATEANYTHING_CHECKPOINT_METADATA = {
    "format": "mlx-cv-locateanything-v1",
    "architecture": "LocateAnything-3B",
    "layout": "mlx-final",
    "dtype": "bfloat16",
    "tensor_count": "769",
}


class LocateAnythingCheckpointError(ValueError):
    """Raised when a production LocateAnything checkpoint is incompatible."""


def remap_key(key: str) -> str | None:
    """Map one reference key to the mlx-cv key, or ``None`` to drop it."""
    if key == "language_model.lm_head.weight":
        return None  # tied to embed_tokens
    if key.startswith("vision_model."):
        return key.replace("vision_model.encoder.", "vision_tower.").replace(
            "vision_model.", "vision_tower."
        )
    if key.startswith("mlp1."):
        return (
            key.replace("mlp1.0.", "multi_modal_projector.layer_norm.")
            .replace("mlp1.1.", "multi_modal_projector.linear_1.")
            .replace("mlp1.3.", "multi_modal_projector.linear_2.")
        )
    return key  # language_model.model.* and everything else: unchanged


def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):]


def _assert_full_tied_lm_head_is_lossless(weights: dict[str, np.ndarray]) -> None:
    lm_head = weights.get("language_model.lm_head.weight")
    if lm_head is None:
        return
    embed = weights.get("language_model.model.embed_tokens.weight")
    if embed is None:
        raise ValueError("cannot drop language_model.lm_head.weight without language_model.model.embed_tokens.weight")
    if not np.array_equal(lm_head, embed):
        raise ValueError("cannot drop language_model.lm_head.weight because it is not tied to embed_tokens")


def _convert_vision(weights: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    from ...backbones.vision.moonvit.convert import convert_moonvit_state_dict

    state = {}
    for k, v in weights.items():
        if k.startswith("vision_model.encoder."):
            state[_strip_prefix(k, "vision_model.")] = v
        elif k.startswith("vision_model."):
            state[_strip_prefix(k, "vision_model.")] = v
    return [(f"vision_tower.{k}", np.array(v)) for k, v in convert_moonvit_state_dict(state)]


def _convert_language(weights: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    from ...backbones.llm.qwen2.convert import convert_qwen2_state_dict

    _assert_full_tied_lm_head_is_lossless(weights)
    state = {
        _strip_prefix(k, "language_model."): v
        for k, v in weights.items()
        if k.startswith("language_model.")
    }
    return [(f"language_model.{k}", np.array(v)) for k, v in convert_qwen2_state_dict(state)]


def _convert_projector(weights: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for key, value in weights.items():
        if not key.startswith("mlp1."):
            continue
        mapped = remap_key(key)
        if mapped is not None:
            out.append((mapped, np.asarray(value)))
    return out


def convert_state_dict(weights: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Convert full reference LocateAnything weights to local MLX parameter paths."""
    items: list[tuple[str, np.ndarray]] = []
    items.extend(_convert_vision(weights))
    items.extend(_convert_language(weights))
    items.extend(_convert_projector(weights))
    for key, value in weights.items():
        if key.startswith(("vision_model.", "language_model.", "mlp1.")) or key.startswith("__"):
            continue
        mapped = remap_key(key)
        if mapped is not None:
            items.append((mapped, np.asarray(value)))
    return items


def _load_weight_arrays(weights_path) -> dict[str, np.ndarray]:
    path = Path(weights_path)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if not index_path.exists():
            raise ValueError(f"LocateAnything weight directory is missing {index_path.name}: {path}")
        import mlx.core as mx

        index = json.loads(index_path.read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"LocateAnything safetensors index has no weight_map entries: {index_path}")
        state: dict[str, np.ndarray] = {}
        for shard in shards:
            shard_path = path / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"LocateAnything safetensors shard is missing: {shard_path}")
            state.update({k: np.array(v) for k, v in mx.load(str(shard_path)).items()})
        return state

    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=False)
        return {k: npz[k] for k in npz.files}

    if path.suffix == ".safetensors":
        import mlx.core as mx

        return {k: np.array(v) for k, v in mx.load(str(path)).items()}

    raise ValueError(f"unsupported LocateAnything weight format: {path}")


def load_locateanything_weights(model, weights_path):
    """Load full LocateAnything weights from ``.npz`` or MLX safetensors."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    path = Path(weights_path)
    if path.suffix == ".safetensors":
        from ...hub.safetensors import read_safetensors_metadata
        from mlx.utils import tree_flatten

        metadata = read_safetensors_metadata(path)
        if metadata.get("format") == LOCATEANYTHING_CHECKPOINT_METADATA["format"]:
            errors = {
                key: (metadata.get(key), expected)
                for key, expected in LOCATEANYTHING_CHECKPOINT_METADATA.items()
                if metadata.get(key) != expected
            }
            if not metadata.get("source_sha256"):
                errors["source_sha256"] = (metadata.get("source_sha256"), "non-empty")
            if errors:
                raise LocateAnythingCheckpointError(
                    "incompatible LocateAnything checkpoint metadata: "
                    + ", ".join(
                        f"{key}={actual!r} (expected {expected!r})"
                        for key, (actual, expected) in errors.items()
                    )
                )
            weights = mx.load(str(path))
            params = dict(tree_flatten(model.parameters()))
            missing = sorted(set(params) - set(weights))
            unexpected = sorted(set(weights) - set(params))
            if missing or unexpected:
                raise LocateAnythingCheckpointError(
                    "LocateAnything parameter names do not match: "
                    f"missing={missing[:5]!r}, unexpected={unexpected[:5]!r}"
                )
            for key, value in weights.items():
                if tuple(value.shape) != tuple(params[key].shape):
                    raise LocateAnythingCheckpointError(
                        f"LocateAnything tensor {key!r} has shape {tuple(value.shape)}, "
                        f"expected {tuple(params[key].shape)}"
                    )
                if value.dtype != mx.bfloat16:
                    raise LocateAnythingCheckpointError(
                        f"LocateAnything tensor {key!r} has dtype {value.dtype}, expected bfloat16"
                    )
            model.update(tree_unflatten(list(weights.items())))
            mx.eval(model.parameters())
            return model

    state = _load_weight_arrays(path)
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in convert_state_dict(state)]))
    mx.eval(model.parameters())
    return model
