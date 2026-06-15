"""Append-only Qwen2 KV cache."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["Qwen2KVCache"]


class Qwen2KVCache:
    """Per-layer append-only cache storing unrepeated RoPE-applied K/V tensors."""

    def __init__(self, num_layers: int) -> None:
        self.keys: list[mx.array | None] = [None] * num_layers
        self.values: list[mx.array | None] = [None] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        key = self.keys[layer_idx]
        return 0 if key is None else int(key.shape[-2])

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
    ) -> tuple[mx.array, mx.array]:
        key = self.keys[layer_idx]
        value = self.values[layer_idx]
        if key is None:
            new_key = key_states
            new_value = value_states
        else:
            new_key = mx.concatenate([key, key_states], axis=2)
            new_value = mx.concatenate([value, value_states], axis=2)
        self.keys[layer_idx] = new_key
        self.values[layer_idx] = new_value
        return new_key, new_value
