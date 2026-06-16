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

    def trim(self, count: int) -> None:
        """Remove the last ``count`` cached positions from every populated layer."""
        count = int(count)
        if count < 0:
            raise ValueError(f"trim count must be non-negative, got {count}")
        if count == 0:
            return
        for i, key in enumerate(self.keys):
            if key is None:
                continue
            keep = max(int(key.shape[-2]) - count, 0)
            self.keys[i] = key[:, :, :keep, :]
            value = self.values[i]
            self.values[i] = None if value is None else value[:, :, :keep, :]
