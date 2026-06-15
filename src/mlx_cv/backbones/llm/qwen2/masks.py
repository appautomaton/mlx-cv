"""LocateAnything/Qwen2 additive attention-mask helpers.

Masks use the local SDPA/manual convention: ``0.0`` means visible and ``-inf``
means masked. Helpers return MLX arrays and avoid importing torch/transformers.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "make_causal_mask_4d",
    "find_prefix_seq_length_by_pe",
    "update_causal_mask_with_pad_non_visible_2d",
    "update_causal_mask_for_one_gen_window_2d",
    "create_block_diff_mask_by_pe_4d",
]


def _mask_value(dtype) -> mx.array:
    return mx.array(-float("inf"), dtype=dtype)


def make_causal_mask_4d(
    batch_size: int,
    query_length: int,
    key_value_length: int | None = None,
    *,
    dtype=mx.float32,
) -> mx.array:
    """Create a 4D additive causal mask of shape ``(B, 1, Q, K)``."""
    key_value_length = query_length if key_value_length is None else key_value_length
    past_length = key_value_length - query_length
    if past_length < 0:
        raise ValueError(
            "key_value_length must be at least query_length, got "
            f"{key_value_length} < {query_length}"
        )

    rows = mx.arange(query_length, dtype=mx.int32).reshape(query_length, 1)
    cols = mx.arange(key_value_length, dtype=mx.int32).reshape(1, key_value_length)
    visible = cols <= (rows + past_length)
    mask = mx.where(
        visible,
        mx.zeros((query_length, key_value_length), dtype=dtype),
        mx.full((query_length, key_value_length), -float("inf"), dtype=dtype),
    )
    return mx.broadcast_to(mask.reshape(1, 1, query_length, key_value_length), (batch_size, 1, query_length, key_value_length))


def find_prefix_seq_length_by_pe(pe: mx.array) -> mx.array:
    """Find the first position-id drop per batch, or ``-1`` if no drop exists."""
    prev = pe[:, :-1]
    curr = pe[:, 1:]
    drop_mask = curr < prev
    first_drop = mx.argmax(drop_mask.astype(mx.int32), axis=1) + 1
    has_drop = mx.any(drop_mask, axis=1)
    return mx.where(has_drop, first_drop, mx.full((pe.shape[0],), -1, dtype=mx.int32))


def update_causal_mask_with_pad_non_visible_2d(
    input_ids: mx.array,
    attn_mask_2d: mx.array,
    text_mask_token_id: int,
    *,
    block_size: int = 4,
    causal_attn: bool = False,
) -> mx.array:
    """Apply LocateAnything non-visible padding/token-mask visibility to a 2D mask."""
    del block_size
    seq_len = input_ids.shape[0]
    input_mask = input_ids == text_mask_token_id
    input_before_mask = mx.concatenate([input_mask[1:], mx.array([False])], axis=0)
    mask_cols = input_mask | input_before_mask
    non_mask = ~mask_cols

    rows = mx.arange(seq_len, dtype=mx.int32).reshape(seq_len, 1)
    cols = mx.arange(seq_len, dtype=mx.int32).reshape(1, seq_len)
    indices = mx.arange(seq_len, dtype=mx.int32)

    prev_non_mask = mx.cummax(mx.where(non_mask, indices, mx.zeros_like(indices)), axis=0)
    max_value = mx.full((seq_len,), seq_len + 1, dtype=mx.int32)
    mask_indices = mx.where(non_mask, indices, max_value)
    next_non_mask = mx.cummin(mask_indices[::-1], axis=0)[::-1]

    infra_mask = (cols > prev_non_mask.reshape(1, seq_len)) & (
        rows >= next_non_mask.reshape(1, seq_len)
    ) & mask_cols.reshape(1, seq_len)
    out = mx.where(infra_mask, _mask_value(attn_mask_2d.dtype), attn_mask_2d)

    if not causal_attn:
        visible_mask = (rows > prev_non_mask.reshape(1, seq_len)) & (rows < cols) & mask_cols.reshape(
            1,
            seq_len,
        )
        out = mx.where(visible_mask, mx.array(0.0, dtype=attn_mask_2d.dtype), out)

    return out


def update_causal_mask_for_one_gen_window_2d(
    input_ids: mx.array,
    attn_mask_2d: mx.array,
    *,
    block_size: int = 4,
    use_cache: bool = True,
    causal_attn: bool = False,
) -> mx.array:
    """Apply the one-window inference mask update used by SDLM generation."""
    del input_ids
    seq_len = attn_mask_2d.shape[0]
    block_start = max(seq_len - block_size, 0)
    rows = mx.arange(seq_len, dtype=mx.int32).reshape(seq_len, 1)
    cols = mx.arange(seq_len, dtype=mx.int32).reshape(1, seq_len)
    out = attn_mask_2d

    if not causal_attn:
        window = (rows >= block_start) & (cols >= block_start)
        out = mx.where(window, mx.array(0.0, dtype=attn_mask_2d.dtype), out)

    if use_cache:
        prev_col = seq_len - block_size - 1
        if prev_col >= 0:
            stale = (rows >= block_start) & (cols == prev_col)
            out = mx.where(stale, _mask_value(attn_mask_2d.dtype), out)

    return out


def create_block_diff_mask_by_pe_4d(
    block_size: int,
    x0_len_list: mx.array,
    position_ids: mx.array,
    *,
    causal_attn: bool = False,
) -> tuple[mx.array, mx.array]:
    """Create LocateAnything block-diff additive and boolean visibility masks."""
    batch_size, seq_len = position_ids.shape
    q_idx = mx.arange(seq_len, dtype=mx.int32).reshape(1, seq_len, 1)
    kv_idx = mx.arange(seq_len, dtype=mx.int32).reshape(1, 1, seq_len)
    x0_len = x0_len_list.astype(mx.int32).reshape(batch_size, 1, 1)

    x0_flag_q = q_idx < x0_len
    x0_flag_kv = kv_idx < x0_len
    q_block_idx = (q_idx - x0_len) // block_size
    kv_block_idx = (kv_idx - x0_len) // block_size

    block_causal = x0_flag_q & x0_flag_kv & (q_idx >= kv_idx)
    mutual_condition = (q_idx >= kv_idx) if causal_attn else mx.ones((1, seq_len, seq_len), dtype=mx.bool_)
    block_mutual = (~x0_flag_q) & (~x0_flag_kv) & (q_block_idx == kv_block_idx) & mutual_condition

    q_blk = (q_idx - x0_len) // block_size
    q_blk_start = x0_len_list.astype(mx.int32).reshape(batch_size, 1) + q_blk[:, :, 0] * block_size
    q_blk_start = mx.clip(q_blk_start, 0, seq_len - 1)
    prefix_len = mx.take_along_axis(position_ids.astype(mx.int32), q_blk_start, axis=1)
    prefix_len = prefix_len.reshape(batch_size, seq_len, 1)
    block_prefix = (~x0_flag_q) & x0_flag_kv & (kv_idx < prefix_len)

    final_mask = block_causal | block_mutual | block_prefix
    additive = mx.where(
        final_mask,
        mx.zeros((batch_size, seq_len, seq_len), dtype=mx.float32),
        mx.full((batch_size, seq_len, seq_len), -float("inf"), dtype=mx.float32),
    )
    return mx.expand_dims(additive, axis=1), mx.expand_dims(final_mask, axis=1)
