"""MoonViT packed-patch primitives.

MoonViT consumes LocateAnything processor output: packed NCHW patch tensors plus
one ``(height, width)`` grid per image. This module is the MLX boundary; package
root config imports stay mlx-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import MoonViTConfig

__all__ = [
    "Learnable2DInterpPosEmb",
    "MoonViTPatchEmbed",
    "Rope2DPosEmb",
    "apply_rope",
    "bicubic_interpolate",
    "cu_seqlens_from_grid_hws",
    "make_block_attention_mask",
    "patch_merger",
]


def _as_hw_shapes(grid_hws) -> list[tuple[int, int]]:
    raw = grid_hws.tolist() if hasattr(grid_hws, "tolist") else np.asarray(grid_hws).tolist()
    shapes = [(int(shape[0]), int(shape[1])) for shape in raw]
    for height, width in shapes:
        if height <= 0 or width <= 0:
            raise ValueError(f"grid_hws entries must be positive, got {(height, width)}")
    return shapes


def _lengths_from_shapes(shapes: Sequence[tuple[int, int]]) -> list[int]:
    return [height * width for height, width in shapes]


def _assert_packed_length(seq_len: int, shapes: Sequence[tuple[int, int]]) -> None:
    expected = sum(_lengths_from_shapes(shapes))
    if seq_len != expected:
        raise ValueError(f"packed sequence length {seq_len} does not match grid_hws total {expected}")


def cu_seqlens_from_grid_hws(grid_hws) -> mx.array:
    """Return cumulative packed sequence lengths: ``[0, len0, len0+len1, ...]``."""
    lengths = [0, *_lengths_from_shapes(_as_hw_shapes(grid_hws))]
    return mx.array(np.cumsum(lengths, dtype=np.int32), dtype=mx.int32)


def make_block_attention_mask(cu_seqlens: mx.array, seq_length: int) -> mx.array:
    """Boolean visibility mask where tokens attend only within their image block."""
    pos = mx.arange(seq_length)
    block_id = mx.sum(pos[None, :] >= cu_seqlens[1:, None], axis=0)
    return block_id[:, None] == block_id[None, :]


def _cubic_weight(t: mx.array, a: float) -> mx.array:
    at = mx.abs(t)
    at2 = at * at
    at3 = at2 * at
    w1 = (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
    w2 = a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
    return mx.where(at <= 1.0, w1, mx.where(at < 2.0, w2, mx.zeros_like(t)))


def bicubic_interpolate(
    x: mx.array,
    *,
    size: tuple[int, int],
    align_corners: bool = False,
    antialias: bool = False,
) -> mx.array:
    """CPU-capable bicubic interpolation for NCHW tensors.

    Matches the PyTorch coordinate convention used by ``F.interpolate`` for this
    change. Antialiasing is present for completeness, but MoonViT pos-emb
    interpolation uses the default non-antialiased path.
    """
    batch, channels, in_h, in_w = x.shape
    out_h, out_w = int(size[0]), int(size[1])
    input_dtype = x.dtype
    x = x.astype(mx.float32)

    scale_h = out_h / in_h
    scale_w = out_w / in_w
    if align_corners and out_h > 1 and out_w > 1:
        y_out = mx.arange(out_h, dtype=mx.float32) * (in_h - 1) / (out_h - 1)
        x_out = mx.arange(out_w, dtype=mx.float32) * (in_w - 1) / (out_w - 1)
    else:
        y_out = (mx.arange(out_h, dtype=mx.float32) + 0.5) / out_h * in_h - 0.5
        x_out = (mx.arange(out_w, dtype=mx.float32) + 0.5) / out_w * in_w - 0.5

    fs_h = (1.0 / scale_h) if (antialias and scale_h < 1.0) else 1.0
    fs_w = (1.0 / scale_w) if (antialias and scale_w < 1.0) else 1.0
    support_h = 2.0 * fs_h
    support_w = 2.0 * fs_w

    def weights_1d(coords: mx.array, in_size: int, fs: float, support: float):
        start = mx.floor(coords - support).astype(mx.int32) + 1
        n_taps = int(2 * support + 1)
        offsets = mx.arange(n_taps, dtype=mx.int32)
        pix = start[:, None] + offsets[None, :]
        dist = coords[:, None] - pix.astype(mx.float32)
        weight = _cubic_weight(dist / fs, -0.5 if antialias else -0.75)
        valid = (pix >= 0) & (pix < in_size)
        weight = weight * valid
        pix = mx.clip(pix, 0, in_size - 1)
        weight = weight / (mx.sum(weight, axis=-1, keepdims=True) + 1e-8)
        return pix, weight

    pix_y, wy = weights_1d(y_out, in_h, fs_h, support_h)
    pix_x, wx = weights_1d(x_out, in_w, fs_w, support_w)
    taps_h = pix_y.shape[1]
    taps_w = pix_x.shape[1]

    gathered_y = x[:, :, pix_y.reshape(-1), :].reshape(batch, channels, out_h, taps_h, in_w)
    tmp = mx.sum(gathered_y * wy[None, None, :, :, None], axis=3)
    gathered_x = tmp[:, :, :, pix_x.reshape(-1)].reshape(batch, channels, out_h, out_w, taps_w)
    result = mx.sum(gathered_x * wx[None, None, None, :, :], axis=4)
    return result.astype(input_dtype) if input_dtype != mx.float32 else result


class Learnable2DInterpPosEmb(nn.Module):
    """Learned 2D position embedding interpolated per packed image grid."""

    def __init__(self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic") -> None:
        super().__init__()
        if interpolation_mode != "bicubic":
            raise NotImplementedError("MoonViT position embedding only supports bicubic interpolation")
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = mx.ones((height, width, dim))

    def _get_pos_emb(self, shape: tuple[int, int]) -> mx.array:
        if shape == (self.height, self.width):
            return self.weight.reshape(self.height * self.width, -1)
        table = mx.expand_dims(mx.transpose(self.weight, (2, 0, 1)), axis=0)
        resized = bicubic_interpolate(table, size=shape)
        return mx.transpose(mx.squeeze(resized, axis=0), (1, 2, 0)).reshape(shape[0] * shape[1], -1)

    def __call__(self, x: mx.array, grid_hws) -> mx.array:
        shapes = _as_hw_shapes(grid_hws)
        _assert_packed_length(x.shape[0], shapes)
        pos = mx.concatenate([self._get_pos_emb(shape) for shape in shapes], axis=0).astype(x.dtype)
        return x + pos


class MoonViTPatchEmbed(nn.Module):
    """Conv patch embed for already-patchified packed NCHW inputs."""

    def __init__(self, config: MoonViTConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )
        self.pos_emb = Learnable2DInterpPosEmb(
            height=config.init_pos_emb_height,
            width=config.init_pos_emb_width,
            dim=config.hidden_size,
        )

    def __call__(self, pixel_values: mx.array, grid_hws) -> mx.array:
        if pixel_values.ndim != 4:
            raise ValueError(f"pixel_values must be packed NCHW patches, got shape {pixel_values.shape}")
        seq_len, channels, height, width = pixel_values.shape
        if channels != self.num_channels:
            raise ValueError(f"pixel_values channel count {channels} does not match config {self.num_channels}")
        if (height, width) != (self.patch_size, self.patch_size):
            raise ValueError(
                "MoonViTPatchEmbed expects already-patchified inputs with "
                f"patch size {(self.patch_size, self.patch_size)}, got {(height, width)}"
            )
        shapes = _as_hw_shapes(grid_hws)
        _assert_packed_length(seq_len, shapes)
        x = mx.transpose(pixel_values, (0, 2, 3, 1))
        x = self.proj(x).reshape(seq_len, -1)
        return self.pos_emb(x, shapes)


def _validate_rope_input(x: mx.array, freqs_cis: mx.array) -> None:
    if x.ndim != freqs_cis.ndim + 1:
        raise ValueError(f"rope input rank mismatch: {x.shape} vs {freqs_cis.shape}")
    if x.shape[:-2] != freqs_cis.shape[:-1]:
        raise ValueError(f"rope leading shape mismatch: {x.shape} vs {freqs_cis.shape}")
    if x.shape[-1] != 2 * freqs_cis.shape[-1]:
        raise ValueError(f"rope head dim mismatch: {x.shape} vs {freqs_cis.shape}")
    if freqs_cis.dtype != mx.complex64:
        raise ValueError(f"freqs_cis must be complex64, got {freqs_cis.dtype}")


def _view_as_complex(x: mx.array) -> mx.array:
    return x[..., 0] + 1j * x[..., 1]


def _view_as_real(x: mx.array) -> mx.array:
    return mx.stack([mx.real(x), mx.imag(x)], axis=-1)


def apply_rope(q: mx.array, k: mx.array, freqs_cis: mx.array) -> tuple[mx.array, mx.array]:
    """Apply MoonViT complex 2D RoPE to query/key tensors."""
    _validate_rope_input(q, freqs_cis)
    _validate_rope_input(k, freqs_cis)
    freqs = mx.expand_dims(freqs_cis, axis=-2)
    q_complex = _view_as_complex(q.astype(mx.float32).reshape(*q.shape[:-1], -1, 2))
    k_complex = _view_as_complex(k.astype(mx.float32).reshape(*k.shape[:-1], -1, 2))
    q_out = _view_as_real(q_complex * freqs).reshape(q.shape)
    k_out = _view_as_real(k_complex * freqs).reshape(k.shape)
    return q_out.astype(q.dtype), k_out.astype(k.dtype)


class Rope2DPosEmb(nn.Module):
    """Reference-style complex 2D rotary position embedding."""

    def __init__(self, dim: int, max_height: int = 512, max_width: int = 512, theta_base: float = 10000.0) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self) -> mx.array:
        total = self.max_height * self.max_width
        flat_pos = mx.arange(total, dtype=mx.float32)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mx.arange(0, self.dim, 4, dtype=mx.float32)[: (self.dim // 4)]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mx.outer(x_pos, freqs)
        y_freqs = mx.outer(y_pos, freqs)
        x_cis = mx.cos(x_freqs) + 1j * mx.sin(x_freqs)
        y_cis = mx.cos(y_freqs) + 1j * mx.sin(y_freqs)
        return mx.stack([x_cis, y_cis], axis=-1).reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(self, grid_hws) -> mx.array:
        shapes = _as_hw_shapes(grid_hws)
        if not all(1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes):
            raise ValueError(f"grid_hws {shapes} exceed RoPE limits {(self.max_height, self.max_width)}")
        full = self._precompute_freqs_cis()
        return mx.concatenate([full[:h, :w].reshape(h * w, self.dim // 2) for h, w in shapes], axis=0)


def patch_merger(
    x: mx.array,
    grid_hws,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[mx.array]:
    """Merge each image grid into non-overlapping 2D windows flattened to ``kH*kW*D``."""
    shapes = _as_hw_shapes(grid_hws)
    _assert_packed_length(x.shape[0], shapes)
    kernel_h, kernel_w = int(merge_kernel_size[0]), int(merge_kernel_size[1])
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError(f"merge_kernel_size must be positive, got {merge_kernel_size}")

    outputs: list[mx.array] = []
    offset = 0
    d_model = x.shape[-1]
    for height, width in shapes:
        if height % kernel_h != 0 or width % kernel_w != 0:
            raise ValueError(
                f"grid {(height, width)} must be divisible by merge kernel {(kernel_h, kernel_w)}"
            )
        length = height * width
        seq = x[offset : offset + length]
        offset += length
        new_h, new_w = height // kernel_h, width // kernel_w
        seq = seq.reshape(new_h, kernel_h, new_w, kernel_w, d_model)
        seq = mx.transpose(seq, (0, 2, 1, 3, 4))
        outputs.append(seq.reshape(new_h * new_w, kernel_h * kernel_w * d_model))
    return outputs
