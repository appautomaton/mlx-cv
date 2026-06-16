"""RF-DETR query decoder and detection head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.necks import RFDETRFeaturePyramid
from ...core.features import HeadOutput
from ...ops import ms_deform_attn_core

__all__ = ["RFDETRDecoderConfig", "RFDETRQueryDecoder", "RFDETRDetectionHead"]


@dataclass(frozen=True)
class RFDETRDecoderConfig:
    hidden_dim: int = 256
    num_queries: int = 300
    num_heads: int = 8
    num_layers: int = 6
    num_points: int = 4
    num_classes: int = 80

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if min(self.num_queries, self.num_heads, self.num_layers, self.num_points, self.num_classes) <= 0:
            raise ValueError("RF-DETR decoder dimensions must be positive")


def _sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-x))


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.maximum(self.layers[0](x), 0)
        x = mx.maximum(self.layers[1](x), 0)
        return self.layers[2](x)


class RFDETRDecoderLayer(nn.Module):
    def __init__(self, cfg: RFDETRDecoderConfig, num_levels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_levels = num_levels
        self.value_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.sampling_offsets = nn.Linear(
            cfg.hidden_dim,
            cfg.num_heads * num_levels * cfg.num_points * 2,
        )
        self.attention_weights = nn.Linear(
            cfg.hidden_dim,
            cfg.num_heads * num_levels * cfg.num_points,
        )
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4)
        self.ffn2 = nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def __call__(
        self,
        query: mx.array,
        memory: mx.array,
        spatial_shapes: np.ndarray,
        reference_points: mx.array,
        *,
        capture_taps: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        batch, spatial_size, hidden_dim = memory.shape
        _, num_queries, _ = query.shape
        value = self.value_proj(memory)
        head_dim = hidden_dim // self.cfg.num_heads
        value = mx.transpose(
            value.reshape(batch, spatial_size, self.cfg.num_heads, head_dim),
            (0, 2, 3, 1),
        )

        offsets = self.sampling_offsets(query).reshape(
            batch,
            num_queries,
            self.cfg.num_heads,
            self.num_levels,
            self.cfg.num_points,
            2,
        )
        weights = self.attention_weights(query).reshape(
            batch,
            num_queries,
            self.cfg.num_heads,
            self.num_levels * self.cfg.num_points,
        )
        weights = mx.softmax(weights.astype(mx.float32), axis=-1).astype(query.dtype)
        weights = weights.reshape(
            batch,
            num_queries,
            self.cfg.num_heads,
            self.num_levels,
            self.cfg.num_points,
        )

        normalizer = mx.array(
            [[float(w), float(h)] for h, w in spatial_shapes.tolist()],
            dtype=query.dtype,
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + offsets / normalizer[None, None, None, :, None, :]
        )
        attended = ms_deform_attn_core(value, spatial_shapes, sampling_locations, weights)
        query = self.norm1(query + self.out_proj(attended))
        ffn = self.ffn2(mx.maximum(self.ffn1(query), 0))
        query = self.norm2(query + ffn)
        if capture_taps:
            return query, attended
        return query


class RFDETRQueryDecoder(nn.Module):
    """Small RF-DETR decoder that consumes the projected pyramid."""

    def __init__(self, cfg: RFDETRDecoderConfig, *, num_levels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_levels = int(num_levels)
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive")
        self.query_embed = mx.zeros((cfg.num_queries, cfg.hidden_dim))
        self.reference_embed = mx.zeros((cfg.num_queries, 2))
        self.layers = [RFDETRDecoderLayer(cfg, self.num_levels) for _ in range(cfg.num_layers)]

    def __call__(self, pyramid: RFDETRFeaturePyramid, *, capture_taps: bool = False) -> dict[str, mx.array]:
        if len(pyramid.levels) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} pyramid levels, got {len(pyramid.levels)}")
        memory_parts = []
        shapes = []
        for level in pyramid.levels:
            if level.data.shape[-1] != self.cfg.hidden_dim:
                raise ValueError(
                    f"pyramid level channels {level.data.shape[-1]} must equal hidden_dim {self.cfg.hidden_dim}"
                )
            batch, height, width, channels = level.data.shape
            memory_parts.append(level.data.reshape(batch, height * width, channels))
            shapes.append((height, width))
        memory = mx.concatenate(memory_parts, axis=1)
        batch = memory.shape[0]
        query = mx.broadcast_to(self.query_embed[None], (batch, self.cfg.num_queries, self.cfg.hidden_dim))
        reference = _sigmoid(mx.broadcast_to(self.reference_embed[None], (batch, self.cfg.num_queries, 2)))
        reference_points = mx.broadcast_to(
            reference[:, :, None, :],
            (batch, self.cfg.num_queries, self.num_levels, 2),
        )
        spatial_shapes = np.array(shapes, dtype=np.int32)
        hidden_states = []
        deformable_attention = []
        for layer in self.layers:
            if capture_taps:
                query, attended = layer(
                    query,
                    memory,
                    spatial_shapes,
                    reference_points,
                    capture_taps=True,
                )
                deformable_attention.append(attended)
            else:
                query = layer(query, memory, spatial_shapes, reference_points)
            hidden_states.append(query)
        out = {
            "hidden_states": query,
            "decoder_hidden_states": mx.stack(hidden_states, axis=0),
            "reference_points": reference_points,
            "spatial_shapes": mx.array(spatial_shapes, dtype=mx.int32),
        }
        if capture_taps:
            out["deformable_attention"] = mx.stack(deformable_attention, axis=0)
        return out


class RFDETRDetectionHead(nn.Module):
    """RF-DETR raw classification and box head."""

    def __init__(self, cfg: RFDETRDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_embed = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.bbox_embed = _MLP(cfg.hidden_dim, cfg.hidden_dim, 4)

    def __call__(self, decoder_out: dict[str, mx.array]) -> HeadOutput:
        hidden = decoder_out["hidden_states"]
        logits = self.class_embed(hidden)
        boxes = _sigmoid(self.bbox_embed(hidden))
        return HeadOutput(
            {
                "logits": logits,
                "boxes": boxes,
                "hidden_states": hidden,
                "decoder_hidden_states": decoder_out["decoder_hidden_states"],
                "reference_points": decoder_out["reference_points"],
                "spatial_shapes": decoder_out["spatial_shapes"],
            }
        )
