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
    ffn_hidden_dim: int | None = None
    self_attn_heads: int | None = None
    group_detr: int = 1
    query_dim: int = 2
    use_self_attention: bool = False
    two_stage: bool = False
    bbox_reparam: bool = False
    lite_refpoint_refine: bool = False
    decoder_final_norm: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.self_attn_heads is not None and self.hidden_dim % self.self_attn_heads != 0:
            raise ValueError("hidden_dim must be divisible by self_attn_heads")
        dims = (
            self.hidden_dim,
            self.num_queries,
            self.num_heads,
            self.num_layers,
            self.num_points,
            self.num_classes,
            self.group_detr,
            self.query_dim,
        )
        if min(dims) <= 0 or (self.ffn_hidden_dim is not None and self.ffn_hidden_dim <= 0):
            raise ValueError("RF-DETR decoder dimensions must be positive")
        if self.use_self_attention and self.hidden_dim % 4 != 0:
            raise ValueError("use_self_attention requires hidden_dim divisible by 4")
        if (self.use_self_attention or self.two_stage or self.bbox_reparam) and self.query_dim != 4:
            raise ValueError("Nano two-stage/ref-point decoder paths require query_dim=4")

    @property
    def total_queries(self) -> int:
        return self.num_queries * self.group_detr

    @property
    def resolved_ffn_hidden_dim(self) -> int:
        return self.ffn_hidden_dim if self.ffn_hidden_dim is not None else self.hidden_dim * 4

    @property
    def resolved_self_attn_heads(self) -> int:
        return self.self_attn_heads if self.self_attn_heads is not None else self.num_heads


def _sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-x))


def _inverse_sigmoid(x: mx.array, eps: float = 1e-6) -> mx.array:
    x = mx.clip(x, eps, 1.0 - eps)
    return mx.log(x / (1 - x))


def _bbox_reparametrize(delta: mx.array, reference: mx.array) -> mx.array:
    center = delta[..., :2] * reference[..., 2:] + reference[..., :2]
    size = mx.exp(delta[..., 2:]) * reference[..., 2:]
    return mx.concatenate([center, size], axis=-1)


def slice_grouped_queries_for_inference(tensor: mx.array, *, num_queries: int, group_detr: int) -> mx.array:
    """Return inference group 0 without flattening across packed query groups."""

    if num_queries <= 0 or group_detr <= 0:
        raise ValueError("num_queries and group_detr must be positive")
    if tensor.shape[0] == num_queries:
        return tensor
    expected = num_queries * group_detr
    if tensor.shape[0] != expected:
        raise ValueError(f"grouped query tensor has {tensor.shape[0]} rows, expected {expected}")
    return tensor.reshape(group_detr, num_queries, *tensor.shape[1:])[0]


def _gen_sineembed_for_position(pos_tensor: mx.array, dim: int) -> mx.array:
    if pos_tensor.shape[-1] not in (2, 4):
        raise ValueError(f"position tensor last dimension must be 2 or 4, got {pos_tensor.shape[-1]}")
    scale = 2 * np.pi
    dim_t = mx.array(10000 ** (2 * (np.arange(dim, dtype=np.float32) // 2) / dim), dtype=pos_tensor.dtype)

    def encode(value: mx.array) -> mx.array:
        embed = value[..., None] * scale / dim_t
        return mx.stack([mx.sin(embed[..., 0::2]), mx.cos(embed[..., 1::2])], axis=-1).reshape(
            *value.shape,
            dim,
        )

    parts = [encode(pos_tensor[..., 1]), encode(pos_tensor[..., 0])]
    if pos_tensor.shape[-1] == 4:
        parts.extend([encode(pos_tensor[..., 2]), encode(pos_tensor[..., 3])])
    return mx.concatenate(parts, axis=-1)


def _take_along_queries(value: mx.array, indices: mx.array) -> mx.array:
    expanded = mx.broadcast_to(indices[..., None], (indices.shape[0], indices.shape[1], value.shape[-1]))
    return mx.take_along_axis(value, expanded, axis=1)


def _topk_indices(scores: mx.array, k: int) -> mx.array:
    # RF-DETR's two-stage decoder makes proposal order observable through
    # self-attention. Bucket tiny score differences before the secondary index
    # tie-break so CPU reference parity is not decided by backend rounding noise.
    tie_tol = 2e-5
    ranked = scores.astype(mx.float32)
    index_bias = mx.arange(scores.shape[1], dtype=mx.float32) * (tie_tol / (2.0 * float(scores.shape[1])))
    ranked = mx.round(ranked / tie_tol) * tie_tol + index_bias[None]
    order = mx.argsort(ranked, axis=1)
    return order[:, -k:][:, ::-1]


def _generate_encoder_output_proposals(
    memory: mx.array,
    spatial_shapes: np.ndarray,
    memory_padding_mask: mx.array | None = None,
    *,
    unsigmoid: bool,
) -> tuple[mx.array, mx.array]:
    batch = memory.shape[0]
    dtype = memory.dtype
    proposals = []
    cur = 0
    for level, (height, width) in enumerate(np.asarray(spatial_shapes, dtype=np.int32).tolist()):
        if memory_padding_mask is not None:
            mask = memory_padding_mask[:, cur : cur + height * width].reshape(batch, height, width, 1)
            valid_height = mx.sum((~mask[:, :, 0, 0]).astype(dtype), axis=1)
            valid_width = mx.sum((~mask[:, 0, :, 0]).astype(dtype), axis=1)
        else:
            valid_height = mx.full((batch,), float(height), dtype=dtype)
            valid_width = mx.full((batch,), float(width), dtype=dtype)

        yy, xx = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        grid = mx.array(np.stack([xx, yy], axis=-1), dtype=dtype)
        scale = mx.stack([valid_width, valid_height], axis=1).reshape(batch, 1, 1, 2)
        grid = (grid[None] + 0.5) / scale
        wh = mx.ones_like(grid) * (0.05 * (2.0**level))
        proposals.append(mx.concatenate([grid, wh], axis=-1).reshape(batch, height * width, 4))
        cur += height * width

    output_proposals = mx.concatenate(proposals, axis=1)
    valid = mx.all((output_proposals > 0.01) & (output_proposals < 0.99), axis=-1, keepdims=True)
    if unsigmoid:
        output_proposals = _inverse_sigmoid(output_proposals)
        invalid_value = mx.array(float("inf"), dtype=dtype)
    else:
        invalid_value = mx.array(0.0, dtype=dtype)
    output_proposals = mx.where(valid, output_proposals, invalid_value)
    output_memory = mx.where(valid, memory, mx.zeros_like(memory))
    if memory_padding_mask is not None:
        keep = ~memory_padding_mask[..., None]
        output_memory = mx.where(keep, output_memory, mx.zeros_like(output_memory))
        output_proposals = mx.where(keep, output_proposals, invalid_value)
    return output_memory.astype(dtype), output_proposals.astype(dtype)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = mx.maximum(x, 0)
        return x


class RFDETRDecoderLayer(nn.Module):
    def __init__(self, cfg: RFDETRDecoderConfig, num_levels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_levels = num_levels
        if cfg.use_self_attention:
            self.self_attn = nn.MultiHeadAttention(
                cfg.hidden_dim,
                cfg.resolved_self_attn_heads,
                bias=True,
            )
            self.norm1 = nn.LayerNorm(cfg.hidden_dim)
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
        if cfg.use_self_attention:
            self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        else:
            self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn1 = nn.Linear(cfg.hidden_dim, cfg.resolved_ffn_hidden_dim)
        self.ffn2 = nn.Linear(cfg.resolved_ffn_hidden_dim, cfg.hidden_dim)
        if cfg.use_self_attention:
            self.norm3 = nn.LayerNorm(cfg.hidden_dim)
        else:
            self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def __call__(
        self,
        query: mx.array,
        memory: mx.array,
        spatial_shapes: np.ndarray,
        reference_points: mx.array,
        *,
        query_pos: mx.array | None = None,
        memory_padding_mask: mx.array | None = None,
        capture_taps: bool = False,
    ) -> mx.array | tuple[mx.array, dict[str, mx.array]]:
        taps: dict[str, mx.array] = {}
        if self.cfg.use_self_attention:
            q = query if query_pos is None else query + query_pos
            attended = self.self_attn(q, q, query)
            query = self.norm1(query + attended)
            if capture_taps:
                taps["self_attention"] = attended

        batch, spatial_size, hidden_dim = memory.shape
        _, num_queries, _ = query.shape
        value = self.value_proj(memory)
        if memory_padding_mask is not None:
            value = mx.where(memory_padding_mask[..., None], mx.zeros_like(value), value)
        head_dim = hidden_dim // self.cfg.num_heads
        value = mx.transpose(
            value.reshape(batch, spatial_size, self.cfg.num_heads, head_dim),
            (0, 2, 3, 1),
        )

        cross_query = query if query_pos is None else query + query_pos
        offsets = self.sampling_offsets(cross_query).reshape(
            batch,
            num_queries,
            self.cfg.num_heads,
            self.num_levels,
            self.cfg.num_points,
            2,
        )
        weights = self.attention_weights(cross_query).reshape(
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

        if reference_points.shape[-1] == 2:
            normalizer = mx.array(
                [[float(w), float(h)] for h, w in spatial_shapes.tolist()],
                dtype=query.dtype,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / self.cfg.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dimension must be 2 or 4, got {reference_points.shape[-1]}")
        attended = ms_deform_attn_core(value, spatial_shapes, sampling_locations, weights)
        if self.cfg.use_self_attention:
            query = self.norm2(query + self.out_proj(attended))
        else:
            query = self.norm1(query + self.out_proj(attended))
        ffn = self.ffn2(mx.maximum(self.ffn1(query), 0))
        if self.cfg.use_self_attention:
            query = self.norm3(query + ffn)
        else:
            query = self.norm2(query + ffn)
        if capture_taps:
            taps["deformable_attention"] = attended
            return query, taps
        return query


class RFDETRQueryDecoder(nn.Module):
    """Small RF-DETR decoder that consumes the projected pyramid."""

    def __init__(self, cfg: RFDETRDecoderConfig, *, num_levels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_levels = int(num_levels)
        if self.num_levels <= 0:
            raise ValueError("num_levels must be positive")
        self.query_embed = mx.zeros((cfg.total_queries, cfg.hidden_dim))
        self.reference_embed = mx.zeros((cfg.total_queries, cfg.query_dim))
        self.layers = [RFDETRDecoderLayer(cfg, self.num_levels) for _ in range(cfg.num_layers)]
        if cfg.use_self_attention:
            self.ref_point_head = _MLP(2 * cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2)
        if cfg.decoder_final_norm:
            self.norm = nn.LayerNorm(cfg.hidden_dim)
        if cfg.two_stage:
            self.enc_output = [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.group_detr)]
            self.enc_output_norm = [nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.group_detr)]
            self.enc_out_class_embed = [nn.Linear(cfg.hidden_dim, cfg.num_classes) for _ in range(cfg.group_detr)]
            self.enc_out_bbox_embed = [
                _MLP(cfg.hidden_dim, cfg.hidden_dim, 4, num_layers=3) for _ in range(cfg.group_detr)
            ]

    def _reference_points_for_attention(self, reference: mx.array) -> mx.array:
        if self.cfg.bbox_reparam:
            ref = reference
        else:
            ref = _sigmoid(reference)
        return mx.broadcast_to(
            ref[:, :, None, :],
            (ref.shape[0], ref.shape[1], self.num_levels, ref.shape[-1]),
        )

    def _two_stage_references(
        self,
        memory: mx.array,
        spatial_shapes: np.ndarray,
        reference: mx.array,
        memory_padding_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        output_memory, output_proposals = _generate_encoder_output_proposals(
            memory,
            spatial_shapes,
            memory_padding_mask,
            unsigmoid=not self.cfg.bbox_reparam,
        )
        group_index = 0
        proposal_memory = self.enc_output_norm[group_index](self.enc_output[group_index](output_memory))
        proposal_logits = self.enc_out_class_embed[group_index](proposal_memory)
        proposal_delta = self.enc_out_bbox_embed[group_index](proposal_memory)
        if self.cfg.bbox_reparam:
            proposal_boxes = _bbox_reparametrize(proposal_delta, output_proposals)
        else:
            proposal_boxes = proposal_delta + output_proposals

        topk = min(self.cfg.num_queries, proposal_logits.shape[1])
        topk_indices = _topk_indices(mx.max(proposal_logits, axis=-1), topk)
        selected_boxes = _take_along_queries(proposal_boxes, topk_indices)
        selected_memory = _take_along_queries(proposal_memory, topk_indices)

        ts_len = selected_boxes.shape[1]
        reference_ts = reference[:, :ts_len, :]
        reference_tail = reference[:, ts_len:, :]
        if self.cfg.bbox_reparam:
            reference_ts = _bbox_reparametrize(reference_ts, selected_boxes)
        else:
            reference_ts = reference_ts + selected_boxes
        if reference_tail.shape[1] > 0:
            reference = mx.concatenate([reference_ts, reference_tail], axis=1)
        else:
            reference = reference_ts
        return reference, selected_memory, selected_boxes

    def __call__(self, pyramid: RFDETRFeaturePyramid, *, capture_taps: bool = False) -> dict[str, mx.array]:
        if len(pyramid.levels) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} pyramid levels, got {len(pyramid.levels)}")
        memory_parts = []
        mask_parts = []
        shapes = []
        for level in pyramid.levels:
            if level.data.shape[-1] != self.cfg.hidden_dim:
                raise ValueError(
                    f"pyramid level channels {level.data.shape[-1]} must equal hidden_dim {self.cfg.hidden_dim}"
                )
            batch, height, width, channels = level.data.shape
            memory_parts.append(level.data.reshape(batch, height * width, channels))
            mask_parts.append(level.mask.reshape(batch, height * width))
            shapes.append((height, width))
        memory = mx.concatenate(memory_parts, axis=1)
        memory_padding_mask = mx.concatenate(mask_parts, axis=1) if mask_parts else None
        batch = memory.shape[0]
        query_weight = slice_grouped_queries_for_inference(
            self.query_embed,
            num_queries=self.cfg.num_queries,
            group_detr=self.cfg.group_detr,
        )
        reference_weight = slice_grouped_queries_for_inference(
            self.reference_embed,
            num_queries=self.cfg.num_queries,
            group_detr=self.cfg.group_detr,
        )
        query = mx.broadcast_to(query_weight[None], (batch, self.cfg.num_queries, self.cfg.hidden_dim))
        reference = mx.broadcast_to(reference_weight[None], (batch, self.cfg.num_queries, self.cfg.query_dim))
        spatial_shapes = np.array(shapes, dtype=np.int32)
        encoder_memory = None
        encoder_boxes = None
        if self.cfg.two_stage:
            reference, encoder_memory, encoder_boxes = self._two_stage_references(
                memory,
                spatial_shapes,
                reference,
                memory_padding_mask,
            )

        hidden_states = []
        self_attention = []
        deformable_attention = []
        for layer in self.layers:
            reference_points = self._reference_points_for_attention(reference)
            query_pos = None
            if self.cfg.use_self_attention:
                ref_for_sine = reference_points[:, :, 0, :]
                query_sine = _gen_sineembed_for_position(ref_for_sine, self.cfg.hidden_dim // 2)
                query_pos = self.ref_point_head(query_sine)
            if capture_taps:
                query, taps = layer(
                    query,
                    memory,
                    spatial_shapes,
                    reference_points,
                    query_pos=query_pos,
                    memory_padding_mask=memory_padding_mask,
                    capture_taps=True,
                )
                if "self_attention" in taps:
                    self_attention.append(taps["self_attention"])
                deformable_attention.append(taps["deformable_attention"])
            else:
                query = layer(
                    query,
                    memory,
                    spatial_shapes,
                    reference_points,
                    query_pos=query_pos,
                    memory_padding_mask=memory_padding_mask,
            )
            hidden_states.append(query)
        if self.cfg.decoder_final_norm:
            hidden_states = [self.norm(state) for state in hidden_states]
            query = hidden_states[-1]
        reference_points = self._reference_points_for_attention(reference)
        out = {
            "hidden_states": query,
            "decoder_hidden_states": mx.stack(hidden_states, axis=0),
            "reference_points": reference_points,
            "reference_boxes": reference_points[:, :, 0, :],
            "spatial_shapes": mx.array(spatial_shapes, dtype=mx.int32),
        }
        if encoder_memory is not None and encoder_boxes is not None:
            out["encoder_hidden_states"] = encoder_memory
            out["encoder_boxes"] = encoder_boxes
        if capture_taps:
            if self_attention:
                out["self_attention"] = mx.stack(self_attention, axis=0)
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
        box_delta = self.bbox_embed(hidden)
        boxes = (
            _bbox_reparametrize(box_delta, decoder_out["reference_boxes"])
            if self.cfg.bbox_reparam
            else _sigmoid(box_delta)
        )
        data = {
            "logits": logits,
            "boxes": boxes,
            "hidden_states": hidden,
            "decoder_hidden_states": decoder_out["decoder_hidden_states"],
            "reference_points": decoder_out["reference_points"],
            "spatial_shapes": decoder_out["spatial_shapes"],
        }
        if "encoder_hidden_states" in decoder_out:
            data["encoder_hidden_states"] = decoder_out["encoder_hidden_states"]
            data["encoder_boxes"] = decoder_out["encoder_boxes"]
        return HeadOutput(data)
