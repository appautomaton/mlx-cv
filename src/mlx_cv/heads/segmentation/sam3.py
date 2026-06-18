"""SAM 3.1 PCS geometry prompt encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.necks import SAM3FeaturePyramid
from ...core.features import HeadOutput
from ...core.geometry import SpatialTransform
from ...prompts import BoxPrompt, ExemplarPrompt, PointPrompt

__all__ = [
    "SAM3DecoderConfig",
    "SAM3EncodedGeometryPrompt",
    "SAM3ImageDecoder",
    "SAM3MaskDecoder",
    "SAM3PCSPromptEncoder",
]


@dataclass(frozen=True)
class SAM3DecoderConfig:
    hidden_dim: int = 256
    num_queries: int = 64
    num_layers: int = 4
    num_heads: int = 8
    num_classes: int = 2
    text_dim: int = 256

    def __post_init__(self) -> None:
        if min(
            self.hidden_dim,
            self.num_queries,
            self.num_layers,
            self.num_heads,
            self.num_classes,
            self.text_dim,
        ) <= 0:
            raise ValueError("SAM3 decoder dimensions must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("SAM3 hidden_dim must be divisible by num_heads")


@dataclass(frozen=True)
class SAM3EncodedGeometryPrompt:
    boxes_cxcywh: np.ndarray
    box_labels: np.ndarray
    exemplar_boxes_cxcywh: np.ndarray | None = None
    exemplar_labels: np.ndarray | None = None


def _sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-x))


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        return self.layers[2](_relu(self.layers[1](_relu(self.layers[0](x)))))


def _empty_summary(batch: int, width: int, dtype) -> mx.array:
    return mx.zeros((batch, width), dtype=dtype)


def _text_summary(text_output, *, batch: int, text_dim: int, dtype) -> mx.array:
    if text_output is None:
        return _empty_summary(batch, text_dim, dtype)
    features = getattr(text_output, "language_features", text_output)
    features = mx.array(features)
    if len(features.shape) == 2:
        if features.shape[0] == batch:
            return features
        if features.shape[0] == 1:
            return mx.broadcast_to(features, (batch, features.shape[-1]))
        raise ValueError(f"SAM3 text batch {features.shape[0]} does not match image batch {batch}")
    if len(features.shape) != 3:
        raise ValueError(f"SAM3 text features must have rank 2 or 3, got {features.shape}")
    if features.shape[1] == batch:
        seq_first = features
    elif features.shape[0] == batch:
        seq_first = features.transpose(1, 0, 2)
    else:
        raise ValueError(f"SAM3 text features shape {features.shape} is incompatible with image batch {batch}")
    if seq_first.shape[-1] != text_dim:
        raise ValueError(f"SAM3 text feature width {seq_first.shape[-1]} does not match text_dim {text_dim}")

    mask = getattr(text_output, "language_mask", None)
    if mask is None:
        return mx.mean(seq_first, axis=0)
    mask = mx.array(mask, dtype=mx.bool_)
    if mask.shape[0] != batch or mask.shape[1] != seq_first.shape[0]:
        raise ValueError(f"SAM3 language mask shape {mask.shape} is incompatible with text features {seq_first.shape}")
    valid = (~mask).transpose(1, 0)[:, :, None].astype(seq_first.dtype)
    counts = mx.maximum(mx.sum(valid, axis=0), mx.array(1.0, dtype=seq_first.dtype))
    return mx.sum(seq_first * valid, axis=0) / counts


def _geometry_summary(prompt: SAM3EncodedGeometryPrompt | None, *, batch: int, dtype) -> mx.array:
    if prompt is None:
        return mx.zeros((batch, 4), dtype=dtype)
    parts = []
    if prompt.boxes_cxcywh.size:
        parts.append(prompt.boxes_cxcywh)
    if prompt.exemplar_boxes_cxcywh is not None and prompt.exemplar_boxes_cxcywh.size:
        parts.append(prompt.exemplar_boxes_cxcywh)
    if not parts:
        return mx.zeros((batch, 4), dtype=dtype)
    merged = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    summary = mx.array(np.mean(merged, axis=0, keepdims=True), dtype=dtype)
    return mx.broadcast_to(summary, (batch, 4))


class SAM3DecoderLayer(nn.Module):
    def __init__(self, cfg: SAM3DecoderConfig) -> None:
        super().__init__()
        self.memory_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.prompt_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.ffn1 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4)
        self.ffn2 = nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def __call__(self, query: mx.array, memory_context: mx.array) -> mx.array:
        query = self.norm1(query + self.memory_proj(memory_context)[:, None, :])
        query = self.norm2(query + self.ffn2(_relu(self.ffn1(self.prompt_proj(query)))))
        return query


class SAM3ImageDecoder(nn.Module):
    """Prompt-conditioned image-mode decoder over a SAM3 feature pyramid."""

    def __init__(self, cfg: SAM3DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.query_embed = mx.zeros((cfg.num_queries, cfg.hidden_dim))
        self.text_projection = nn.Linear(cfg.text_dim, cfg.hidden_dim)
        self.geometry_projection = nn.Linear(4, cfg.hidden_dim)
        self.layers = [SAM3DecoderLayer(cfg) for _ in range(cfg.num_layers)]

    def __call__(
        self,
        pyramid: SAM3FeaturePyramid,
        *,
        prompt: SAM3EncodedGeometryPrompt | None = None,
        text_output=None,
        capture_taps: bool = False,
    ) -> dict[str, mx.array]:
        if not pyramid.levels:
            raise ValueError("SAM3 decoder requires at least one feature level")
        for level in pyramid.levels:
            if level.data.shape[-1] != self.cfg.hidden_dim:
                raise ValueError(
                    f"SAM3 pyramid channels {level.data.shape[-1]} must equal hidden_dim {self.cfg.hidden_dim}"
                )

        memory_parts = []
        shapes = []
        for level in pyramid.levels:
            batch, height, width, channels = level.data.shape
            memory_parts.append(level.data.reshape(batch, height * width, channels))
            shapes.append((height, width))
        memory = mx.concatenate(memory_parts, axis=1)
        batch = memory.shape[0]
        dtype = memory.dtype
        query = mx.broadcast_to(self.query_embed[None], (batch, self.cfg.num_queries, self.cfg.hidden_dim))
        text = self.text_projection(_text_summary(text_output, batch=batch, text_dim=self.cfg.text_dim, dtype=dtype))
        geometry = self.geometry_projection(_geometry_summary(prompt, batch=batch, dtype=dtype))
        query = query + text[:, None, :].astype(query.dtype) + geometry[:, None, :].astype(query.dtype)

        memory_context = mx.mean(memory, axis=1)
        hidden_states = []
        for layer in self.layers:
            query = layer(query, memory_context)
            hidden_states.append(query)
        out = {
            "hidden_states": query,
            "decoder_hidden_states": mx.stack(hidden_states, axis=0),
            "spatial_shapes": mx.array(np.array(shapes, dtype=np.int32)),
            "text_summary": text,
            "geometry_summary": geometry,
        }
        if capture_taps:
            out["memory"] = memory
        return out


class SAM3MaskDecoder(nn.Module):
    """Project decoder tokens and image features into mask logits and grounding metadata."""

    def __init__(self, cfg: SAM3DecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_feature_proj = nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.mask_token_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.score_embed = nn.Linear(cfg.hidden_dim, 1)
        self.class_embed = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.box_embed = _MLP(cfg.hidden_dim, cfg.hidden_dim, 4)

    def __call__(self, decoder_out: dict[str, mx.array], pyramid: SAM3FeaturePyramid) -> HeadOutput:
        hidden = decoder_out["hidden_states"]
        mask_features = self.mask_feature_proj(pyramid.levels[0].data)
        mask_tokens = self.mask_token_proj(hidden)
        mask_logits = mx.sum(mask_features[:, None, :, :, :] * mask_tokens[:, :, None, None, :], axis=-1)
        object_scores = _sigmoid(self.score_embed(hidden)[..., 0])
        class_logits = self.class_embed(hidden)
        labels = mx.argmax(class_logits, axis=-1)
        boxes = _sigmoid(self.box_embed(hidden))
        return HeadOutput(
            {
                "mask_logits": mask_logits,
                "object_scores": object_scores,
                "class_logits": class_logits,
                "labels": labels,
                "boxes": boxes,
                "hidden_states": hidden,
                "decoder_hidden_states": decoder_out["decoder_hidden_states"],
                "spatial_shapes": decoder_out["spatial_shapes"],
                "text_summary": decoder_out["text_summary"],
                "geometry_summary": decoder_out["geometry_summary"],
            }
        )


def _xyxy_to_cxcywh_norm(boxes: np.ndarray, *, model_size: tuple[int, int]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h, w = model_size
    x0, y0, x1, y1 = boxes.T if len(boxes) else [np.array([], dtype=np.float64)] * 4
    out = np.stack(
        [
            (x0 + x1) * 0.5 / w,
            (y0 + y1) * 0.5 / h,
            (x1 - x0) / w,
            (y1 - y0) / h,
        ],
        axis=-1,
    )
    return np.clip(out, 0.0, 1.0)


class SAM3PCSPromptEncoder:
    """Encode SAM3 image-mode PCS boxes into normalized model-space prompts."""

    def __init__(self, model_size: tuple[int, int]) -> None:
        h, w = int(model_size[0]), int(model_size[1])
        if h <= 0 or w <= 0:
            raise ValueError("SAM3 PCS prompt encoder model_size must be positive")
        self.model_size = (h, w)

    def encode_boxes(
        self,
        prompt: BoxPrompt,
        transform: SpatialTransform,
        *,
        labels: np.ndarray | list[bool] | list[int] | None = None,
    ) -> SAM3EncodedGeometryPrompt:
        boxes = transform.apply_boxes(prompt.boxes)
        encoded = _xyxy_to_cxcywh_norm(boxes, model_size=self.model_size)
        if labels is None:
            label_arr = np.ones((encoded.shape[0],), dtype=np.bool_)
        else:
            label_arr = np.asarray(labels, dtype=np.bool_).reshape(-1)
            if len(label_arr) != encoded.shape[0]:
                raise ValueError(f"SAM3 box labels length {len(label_arr)} does not match {encoded.shape[0]} boxes")
        return SAM3EncodedGeometryPrompt(encoded, label_arr)

    def encode_exemplar(
        self,
        prompt: ExemplarPrompt,
        *,
        exemplar_transform: SpatialTransform | None = None,
    ) -> SAM3EncodedGeometryPrompt:
        if exemplar_transform is None:
            exemplar_transform = SpatialTransform.resize(prompt.image.shape[:2], self.model_size)
        boxes = exemplar_transform.apply_boxes(prompt.boxes)
        encoded = _xyxy_to_cxcywh_norm(boxes, model_size=self.model_size)
        labels = np.ones((encoded.shape[0],), dtype=np.bool_)
        return SAM3EncodedGeometryPrompt(
            boxes_cxcywh=np.zeros((0, 4), dtype=np.float64),
            box_labels=np.zeros((0,), dtype=np.bool_),
            exemplar_boxes_cxcywh=encoded,
            exemplar_labels=labels,
        )

    def encode(self, prompt, transform: SpatialTransform) -> SAM3EncodedGeometryPrompt:
        if isinstance(prompt, BoxPrompt):
            return self.encode_boxes(prompt, transform)
        if isinstance(prompt, ExemplarPrompt):
            return self.encode_exemplar(prompt)
        if isinstance(prompt, PointPrompt):
            raise NotImplementedError("SAM 3.1 PCS grounding does not support PointPrompt; interactive points are deferred")
        raise TypeError(f"unsupported SAM3 geometry prompt type: {type(prompt).__name__}")
