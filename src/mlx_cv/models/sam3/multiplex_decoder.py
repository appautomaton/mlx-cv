"""Object Multiplex mask decoder for SAM3 video propagation."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import SAM3MultiplexDecoderConfig
from .video_memory import _conv_nchw, _resize_nchw_nearest

__all__ = [
    "MLP",
    "SAM3MultiplexDecoderOutput",
    "SAM3MultiplexMaskDecoder",
]


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _sigmoid(x: mx.array) -> mx.array:
    return 1 / (1 + mx.exp(-x))


@dataclass
class SAM3MultiplexDecoderOutput:
    masks: mx.array
    iou_pred: mx.array
    object_score_logits: mx.array
    sam_tokens_out: mx.array
    taps: dict[str, mx.array]

    def __getitem__(self, key: str) -> mx.array:
        if key == "masks":
            return self.masks
        if key == "iou_pred":
            return self.iou_pred
        if key == "object_score_logits":
            return self.object_score_logits
        if key in {"sam_tokens_out", "mask_tokens_out"}:
            return self.sam_tokens_out
        raise KeyError(key)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        *,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("MLP requires at least one layer")
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        self.sigmoid_output = bool(sigmoid_output)

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = _relu(x)
        if self.sigmoid_output:
            x = _sigmoid(x)
        return x


class SAM3MultiplexMaskDecoder(nn.Module):
    """Bucket-space mask decoder for propagation frames."""

    def __init__(self, cfg: SAM3MultiplexDecoderConfig) -> None:
        super().__init__()
        if cfg.decode_mask_with_shared_tokens or cfg.decode_mask_attribute_with_shared_tokens:
            raise NotImplementedError("shared Object Multiplex token decoding is not ported in MLX yet")
        self.cfg = cfg
        self.slot_embed = mx.zeros((cfg.multiplex_count, cfg.hidden_dim), dtype=mx.float32)
        self.context_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.slot_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.mask_feature_proj = nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, 1)
        self.hyper = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=3)
        self.iou_prediction_head = MLP(
            cfg.hidden_dim,
            cfg.hidden_dim,
            1,
            num_layers=3,
            sigmoid_output=cfg.iou_prediction_use_sigmoid,
        )
        self.object_score_head = MLP(cfg.hidden_dim, cfg.hidden_dim, 1, num_layers=3)

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array | None = None,
        *,
        multimask_output: bool = False,
        high_res_features: tuple[mx.array, ...] | list[mx.array] | None = None,
        extra_per_object_embeddings: mx.array | None = None,
        capture_taps: bool = False,
    ) -> SAM3MultiplexDecoderOutput:
        if multimask_output or self.cfg.num_multimask_outputs != 0:
            raise NotImplementedError("SAM3 MLX multiplex decoder currently supports single-mask propagation")
        if len(image_embeddings.shape) != 4:
            raise ValueError(f"SAM3 multiplex decoder expects NCHW image embeddings, got {tuple(image_embeddings.shape)}")
        batch, channels, height, width = (int(v) for v in image_embeddings.shape)
        if channels != self.cfg.hidden_dim:
            raise ValueError(f"SAM3 multiplex decoder hidden width {channels} must equal {self.cfg.hidden_dim}")

        src = image_embeddings
        if image_pe is not None:
            if len(image_pe.shape) == 3:
                hw, pe_batch, pe_channels = (int(v) for v in image_pe.shape)
                if hw != height * width or pe_channels != channels:
                    raise ValueError(f"SAM3 multiplex image_pe shape is incompatible: {tuple(image_pe.shape)}")
                image_pe = mx.transpose(image_pe, (1, 2, 0)).reshape(pe_batch, pe_channels, height, width)
            if len(image_pe.shape) != 4 or int(image_pe.shape[1]) != channels:
                raise ValueError(f"SAM3 multiplex image_pe shape is incompatible: {tuple(image_pe.shape)}")
            if int(image_pe.shape[0]) == 1 and batch != 1:
                image_pe = mx.broadcast_to(image_pe, (batch, channels, int(image_pe.shape[2]), int(image_pe.shape[3])))
            elif int(image_pe.shape[0]) != batch:
                raise ValueError(f"SAM3 multiplex image_pe batch {int(image_pe.shape[0])} must be 1 or {batch}")
            src = src + _resize_nchw_nearest(image_pe, (int(src.shape[2]), int(src.shape[3])))

        pooled = mx.mean(src, axis=(2, 3))
        context = self.context_proj(pooled)
        tokens = mx.broadcast_to(
            context[:, None, :],
            (batch, self.cfg.multiplex_count, self.cfg.hidden_dim),
        )
        tokens = tokens + mx.broadcast_to(
            self.slot_embed[None, :, :],
            (batch, self.cfg.multiplex_count, self.cfg.hidden_dim),
        )
        if extra_per_object_embeddings is not None:
            if tuple(extra_per_object_embeddings.shape) != tuple(tokens.shape):
                raise ValueError(
                    "SAM3 multiplex extra_per_object_embeddings must have shape "
                    f"{tuple(tokens.shape)}, got {tuple(extra_per_object_embeddings.shape)}"
                )
            tokens = tokens + extra_per_object_embeddings
        tokens = self.slot_proj(tokens)

        mask_features = _resize_nchw_nearest(image_embeddings, self.cfg.low_res_mask_size)
        mask_features = _conv_nchw(self.mask_feature_proj, mask_features)
        if high_res_features:
            high = high_res_features[0]
            if tuple(high.shape[:2]) == tuple(mask_features.shape[:2]):
                mask_features = mask_features + _resize_nchw_nearest(high, self.cfg.low_res_mask_size)

        hyper = self.hyper(tokens)
        masks = mx.sum(mask_features[:, None, :, :, :] * hyper[:, :, :, None, None], axis=2)
        masks = masks[:, :, None, :, :]
        iou_pred = self.iou_prediction_head(tokens)
        object_score_logits = self.object_score_head(tokens) if self.cfg.pred_obj_scores else mx.ones_like(iou_pred) * 10.0
        sam_tokens_out = tokens[:, :, None, :]

        taps = {}
        if capture_taps:
            taps["mask_decoder.masks.bucket_space"] = masks
            taps["mask_decoder.iou_pred.bucket_space"] = iou_pred
        return SAM3MultiplexDecoderOutput(
            masks=masks,
            iou_pred=iou_pred,
            object_score_logits=object_score_logits,
            sam_tokens_out=sam_tokens_out,
            taps=taps,
        )
