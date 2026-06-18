"""SAM3 video Object Multiplex tracker core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn

from .config import SAM3MultiplexDecoderConfig, SAM3VideoConfig, SAM3VideoMemoryConfig, SAM3VideoTrackerConfig
from .multiplex_decoder import SAM3MultiplexMaskDecoder
from .multiplex_state import SAM3MultiplexState
from .video_memory import (
    SAM3MemoryEncoder,
    SAM3MemoryEncoderOutput,
    SAM3MemoryMaskInput,
    _resize_nchw_nearest,
    bucket_features_to_object_space,
    build_multiplex_memory_mask_input,
    mask_logits_for_memory,
)

__all__ = ["SAM3VideoMultiplexTrackerCore", "SAM3VideoStageOutput"]


@dataclass
class SAM3VideoStageOutput:
    frame_index: int
    low_res_masks: mx.array
    high_res_masks: mx.array
    iou_pred: mx.array
    object_score_logits: mx.array
    obj_ptr: mx.array
    obj_ptr_mux: mx.array
    maskmem_features: mx.array | None = None
    maskmem_pos_enc: mx.array | None = None
    current_out: dict[str, Any] = field(default_factory=dict)
    taps: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class SAM3VideoMultiplexTrackerCore(nn.Module):
    """Shape-compatible MLX tracker core for SAM3 video inference tests."""

    def __init__(
        self,
        cfg: SAM3VideoTrackerConfig | SAM3VideoConfig | None = None,
        *,
        memory_cfg: SAM3VideoMemoryConfig | None = None,
        decoder_cfg: SAM3MultiplexDecoderConfig | None = None,
        memory_encoder: SAM3MemoryEncoder | None = None,
        mask_decoder: SAM3MultiplexMaskDecoder | None = None,
    ) -> None:
        super().__init__()
        if cfg is None:
            video_cfg = SAM3VideoConfig()
            tracker_cfg = video_cfg.tracker
            memory_cfg = memory_cfg or video_cfg.memory
            decoder_cfg = decoder_cfg or video_cfg.decoder
        elif isinstance(cfg, SAM3VideoConfig):
            tracker_cfg = cfg.tracker
            memory_cfg = memory_cfg or cfg.memory
            decoder_cfg = decoder_cfg or cfg.decoder
        else:
            tracker_cfg = cfg
            memory_cfg = memory_cfg or SAM3VideoMemoryConfig(
                hidden_dim=tracker_cfg.hidden_dim,
                image_size=tracker_cfg.image_size,
                feature_grid=tracker_cfg.feature_grid,
                multiplex_count=tracker_cfg.multiplex_count,
                condition_as_mask_input=tracker_cfg.condition_as_mask_input,
            )
            decoder_cfg = decoder_cfg or SAM3MultiplexDecoderConfig(
                hidden_dim=tracker_cfg.hidden_dim,
                multiplex_count=tracker_cfg.multiplex_count,
                high_res_mask_size=tracker_cfg.image_size,
            )
        self.cfg = tracker_cfg
        self.memory_encoder = memory_encoder or SAM3MemoryEncoder(memory_cfg)
        self.mask_decoder = mask_decoder or SAM3MultiplexMaskDecoder(decoder_cfg)
        self.obj_ptr_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)

    def _features_to_nchw(self, features: mx.array) -> mx.array:
        if len(features.shape) == 3:
            hw, batch, channels = (int(v) for v in features.shape)
            height, width = self.cfg.feature_grid
            if hw != height * width:
                raise ValueError(f"SAM3 tracker token count {hw} does not match grid {self.cfg.feature_grid}")
            if channels != self.cfg.hidden_dim:
                raise ValueError(f"SAM3 tracker feature width {channels} must equal {self.cfg.hidden_dim}")
            return mx.transpose(features, (1, 2, 0)).reshape(batch, channels, height, width)
        if len(features.shape) == 4:
            if int(features.shape[1]) == self.cfg.hidden_dim:
                return features
            if int(features.shape[-1]) == self.cfg.hidden_dim:
                return mx.transpose(features, (0, 3, 1, 2))
        raise ValueError(f"SAM3 tracker features must be (HW,B,C), NCHW, or NHWC; got {tuple(features.shape)}")

    def _expand_to_buckets(self, features: mx.array, multiplex_state: SAM3MultiplexState) -> mx.array:
        if int(features.shape[0]) == multiplex_state.num_buckets:
            return features
        if int(features.shape[0]) != 1:
            raise ValueError(
                f"SAM3 tracker expected image batch 1 or {multiplex_state.num_buckets} buckets, got {features.shape[0]}"
            )
        return mx.broadcast_to(
            features,
            (
                multiplex_state.num_buckets,
                int(features.shape[1]),
                int(features.shape[2]),
                int(features.shape[3]),
            ),
        )

    def _collect_prior_memory(self, output_dict: dict[str, Any], multiplex_state: SAM3MultiplexState) -> mx.array | None:
        memories = []
        for group in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for prev in output_dict.get(group, {}).values():
                feats = prev.get("maskmem_features") if isinstance(prev, dict) else None
                if feats is None:
                    continue
                if len(feats.shape) == 4 and int(feats.shape[0]) == multiplex_state.num_buckets:
                    memories.append(feats)
                elif len(feats.shape) == 4 and int(feats.shape[0]) == multiplex_state.total_valid_entries:
                    muxed = multiplex_state.mux(feats)
                    valid = multiplex_state.valid_mask.astype(feats.dtype)[:, :, None, None, None]
                    denom = mx.maximum(mx.sum(valid, axis=1), mx.array(1.0, dtype=feats.dtype))
                    memories.append(mx.sum(muxed * valid, axis=1) / denom)
        if not memories:
            return None
        return mx.mean(mx.stack(memories, axis=0), axis=0)

    def _prepare_memory_conditioned_features(
        self,
        image_features: mx.array,
        *,
        output_dict: dict[str, Any],
        multiplex_state: SAM3MultiplexState,
        capture_taps: bool,
    ) -> tuple[mx.array, dict[str, Any]]:
        image_space = self._features_to_nchw(image_features)
        bucket_space = self._expand_to_buckets(image_space, multiplex_state)
        prior = self._collect_prior_memory(output_dict, multiplex_state)
        if prior is not None:
            prior = _resize_nchw_nearest(prior, (int(bucket_space.shape[2]), int(bucket_space.shape[3])))
            bucket_space = bucket_space + prior * 0.1
        taps = {}
        if capture_taps:
            taps["tracker.backbone.top_features.image_space"] = image_space
            taps["tracker.memory_conditioned_features.bucket_space"] = bucket_space
        return bucket_space, taps

    def _default_high_res_features(
        self,
        high_res_features: tuple[mx.array, ...] | list[mx.array] | None,
        *,
        multiplex_state: SAM3MultiplexState,
        dtype,
    ) -> tuple[mx.array, mx.array]:
        if high_res_features is not None:
            if len(high_res_features) < 2:
                raise ValueError("SAM3 tracker high_res_features must contain two levels")
            return high_res_features[0], high_res_features[1]
        low_h, low_w = self.mask_decoder.cfg.low_res_mask_size
        return (
            mx.zeros((multiplex_state.num_buckets, self.cfg.hidden_dim, low_h, low_w), dtype=dtype),
            mx.zeros((multiplex_state.num_buckets, self.cfg.hidden_dim, max(1, low_h // 2), max(1, low_w // 2)), dtype=dtype),
        )

    def _object_pointer_from_masks(self, high_res_masks: mx.array) -> mx.array:
        summary = mx.mean(high_res_masks, axis=(2, 3))
        summary = mx.broadcast_to(summary, (int(summary.shape[0]), self.cfg.hidden_dim))
        return self.obj_ptr_proj(summary)

    def _use_mask_as_output(self, mask_inputs: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        if len(mask_inputs.shape) != 4 or int(mask_inputs.shape[1]) != 1:
            raise ValueError(f"SAM3 mask-as-output expects (O,1,H,W), got {tuple(mask_inputs.shape)}")
        high_res_masks = mask_inputs.astype(mx.float32) * 20.0 - 10.0
        low_res_masks = _resize_nchw_nearest(high_res_masks, self.mask_decoder.cfg.low_res_mask_size)
        iou_pred = mx.ones((int(mask_inputs.shape[0]), 1), dtype=high_res_masks.dtype)
        object_presence = (mx.max(mask_inputs.reshape(int(mask_inputs.shape[0]), -1), axis=1, keepdims=True) > 0).astype(
            high_res_masks.dtype
        )
        object_score_logits = object_presence * 20.0 - 10.0
        obj_ptr = self._object_pointer_from_masks(high_res_masks)
        return low_res_masks, high_res_masks, iou_pred, object_score_logits, obj_ptr

    def _encode_new_memory(
        self,
        memory_conditioned_features: mx.array,
        high_res_masks: mx.array,
        *,
        multiplex_state: SAM3MultiplexState,
        conditioning_objects: Iterable[int] | None,
        capture_taps: bool,
    ) -> tuple[SAM3MemoryEncoderOutput, SAM3MemoryMaskInput, dict[str, Any]]:
        mask_for_mem = mask_logits_for_memory(
            high_res_masks,
            apply_sigmoid=self.cfg.apply_sigmoid_to_mask_logits_for_mem_enc,
            scale=self.cfg.sigmoid_scale_for_mem_enc,
            bias=self.cfg.sigmoid_bias_for_mem_enc,
        )
        mask_input = build_multiplex_memory_mask_input(
            mask_for_mem,
            multiplex_state,
            condition_as_mask_input=self.cfg.condition_as_mask_input,
            conditioning_objects=conditioning_objects,
            condition_fg=self.cfg.condition_as_mask_input_fg,
            condition_bg=self.cfg.condition_as_mask_input_bg,
        )
        memory = self.memory_encoder(
            memory_conditioned_features,
            mask_input.encoder_input_channels,
            skip_mask_sigmoid=True,
            capture_taps=capture_taps,
        )
        taps = {}
        if capture_taps:
            taps["memory.mask_for_mem.object_space"] = mask_input.mask_for_mem_object_space
            taps["memory.mask_for_mem.mux_space"] = mask_input.mask_for_mem_mux_space
            if mask_input.condition_mask_channels is not None:
                taps["memory.condition_mask_channels"] = mask_input.condition_mask_channels
            taps["memory.encoder_input_channels"] = mask_input.encoder_input_channels
            taps["memory.features.bucket_space"] = memory.features
            taps["memory.features.object_space"] = bucket_features_to_object_space(memory.features, multiplex_state)
            taps["memory.pos_enc.object_space"] = bucket_features_to_object_space(memory.pos_enc, multiplex_state)
        return memory, mask_input, taps

    def track_step(
        self,
        *,
        frame_index: int,
        image_features: mx.array,
        image_pos_enc: mx.array | None = None,
        high_res_features: tuple[mx.array, ...] | list[mx.array] | None = None,
        mask_inputs: mx.array | None = None,
        is_init_cond_frame: bool = False,
        output_dict: dict[str, Any] | None = None,
        run_mem_encoder: bool = True,
        multiplex_state: SAM3MultiplexState,
        capture_taps: bool = False,
    ) -> SAM3VideoStageOutput:
        output_dict = output_dict if output_dict is not None else {}
        output_dict.setdefault("cond_frame_outputs", {})
        output_dict.setdefault("non_cond_frame_outputs", {})

        taps = multiplex_state.capture_taps() if capture_taps else {}
        memory_conditioned, prep_taps = self._prepare_memory_conditioned_features(
            image_features,
            output_dict=output_dict,
            multiplex_state=multiplex_state,
            capture_taps=capture_taps,
        )
        taps.update(prep_taps)
        high_res = self._default_high_res_features(
            high_res_features,
            multiplex_state=multiplex_state,
            dtype=memory_conditioned.dtype,
        )

        decoder_bucket = None
        if is_init_cond_frame or mask_inputs is not None:
            if mask_inputs is None:
                raise ValueError("SAM3 tracker initialization requires mask_inputs")
            low_res_masks, high_res_masks, iou_pred, object_score_logits, obj_ptr = self._use_mask_as_output(mask_inputs)
            conditioning_objects = range(multiplex_state.total_valid_entries)
        else:
            image_pe = None
            if image_pos_enc is not None:
                image_pe = self._features_to_nchw(image_pos_enc)
            decoder_bucket = self.mask_decoder(
                memory_conditioned,
                image_pe,
                high_res_features=high_res,
                multimask_output=False,
                capture_taps=capture_taps,
            )
            low_res_masks = multiplex_state.demux(decoder_bucket.masks)
            iou_pred = multiplex_state.demux(decoder_bucket.iou_pred)
            object_score_logits = multiplex_state.demux(decoder_bucket.object_score_logits)
            sam_tokens = multiplex_state.demux(decoder_bucket.sam_tokens_out)[:, 0, :]
            high_res_masks = _resize_nchw_nearest(low_res_masks, self.cfg.image_size)
            obj_ptr = self.obj_ptr_proj(sam_tokens)
            conditioning_objects = ()
            if capture_taps:
                taps.update(decoder_bucket.taps)
                taps["mask_decoder.iou_pred.bucket_space"] = decoder_bucket.iou_pred
                taps["mask_decoder.object_score_logits.bucket_space"] = decoder_bucket.object_score_logits

        obj_ptr_mux = multiplex_state.mux(obj_ptr)
        memory = None
        if run_mem_encoder:
            memory, _, memory_taps = self._encode_new_memory(
                memory_conditioned,
                high_res_masks,
                multiplex_state=multiplex_state,
                conditioning_objects=conditioning_objects,
                capture_taps=capture_taps,
            )
            taps.update(memory_taps)

        current_out: dict[str, Any] = {
            "frame_index": int(frame_index),
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "ious": iou_pred,
            "object_score_logits": object_score_logits,
            "obj_ptr": obj_ptr_mux,
        }
        if memory is not None:
            current_out["maskmem_features"] = memory.features
            current_out["maskmem_pos_enc"] = [memory.pos_enc]
        if self.cfg.save_image_features:
            current_out["image_features"] = image_features
            current_out["image_pos_enc"] = image_features if image_pos_enc is None else image_pos_enc

        target_group = "cond_frame_outputs" if is_init_cond_frame else "non_cond_frame_outputs"
        output_dict[target_group][int(frame_index)] = current_out

        if capture_taps:
            taps["mask_decoder.low_res_masks.object_space"] = low_res_masks
            taps["mask_decoder.high_res_masks.object_space"] = high_res_masks
            taps["mask_decoder.iou_pred.object_space"] = iou_pred
            taps["mask_decoder.object_score_logits.object_space"] = object_score_logits
            taps["tracker.obj_ptr.object_space"] = obj_ptr
            taps["tracker.obj_ptr.mux_space"] = obj_ptr_mux
            if self.cfg.save_image_features:
                taps["memory.image_features"] = image_features
                taps["memory.image_pos_enc"] = image_features if image_pos_enc is None else image_pos_enc

        return SAM3VideoStageOutput(
            frame_index=int(frame_index),
            low_res_masks=low_res_masks,
            high_res_masks=high_res_masks,
            iou_pred=iou_pred,
            object_score_logits=object_score_logits,
            obj_ptr=obj_ptr,
            obj_ptr_mux=obj_ptr_mux,
            maskmem_features=None if memory is None else memory.features,
            maskmem_pos_enc=None if memory is None else memory.pos_enc,
            current_out=current_out,
            taps=taps,
        )
