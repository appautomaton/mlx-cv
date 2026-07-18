"""One-time official SAM 3.1 detector-to-MLX layout conversion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = [
    "SAM31ConversionError",
    "convert_sam31_detector_state_dict",
    "convert_sam31_tracker_state_dict",
    "map_sam31_detector_key",
    "map_sam31_tracker_key",
]


class SAM31ConversionError(ValueError):
    pass


_BLOCK = re.compile(r"^backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.(.+)$")
_NECK = re.compile(
    r"^backbone\.vision_backbone\.(convs|interactive_convs|propagation_convs)\.(\d+)\.(.+)$"
)
_TEXT_BLOCK = re.compile(
    r"^backbone\.language_backbone\.encoder\.transformer\.resblocks\.(\d+)\.(.+)$"
)
_ENCODER_LAYER = re.compile(r"^transformer\.encoder\.layers\.(\d+)\.(.+)$")
_DECODER_LAYER = re.compile(r"^transformer\.decoder\.layers\.(\d+)\.(.+)$")
_GEOMETRY_LAYER = re.compile(r"^geometry_encoder\.encode\.(\d+)\.(.+)$")


def _split_qkv(prefix: str, suffix: str) -> tuple[str, ...] | None:
    if suffix in {"in_proj_weight", "qkv.weight"}:
        ending = "weight"
    elif suffix in {"in_proj_bias", "qkv.bias"}:
        ending = "bias"
    else:
        return None
    return tuple(f"{prefix}.{name}_proj.{ending}" for name in ("q", "k", "v"))


def _map_attention(prefix: str, suffix: str) -> tuple[str, ...] | None:
    split = _split_qkv(prefix, suffix)
    if split is not None:
        return split
    if suffix.startswith("out_proj."):
        return (f"{prefix}.o_proj.{suffix.removeprefix('out_proj.')}",)
    if suffix.startswith("proj."):
        return (f"{prefix}.o_proj.{suffix.removeprefix('proj.')}",)
    return None


def _neck_layer(index: int, suffix: str) -> str:
    if suffix.startswith("dconv_2x2_0."):
        return f"scale_layers.0.{suffix.removeprefix('dconv_2x2_0.')}"
    if suffix.startswith("dconv_2x2_1."):
        return f"scale_layers.2.{suffix.removeprefix('dconv_2x2_1.')}"
    if suffix.startswith("dconv_2x2."):
        return f"scale_layers.0.{suffix.removeprefix('dconv_2x2.')}"
    if suffix.startswith("conv_1x1."):
        return f"proj1.{suffix.removeprefix('conv_1x1.')}"
    if suffix.startswith("conv_3x3."):
        return f"proj2.{suffix.removeprefix('conv_3x3.')}"
    raise SAM31ConversionError(f"unsupported SAM 3.1 neck key at level {index}: {suffix}")


def map_sam31_detector_key(key: str) -> tuple[str, ...]:
    """Map one official source key to zero, one, or three final MLX keys."""

    if not key.startswith("detector."):
        return ()
    key = key.removeprefix("detector.")

    if key == "backbone.vision_backbone.trunk.pos_embed":
        return ("vision_encoder.backbone.embeddings.position_embeddings",)
    if key == "backbone.vision_backbone.trunk.patch_embed.proj.weight":
        return ("vision_encoder.backbone.embeddings.patch_embeddings.projection.weight",)
    if key.startswith("backbone.vision_backbone.trunk.ln_pre."):
        return (
            "vision_encoder.backbone.layer_norm."
            + key.removeprefix("backbone.vision_backbone.trunk.ln_pre."),
        )

    match = _BLOCK.match(key)
    if match:
        index, suffix = match.groups()
        prefix = f"vision_encoder.backbone.layers.{index}"
        if suffix == "attn.freqs_cis":
            return ()  # deterministic buffer, regenerated from the fixed config
        if suffix.startswith("norm1."):
            return (f"{prefix}.layer_norm1.{suffix.removeprefix('norm1.')}",)
        if suffix.startswith("norm2."):
            return (f"{prefix}.layer_norm2.{suffix.removeprefix('norm2.')}",)
        if suffix.startswith("attn."):
            mapped = _map_attention(f"{prefix}.attention", suffix.removeprefix("attn."))
            if mapped:
                return mapped
        if suffix.startswith("mlp.fc1.") or suffix.startswith("mlp.fc2."):
            return (f"{prefix}.{suffix}",)
        raise SAM31ConversionError(f"unsupported SAM 3.1 vision block key: {key}")

    match = _NECK.match(key)
    if match:
        family, index_text, suffix = match.groups()
        index = int(index_text)
        neck = {
            "convs": "neck",
            "interactive_convs": "interactive_neck",
            "propagation_convs": "propagation_neck",
        }[family]
        return (f"vision_encoder.{neck}.fpn_layers.{index}.{_neck_layer(index, suffix)}",)

    text_prefix = "backbone.language_backbone."
    if key == text_prefix + "encoder.positional_embedding":
        return ("text_encoder.text_model.embeddings.position_embedding.weight",)
    if key == text_prefix + "encoder.token_embedding.weight":
        return ("text_encoder.text_model.embeddings.token_embedding.weight",)
    if key == text_prefix + "encoder.text_projection":
        return ("text_encoder.text_projection.weight",)
    if key.startswith(text_prefix + "encoder.ln_final."):
        return (
            "text_encoder.text_model.final_layer_norm."
            + key.removeprefix(text_prefix + "encoder.ln_final."),
        )
    if key.startswith(text_prefix + "resizer."):
        return ("text_projection." + key.removeprefix(text_prefix + "resizer."),)

    match = _TEXT_BLOCK.match(key)
    if match:
        index, suffix = match.groups()
        prefix = f"text_encoder.text_model.encoder.layers.{index}"
        if suffix.startswith("attn."):
            attention_suffix = suffix.removeprefix("attn.")
            if attention_suffix.startswith("out_proj."):
                return (f"{prefix}.self_attn.{attention_suffix}",)
            mapped = _map_attention(f"{prefix}.self_attn", attention_suffix)
            if mapped:
                return mapped
        replacements = {
            "ln_1.": "layer_norm1.",
            "ln_2.": "layer_norm2.",
            "mlp.c_fc.": "mlp.fc1.",
            "mlp.c_proj.": "mlp.fc2.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"{prefix}.{target}{suffix.removeprefix(source)}",)
        raise SAM31ConversionError(f"unsupported SAM 3.1 text block key: {key}")

    match = _ENCODER_LAYER.match(key)
    if match:
        index, suffix = match.groups()
        prefix = f"detr_encoder.layers.{index}"
        for source, target in (
            ("self_attn.", "self_attn"),
            ("cross_attn_image.", "cross_attn"),
        ):
            if suffix.startswith(source):
                mapped = _map_attention(f"{prefix}.{target}", suffix.removeprefix(source))
                if mapped:
                    return mapped
        replacements = {
            "linear1.": "mlp.fc1.",
            "linear2.": "mlp.fc2.",
            "norm1.": "layer_norm1.",
            "norm2.": "layer_norm2.",
            "norm3.": "layer_norm3.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"{prefix}.{target}{suffix.removeprefix(source)}",)
        raise SAM31ConversionError(f"unsupported SAM 3.1 encoder key: {key}")

    match = _DECODER_LAYER.match(key)
    if match:
        index, suffix = match.groups()
        prefix = f"detr_decoder.layers.{index}"
        for source, target in (
            ("self_attn.", "self_attn"),
            ("ca_text.", "text_cross_attn"),
            ("cross_attn.", "vision_cross_attn"),
        ):
            if suffix.startswith(source):
                mapped = _map_attention(f"{prefix}.{target}", suffix.removeprefix(source))
                if mapped:
                    return mapped
        replacements = {
            "norm2.": "self_attn_layer_norm.",
            "catext_norm.": "text_cross_attn_layer_norm.",
            "norm1.": "vision_cross_attn_layer_norm.",
            "linear1.": "mlp.fc1.",
            "linear2.": "mlp.fc2.",
            "norm3.": "mlp_layer_norm.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"{prefix}.{target}{suffix.removeprefix(source)}",)
        raise SAM31ConversionError(f"unsupported SAM 3.1 decoder layer key: {key}")

    decoder_prefix = "transformer.decoder."
    decoder_replacements = {
        "norm.": "output_layer_norm.",
        "bbox_embed.layers.0.": "box_head.layer1.",
        "bbox_embed.layers.1.": "box_head.layer2.",
        "bbox_embed.layers.2.": "box_head.layer3.",
        "presence_token_head.layers.0.": "presence_head.layer1.",
        "presence_token_head.layers.1.": "presence_head.layer2.",
        "presence_token_head.layers.2.": "presence_head.layer3.",
        "presence_token_out_norm.": "presence_layer_norm.",
        "ref_point_head.layers.0.": "ref_point_head.layer1.",
        "ref_point_head.layers.1.": "ref_point_head.layer2.",
        "boxRPB_embed_x.layers.0.": "box_rpb_embed_x.layer1.",
        "boxRPB_embed_x.layers.1.": "box_rpb_embed_x.layer2.",
        "boxRPB_embed_y.layers.0.": "box_rpb_embed_y.layer1.",
        "boxRPB_embed_y.layers.1.": "box_rpb_embed_y.layer2.",
    }
    if key.startswith(decoder_prefix):
        suffix = key.removeprefix(decoder_prefix)
        if suffix in {"query_embed.weight", "reference_points.weight", "presence_token.weight"}:
            return (f"detr_decoder.{suffix}",)
        for source, target in decoder_replacements.items():
            if suffix.startswith(source):
                return (f"detr_decoder.{target}{suffix.removeprefix(source)}",)

    match = _GEOMETRY_LAYER.match(key)
    if match:
        index, suffix = match.groups()
        prefix = f"geometry_encoder.layers.{index}"
        for source, target in (
            ("self_attn.", "self_attn"),
            ("cross_attn_image.", "cross_attn"),
        ):
            if suffix.startswith(source):
                mapped = _map_attention(f"{prefix}.{target}", suffix.removeprefix(source))
                if mapped:
                    return mapped
        replacements = {
            "linear1.": "mlp.fc1.",
            "linear2.": "mlp.fc2.",
            "norm1.": "layer_norm1.",
            "norm2.": "layer_norm2.",
            "norm3.": "layer_norm3.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"{prefix}.{target}{suffix.removeprefix(source)}",)
        raise SAM31ConversionError(f"unsupported SAM 3.1 geometry layer key: {key}")

    geometry_prefix = "geometry_encoder."
    if key.startswith(geometry_prefix):
        suffix = key.removeprefix(geometry_prefix)
        replacements = {
            "norm.": "prompt_layer_norm.",
            "img_pre_norm.": "vision_layer_norm.",
            "encode_norm.": "output_layer_norm.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"geometry_encoder.{target}{suffix.removeprefix(source)}",)
        allowed = (
            "label_embed.",
            "cls_embed.",
            "points_direct_project.",
            "points_pool_project.",
            "points_pos_enc_project.",
            "boxes_direct_project.",
            "boxes_pool_project.",
            "boxes_pos_enc_project.",
            "final_proj.",
        )
        if suffix.startswith(allowed):
            return (f"geometry_encoder.{suffix}",)

    mask_prefix = "segmentation_head."
    if key.startswith(mask_prefix):
        suffix = key.removeprefix(mask_prefix)
        if suffix.startswith("cross_attend_prompt."):
            mapped = _map_attention(
                "mask_decoder.prompt_cross_attn",
                suffix.removeprefix("cross_attend_prompt."),
            )
            if mapped:
                return mapped
        replacements = {
            "pixel_decoder.": "pixel_decoder.",
            "mask_predictor.mask_embed.": "mask_embedder.",
            "cross_attn_norm.": "prompt_cross_attn_norm.",
            "semantic_seg_head.": "semantic_projection.",
            "instance_seg_head.": "instance_projection.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"mask_decoder.{target}{suffix.removeprefix(source)}",)

    scoring_prefix = "dot_prod_scoring."
    if key.startswith(scoring_prefix):
        suffix = key.removeprefix(scoring_prefix)
        replacements = {
            "prompt_mlp.layers.0.": "text_mlp.layer1.",
            "prompt_mlp.layers.1.": "text_mlp.layer2.",
            "prompt_mlp.out_norm.": "text_mlp_out_norm.",
            "prompt_proj.": "text_proj.",
            "hs_proj.": "query_proj.",
        }
        for source, target in replacements.items():
            if suffix.startswith(source):
                return (f"dot_product_scoring.{target}{suffix.removeprefix(source)}",)

    raise SAM31ConversionError(f"unsupported SAM 3.1 detector key: detector.{key}")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _convert_layout(source_key: str, target_key: str, value: np.ndarray) -> np.ndarray:
    if source_key.endswith("trunk.pos_embed"):
        value = value[:, 1:, :]
    if source_key.endswith("encoder.text_projection"):
        value = value.T
    if value.ndim == 4 and target_key.endswith(".weight"):
        if ".scale_layers." in target_key:
            value = value.transpose(1, 2, 3, 0)
        else:
            value = value.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(value)


def convert_sam31_detector_state_dict(
    state: Mapping[str, Any], *, dtype: np.dtype | None = None
) -> dict[str, np.ndarray]:
    """Convert all 1166 source detector entries into final MLX parameter arrays."""

    if isinstance(state.get("model"), Mapping):
        state = state["model"]
    converted: dict[str, np.ndarray] = {}
    source_count = 0
    for source_key, raw in state.items():
        if not source_key.startswith("detector."):
            continue
        source_count += 1
        targets = map_sam31_detector_key(source_key)
        if not targets:
            continue
        value = _as_numpy(raw)
        values = np.split(value, 3, axis=0) if len(targets) == 3 else (value,)
        for target_key, target_value in zip(targets, values, strict=True):
            if target_key in converted:
                raise SAM31ConversionError(f"duplicate converted key: {target_key}")
            array = _convert_layout(source_key, target_key, target_value)
            if dtype is not None and array.dtype.kind == "f":
                array = array.astype(dtype)
            converted[target_key] = array
    if source_count != 1166:
        raise SAM31ConversionError(
            f"expected 1166 SAM 3.1 detector tensors, found {source_count}"
        )
    if len(converted) != 1506:
        raise SAM31ConversionError(
            f"expected 1506 final MLX detector parameters, produced {len(converted)}"
        )
    return converted


_PROMPT_DOWNSCALE = {
    "0": "conv1",
    "1": "layer_norm1",
    "3": "conv2",
    "4": "layer_norm2",
    "6": "conv3",
}


def _map_tracker_mlp(key: str) -> str:
    """Map an official three-linear MLP into ``Sam3TrackerFeedForward``."""

    replacements = {
        ".layers.0.": ".proj_in.",
        ".layers.1.": ".layers.0.",
        ".layers.2.": ".proj_out.",
    }
    for source, target in replacements.items():
        if source in key:
            return key.replace(source, target)
    return key


def _map_tracker_decoder(key: str) -> str:
    key = key.replace(".out_proj.", ".o_proj.")
    key = key.replace(".norm_final_attn.", ".layer_norm_final_attn.")
    for index in range(1, 5):
        key = key.replace(f".norm{index}.", f".layer_norm{index}.")
    key = key.replace(".mlp.lin1.", ".mlp.proj_in.")
    key = key.replace(".mlp.lin2.", ".mlp.proj_out.")
    key = key.replace(".output_upscaling.0.", ".upscale_conv1.")
    key = key.replace(".output_upscaling.1.", ".upscale_layer_norm.")
    key = key.replace(".output_upscaling.3.", ".upscale_conv2.")
    if any(
        stem in key
        for stem in (
            ".output_hypernetworks_mlps.",
            ".iou_prediction_head.",
            ".pred_obj_score_head.",
        )
    ):
        key = _map_tracker_mlp(key)
    return key


def map_sam31_tracker_key(key: str) -> tuple[str, ...]:
    """Map one official multiplex-tracker key to its final MLX parameter key."""

    if not key.startswith("tracker.model."):
        return ()
    key = key.removeprefix("tracker.model.")

    prompt_prefix = "interactive_sam_prompt_encoder.mask_downscaling."
    if key.startswith(prompt_prefix):
        suffix = key.removeprefix(prompt_prefix)
        index, separator, rest = suffix.partition(".")
        target = _PROMPT_DOWNSCALE.get(index)
        if target is None or not separator:
            raise SAM31ConversionError(
                f"unsupported SAM 3.1 prompt downscaling key: {key}"
            )
        return (f"{prompt_prefix}{target}.{rest}",)

    if key.startswith(("interactive_sam_mask_decoder.", "sam_mask_decoder.")):
        return (_map_tracker_decoder(key),)
    if key.startswith(("obj_ptr_proj.", "interactive_obj_ptr_proj.")):
        return (_map_tracker_mlp(key),)

    # The remaining official tracker modules deliberately preserve source names.
    return (key,)


def _convert_tracker_layout(
    source_key: str, target_key: str, value: np.ndarray
) -> np.ndarray:
    if value.ndim != 4 or not target_key.endswith(".weight"):
        return np.ascontiguousarray(value)
    if ".output_upscaling." in source_key:
        # PyTorch ConvTranspose2d: [in, out, h, w]; MLX: [out, h, w, in].
        value = value.transpose(1, 2, 3, 0)
    else:
        # PyTorch Conv2d: [out, in/groups, h, w]; MLX: [out, h, w, in/groups].
        value = value.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(value)


def convert_sam31_tracker_state_dict(
    state: Mapping[str, Any], *, dtype: np.dtype | None = None
) -> dict[str, np.ndarray]:
    """Convert all 457 official tracker entries to final MLX name/layout arrays."""

    if isinstance(state.get("model"), Mapping):
        state = state["model"]
    converted: dict[str, np.ndarray] = {}
    source_count = 0
    for source_key, raw in state.items():
        if not source_key.startswith("tracker.model."):
            continue
        source_count += 1
        targets = map_sam31_tracker_key(source_key)
        if len(targets) != 1:
            raise SAM31ConversionError(
                f"tracker key must map one-to-one, got {source_key}: {targets}"
            )
        target_key = targets[0]
        if target_key in converted:
            raise SAM31ConversionError(f"duplicate converted key: {target_key}")
        array = _convert_tracker_layout(source_key, target_key, _as_numpy(raw))
        if dtype is not None and array.dtype.kind == "f":
            array = array.astype(dtype)
        converted[target_key] = array
    if source_count != 457:
        raise SAM31ConversionError(
            f"expected 457 SAM 3.1 tracker tensors, found {source_count}"
        )
    if len(converted) != 457:
        raise SAM31ConversionError(
            f"expected 457 final MLX tracker parameters, produced {len(converted)}"
        )
    return converted
