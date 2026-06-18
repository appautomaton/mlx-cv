"""SAM 3.1 image-mode prompt normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.geometry import SpatialTransform
from ...heads.segmentation import SAM3EncodedGeometryPrompt, SAM3PCSPromptEncoder
from ...prompts import BoxPrompt, ExemplarPrompt, PointPrompt, TextPrompt

__all__ = [
    "SAM3PromptBundle",
    "SAM3PreparedPrompt",
    "normalize_sam3_prompt",
    "prepare_sam3_prompt",
]


_UNSUPPORTED_DICT_KEYS = {
    "mask",
    "masks",
    "mask_prompt",
    "mask_prompts",
    "video",
    "video_state",
    "tracker",
    "tracker_state",
    "memory",
}


@dataclass(frozen=True)
class SAM3PromptBundle:
    texts: tuple[str, ...] = ()
    boxes: BoxPrompt | None = None
    exemplar: ExemplarPrompt | None = None


@dataclass(frozen=True)
class SAM3PreparedPrompt:
    texts: tuple[str, ...]
    geometry: SAM3EncodedGeometryPrompt | None


def _merge(left: SAM3PromptBundle, right: SAM3PromptBundle) -> SAM3PromptBundle:
    if left.boxes is not None and right.boxes is not None:
        boxes = BoxPrompt(np.concatenate([left.boxes.boxes, right.boxes.boxes], axis=0))
    else:
        boxes = left.boxes or right.boxes
    if left.exemplar is not None and right.exemplar is not None:
        raise ValueError("SAM3 supports one exemplar prompt bundle at a time")
    return SAM3PromptBundle(
        texts=left.texts + right.texts,
        boxes=boxes,
        exemplar=left.exemplar or right.exemplar,
    )


def _from_dict(prompt: dict[str, Any]) -> SAM3PromptBundle:
    unsupported = sorted(_UNSUPPORTED_DICT_KEYS & set(prompt))
    if unsupported:
        raise NotImplementedError(f"SAM 3.1 image-mode path does not support {unsupported[0]!r} prompt state")
    if "points" in prompt or "point" in prompt:
        raise NotImplementedError("SAM 3.1 PCS grounding does not support point prompts; interactive points are deferred")

    bundle = SAM3PromptBundle()
    text = prompt.get("text", prompt.get("prompt"))
    if text is not None:
        texts = (text,) if isinstance(text, str) else tuple(str(t) for t in text)
        bundle = _merge(bundle, SAM3PromptBundle(texts=texts))

    boxes = prompt.get("boxes", prompt.get("box"))
    if boxes is not None:
        bundle = _merge(bundle, SAM3PromptBundle(boxes=BoxPrompt(boxes)))

    exemplar = prompt.get("exemplar")
    if exemplar is None and ("exemplar_image" in prompt or "exemplar_boxes" in prompt):
        exemplar = {
            "image": prompt.get("exemplar_image"),
            "boxes": prompt.get("exemplar_boxes"),
        }
    if exemplar is not None:
        if isinstance(exemplar, ExemplarPrompt):
            exemplar_prompt = exemplar
        elif isinstance(exemplar, dict):
            if exemplar.get("image") is None or exemplar.get("boxes") is None:
                raise ValueError("SAM3 exemplar prompts require image and boxes")
            exemplar_prompt = ExemplarPrompt(image=np.asarray(exemplar["image"]), boxes=exemplar["boxes"])
        else:
            raise TypeError(f"unsupported SAM3 exemplar prompt type: {type(exemplar).__name__}")
        bundle = _merge(bundle, SAM3PromptBundle(exemplar=exemplar_prompt))
    return bundle


def normalize_sam3_prompt(prompt: Any) -> SAM3PromptBundle:
    """Normalize string/dict/prompt inputs into the SAM3 image-mode prompt bundle."""
    if prompt is None:
        return SAM3PromptBundle()
    if isinstance(prompt, str):
        return SAM3PromptBundle(texts=(prompt,))
    if isinstance(prompt, TextPrompt):
        return SAM3PromptBundle(texts=(prompt.text,))
    if isinstance(prompt, BoxPrompt):
        return SAM3PromptBundle(boxes=prompt)
    if isinstance(prompt, ExemplarPrompt):
        return SAM3PromptBundle(exemplar=prompt)
    if isinstance(prompt, PointPrompt):
        raise NotImplementedError("SAM 3.1 PCS grounding does not support PointPrompt; interactive points are deferred")
    if isinstance(prompt, dict):
        return _from_dict(prompt)
    if isinstance(prompt, (list, tuple)):
        bundle = SAM3PromptBundle()
        for item in prompt:
            bundle = _merge(bundle, normalize_sam3_prompt(item))
        return bundle
    raise TypeError(f"unsupported SAM3 prompt type: {type(prompt).__name__}")


def prepare_sam3_prompt(
    prompt: Any,
    *,
    transform: SpatialTransform,
    model_size: tuple[int, int],
) -> SAM3PreparedPrompt:
    bundle = normalize_sam3_prompt(prompt)
    encoder = SAM3PCSPromptEncoder(model_size)
    geometry = None
    if bundle.boxes is not None:
        geometry = encoder.encode_boxes(bundle.boxes, transform)
    if bundle.exemplar is not None:
        encoded_exemplar = encoder.encode_exemplar(bundle.exemplar)
        if geometry is None:
            geometry = encoded_exemplar
        else:
            geometry = SAM3EncodedGeometryPrompt(
                boxes_cxcywh=geometry.boxes_cxcywh,
                box_labels=geometry.box_labels,
                exemplar_boxes_cxcywh=encoded_exemplar.exemplar_boxes_cxcywh,
                exemplar_labels=encoded_exemplar.exemplar_labels,
            )
    return SAM3PreparedPrompt(texts=bundle.texts, geometry=geometry)
