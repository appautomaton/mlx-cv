"""Canonical text-prompt image inference for official SAM 3.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .sam31_checkpoint import load_sam3_weights
from .sam31_modeling import SAM3Model
from .sam31_session import _resize_bilinear_nhwc
from .tokenizer import SAM3Tokenizer
from .sam31_processor import SAM3VideoProcessor

__all__ = ["SAM3ImagePrediction", "SAM3Processor"]


@dataclass(frozen=True)
class SAM3ImagePrediction:
    boxes: np.ndarray
    scores: np.ndarray
    masks: np.ndarray
    query_indices: np.ndarray


class SAM3Processor:
    def __init__(
        self,
        model: SAM3Model,
        *,
        bpe_path: str | Path,
        score_threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = SAM3Tokenizer(bpe_path, clean="lower")
        self.score_threshold = float(score_threshold)
        self.processor = SAM3VideoProcessor()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        bpe_path: str | Path | None = None,
        score_threshold: float = 0.5,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool | None = None,
        token: str | bool | None = None,
    ) -> "SAM3Processor":
        from ...hub import resolve_pretrained

        resolved = resolve_pretrained(
            checkpoint,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        if resolved.is_dir():
            checkpoint = resolved / "model.safetensors"
            bpe_path = bpe_path or resolved / "bpe_simple_vocab_16e6.txt.gz"
        else:
            checkpoint = resolved
        if bpe_path is None:
            raise ValueError(
                "bpe_path is required when loading a direct SAM 3.1 checkpoint file"
            )
        if not Path(bpe_path).is_file():
            raise FileNotFoundError(f"SAM 3.1 BPE vocabulary is missing: {bpe_path}")
        model = load_sam3_weights(SAM3Model(), checkpoint)
        return cls(model, bpe_path=bpe_path, score_threshold=score_threshold)

    def predict(self, image: Any, text: str) -> SAM3ImagePrediction:
        processed, context = self.processor.preprocess([image])
        token_ids = self.tokenizer(text)
        attention_mask = token_ids != 0
        output = self.model(
            mx.array(processed["pixel_values"]),
            mx.array(token_ids),
            mx.array(attention_mask),
        )
        scores = (1.0 / (1.0 + mx.exp(-output.pred_logits))) * (
            1.0 / (1.0 + mx.exp(-output.presence_logits))
        )
        keep = np.flatnonzero(np.asarray(scores[0]) >= self.score_threshold)
        boxes = np.asarray(output.pred_boxes[0], dtype=np.float32)[keep]
        cx, cy, width, height = np.moveaxis(boxes, -1, 0)
        boxes = np.stack(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            axis=-1,
        ) * 1008.0
        boxes = context.frames[0].transform.invert_boxes(boxes, clip=True).astype(
            np.float32
        )
        raw_masks = mx.take(output.pred_masks[0], mx.array(keep), axis=0)
        if len(keep):
            resized = _resize_bilinear_nhwc(
                raw_masks[..., None], (1008, 1008)
            )[..., 0]
            mx.eval(resized)
            masks = np.stack(
                [
                    context.frames[0].transform.invert_mask(np.asarray(mask) > 0)
                    for mask in resized
                ]
            ).astype(bool)
        else:
            masks = np.zeros((0,) + context.frames[0].image_size, dtype=bool)
        return SAM3ImagePrediction(
            boxes=boxes,
            scores=np.asarray(scores[0], dtype=np.float32)[keep],
            masks=masks,
            query_indices=keep.astype(np.int32),
        )
