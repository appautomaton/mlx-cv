#!/usr/bin/env python3
"""Capture the official Transformers EoMT oracle into one temporary NPZ file.

This repository-only tool requires PyTorch and Transformers. It does not belong
to the installable ``mlx_cv`` package and must be run from ``.venv-torch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "tue-mps/eomt-dinov3-coco-panoptic-small-640"
SCHEMA_VERSION = 1
EXPECTED_BLOCK_COUNT = 12
PANOPTIC_THRESHOLD = 0.8
MASK_THRESHOLD = 0.5
OVERLAP_MASK_AREA_THRESHOLD = 0.8
COCO_STUFF_CLASSES = tuple(range(80, 133))


def _numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _block_outputs(outputs: Any, block_count: int) -> list[np.ndarray]:
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Transformers did not return hidden_states")
    # Transformers documents an embedding output followed by every layer output.
    # Taking the final N entries therefore excludes the embedding if it is present.
    if len(hidden_states) < block_count:
        raise RuntimeError(
            f"Transformers returned {len(hidden_states)} hidden states for "
            f"{block_count} model blocks"
        )
    blocks = list(hidden_states[-block_count:])
    return [_numpy(value) for value in blocks]


def capture_reference(model_source: str, image_path: Path, output_path: Path) -> None:
    """Run the official CPU reference and write the complete gate oracle."""

    try:
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
    except ImportError as exc:
        raise SystemExit(
            "EoMT reference capture requires torch, transformers, and Pillow in .venv-torch"
        ) from exc

    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(model_source)
    model = AutoModelForUniversalSegmentation.from_pretrained(model_source)
    model.eval()
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    panoptic = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
        threshold=PANOPTIC_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        overlap_mask_area_threshold=OVERLAP_MASK_AREA_THRESHOLD,
        stuff_classes=list(COCO_STUFF_CLASSES),
    )[0]

    block_count = int(model.config.num_hidden_layers)
    if block_count != EXPECTED_BLOCK_COUNT:
        raise RuntimeError(
            f"expected EoMT-DINOv3 small to have {EXPECTED_BLOCK_COUNT} blocks, "
            f"found {block_count}"
        )
    blocks = _block_outputs(outputs, block_count)
    segments_info = list(panoptic["segments_info"])
    image_array = np.asarray(image, dtype=np.uint8)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "image": image_array,
        "image_size": np.asarray(image_array.shape[:2], dtype=np.int64),
        "pixel_values": _numpy(inputs["pixel_values"]).astype(np.float32, copy=False),
        "masks": _numpy(outputs.masks_queries_logits).astype(np.float32, copy=False),
        "classes": _numpy(outputs.class_queries_logits).astype(np.float32, copy=False),
        "panoptic_map": _numpy(panoptic["segmentation"]).astype(np.int64, copy=False),
        "panoptic_segment_ids": np.asarray(
            [segment["id"] for segment in segments_info], dtype=np.int64
        ),
        "panoptic_label_ids": np.asarray(
            [segment["label_id"] for segment in segments_info], dtype=np.int64
        ),
        "segments_json": np.asarray(json.dumps(segments_info, sort_keys=True)),
    }
    for index, block in enumerate(blocks):
        arrays[f"block_{index:02d}"] = block.astype(np.float32, copy=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_source": model_source,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "block_count": block_count,
        "panoptic_threshold": PANOPTIC_THRESHOLD,
        "mask_threshold": MASK_THRESHOLD,
        "overlap_mask_area_threshold": OVERLAP_MASK_AREA_THRESHOLD,
        "stuff_classes": COCO_STUFF_CLASSES,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a complete Transformers EoMT parity oracle"
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()
    capture_reference(args.model, args.image, args.output)


if __name__ == "__main__":
    main()
