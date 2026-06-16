"""RF-DETR Nano upstream reference capture.

This module may import PyTorch, torchvision, supervision, and the RF-DETR
reference checkout. It must stay under ``tools/`` and tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from rfdetr_checkpoint import CheckpointInfo, print_checkpoint_evidence, resolve_rfdetr_nano_checkpoint  # noqa: E402
from mlx_cv.parity.fixtures import rfdetr_fixed_image  # noqa: E402


class ReferenceDependencyError(RuntimeError):
    """Raised when the RF-DETR reference environment is unavailable."""


@dataclass(frozen=True)
class RFDETRReferenceCapture:
    checkpoint: CheckpointInfo
    input_image: np.ndarray
    raw_logits: np.ndarray
    raw_boxes: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    tap_gaps: tuple[str, ...]

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "input_image": self.input_image,
            "raw_logits": self.raw_logits,
            "raw_boxes": self.raw_boxes,
            "boxes": self.boxes,
            "scores": self.scores,
            "class_ids": self.class_ids,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint.path),
            "checkpoint_md5": self.checkpoint.md5,
            "input_shape": list(self.input_image.shape),
            "raw_logits_shape": list(self.raw_logits.shape),
            "raw_boxes_shape": list(self.raw_boxes.shape),
            "boxes_shape": list(self.boxes.shape),
            "scores_shape": list(self.scores.shape),
            "class_ids_shape": list(self.class_ids.shape),
            "tap_gaps": list(self.tap_gaps),
        }


def _np(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _import_reference():
    try:
        import torch
        import torchvision.transforms.functional as tvf
        from PIL import Image
        from rfdetr import RFDETRNano
    except Exception as exc:  # pragma: no cover - depends on optional reference env.
        raise ReferenceDependencyError(
            "RF-DETR upstream capture requires torch, torchvision, PIL, and "
            "PYTHONPATH=references/rf-detr/src."
        ) from exc
    return torch, tvf, Image, RFDETRNano


def capture_rfdetr_nano_reference(checkpoint: CheckpointInfo) -> RFDETRReferenceCapture:
    torch, tvf, Image, RFDETRNano = _import_reference()

    image = rfdetr_fixed_image()
    pil_image = Image.fromarray(image)
    model = RFDETRNano(pretrain_weights=str(checkpoint.path), device="cpu")
    model.model.model.eval()

    tensor = tvf.to_tensor(pil_image)
    resized = tvf.resize(tensor, [model.model.resolution, model.model.resolution])
    normalized = tvf.normalize(resized, model.means, model.stds)
    batch = normalized.unsqueeze(0)

    with torch.no_grad():
        raw = model.model.model(batch)
        target_sizes = torch.tensor([image.shape[:2]], device=model.model.device)
        result = model.model.postprocess(raw, target_sizes=target_sizes)[0]

    tap_gaps = (
        "RF-DETR reference exposes final raw logits/boxes through the public model output; "
        "stable intermediate taps are not exposed without invasive hooks.",
    )
    return RFDETRReferenceCapture(
        checkpoint=checkpoint,
        input_image=image,
        raw_logits=_np(raw["pred_logits"]),
        raw_boxes=_np(raw["pred_boxes"]),
        boxes=_np(result["boxes"]),
        scores=_np(result["scores"]),
        class_ids=_np(result["labels"]).astype(np.int64),
        tap_gaps=tap_gaps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RF-DETR Nano upstream reference capture.")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--save-npz", type=Path, default=None)
    args = parser.parse_args(argv)

    checkpoint = resolve_rfdetr_nano_checkpoint(cache_root=args.cache_root, required=True)
    if checkpoint is None:  # pragma: no cover - required=True raises instead.
        raise RuntimeError("required checkpoint unexpectedly resolved to None")
    print_checkpoint_evidence(checkpoint)
    capture = capture_rfdetr_nano_reference(checkpoint)
    print(json.dumps(capture.summary(), indent=2, sort_keys=True))
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_npz, **capture.as_arrays())
        print(f"saved {args.save_npz}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by direct CLI use.
    raise SystemExit(main())
