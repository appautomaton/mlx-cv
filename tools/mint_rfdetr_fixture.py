"""Mint RF-DETR parity fixtures out-of-band.

This script may import torch and the RF-DETR reference checkout. Those imports are
never package runtime dependencies. The committed MLX tests use fixed tiny inputs
from ``mlx_cv.parity.fixtures`` and compare against the saved reference outputs.

Usage in a throwaway torch env:

    PYTHONPATH=references/rf-detr/src python tools/mint_rfdetr_fixture.py --target ms-deform

Usage for the committed tiny detector fixture:

    uv run python tools/mint_rfdetr_fixture.py --target detector
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures"

sys.path.insert(0, str(REPO / "src"))

from mlx_cv.parity import ParityCase, save_case  # noqa: E402
from mlx_cv.parity.fixtures import (  # noqa: E402
    RFDETR_FIXTURE_CONFIG,
    RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG,
    rfdetr_fixed_input,
    rfdetr_tap_order,
    rfdetr_ms_deform_attn_fixed_inputs,
)


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _atomic_savez(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def mint_ms_deform_attn() -> None:
    try:
        import torch
        from rfdetr.models.ops.functions import ms_deform_attn_core_pytorch
    except Exception as exc:  # pragma: no cover - mint environment only.
        raise SystemExit(
            "RF-DETR deformable-attention minting requires torch and the RF-DETR "
            "reference package on PYTHONPATH. Do not add them as runtime deps. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    inputs = rfdetr_ms_deform_attn_fixed_inputs()
    with torch.no_grad():
        expected = ms_deform_attn_core_pytorch(
            torch.from_numpy(inputs["value"]),
            torch.from_numpy(inputs["value_spatial_shapes"]),
            torch.from_numpy(inputs["sampling_locations"]),
            torch.from_numpy(inputs["attention_weights"]),
            value_spatial_shapes_hw=[
                tuple(int(x) for x in row) for row in inputs["value_spatial_shapes"].tolist()
            ],
        ).numpy()

    out = dict(inputs)
    out["expected"] = expected.astype(np.float32)
    out["__fixture_name__"] = np.asarray(RFDETR_MS_DEFORM_ATTN_FIXTURE_CONFIG["name"])
    path = FIXTURE_DIR / "rfdetr_ms_deform_attn_tiny_fixture.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"wrote {path}")


def _model_config():
    from mlx_cv.backbones.vision.dinov2 import DINOv2Config
    from mlx_cv.heads.detection import RFDETRDecoderConfig
    from mlx_cv.models.rfdetr import RFDETRConfig

    cfg = RFDETR_FIXTURE_CONFIG
    return RFDETRConfig(
        backbone=DINOv2Config(**cfg["backbone"]),
        out_layers=tuple(cfg["out_layers"]),
        projector_out_channels=cfg["projector_out_channels"],
        projector_scale_factors=tuple(cfg["projector_scale_factors"]),
        decoder=RFDETRDecoderConfig(**cfg["decoder"]),
    )


def _reference_postprocess(raw: dict[str, np.ndarray], target_size: tuple[int, int], num_select: int) -> dict:
    import torch
    from rfdetr.models.postprocess import PostProcess

    post = PostProcess(num_select=num_select)
    outputs = {
        "pred_logits": torch.from_numpy(raw["logits"]),
        "pred_boxes": torch.from_numpy(raw["boxes"]),
    }
    result = post(outputs, torch.tensor([target_size]))[0]
    return {
        "boxes": _np(result["boxes"]),
        "scores": _np(result["scores"]),
        "class_ids": _np(result["labels"]).astype(np.int64),
    }


def mint_detector(*, reference_postprocess: bool = False) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_cv.core.geometry import SpatialTransform
    from mlx_cv.models.rfdetr import RFDETRModel, RFDETRProcessor, RFDETRProcessorConfig

    cfg = RFDETR_FIXTURE_CONFIG
    with mx.stream(mx.cpu):
        mx.random.seed(int(cfg["seed"]))
        model = RFDETRModel(_model_config())
        mx.eval(model.parameters())

        x_np = rfdetr_fixed_input()
        raw = model(mx.array(x_np), capture_taps=True)
        mx.eval(raw.data)
    raw_np = {
        "logits": _np(raw["logits"]),
        "boxes": _np(raw["boxes"]),
    }

    target_size = tuple(int(x) for x in cfg["image_size"])
    processor = RFDETRProcessor(
        RFDETRProcessorConfig(
            image_size=target_size,
            top_k=int(cfg["num_select"]),
            labels=tuple(cfg["labels"]),
        )
    )
    if reference_postprocess:
        post = _reference_postprocess(raw_np, target_size, int(cfg["num_select"]))
    else:
        result = processor.postprocess(raw, SpatialTransform.identity(target_size))
        post = {
            "boxes": result.detections.boxes,
            "scores": result.detections.scores,
            "class_ids": result.detections.class_ids,
        }

    taps = {key: _np(value) for key, value in raw["taps"].items()}
    taps["result.boxes"] = np.asarray(post["boxes"], dtype=np.float64)
    taps["result.scores"] = np.asarray(post["scores"], dtype=np.float64)
    taps["result.class_ids"] = np.asarray(post["class_ids"], dtype=np.int64)
    expected_order = rfdetr_tap_order(
        num_levels=len(cfg["projector_scale_factors"]),
        num_layers=cfg["decoder"]["num_layers"],
    )
    if list(taps) != expected_order:
        raise RuntimeError(f"unexpected RF-DETR tap order: {list(taps)} != {expected_order}")

    expected = {
        "logits": raw_np["logits"],
        "boxes": raw_np["boxes"],
        "result_boxes": taps["result.boxes"],
        "scores": taps["result.scores"],
        "class_ids": taps["result.class_ids"],
    }
    case = ParityCase(name=cfg["name"], inputs={"x": x_np}, expected=expected, taps=taps)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{cfg['name']}.npz"
    weights_path = FIXTURE_DIR / f"{cfg['name']}_weights.npz"
    tmp_fixture = fixture_path.with_name(f"{fixture_path.name}.tmp.npz")
    save_case(case, tmp_fixture)
    tmp_fixture.replace(fixture_path)

    weights = {key: _np(value) for key, value in tree_flatten(model.parameters())}
    weights["__config_json__"] = np.asarray(json.dumps(cfg, sort_keys=True))
    _atomic_savez(weights_path, **weights)
    print(f"fixture -> {fixture_path} ({fixture_path.stat().st_size / 1e6:.2f} MB)")
    print(f"weights -> {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print(f"postprocess -> {'reference' if reference_postprocess else 'local'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["ms-deform", "detector", "all"], default="ms-deform")
    parser.add_argument(
        "--reference-postprocess",
        action="store_true",
        help="Use RF-DETR reference PostProcess for detector result taps. Requires torch/reference PYTHONPATH.",
    )
    args = parser.parse_args()

    if args.target in ("ms-deform", "all"):
        sys.path.insert(0, str(REPO / "references" / "rf-detr" / "src"))
        mint_ms_deform_attn()
    if args.target in ("detector", "all"):
        if args.reference_postprocess:
            sys.path.insert(0, str(REPO / "references" / "rf-detr" / "src"))
        mint_detector(reference_postprocess=args.reference_postprocess)


if __name__ == "__main__":
    main()
