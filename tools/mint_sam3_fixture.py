"""Mint SAM3 image-mode parity fixtures.

The committed tiny fixtures are generated from the local MLX image-mode oracle
because this workspace does not ship a SAM3 reference checkout with stable public
tap points. The fixture metadata records that boundary; the runtime package still
has no PyTorch/Transformers dependency.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures"

sys.path.insert(0, str(REPO / "src"))

from mlx_cv.parity import ParityCase, save_case  # noqa: E402
from mlx_cv.parity.fixtures import (  # noqa: E402
    SAM3_FIXTURE_CONFIG,
    sam3_fixed_image,
    sam3_pcs_prompt,
    sam3_tap_order,
    sam3_text_prompt,
)


def _np(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _atomic_savez(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _model_config():
    from mlx_cv.heads.segmentation import SAM3DecoderConfig
    from mlx_cv.models.sam3 import SAM3Config, SAM3ImageBackboneConfig, SAM3TextConfig, SAM3Tokenizer

    cfg = SAM3_FIXTURE_CONFIG
    tokenizer = SAM3Tokenizer(context_length=int(cfg["text"]["context_length"]))
    text_cfg = dict(cfg["text"])
    text_cfg["vocab_size"] = tokenizer.vocab_size
    return SAM3Config(
        image=SAM3ImageBackboneConfig(**cfg["image"]),
        text=SAM3TextConfig(**text_cfg),
        decoder=SAM3DecoderConfig(**cfg["decoder"]),
    )


def _processor():
    from mlx_cv.models.sam3 import SAM3Processor, SAM3ProcessorConfig

    cfg = SAM3_FIXTURE_CONFIG
    return SAM3Processor(
        SAM3ProcessorConfig(
            image_size=tuple(cfg["image_size"]),
            top_k=int(cfg["num_select"]),
            labels=tuple(cfg["labels"]),
        )
    )


def _ordered_taps(raw, result, *, include_text: bool, include_geometry: bool) -> dict[str, np.ndarray]:
    taps = {key: _np(value) for key, value in raw["taps"].items()}
    taps["result.masks"] = np.asarray(result.masks.data, dtype=np.float32)
    taps["result.boxes"] = np.asarray(result.detections.boxes, dtype=np.float32)
    taps["result.scores"] = np.asarray(result.detections.scores, dtype=np.float32)
    taps["result.class_ids"] = np.asarray(result.detections.class_ids, dtype=np.int64)
    expected = sam3_tap_order(include_text=include_text, include_geometry=include_geometry)
    if list(taps) != expected:
        raise RuntimeError(f"unexpected SAM3 tap order: {list(taps)} != {expected}")
    return taps


def _case(model, *, name: str, prompt, include_text: bool, include_geometry: bool) -> ParityCase:
    import mlx.core as mx

    processor = _processor()
    image = sam3_fixed_image()
    model_inputs, ctx = processor.preprocess({"image": image, "prompt": prompt})
    raw = model(model_inputs["pixel_values"], model_inputs["prompt"], capture_taps=True)
    result = processor.postprocess(raw, ctx)
    mx.eval(raw["mask_logits"], raw["object_scores"], raw["boxes"])

    taps = _ordered_taps(raw, result, include_text=include_text, include_geometry=include_geometry)
    expected = {
        "mask_logits": _np(raw["mask_logits"]),
        "object_scores": _np(raw["object_scores"]),
        "boxes": _np(raw["boxes"]),
        "result_masks": taps["result.masks"],
        "result_boxes": taps["result.boxes"],
        "scores": taps["result.scores"],
        "class_ids": taps["result.class_ids"],
    }
    return ParityCase(name=name, inputs={"image": image}, expected=expected, taps=taps)


def main() -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_cv.models.sam3 import SAM3Model, SAM3Tokenizer

    cfg = SAM3_FIXTURE_CONFIG
    with mx.stream(mx.cpu):
        mx.random.seed(int(cfg["seed"]))
        tokenizer = SAM3Tokenizer(context_length=int(cfg["text"]["context_length"]))
        model = SAM3Model(_model_config(), tokenizer=tokenizer)
        mx.eval(model.parameters())

        text_case = _case(
            model,
            name=f"{cfg['name']}_text",
            prompt=sam3_text_prompt(),
            include_text=True,
            include_geometry=False,
        )
        pcs_case = _case(
            model,
            name=f"{cfg['name']}_pcs",
            prompt=sam3_pcs_prompt(),
            include_text=False,
            include_geometry=True,
        )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for case in (text_case, pcs_case):
        fixture_path = FIXTURE_DIR / f"{case.name}.npz"
        tmp_fixture = fixture_path.with_name(f"{fixture_path.name}.tmp.npz")
        save_case(case, tmp_fixture)
        tmp_fixture.replace(fixture_path)
        print(f"fixture -> {fixture_path} ({fixture_path.stat().st_size / 1e6:.2f} MB)")

    weights_path = FIXTURE_DIR / f"{cfg['name']}_weights.npz"
    weights = {key: _np(value) for key, value in tree_flatten(model.parameters())}
    metadata = dict(cfg)
    metadata["oracle"] = "local_mlx_image_mode"
    metadata["tap_points"] = "submodule-level local taps; no reference checkout available in this workspace"
    weights["__config_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    _atomic_savez(weights_path, **weights)
    print(f"weights -> {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
