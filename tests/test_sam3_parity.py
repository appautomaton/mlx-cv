import pathlib

import numpy as np
import pytest

from mlx_cv.parity import (
    SAM3_FIXTURE_CONFIG,
    assert_parity,
    bisect,
    load_case,
    sam3_fixed_image,
    sam3_pcs_prompt,
    sam3_tap_order,
    sam3_text_prompt,
)

mx = pytest.importorskip("mlx.core")

from mlx_cv.heads.segmentation import SAM3DecoderConfig  # noqa: E402
from mlx_cv.models.sam3 import (  # noqa: E402
    SAM3Config,
    SAM3ImageBackboneConfig,
    SAM3Model,
    SAM3Processor,
    SAM3ProcessorConfig,
    SAM3TextConfig,
    SAM3Tokenizer,
    load_sam3_weights,
)


_FIX = pathlib.Path(__file__).parent / "fixtures"
_ATOL = 1e-5


def _cfg():
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
    cfg = SAM3_FIXTURE_CONFIG
    return SAM3Processor(
        SAM3ProcessorConfig(
            image_size=tuple(cfg["image_size"]),
            top_k=int(cfg["num_select"]),
            labels=tuple(cfg["labels"]),
        )
    )


def _model():
    tokenizer = SAM3Tokenizer(context_length=int(SAM3_FIXTURE_CONFIG["text"]["context_length"]))
    return load_sam3_weights(SAM3Model(_cfg(), tokenizer=tokenizer), _FIX / "sam3_tiny_fixture_weights.npz")


def _run_parity(kind: str):
    prompt = sam3_text_prompt() if kind == "text" else sam3_pcs_prompt()
    with mx.stream(mx.cpu):
        case = load_case(_FIX / f"sam3_tiny_fixture_{kind}.npz")
        processor = _processor()
        model_inputs, ctx = processor.preprocess({"image": case.inputs["image"], "prompt": prompt})
        raw = _model()(model_inputs["pixel_values"], model_inputs["prompt"], capture_taps=True)
        result = processor.postprocess(raw, ctx)
        mx.eval(raw["mask_logits"], raw["object_scores"], raw["boxes"])

    got = {
        "mask_logits": np.array(raw["mask_logits"]),
        "object_scores": np.array(raw["object_scores"]),
        "boxes": np.array(raw["boxes"]),
        "result_masks": np.asarray(result.masks.data, dtype=np.float32),
        "result_boxes": np.asarray(result.detections.boxes, dtype=np.float32),
        "scores": np.asarray(result.detections.scores, dtype=np.float32),
        "class_ids": np.asarray(result.detections.class_ids, dtype=np.int64),
    }
    taps = {key: np.asarray(value) for key, value in raw["taps"].items()}
    taps["result.masks"] = got["result_masks"]
    taps["result.boxes"] = got["result_boxes"]
    taps["result.scores"] = got["scores"]
    taps["result.class_ids"] = got["class_ids"]
    return case, got, taps


@pytest.mark.parametrize("kind", ["text", "pcs"])
def test_sam3_tiny_fixture_raw_and_result_outputs_match(kind):
    case, got, _ = _run_parity(kind)
    assert_parity(got, case.expected, atol=_ATOL, rtol=_ATOL, name=case.name)


def test_sam3_text_and_pcs_taps_match_schema_and_bisect_clean():
    text_case, _, text_taps = _run_parity("text")
    assert list(text_taps.keys()) == sam3_tap_order(include_text=True, include_geometry=False)
    assert bisect(text_case.taps, text_taps, atol=_ATOL, rtol=_ATOL) is None

    pcs_case, _, pcs_taps = _run_parity("pcs")
    assert list(pcs_taps.keys()) == sam3_tap_order(include_text=False, include_geometry=True)
    assert bisect(pcs_case.taps, pcs_taps, atol=_ATOL, rtol=_ATOL) is None


def test_sam3_bisect_localizes_injected_drift():
    case, _, taps = _run_parity("pcs")
    corrupted = dict(taps)
    corrupted["head.mask_logits"] = corrupted["head.mask_logits"] + 1.0
    assert bisect(case.taps, corrupted, atol=_ATOL, rtol=_ATOL) == "head.mask_logits"


def test_sam3_fixture_uses_fixed_image_input():
    assert sam3_fixed_image().shape == tuple(SAM3_FIXTURE_CONFIG["image_size"]) + (3,)
