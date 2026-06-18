"""Mint the deterministic local LocateAnything integration fixture."""

from __future__ import annotations

import json
import pathlib
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures"
sys.path.insert(0, str(REPO / "src"))

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config  # noqa: E402
from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig  # noqa: E402
from mlx_cv.core.geometry import SpatialTransform  # noqa: E402
from mlx_cv.models.locateanything.config import LocateAnythingConfig  # noqa: E402
from mlx_cv.models.locateanything.modeling import LocateAnythingModel  # noqa: E402
from mlx_cv.models.locateanything.pbd import get_token_ids, sample_block  # noqa: E402
from mlx_cv.models.locateanything.processor import (  # noqa: E402
    LocateAnythingProcessor,
    LocateAnythingProcessorContext,
)
from mlx_cv.parity import (  # noqa: E402
    LOCATEANYTHING_FIXTURE_CONFIG,
    ParityCase,
    locateanything_fixed_inputs,
    locateanything_tap_order,
    save_case,
)


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{int(i)}" for i in ids)


def build_config() -> LocateAnythingConfig:
    cfg = LOCATEANYTHING_FIXTURE_CONFIG
    return LocateAnythingConfig(
        vision_config=MoonViTConfig(**cfg["vision"]),
        text_config=Qwen2Config(**cfg["text"]),
        vocab_size=cfg["text"]["vocab_size"],
        image_token_index=cfg["image_token_index"],
        box_start_token_id=cfg["box_start_token_id"],
        box_end_token_id=cfg["box_end_token_id"],
        coord_start_token_id=cfg["coord_start_token_id"],
        coord_end_token_id=cfg["coord_end_token_id"],
        ref_start_token_id=cfg["ref_start_token_id"],
        ref_end_token_id=cfg["ref_end_token_id"],
        none_token_id=cfg["none_token_id"],
        text_mask_token_id=cfg["text_mask_token_id"],
    )


def deterministic_weights(model: LocateAnythingModel) -> dict[str, np.ndarray]:
    out = {}
    for i, (key, value) in enumerate(tree_flatten(model.parameters())):
        size = int(np.prod(value.shape))
        arr = np.linspace(-0.25, 0.25, num=size, dtype=np.float32).reshape(value.shape)
        arr = arr + np.float32(i * 0.001)
        out[key] = arr
    return out


def build_case(model: LocateAnythingModel) -> ParityCase:
    cfg = LOCATEANYTHING_FIXTURE_CONFIG
    inputs = locateanything_fixed_inputs()
    input_ids = mx.array(inputs["input_ids"], dtype=mx.int32)
    cached = mx.array(inputs["cached_image_features"], dtype=mx.float32)
    block_logits = mx.array(inputs["pbd_block_logits"], dtype=mx.float32)
    generated_ids = inputs["generated_ids"]

    processor = LocateAnythingProcessor(model.config, tokenizer=TinyTokenizer())
    ctx = LocateAnythingProcessorContext(
        transform=SpatialTransform.resize(tuple(cfg["image_size"]), tuple(cfg["model_size"])),
        image_size=tuple(cfg["image_size"]),
        model_size=tuple(cfg["model_size"]),
        image_grid_hws=((2, 2),),
    )

    with mx.stream(mx.cpu):
        projector = model.multi_modal_projector(cached)
        inputs_embeds = model.get_input_embeddings(input_ids, cached_image_features=cached)
        sampled = np.array(sample_block(block_logits, get_token_ids(model.config)), dtype=np.int32)
        mx.eval(projector, inputs_embeds)

    result = processor.postprocess(generated_ids, ctx)
    boxes = result.detections.boxes if result.detections is not None else np.zeros((0, 4), dtype=np.float64)
    points = result.points.points if result.points is not None else np.zeros((0, 2), dtype=np.float64)
    taps = {
        "projector": np.array(projector),
        "inputs_embeds": np.array(inputs_embeds),
        "pbd_block_logits": inputs["pbd_block_logits"],
        "sampled_tokens": sampled,
        "generated_ids": generated_ids,
        "boxes": boxes,
        "points": points,
    }
    if list(taps) != locateanything_tap_order():
        raise RuntimeError("unexpected LocateAnything tap order")
    return ParityCase(
        name=cfg["name"],
        inputs=inputs,
        expected={
            "inputs_embeds": taps["inputs_embeds"],
            "boxes": boxes,
            "points": points,
        },
        taps=taps,
    )


def main() -> None:
    cfg = LOCATEANYTHING_FIXTURE_CONFIG
    model = LocateAnythingModel(build_config())
    weights = deterministic_weights(model)
    model.update(tree_unflatten([(k, mx.array(v)) for k, v in weights.items()]))
    mx.eval(model.parameters())

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{cfg['name']}.npz"
    weights_path = FIXTURE_DIR / f"{cfg['name']}_weights.npz"
    tmp_fixture = fixture_path.with_name(f"{fixture_path.name}.tmp.npz")
    save_case(build_case(model), tmp_fixture)
    tmp_fixture.replace(fixture_path)

    weights["__config_json__"] = np.asarray(json.dumps(cfg, sort_keys=True))
    np.savez(weights_path, **weights)
    print(f"fixture -> {fixture_path} ({fixture_path.stat().st_size / 1e6:.2f} MB)")
    print(f"weights -> {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
