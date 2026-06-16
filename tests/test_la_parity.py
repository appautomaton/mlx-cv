import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.core.geometry import SpatialTransform
from mlx_cv.models.locateanything.config import LocateAnythingConfig
from mlx_cv.models.locateanything.modeling import LocateAnythingModel
from mlx_cv.models.locateanything.pbd import get_token_ids, sample_block
from mlx_cv.models.locateanything.processor import (
    LocateAnythingProcessor,
    LocateAnythingProcessorContext,
)
from mlx_cv.parity import LOCATEANYTHING_FIXTURE_CONFIG, bisect, load_case

FIXTURE = "tests/fixtures/locateanything_tiny_fixture.npz"
WEIGHTS = "tests/fixtures/locateanything_tiny_fixture_weights.npz"


class TinyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{int(i)}" for i in ids)


def _config():
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


def test_locateanything_local_integration_fixture_parity():
    cfg = LOCATEANYTHING_FIXTURE_CONFIG
    case = load_case(FIXTURE)
    weights = np.load(WEIGHTS, allow_pickle=False)
    model = LocateAnythingModel(_config())
    params = [(k, mx.array(weights[k])) for k in weights.files if not k.startswith("__")]
    model.update(tree_unflatten(params))

    input_ids = mx.array(case.inputs["input_ids"], dtype=mx.int32)
    cached = mx.array(case.inputs["cached_image_features"], dtype=mx.float32)
    block_logits = mx.array(case.inputs["pbd_block_logits"], dtype=mx.float32)
    generated_ids = case.inputs["generated_ids"].astype(np.int32)
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
    got_taps = {
        "projector": np.array(projector),
        "inputs_embeds": np.array(inputs_embeds),
        "pbd_block_logits": case.inputs["pbd_block_logits"],
        "sampled_tokens": sampled,
        "generated_ids": generated_ids,
        "boxes": result.detections.boxes,
        "points": result.points.points,
    }

    assert bisect(case.taps, got_taps, atol=1e-6, rtol=1e-6) is None
