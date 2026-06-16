import subprocess
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.backbones.llm.qwen2.modeling import Qwen2ForCausalLM
from mlx_cv.backbones.vision.moonvit.config import MoonViTConfig
from mlx_cv.backbones.vision.moonvit.modeling import MoonViTBackbone
from mlx_cv.models.locateanything.config import LocateAnythingConfig
from mlx_cv.models.locateanything.modeling import (
    LocateAnythingModel,
    LocateAnythingProjector,
)


def _tiny_config(**overrides):
    text = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        block_size=2,
        text_mask_token_id=7,
        null_token_id=14,
        switch_token_id=15,
    )
    vision = MoonViTConfig(
        hidden_size=8,
        num_hidden_layers=0,
        num_attention_heads=2,
        intermediate_size=16,
        patch_size=2,
        num_channels=1,
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        merge_kernel_size=(2, 2),
    )
    values = dict(
        vision_config=vision,
        text_config=text,
        vocab_size=16,
        image_token_index=5,
        text_mask_token_id=7,
    )
    values.update(overrides)
    return LocateAnythingConfig(**values)


def test_locateanything_model_constructs_backbones_and_projector():
    model = LocateAnythingModel(_tiny_config())
    assert isinstance(model.vision_tower, MoonViTBackbone)
    assert isinstance(model.language_model, Qwen2ForCausalLM)
    assert isinstance(model.multi_modal_projector, LocateAnythingProjector)
    assert model.multi_modal_projector.input_dim == 32
    assert model.multi_modal_projector.output_dim == 8
    assert model.image_token_index == 5


def test_get_input_embeddings_without_image_matches_language_embeddings():
    model = LocateAnythingModel(_tiny_config())
    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    with mx.stream(mx.cpu):
        got = model.get_input_embeddings(input_ids)
        expected = model.language_model.get_input_embeddings()(input_ids)
        mx.eval(got, expected)
    assert np.allclose(np.array(got), np.array(expected), rtol=1e-6, atol=1e-6)


def test_get_input_embeddings_scatters_cached_image_features():
    model = LocateAnythingModel(_tiny_config())
    input_ids = mx.array([[1, 5, 2]], dtype=mx.int32)
    cached = [mx.ones((1, 32), dtype=mx.float32)]
    with mx.stream(mx.cpu):
        got = model.get_input_embeddings(input_ids, cached_image_features=cached)
        text = model.language_model.get_input_embeddings()(input_ids)
        projected = model.multi_modal_projector(cached)
        mx.eval(got, text, projected)

    got_np = np.array(got)
    text_np = np.array(text)
    assert np.allclose(got_np[0, 0], text_np[0, 0], rtol=1e-6, atol=1e-6)
    assert np.allclose(got_np[0, 2], text_np[0, 2], rtol=1e-6, atol=1e-6)
    assert np.allclose(got_np[0, 1], np.array(projected)[0], rtol=1e-6, atol=1e-6)


def test_get_input_embeddings_accepts_live_moonvit_features():
    model = LocateAnythingModel(_tiny_config())
    input_ids = mx.array([[1, 5, 2]], dtype=mx.int32)
    pixel_values = mx.zeros((4, 1, 2, 2), dtype=mx.float32)
    grid_hws = mx.array([[2, 2]], dtype=mx.int32)
    with mx.stream(mx.cpu):
        got = model.get_input_embeddings(input_ids, pixel_values, image_grid_hws=grid_hws)
        mx.eval(got)
    assert got.shape == (1, 3, 8)


def test_get_input_embeddings_rejects_image_token_feature_mismatch():
    model = LocateAnythingModel(_tiny_config())
    input_ids = mx.array([[1, 5, 5]], dtype=mx.int32)
    cached = [mx.ones((1, 32), dtype=mx.float32)]
    with pytest.raises(ValueError, match="image token count"):
        model.get_input_embeddings(input_ids, cached_image_features=cached)


def test_forward_delegates_to_qwen2_with_prepared_embeddings():
    model = LocateAnythingModel(_tiny_config())
    input_ids = mx.array([[1, 5, 2]], dtype=mx.int32)
    cached = [mx.ones((1, 32), dtype=mx.float32)]
    with mx.stream(mx.cpu):
        out = model(input_ids, cached_image_features=cached)
        mx.eval(*out)
    assert len(out) == 1
    assert out[0].shape == (1, 3, 16)


def test_forward_preserves_input_ids_for_language_mask_dispatch():
    model = LocateAnythingModel(_tiny_config())
    input_ids = mx.array([[1, 5, 2]], dtype=mx.int32)
    seen = {}

    class CaptureLM:
        def __init__(self):
            self.model = type("FakeInner", (), {"layers": []})()

        def get_input_embeddings(self):
            def embed(ids):
                return mx.zeros((*ids.shape, 8), dtype=mx.float32)

            return embed

        def __call__(self, *args, **kwargs):
            del args
            seen["input_ids"] = kwargs.get("input_ids")
            seen["inputs_embeds"] = kwargs.get("inputs_embeds")
            return (mx.zeros((1, 3, 16), dtype=mx.float32),)

    model.language_model = CaptureLM()
    with mx.stream(mx.cpu):
        out = model(input_ids)
        mx.eval(*out)

    assert seen["input_ids"] is input_ids
    assert seen["inputs_embeds"].shape == (1, 3, 8)


def test_locateanything_package_root_import_is_mlx_free():
    code = (
        "import sys\n"
        "import mlx_cv.models.locateanything\n"
        "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)\n"
    )
    subprocess.check_call([sys.executable, "-c", code])
