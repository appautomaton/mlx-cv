import numpy as np
import mlx.core as mx
import pytest

from mlx_cv.heads.segmentation import SAM3DecoderConfig
from mlx_cv.models.sam3 import SAM3Config, SAM3ImageBackboneConfig, SAM3Model, SAM3TextConfig, SAM3Tokenizer
from mlx_cv.prompts import BoxPrompt, ExemplarPrompt, PointPrompt


def _cfg():
    tokenizer = SAM3Tokenizer(context_length=8)
    return SAM3Config(
        image=SAM3ImageBackboneConfig(
            image_size=32,
            patch_size=4,
            embed_dim=8,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
            text_dim=6,
            out_layers=(0, 1),
            neck_channels=4,
            neck_scales=(1.0, 0.5),
        ),
        text=SAM3TextConfig(
            d_model=6,
            context_length=8,
            vocab_size=tokenizer.vocab_size,
            width=8,
            heads=2,
            layers=1,
            mlp_ratio=2.0,
        ),
        decoder=SAM3DecoderConfig(hidden_dim=4, num_queries=3, num_layers=1, num_heads=1, num_classes=2, text_dim=6),
    )


def test_sam3_model_forwards_text_prompt_through_vl_path():
    tokenizer = SAM3Tokenizer(context_length=8)
    model = SAM3Model(_cfg(), tokenizer=tokenizer)
    out = model(mx.ones((1, 3, 32, 32), dtype=mx.float32), "cat", capture_taps=True)
    mx.eval(out["mask_logits"], out["object_scores"], out["boxes"])

    assert out["mask_logits"].shape == (1, 3, 8, 8)
    assert out["object_scores"].shape == (1, 3)
    assert out["boxes"].shape == (1, 3, 4)
    assert out["prompt_texts"] == ("cat",)
    assert out["text_fused"] is True
    assert "pyramid" in out.data


def test_sam3_model_forwards_pcs_box_and_exemplar_prompts():
    tokenizer = SAM3Tokenizer(context_length=8)
    model = SAM3Model(_cfg(), tokenizer=tokenizer)
    prompt = [
        BoxPrompt([[4, 4, 20, 24]]),
        ExemplarPrompt(image=np.zeros((16, 16, 3), dtype=np.uint8), boxes=[[2, 2, 10, 12]]),
    ]
    out = model(mx.ones((1, 3, 32, 32), dtype=mx.float32), prompt)
    mx.eval(out["mask_logits"], out["geometry_summary"])

    assert out["mask_logits"].shape == (1, 3, 8, 8)
    assert out["text_fused"] is False
    assert out["prepared_geometry"].boxes_cxcywh.shape == (1, 4)
    assert out["prepared_geometry"].exemplar_boxes_cxcywh.shape == (1, 4)


def test_sam3_model_rejects_deferred_interactive_mask_and_video_paths():
    tokenizer = SAM3Tokenizer(context_length=8)
    model = SAM3Model(_cfg(), tokenizer=tokenizer)
    image = mx.ones((1, 3, 32, 32), dtype=mx.float32)

    with pytest.raises(NotImplementedError, match="point"):
        model(image, PointPrompt([[1, 2]], labels=[1]))
    with pytest.raises(NotImplementedError, match="mask"):
        model(image, {"mask_prompt": np.zeros((4, 4))})
    with pytest.raises(NotImplementedError, match="video"):
        model(image, {"video_state": {}})
