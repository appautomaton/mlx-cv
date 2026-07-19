import json

import pytest

from mlx_cv.models.locateanything import LocateAnythingConfig


def test_locateanything_config_normalized_round_trip():
    config = LocateAnythingConfig()
    loaded = LocateAnythingConfig.from_dict(config.to_dict())
    assert loaded == config


def test_pipeline_requires_complete_local_package(tmp_path):
    pytest.importorskip("mlx")
    from mlx_cv.models.locateanything import LocateAnythingPipeline

    (tmp_path / "config.json").write_text(json.dumps(LocateAnythingConfig().to_dict()))
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        LocateAnythingPipeline.from_pretrained(tmp_path)


def test_pipeline_formats_the_upstream_single_image_chat_template():
    pytest.importorskip("mlx")
    from mlx_cv.models.locateanything import LocateAnythingPipeline

    prompt = LocateAnythingPipeline.format_prompt("find signs in <image-0>")
    assert prompt == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<image-1>find signs in<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
