from pathlib import Path

import pytest


def test_sam_image_direct_checkpoint_still_requires_explicit_bpe(tmp_path, monkeypatch):
    pytest.importorskip("mlx")
    from mlx_cv.models.sam3 import SAM3Processor

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    with pytest.raises(ValueError, match="bpe_path is required"):
        SAM3Processor.from_pretrained(checkpoint)


def test_sam_package_resolves_standard_files(tmp_path, monkeypatch):
    pytest.importorskip("mlx")
    import mlx_cv.models.sam3.sam31_predictor as predictor

    checkpoint = tmp_path / "model.safetensors"
    bpe = tmp_path / "bpe_simple_vocab_16e6.txt.gz"
    checkpoint.touch()
    bpe.touch()
    sentinel = object()
    monkeypatch.setattr(predictor, "SAM3Model", lambda: sentinel)
    monkeypatch.setattr(predictor, "load_sam3_weights", lambda model, path: (model, Path(path)))
    monkeypatch.setattr(predictor.SAM3Processor, "__init__", lambda self, model, **kwargs: setattr(self, "loaded", (model, kwargs)))

    instance = predictor.SAM3Processor.from_pretrained(tmp_path)
    assert instance.loaded[0] == (sentinel, checkpoint)
    assert instance.loaded[1]["bpe_path"] == bpe


def test_sam_video_package_loads_standard_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("mlx")
    import mlx_cv.models.sam3.sam31_checkpoint as checkpoint_module
    import mlx_cv.models.sam3.sam31_session as session_module

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    sentinel = object()
    monkeypatch.setattr(session_module, "SAM3VideoModel", lambda: sentinel)
    monkeypatch.setattr(checkpoint_module, "load_sam3_video_weights", lambda model, path: (model, Path(path)))
    monkeypatch.setattr(session_module.SAM3VideoSession, "__init__", lambda self, model=None, **kwargs: setattr(self, "loaded", model))

    instance = session_module.SAM3VideoSession.from_pretrained(tmp_path)
    assert instance.loaded == (sentinel, checkpoint)
