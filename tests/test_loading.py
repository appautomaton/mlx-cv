import subprocess
import sys

import pytest

import mlx_cv.loading as loading
from mlx_cv import MODEL_LOADERS, Task, available_models, load


def test_builtin_loader_catalog_covers_supported_public_runtimes():
    assert available_models() == (
        "depth-anything-v3-monocular",
        "depth-anything-v3-multiview",
        "eomt-dinov3-coco-panoptic-small-640",
        "locateanything-3b",
        "rfdetr-nano",
        "sam3.1-image",
        "sam3.1-video",
    )
    assert MODEL_LOADERS.get("rfdetr-nano").task is Task.DETECTION
    assert MODEL_LOADERS.get("eomt-dinov3-coco-panoptic-small-640").task is Task.SEGMENTATION
    assert MODEL_LOADERS.get("sam3.1-video").task is Task.TRACKING


def test_load_dispatches_alias_and_options_lazily(monkeypatch):
    calls = []

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, identifier, **kwargs):
            calls.append((identifier, kwargs))
            return "loaded"

    monkeypatch.setattr(
        loading.ModelLoaderSpec,
        "resolve_loader",
        lambda self: FakeLoader,
    )
    assert load("rf-detr-nano", "/models/rfdetr", strict=False) == "loaded"
    assert calls == [("/models/rfdetr", {"strict": False})]


def test_load_uses_only_official_configured_default(monkeypatch):
    calls = []

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, identifier, **kwargs):
            calls.append(identifier)
            return identifier

    monkeypatch.setattr(
        loading.ModelLoaderSpec,
        "resolve_loader",
        lambda self: FakeLoader,
    )
    assert load("eomt-dinov3") == "tue-mps/eomt-dinov3-coco-panoptic-small-640"
    assert calls == ["tue-mps/eomt-dinov3-coco-panoptic-small-640"]
    with pytest.raises(ValueError, match="requires a local package path"):
        load("sam3-video")
    with pytest.raises(ValueError, match="requires a local package path"):
        load("da3-small")


def test_load_rejects_unknown_model_key():
    with pytest.raises(KeyError, match="unknown model"):
        load("not-a-model", "/tmp/package")


def test_public_loader_catalog_is_mlx_free_at_import_time():
    code = (
        "import sys, mlx_cv; "
        "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules); "
        "assert 'rfdetr-nano' in mlx_cv.available_models()"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
