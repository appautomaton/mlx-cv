from pathlib import Path

import pytest

from mlx_cv.hub import PretrainedResolutionError, resolve_pretrained


def test_resolver_returns_local_path_without_hub_import(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    assert resolve_pretrained(package) == package.resolve()


def test_resolver_expands_alias_and_forwards_revision(monkeypatch, tmp_path):
    calls = {}

    def snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    resolved = resolve_pretrained(
        "sam3.1", revision="abc123", cache_dir=tmp_path / "cache"
    )
    assert resolved == tmp_path.resolve()
    assert calls["repo_id"] == "appautomaton/sam3.1-multiplex-bf16-mlx"
    assert calls["revision"] == "abc123"
    assert calls["local_files_only"] is False


def test_resolver_honors_offline_environment(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.update(kwargs) or str(tmp_path),
    )
    resolve_pretrained("appautomaton/example")
    assert calls["local_files_only"] is True


def test_resolver_rejects_unknown_short_name():
    with pytest.raises(PretrainedResolutionError, match="unknown pretrained"):
        resolve_pretrained("mystery-model")


def test_resolver_wraps_offline_cache_miss(monkeypatch):
    def missing(**kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    with pytest.raises(PretrainedResolutionError, match="offline cache"):
        resolve_pretrained("appautomaton/example", local_files_only=True)
