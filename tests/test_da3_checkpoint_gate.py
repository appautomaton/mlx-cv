from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("da3_checkpoint", REPO / "tools" / "da3_checkpoint.py")
assert SPEC is not None and SPEC.loader is not None
da3_checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = da3_checkpoint
SPEC.loader.exec_module(da3_checkpoint)

DA3CheckpointError = da3_checkpoint.DA3CheckpointError
DA3_CHECKPOINT_ENV = da3_checkpoint.DA3_CHECKPOINT_ENV
DA3_CHECKPOINT_FILENAME = da3_checkpoint.DA3_CHECKPOINT_FILENAME
DA3_CONFIG_ENV = da3_checkpoint.DA3_CONFIG_ENV
DA3_CONFIG_FILENAME = da3_checkpoint.DA3_CONFIG_FILENAME
DA3_DEFAULT_MODEL_ID = da3_checkpoint.DA3_DEFAULT_MODEL_ID
DA3_DOWNLOAD_ENV = da3_checkpoint.DA3_DOWNLOAD_ENV
DA3_FALLBACK_MODEL_ID = da3_checkpoint.DA3_FALLBACK_MODEL_ID
DA3_MODEL_ID_ENV = da3_checkpoint.DA3_MODEL_ID_ENV
DA3_REQUIRED_GATE_ENV = da3_checkpoint.DA3_REQUIRED_GATE_ENV
model_cache_dir = da3_checkpoint.model_cache_dir
print_checkpoint_evidence = da3_checkpoint.print_checkpoint_evidence
resolve_da3_checkpoint = da3_checkpoint.resolve_da3_checkpoint


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_da3_checkpoint_from_env_verifies_pair_and_prints_evidence(tmp_path, capsys):
    checkpoint = tmp_path / "manual.safetensors"
    config = tmp_path / "config.json"
    checkpoint_sha = _write(checkpoint, b"da3-small-weights")
    config_sha = _write(config, b'{"model_name":"da3-small"}')

    info = resolve_da3_checkpoint(
        environ={
            DA3_CHECKPOINT_ENV: str(checkpoint),
            DA3_CONFIG_ENV: str(config),
        },
    )

    assert info is not None
    assert info.model_id == DA3_DEFAULT_MODEL_ID
    assert info.checkpoint_path == checkpoint
    assert info.config_path == config
    assert info.checkpoint_sha256 == checkpoint_sha
    assert info.config_sha256 == config_sha
    assert info.license_note == "Apache-2.0"

    print_checkpoint_evidence(info)
    out = capsys.readouterr().out
    assert DA3_DEFAULT_MODEL_ID in out
    assert str(checkpoint) in out
    assert str(config) in out
    assert checkpoint_sha in out
    assert config_sha in out
    assert "Apache-2.0" in out


def test_da3_checkpoint_requires_config_and_checkpoint_env_pair(tmp_path):
    checkpoint = tmp_path / "manual.safetensors"
    _write(checkpoint, b"da3-small-weights")

    with pytest.raises(DA3CheckpointError, match=f"set both {DA3_CHECKPOINT_ENV} and {DA3_CONFIG_ENV}"):
        resolve_da3_checkpoint(environ={DA3_CHECKPOINT_ENV: str(checkpoint)})


def test_da3_checkpoint_missing_is_optional_without_required_gate(tmp_path):
    info = resolve_da3_checkpoint(environ={}, cache_root=tmp_path)
    assert info is None


def test_da3_checkpoint_missing_fails_with_required_gate(tmp_path):
    with pytest.raises(DA3CheckpointError, match="checkpoint is required but missing"):
        resolve_da3_checkpoint(
            environ={DA3_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
        )


def test_da3_checkpoint_cache_resolves_model_directory(tmp_path):
    model_dir = model_cache_dir(tmp_path, DA3_DEFAULT_MODEL_ID)
    checkpoint = model_dir / DA3_CHECKPOINT_FILENAME
    config = model_dir / DA3_CONFIG_FILENAME
    checkpoint_sha = _write(checkpoint, b"cached-weights")
    config_sha = _write(config, b"cached-config")

    info = resolve_da3_checkpoint(environ={}, cache_root=tmp_path)

    assert info is not None
    assert info.checkpoint_path == checkpoint
    assert info.config_path == config
    assert info.checkpoint_sha256 == checkpoint_sha
    assert info.config_sha256 == config_sha


def test_da3_checkpoint_fallback_model_id_uses_explicit_cache_dir(tmp_path):
    model_dir = model_cache_dir(tmp_path, DA3_FALLBACK_MODEL_ID)
    checkpoint = model_dir / DA3_CHECKPOINT_FILENAME
    config = model_dir / DA3_CONFIG_FILENAME
    _write(checkpoint, b"base-weights")
    _write(config, b"base-config")

    info = resolve_da3_checkpoint(
        environ={DA3_MODEL_ID_ENV: DA3_FALLBACK_MODEL_ID},
        cache_root=tmp_path,
    )

    assert info is not None
    assert info.model_id == DA3_FALLBACK_MODEL_ID
    assert info.checkpoint_path == checkpoint
    assert "DA3-BASE" in info.checkpoint_url


def test_da3_checkpoint_rejects_unsupported_model_id(tmp_path):
    with pytest.raises(DA3CheckpointError, match="unsupported DA3 model id"):
        resolve_da3_checkpoint(environ={DA3_MODEL_ID_ENV: "depth-anything/DA3-LARGE"}, cache_root=tmp_path)


def test_da3_checkpoint_download_is_opt_in(tmp_path, monkeypatch):
    def fail_download(*_args, **_kwargs):
        raise AssertionError("download should not run without opt-in")

    monkeypatch.setattr(da3_checkpoint, "_download_file", fail_download)

    info = resolve_da3_checkpoint(environ={}, cache_root=tmp_path)
    assert info is None


def test_da3_checkpoint_download_writes_model_cache_pair(tmp_path, monkeypatch):
    def fake_download(url: str, dest: Path) -> None:
        if url.endswith(DA3_CONFIG_FILENAME):
            dest.write_bytes(b"downloaded-config")
        elif url.endswith(DA3_CHECKPOINT_FILENAME):
            dest.write_bytes(b"downloaded-weights")
        else:
            raise AssertionError(url)

    monkeypatch.setattr(da3_checkpoint, "_download_file", fake_download)

    info = resolve_da3_checkpoint(
        environ={DA3_DOWNLOAD_ENV: "1"},
        cache_root=tmp_path,
    )

    assert info is not None
    model_dir = model_cache_dir(tmp_path, DA3_DEFAULT_MODEL_ID)
    assert info.config_path == model_dir / DA3_CONFIG_FILENAME
    assert info.checkpoint_path == model_dir / DA3_CHECKPOINT_FILENAME
    assert info.config_path.read_bytes() == b"downloaded-config"
    assert info.checkpoint_path.read_bytes() == b"downloaded-weights"
    assert info.source == f"{DA3_DEFAULT_MODEL_ID}@main"
