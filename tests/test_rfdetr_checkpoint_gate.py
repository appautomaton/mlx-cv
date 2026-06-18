from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rfdetr_checkpoint", REPO / "tools" / "rfdetr_checkpoint.py")
assert SPEC is not None and SPEC.loader is not None
rfdetr_checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rfdetr_checkpoint
SPEC.loader.exec_module(rfdetr_checkpoint)

CheckpointError = rfdetr_checkpoint.CheckpointError
RFDETR_DOWNLOAD_ENV = rfdetr_checkpoint.RFDETR_DOWNLOAD_ENV
RFDETR_NANO_CHECKPOINT_ENV = rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_ENV
RFDETR_NANO_CHECKPOINT_FILENAME = rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_FILENAME
RFDETR_NANO_CHECKPOINT_URL_BASENAME = rfdetr_checkpoint.RFDETR_NANO_CHECKPOINT_URL_BASENAME
RFDETR_REQUIRED_GATE_ENV = rfdetr_checkpoint.RFDETR_REQUIRED_GATE_ENV
print_checkpoint_evidence = rfdetr_checkpoint.print_checkpoint_evidence
resolve_rfdetr_nano_checkpoint = rfdetr_checkpoint.resolve_rfdetr_nano_checkpoint


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.md5(data).hexdigest()


def test_rfdetr_checkpoint_from_env_verifies_and_prints_evidence(tmp_path, capsys):
    checkpoint = tmp_path / "manual.pth"
    expected_md5 = _write(checkpoint, b"rfdetr-nano")

    info = resolve_rfdetr_nano_checkpoint(
        environ={RFDETR_NANO_CHECKPOINT_ENV: str(checkpoint)},
        expected_md5=expected_md5,
    )
    assert info is not None
    assert info.path == checkpoint
    assert info.md5 == expected_md5

    print_checkpoint_evidence(info)
    out = capsys.readouterr().out
    assert str(checkpoint) in out
    assert expected_md5 in out


def test_rfdetr_checkpoint_missing_is_optional_without_required_gate(tmp_path):
    info = resolve_rfdetr_nano_checkpoint(environ={}, cache_root=tmp_path, expected_md5="unused")
    assert info is None


def test_rfdetr_checkpoint_missing_fails_with_required_gate(tmp_path):
    with pytest.raises(CheckpointError, match="checkpoint is required but missing"):
        resolve_rfdetr_nano_checkpoint(
            environ={RFDETR_REQUIRED_GATE_ENV: "1"},
            cache_root=tmp_path,
            expected_md5="unused",
        )


def test_rfdetr_checkpoint_checksum_mismatch_is_hard_failure(tmp_path):
    checkpoint = tmp_path / RFDETR_NANO_CHECKPOINT_FILENAME
    _write(checkpoint, b"wrong")

    with pytest.raises(CheckpointError, match="expected"):
        resolve_rfdetr_nano_checkpoint(environ={}, cache_root=tmp_path, expected_md5="0" * 32)


def test_rfdetr_checkpoint_cache_canonicalizes_url_basename(tmp_path):
    source = tmp_path / RFDETR_NANO_CHECKPOINT_URL_BASENAME
    expected_md5 = _write(source, b"from-url-basename")

    info = resolve_rfdetr_nano_checkpoint(environ={}, cache_root=tmp_path, expected_md5=expected_md5)

    canonical = tmp_path / RFDETR_NANO_CHECKPOINT_FILENAME
    assert info is not None
    assert info.path == canonical
    assert canonical.read_bytes() == source.read_bytes()
    assert info.md5 == expected_md5


def test_rfdetr_checkpoint_download_is_opt_in(tmp_path, monkeypatch):
    def fail_download(*_args, **_kwargs):
        raise AssertionError("download should not run without opt-in")

    monkeypatch.setattr(rfdetr_checkpoint, "_download_file", fail_download)

    info = resolve_rfdetr_nano_checkpoint(environ={}, cache_root=tmp_path, expected_md5="unused")
    assert info is None


def test_rfdetr_checkpoint_download_uses_canonical_filename(tmp_path, monkeypatch):
    expected_md5 = hashlib.md5(b"downloaded").hexdigest()

    def fake_download(_url: str, dest: Path) -> None:
        dest.write_bytes(b"downloaded")

    monkeypatch.setattr(rfdetr_checkpoint, "_download_file", fake_download)

    info = resolve_rfdetr_nano_checkpoint(
        environ={RFDETR_DOWNLOAD_ENV: "1"},
        cache_root=tmp_path,
        expected_md5=expected_md5,
    )

    assert info is not None
    assert info.path == tmp_path / RFDETR_NANO_CHECKPOINT_FILENAME
    assert info.path.read_bytes() == b"downloaded"
    assert info.md5 == expected_md5
