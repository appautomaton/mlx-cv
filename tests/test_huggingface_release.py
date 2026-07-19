import json
from pathlib import Path

import pytest

from mlx_cv.hub.release import (
    MODEL_RELEASES,
    ModelRelease,
    ReleaseVerificationError,
    stage_release,
    verify_staged_release,
)


def _tiny_safetensors(path, metadata):
    header = {
        "weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "__metadata__": metadata,
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0\0")


def _release():
    return ModelRelease(
        name="example",
        repo_id="appautomaton/example-bf16-mlx",
        checkpoint="source/model.safetensors",
        card="source/card.md",
        license_file="source/LICENSE",
        assets=(("source/config.json", "config.json"),),
        required_metadata=(("format", "example-v1"), ("dtype", "bfloat16")),
    )


def test_registry_uses_exact_public_repo_names():
    assert MODEL_RELEASES["locateanything-3b-bf16"].repo_id == "appautomaton/locateanything-3b-bf16-mlx"
    assert MODEL_RELEASES["sam3.1-multiplex-bf16"].repo_id == "appautomaton/sam3.1-multiplex-bf16-mlx"


def test_stage_is_flat_allowlisted_and_manifest_verified(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_safetensors(
        source / "model.safetensors",
        {"format": "example-v1", "dtype": "bfloat16", "source_sha256": "abc"},
    )
    (source / "card.md").write_text("---\nlibrary_name: mlx\n---\n# Example\n")
    (source / "LICENSE").write_text("license")
    (source / "config.json").write_text("{}")
    package = stage_release(_release(), source_root=tmp_path, staging_root=tmp_path / "stage")
    manifest = verify_staged_release(_release(), package)
    assert {entry["path"] for entry in manifest["files"]} == {
        "LICENSE", "README.md", "config.json", "model.safetensors"
    }
    assert {path.name for path in package.iterdir()} == {
        "LICENSE", "README.md", "config.json", "manifest.json", "model.safetensors"
    }


def test_verify_rejects_unexpected_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_safetensors(
        source / "model.safetensors",
        {"format": "example-v1", "dtype": "bfloat16", "source_sha256": "abc"},
    )
    (source / "card.md").write_text("---\nlibrary_name: mlx\n---\n")
    (source / "LICENSE").write_text("license")
    (source / "config.json").write_text("{}")
    package = stage_release(_release(), source_root=tmp_path, staging_root=tmp_path / "stage")
    (package / "surprise.bin").write_bytes(b"x")
    with pytest.raises(ReleaseVerificationError, match="allowlist"):
        verify_staged_release(_release(), package)


def test_cards_have_yaml_licenses_and_non_404_backlinks():
    root = Path(__file__).parents[1]
    for release in MODEL_RELEASES.values():
        card = (root / release.card).read_text()
        assert card.startswith("---\n")
        assert "library_name: mlx" in card
        assert "license:" in card
        assert "https://github.com/appautomaton/mlx-cv" in card
        assert "https://appautomaton.github.io/mlx-cv/" not in card


def test_uploader_has_safe_public_resume_contract():
    source = (Path(__file__).parents[1] / "tools/huggingface_release.py").read_text()
    assert "private=False" in source
    assert "exist_ok=False" in source
    assert "num_workers=1" in source
    assert "if exists and not resume" in source
    assert "delete_repo" not in source
    assert "delete_file" not in source
    assert "update_repo_settings" not in source
