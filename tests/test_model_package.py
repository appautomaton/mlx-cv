import json

import pytest

from mlx_cv.hub.package import resolve_model_package


def test_resolve_model_package_reads_standard_files(tmp_path):
    weights = tmp_path / "model.npz"
    weights.touch()
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fixture"}))

    package = resolve_model_package(tmp_path)

    assert package.source == tmp_path.resolve()
    assert package.root == tmp_path.resolve()
    assert package.weights == weights.resolve()
    assert package.config == {"model_type": "fixture"}


def test_resolve_model_package_accepts_direct_checkpoint_without_config(tmp_path):
    weights = tmp_path / "weights.npz"
    weights.touch()

    package = resolve_model_package(weights, require_config=False)

    assert package.weights == weights.resolve()
    assert package.config is None


def test_resolve_model_package_rejects_missing_or_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing a checkpoint"):
        resolve_model_package(tmp_path)

    (tmp_path / "model.npz").touch()
    (tmp_path / "config.json").write_text("[]")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        resolve_model_package(tmp_path)
