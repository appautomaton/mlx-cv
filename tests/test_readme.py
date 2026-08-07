from pathlib import Path


def test_readme_documents_current_supported_surface():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    for model in (
        "LocateAnything-3B",
        "RF-DETR Nano",
        "Depth Anything V3",
        "SAM 3.1",
        "EoMT",
    ):
        assert model in readme
    assert "mlx_cv.load(...)" in readme
    assert "mlx_cv.available_models()" in readme
    assert "`Result.draw()` returns a new RGB Pillow image" in readme
    assert "SAM video propagation returns `VideoResult`" in readme
    assert "docs/model-packages.md" in readme
    assert "mktemp -d" in readme
    assert "trap 'rm -rf" in readme
    assert "--basetemp" in readme
    assert "HF_HUB_OFFLINE=1" in readme
    assert "Published MLX weights" not in readme
