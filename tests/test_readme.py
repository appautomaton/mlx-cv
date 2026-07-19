from pathlib import Path


def test_readme_documents_003_hub_release():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "`v0.0.3`" in readme
    assert "mlx-cv[mlx,hub]==0.0.3" in readme
    assert "appautomaton/locateanything-3b-bf16-mlx" in readme
    assert "appautomaton/sam3.1-multiplex-bf16-mlx" in readme
    assert "BF16, unquantized" in readme
    assert "HF_HUB_OFFLINE=1" in readme
