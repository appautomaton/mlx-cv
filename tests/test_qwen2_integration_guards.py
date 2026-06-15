import subprocess
import sys
from pathlib import Path


def test_qwen2_modeling_import_registers_concrete_llm_builder_in_fresh_process():
    code = (
        "import mlx_cv.backbones.llm.qwen2.modeling\n"
        "from mlx_cv.core.registry import BACKBONES\n"
        "assert BACKBONES.list(kind='llm') == ['qwen2.5-3b']\n"
        "builder = BACKBONES.get('qwen2.5-3b')\n"
        "model = builder({'vocab_size': 8, 'hidden_size': 4, 'intermediate_size': 8, "
        "'num_hidden_layers': 0, 'num_attention_heads': 2, 'num_key_value_heads': 1})\n"
        "assert type(model).__name__ == 'Qwen2ForCausalLM'\n"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_runtime_dependencies_do_not_include_torch_or_transformers():
    text = Path("pyproject.toml").read_text()
    assert "torch" not in text
    assert "transformers" not in text


def test_core_import_and_sources_remain_mlx_free():
    code = (
        "import sys\n"
        "import mlx_cv.core\n"
        "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)\n"
    )
    subprocess.check_call([sys.executable, "-c", code])

    for path in Path("src/mlx_cv/core").glob("*.py"):
        text = path.read_text()
        assert "import mlx" not in text
        assert "from mlx" not in text
