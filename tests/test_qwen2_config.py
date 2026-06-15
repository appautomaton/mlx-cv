import subprocess
import sys

from mlx_cv.backbones.llm.qwen2.config import Qwen2Config
from mlx_cv.core.registry import BACKBONES


def test_qwen2_config_defaults_and_reference_reconciliation():
    cfg = Qwen2Config()
    assert (cfg.hidden_size, cfg.num_hidden_layers) == (2048, 36)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads) == (16, 2)
    assert cfg.intermediate_size == 11008
    assert cfg.vocab_size == 152681
    assert cfg.rope_theta == 1_000_000.0
    assert cfg.block_size == 6
    assert cfg.causal_attn is False
    assert cfg.use_cache is False
    assert cfg.attn_implementation == "sdpa"
    assert cfg._attn_implementation == "sdpa"
    assert cfg.text_mask_token_id == 151676

    ref = Qwen2Config.from_dict(
        {
            "_attn_implementation": "magi",
            "use_cache": False,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
        }
    )
    assert ref.attn_implementation == "sdpa"
    assert ref.use_cache is False


def test_qwen2_config_imports_are_mlx_free():
    code = (
        "import sys\n"
        "import mlx_cv.backbones.llm.qwen2\n"
        "import mlx_cv.backbones.llm.qwen2.config\n"
        "import mlx_cv.models.locateanything\n"
        "assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)\n"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_qwen2_not_registered_by_config_import():
    assert "qwen2.5-3b" not in BACKBONES.list(kind="llm")
