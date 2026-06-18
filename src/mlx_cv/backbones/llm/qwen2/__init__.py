"""Qwen2.5 language backbone package.

Package-root imports are mlx-free so LocateAnything Stage-1 config/decode/convert
can import ``Qwen2Config`` without requiring the MLX extra.  Import
``mlx_cv.backbones.llm.qwen2.modeling`` once the MLX model exists; that submodule
owns concrete registration.
"""

from __future__ import annotations

from .config import Qwen2Config

__all__ = [
    "Qwen2Config",
    "QWEN2_CONVERT_RULES",
    "convert_qwen2_state_dict",
    "load_qwen2_weights",
]


def __getattr__(name: str):
    if name in {"QWEN2_CONVERT_RULES", "convert_qwen2_state_dict", "load_qwen2_weights"}:
        from .convert import QWEN2_CONVERT_RULES, convert_qwen2_state_dict, load_qwen2_weights

        exports = {
            "QWEN2_CONVERT_RULES": QWEN2_CONVERT_RULES,
            "convert_qwen2_state_dict": convert_qwen2_state_dict,
            "load_qwen2_weights": load_qwen2_weights,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
