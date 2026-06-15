"""Qwen2.5 language backbone package.

Package-root imports are mlx-free so LocateAnything Stage-1 config/decode/convert
can import ``Qwen2Config`` without requiring the MLX extra.  Import
``mlx_cv.backbones.llm.qwen2.modeling`` once the MLX model exists; that submodule
owns concrete registration.
"""

from __future__ import annotations

from .config import Qwen2Config

__all__ = ["Qwen2Config"]
