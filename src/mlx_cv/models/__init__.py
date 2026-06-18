"""Model families.

Each family lives in ``models/<family>/`` with config / modeling / processor /
convert, and registers its name (§7, §10). Concrete MLX-backed models are
resolved lazily so mlx-free subpackages can be imported without loading ``mlx``.
"""

from __future__ import annotations

__all__ = ["DA3MonocularConfig", "DepthAnythingV3Monocular"]


def __getattr__(name: str):
    if name in __all__:
        from .depth_anything_v3 import DA3MonocularConfig, DepthAnythingV3Monocular

        exports = {
            "DA3MonocularConfig": DA3MonocularConfig,
            "DepthAnythingV3Monocular": DepthAnythingV3Monocular,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
