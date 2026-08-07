"""Model families.

Each implementation owns its configuration, compute graph, preprocessing, weight
conversion, and public orchestration as needed. Concrete MLX-backed exports are
resolved lazily. User-facing pretrained runtimes are cataloged uniformly in
``mlx_cv.loading``; low-level compute builders remain model-specific.
"""

from __future__ import annotations

__all__ = [
    "DA3MonocularConfig",
    "DepthAnythingV3Monocular",
    "EoMTDINOv3",
    "EoMTDINOv3Config",
]


def __getattr__(name: str):
    if name in __all__:
        if name in {"DA3MonocularConfig", "DepthAnythingV3Monocular"}:
            from .depth_anything_v3 import DA3MonocularConfig, DepthAnythingV3Monocular

            return {
                "DA3MonocularConfig": DA3MonocularConfig,
                "DepthAnythingV3Monocular": DepthAnythingV3Monocular,
            }[name]
        from .eomt_dinov3 import EoMTDINOv3, EoMTDINOv3Config

        return {"EoMTDINOv3": EoMTDINOv3, "EoMTDINOv3Config": EoMTDINOv3Config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
