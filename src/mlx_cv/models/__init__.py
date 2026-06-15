"""Model families. Each lives in ``models/<family>/`` with config / modeling /
processor / convert, and registers its name (§7, §10). No spine edits required."""

from __future__ import annotations

from .depth_anything_v3 import DA3MonocularConfig, DepthAnythingV3Monocular

__all__ = ["DA3MonocularConfig", "DepthAnythingV3Monocular"]
