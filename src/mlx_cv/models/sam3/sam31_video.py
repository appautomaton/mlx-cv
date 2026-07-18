"""Official SAM 3.1 detector + multiplex tracker assembly."""

from __future__ import annotations

import mlx.nn as nn

from .sam31_modeling import SAM3Model
from .sam31_tracker import SAM31MultiplexTracker

__all__ = ["SAM31VideoModel"]


class SAM31VideoModel(nn.Module):
    """One parameter tree matching the final combined SAM 3.1 checkpoint."""

    def __init__(self):
        super().__init__()
        self.detector = SAM3Model()
        self.tracker = SAM31MultiplexTracker()
