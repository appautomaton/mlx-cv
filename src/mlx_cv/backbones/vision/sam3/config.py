"""SAM 3.1 image/VL backbone configuration."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SAM3ImageBackboneConfig"]


@dataclass(frozen=True)
class SAM3ImageBackboneConfig:
    """Tiny-config friendly SAM3 image/VL backbone axes."""

    image_size: int = 1024
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 256
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    text_dim: int = 256
    out_layers: tuple[int, ...] = (8, 10, 11)
    neck_channels: int = 256
    neck_scales: tuple[float, ...] = (1.0, 0.5, 0.25)

    def __post_init__(self) -> None:
        positive = (
            self.image_size,
            self.patch_size,
            self.in_chans,
            self.embed_dim,
            self.depth,
            self.num_heads,
            self.text_dim,
            self.neck_channels,
        )
        if min(positive) <= 0:
            raise ValueError("SAM3 image backbone dimensions must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("SAM3 image_size must be divisible by patch_size")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("SAM3 embed_dim must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("SAM3 mlp_ratio must be positive")
        if not self.out_layers:
            raise ValueError("SAM3 out_layers must not be empty")
        invalid_layers = [i for i in self.out_layers if i < 0 or i >= self.depth]
        if invalid_layers:
            raise ValueError(f"SAM3 out_layers outside valid range 0..{self.depth - 1}: {invalid_layers}")
        if not self.neck_scales or any(scale <= 0 for scale in self.neck_scales):
            raise ValueError("SAM3 neck_scales must be positive")

    @property
    def pretrain_grid(self) -> int:
        return self.image_size // self.patch_size
