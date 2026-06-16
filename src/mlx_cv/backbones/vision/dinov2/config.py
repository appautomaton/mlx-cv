"""DINOv2 (with registers) ViT config — the knobs the shared `ViTBackbone` needs."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DINOv2Config"]


@dataclass(frozen=True)
class DINOv2Config:
    """Architecture config for a DINOv2-with-registers vision transformer.

    Differs from DINOv3 on exactly the parameterized axes the families expose:
    learned-absolute (interpolated) pos-emb instead of RoPE, LayerScale on, and
    ``patch_size`` 14. ``pretrain_grid`` is the pos-emb table side (``image_size //
    patch_size``); the table is bicubic-interpolated to the runtime grid.
    """

    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 14
    in_chans: int = 3
    n_register_tokens: int = 4
    pretrain_grid: int = 37          # 518 // 14 for with-registers checkpoints
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    final_norm_eps: float = 1e-5
    layerscale_init: float = 1.0
    num_windows: int = 1
    windowed_full_attention_layers: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def from_dict(cls, d: dict) -> "DINOv2Config":
        """Build from an HF ``dinov2_with_registers`` config dict (`references/rf-detr/...`)."""
        patch = d.get("patch_size", 14)
        windowed_full_attention_layers = d.get("windowed_full_attention_layers", ())
        if not windowed_full_attention_layers and "window_block_indexes" in d:
            window_blocks = {int(i) for i in d.get("window_block_indexes", ())}
            depth = int(d["num_hidden_layers"])
            windowed_full_attention_layers = tuple(i for i in range(depth) if i not in window_blocks)
        return cls(
            embed_dim=d["hidden_size"],
            depth=d["num_hidden_layers"],
            num_heads=d["num_attention_heads"],
            patch_size=patch,
            in_chans=d.get("num_channels", 3),
            n_register_tokens=d.get("num_register_tokens", 4),
            pretrain_grid=d.get("image_size", 518) // patch,
            ffn_ratio=d.get("mlp_ratio", 4.0),
            qkv_bias=d.get("qkv_bias", True),
            layer_norm_eps=d.get("layer_norm_eps", 1e-6),
            final_norm_eps=d.get("final_norm_eps", 1e-5),
            layerscale_init=d.get("layerscale_value", 1.0),
            num_windows=int(d.get("num_windows", 1)),
            windowed_full_attention_layers=tuple(int(i) for i in windowed_full_attention_layers),
        )

    @classmethod
    def rfdetr_nano(cls) -> "DINOv2Config":
        """RF-DETR Nano's windowed DINOv2-small encoder contract.

        Upstream names this encoder ``dinov2_windowed_small`` and implements it
        with the WindowedDinov2WithRegisters class configured with zero register
        tokens. The local MLX path mirrors that inference contract: patch-16,
        a 24x24 learned positional table, two windows per axis, and upstream's
        runnable full-attention blocks for stage boundaries 3, 6, and 9.
        """
        return cls(
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_size=16,
            n_register_tokens=0,
            pretrain_grid=24,
            final_norm_eps=1e-6,
            num_windows=2,
            windowed_full_attention_layers=(3, 6, 9),
        )
