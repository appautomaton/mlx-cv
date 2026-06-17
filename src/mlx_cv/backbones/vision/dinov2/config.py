"""DINOv2 (with registers) ViT config — the knobs the shared `ViTBackbone` needs."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DA3AnyViewDINOv2Config", "DINOv2Config"]


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


@dataclass(frozen=True)
class DA3AnyViewDINOv2Config:
    """DA3 Small/Base any-view DINOv2 backbone contract.

    The regular :class:`DINOv2Config` remains the monocular/RF-DETR contract.
    DA3's real any-view checkpoint adds view-axis dispatch, camera tokens,
    q/k-normalized blocks, and DA3's own 2D RoPE. Keeping these knobs in a
    separate config prevents accidental behavior changes in the existing DINOv2
    path while preserving the checkpoint-visible parameter names.
    """

    embed_dim: int
    depth: int
    num_heads: int
    patch_size: int = 14
    in_chans: int = 3
    n_register_tokens: int = 0
    pretrain_grid: int = 37
    ffn_ratio: float = 4.0
    qkv_bias: bool = True
    layer_norm_eps: float = 1e-6
    final_norm_eps: float = 1e-5
    layerscale_init: float = 1.0
    out_layers: tuple[int, ...] = (5, 7, 9, 11)
    alt_start: int = 4
    qknorm_start: int = 4
    rope_start: int = 4
    rope_frequency: float = 100.0
    cat_token: bool = True
    ref_selection_threshold: int = 3

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def head_input_dim(self) -> int:
        return self.embed_dim * 2 if self.cat_token else self.embed_dim

    @classmethod
    def from_dict(cls, d: dict) -> "DA3AnyViewDINOv2Config":
        if "hidden_size" not in d and "name" in d:
            variants = {
                "vits": (384, 12, 6),
                "vitb": (768, 12, 12),
            }
            name = str(d["name"])
            if name not in variants:
                raise ValueError(f"unsupported DA3 any-view DINOv2 variant {name!r}")
            embed_dim, depth, num_heads = variants[name]
            patch = int(d.get("patch_size", 14))
            return cls(
                embed_dim=int(d.get("embed_dim", embed_dim)),
                depth=int(d.get("depth", depth)),
                num_heads=int(d.get("num_heads", num_heads)),
                patch_size=patch,
                in_chans=int(d.get("in_chans", d.get("num_channels", 3))),
                n_register_tokens=int(d.get("num_register_tokens", 0)),
                pretrain_grid=int(d.get("image_size", 518)) // patch,
                ffn_ratio=float(d.get("mlp_ratio", 4.0)),
                qkv_bias=bool(d.get("qkv_bias", True)),
                layer_norm_eps=float(d.get("layer_norm_eps", 1e-6)),
                final_norm_eps=float(d.get("final_norm_eps", 1e-5)),
                layerscale_init=float(d.get("layerscale_value", 1.0)),
                out_layers=tuple(int(i) for i in d.get("out_layers", (5, 7, 9, 11))),
                alt_start=int(d.get("alt_start", 4)),
                qknorm_start=int(d.get("qknorm_start", 4)),
                rope_start=int(d.get("rope_start", 4)),
                rope_frequency=float(d.get("rope_frequency", d.get("rope_freq", 100.0))),
                cat_token=bool(d.get("cat_token", True)),
                ref_selection_threshold=int(d.get("ref_selection_threshold", 3)),
            )
        patch = int(d.get("patch_size", 14))
        return cls(
            embed_dim=int(d["hidden_size"]),
            depth=int(d["num_hidden_layers"]),
            num_heads=int(d["num_attention_heads"]),
            patch_size=patch,
            in_chans=int(d.get("num_channels", 3)),
            n_register_tokens=int(d.get("num_register_tokens", 0)),
            pretrain_grid=int(d.get("image_size", 518)) // patch,
            ffn_ratio=float(d.get("mlp_ratio", 4.0)),
            qkv_bias=bool(d.get("qkv_bias", True)),
            layer_norm_eps=float(d.get("layer_norm_eps", 1e-6)),
            final_norm_eps=float(d.get("final_norm_eps", 1e-5)),
            layerscale_init=float(d.get("layerscale_value", 1.0)),
            out_layers=tuple(int(i) for i in d.get("out_layers", (5, 7, 9, 11))),
            alt_start=int(d.get("alt_start", 4)),
            qknorm_start=int(d.get("qknorm_start", 4)),
            rope_start=int(d.get("rope_start", 4)),
            rope_frequency=float(d.get("rope_frequency", d.get("rope_freq", 100.0))),
            cat_token=bool(d.get("cat_token", True)),
            ref_selection_threshold=int(d.get("ref_selection_threshold", 3)),
        )

    @classmethod
    def small(cls) -> "DA3AnyViewDINOv2Config":
        return cls(embed_dim=384, depth=12, num_heads=6)

    @classmethod
    def base(cls) -> "DA3AnyViewDINOv2Config":
        return cls(embed_dim=768, depth=12, num_heads=12)
