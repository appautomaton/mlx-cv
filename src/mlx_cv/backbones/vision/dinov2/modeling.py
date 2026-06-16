"""DINOv2 (with registers) ViT in MLX — a thin config-binding over `ViTBackbone`.

The generalization proof for Phase 2: DINOv2 is a *second* real ViT config that
reuses the shared families with **no new block code**. It differs from DINOv3 on
exactly two parameterized axes — learned-absolute (interpolated) pos-emb via
`AbsPosStrategy` instead of RoPE, and LayerScale on — plus dims/patch/registers.
Registers are carried as the assembly's storage tokens, so token order is
``[cls, register…, patch]`` and they receive no positional embedding.

This module defines **no** attention/block/mlp/rope of its own — it subclasses
`ViTBackbone` (keeping param paths top-level) and wires the config. Weight
conversion + numerical parity are deferred to the phase that consumes DINOv2
weights (Phase 3, Depth Anything V3); this phase proves structural instantiation.

`core/` stays mlx-free: mlx lives only here, behind the ``[mlx]`` extra.
"""

from __future__ import annotations

import mlx.core as mx

from ....core.registry import register_backbone
from ....core.features import BackboneFeatures, FeatureMap, Layout, TokenLayout
from ..vit import AbsPosStrategy, ViTBackbone
from .config import DINOv2Config

__all__ = ["DINOv2ViT", "build_dinov2"]


class DINOv2ViT(ViTBackbone):
    """DINOv2-with-registers vision transformer (MLX) = `ViTBackbone` + DINOv2 config."""

    def __init__(self, cfg: DINOv2Config) -> None:
        super().__init__(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            n_storage=cfg.n_register_tokens,   # registers ride the storage-token slot
            mlp_ratio=cfg.ffn_ratio,
            qkv_bias=cfg.qkv_bias,
            norm="layernorm",
            norm_eps=cfg.layer_norm_eps,
            final_norm_eps=cfg.final_norm_eps,
            ffn="gelu",
            layerscale=True,                   # DINOv2: LayerScale on
            layerscale_init=cfg.layerscale_init,
            position=AbsPosStrategy(cfg.pretrain_grid),
        )
        self.cfg = cfg

    def forward_features(
        self,
        x: mx.array,
        *,
        intermediate_layers: int | list[int] | tuple[int, ...] | None = None,
        capture_taps: bool = False,
    ) -> BackboneFeatures:
        if self.cfg.num_windows <= 1:
            return super().forward_features(
                x,
                intermediate_layers=intermediate_layers,
                capture_taps=capture_taps,
            )
        return self._forward_windowed_features(
            x,
            intermediate_layers=intermediate_layers,
            capture_taps=capture_taps,
        )

    def _layers_to_take(
        self,
        intermediate_layers: int | list[int] | tuple[int, ...] | None,
    ) -> set[int]:
        if intermediate_layers is None:
            return set()
        if isinstance(intermediate_layers, int):
            if intermediate_layers < 0 or intermediate_layers > self.depth:
                raise ValueError(
                    f"intermediate_layers={intermediate_layers} outside valid range 0..{self.depth}"
                )
            return set(range(self.depth - intermediate_layers, self.depth))
        layers = set(int(i) for i in intermediate_layers)
        invalid = sorted(i for i in layers if i < 0 or i >= self.depth)
        if invalid:
            raise ValueError(f"intermediate layer indices {invalid} outside valid range 0..{self.depth - 1}")
        return layers

    def _window_patch_tokens(self, patches: mx.array, grid: tuple[int, int]) -> mx.array:
        batch, _, channels = patches.shape
        height, width = grid
        num_windows = self.cfg.num_windows
        if height % num_windows or width % num_windows:
            raise ValueError(
                "windowed DINOv2 requires the patch grid to be divisible by "
                f"num_windows={num_windows}, got grid={grid}"
            )
        h_per_window = height // num_windows
        w_per_window = width // num_windows
        patches = patches.reshape(batch, num_windows, h_per_window, num_windows, w_per_window, channels)
        patches = mx.transpose(patches, (0, 1, 3, 2, 4, 5))
        return patches.reshape(batch * num_windows * num_windows, h_per_window * w_per_window, channels)

    def _unwindow_patch_tokens(
        self,
        patches: mx.array,
        *,
        batch: int,
        grid: tuple[int, int],
    ) -> mx.array:
        height, width = grid
        num_windows = self.cfg.num_windows
        h_per_window = height // num_windows
        w_per_window = width // num_windows
        channels = patches.shape[-1]
        patches = patches.reshape(batch, num_windows, num_windows, h_per_window, w_per_window, channels)
        patches = mx.transpose(patches, (0, 1, 3, 2, 4, 5))
        return patches.reshape(batch, height * width, channels)

    def _forward_windowed_features(
        self,
        x: mx.array,
        *,
        intermediate_layers: int | list[int] | tuple[int, ...] | None = None,
        capture_taps: bool = False,
    ) -> BackboneFeatures:
        layers_to_take = self._layers_to_take(intermediate_layers)
        num_windows = self.cfg.num_windows
        num_window_groups = num_windows * num_windows
        full_attention_layers = set(int(i) for i in self.cfg.windowed_full_attention_layers)

        taps: dict[str, mx.array] = {}
        patches, (hp, wp) = self.patch_embed(x)
        batch, _, channels = patches.shape
        cls = mx.broadcast_to(self.cls_token, (batch, 1, channels))

        tokens = mx.concatenate([cls, patches], axis=1)
        tokens = self.position.add_pos(self, tokens, (hp, wp))
        cls = tokens[:, :1]
        patch_tokens = self._window_patch_tokens(tokens[:, 1:], (hp, wp))
        cls = mx.broadcast_to(cls[:, None, :, :], (batch, num_window_groups, 1, channels))
        cls = cls.reshape(batch * num_window_groups, 1, channels)
        if self.n_storage > 0:
            storage = mx.broadcast_to(
                self.storage_tokens,
                (batch * num_window_groups, self.n_storage, channels),
            )
            z = mx.concatenate([cls, storage, patch_tokens], axis=1)
        else:
            z = mx.concatenate([cls, patch_tokens], axis=1)
        if capture_taps:
            taps["patch_embed"] = z

        n_prefix = 1 + self.n_storage
        selected_tokens: list[tuple[int, mx.array]] = []
        for i, block in enumerate(self.blocks):
            if i in full_attention_layers:
                tokens_per_window = z.shape[1]
                z = z.reshape(batch, num_window_groups * tokens_per_window, channels)
                z = block(z, rope=None, n_prefix=0)
                z = z.reshape(batch * num_window_groups, tokens_per_window, channels)
            else:
                z = block(z, rope=None, n_prefix=n_prefix)
            if capture_taps:
                taps[f"block_{i:02d}"] = z
            if i in layers_to_take:
                selected_tokens.append((i, z))

        z_norm = self.norm(z)
        intermediates: list[FeatureMap] = []
        for i, selected in selected_tokens:
            selected_norm = self.norm(selected)
            patch_selected = self._unwindow_patch_tokens(
                selected_norm[:, n_prefix:],
                batch=batch,
                grid=(hp, wp),
            )
            intermediates.append(
                FeatureMap(
                    patch_selected,
                    layout=Layout.BNC,
                    grid=(hp, wp),
                    stride=self.patch_size,
                )
            )
            if capture_taps:
                taps[f"intermediate_{i:02d}"] = patch_selected

        cls_windows = z_norm[:, 0].reshape(batch, num_window_groups, channels)
        cls_out = mx.mean(cls_windows, axis=1)
        storage_out = None
        if self.n_storage > 0:
            storage_windows = z_norm[:, 1:n_prefix].reshape(batch, num_window_groups, self.n_storage, channels)
            storage_out = mx.mean(storage_windows, axis=1)
        patch_out = self._unwindow_patch_tokens(z_norm[:, n_prefix:], batch=batch, grid=(hp, wp))
        if capture_taps:
            taps["norm"] = z_norm
            taps["cls"] = cls_out
            if storage_out is not None:
                taps["storage"] = storage_out
            taps["patch"] = patch_out

        extras: dict = {
            "x_prenorm": z,
            "windowed_dinov2": {
                "num_windows": num_windows,
                "full_attention_layers": tuple(sorted(full_attention_layers)),
                "unsupported_training_behaviors": ("drop_path", "gradient_checkpointing"),
            },
        }
        if capture_taps:
            extras["taps"] = taps
        return BackboneFeatures(
            patch_tokens=FeatureMap(patch_out, layout=Layout.BNC, grid=(hp, wp), stride=self.patch_size),
            cls_token=cls_out,
            storage_tokens=storage_out,
            token_layout=TokenLayout.vit(n_storage=self.n_storage),
            intermediates=intermediates,
            extras=extras,
        )


@register_backbone("dinov2", kind="vision")
def build_dinov2(config) -> DINOv2ViT:
    """Registry builder: a config dict (HF `dinov2_with_registers`) or a `DINOv2Config`."""
    cfg = config if isinstance(config, DINOv2Config) else DINOv2Config.from_dict(config)
    return DINOv2ViT(cfg)
