"""DA3 any-view DINOv2 backbone admission.

This module mirrors the Depth Anything 3 any-view DINOv2 control path while
leaving the existing monocular/RF-DETR ``DINOv2ViT`` class untouched:

* input is ``(B, V, 3, H, W)`` and patch outputs keep ``(B, V, N, C)`` layout
* blocks before ``alt_start`` run per-view local attention
* blocks from ``alt_start`` alternate local even layers and cross-view global
  odd layers
* q/k LayerNorm and DA3 2D RoPE are enabled per block from their configured
  start layers
* camera tokens replace cls tokens at ``alt_start``
* selected outputs restore original view order and, when ``cat_token=True``,
  concatenate local and global features with upstream-compatible split norm
"""

from __future__ import annotations

from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from ....core.features import BackboneFeatures, FeatureMap, Layout, TokenLayout
from ...layers import LayerScale, MlpFFN, PatchEmbed
from ...layers.position import LearnedAbsPosEmb
from .config import DA3AnyViewDINOv2Config

__all__ = [
    "DA3AnyViewDINOv2",
    "DA3Attention",
    "DA3TransformerBlock",
    "select_reference_view",
]

RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_1d_rope(tokens: mx.array, positions: mx.array, *, frequency: float) -> mx.array:
    """Apply the 1D half of DA3's 2D RoPE to ``(B, heads, N, D)`` tokens."""

    dim = tokens.shape[-1]
    inv_freq = 1.0 / (frequency ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    angles = positions.astype(mx.float32)[:, :, None] * inv_freq.reshape(1, 1, -1)
    angles = mx.concatenate([angles, angles], axis=-1)
    cos = mx.cos(angles)[:, None, :, :]
    sin = mx.sin(angles)[:, None, :, :]
    return tokens * cos + _rotate_half(tokens) * sin


def apply_da3_rope(tokens: mx.array, positions: mx.array, *, frequency: float = 100.0) -> mx.array:
    """Apply upstream DA3 2D RoPE using explicit ``(y, x)`` integer positions.

    ``tokens`` is ``(B, heads, N, head_dim)`` and ``positions`` is ``(B, N, 2)``.
    The head dimension is split in half: vertical features receive y positions
    and horizontal features receive x positions.
    """

    if tokens.shape[-1] % 4:
        raise ValueError(f"DA3 RoPE requires head_dim divisible by 4, got {tokens.shape[-1]}")
    vertical, horizontal = mx.split(tokens, 2, axis=-1)
    vertical = _apply_1d_rope(vertical, positions[:, :, 0], frequency=frequency)
    horizontal = _apply_1d_rope(horizontal, positions[:, :, 1], frequency=frequency)
    return mx.concatenate([vertical, horizontal], axis=-1)


def _patch_positions(hp: int, wp: int) -> mx.array:
    y = mx.broadcast_to(mx.arange(hp, dtype=mx.int32).reshape(hp, 1), (hp, wp))
    x = mx.broadcast_to(mx.arange(wp, dtype=mx.int32).reshape(1, wp), (hp, wp))
    return mx.stack([y, x], axis=-1).reshape(hp * wp, 2)


def da3_rope_positions(
    *,
    batch: int,
    views: int,
    grid: tuple[int, int],
    n_prefix: int,
) -> tuple[mx.array, mx.array]:
    """Return local and global DA3 RoPE positions in ``(B, V, N, 2)`` layout."""

    hp, wp = grid
    patch = _patch_positions(hp, wp)
    special = mx.zeros((n_prefix, 2), dtype=mx.int32)
    local_one = mx.concatenate([special, patch + 1], axis=0)
    global_one = mx.concatenate([special, mx.ones_like(patch)], axis=0)
    shape = (batch, views, local_one.shape[0], 2)
    return (
        mx.broadcast_to(local_one.reshape(1, 1, local_one.shape[0], 2), shape),
        mx.broadcast_to(global_one.reshape(1, 1, global_one.shape[0], 2), shape),
    )


def select_reference_view(x: mx.array, *, strategy: RefViewStrategy = "saddle_balanced") -> mx.array:
    """Select a reference view from ``(B, V, N, C)`` tokens.

    The feature-based strategies match the upstream class-token logic closely
    enough for parity localization; tests can use ``middle`` to force a
    deterministic non-first reference without relying on real checkpoint weights.
    """

    batch, views = x.shape[0], x.shape[1]
    if views <= 1 or strategy == "first":
        return mx.zeros((batch,), dtype=mx.int32)
    if strategy == "middle":
        return mx.full((batch,), views // 2, dtype=mx.int32)

    cls = x[:, :, 0]
    cls_norm = cls / mx.maximum(mx.sqrt(mx.sum(cls * cls, axis=-1, keepdims=True)), 1e-8)
    sim = cls_norm @ mx.transpose(cls_norm, (0, 2, 1))
    sim_no_diag = sim - mx.eye(views, dtype=sim.dtype).reshape(1, views, views)

    if strategy == "saddle_balanced":
        sim_score = mx.sum(sim_no_diag, axis=-1) / float(views - 1)
        feat_norm = mx.sqrt(mx.sum(cls * cls, axis=-1))
        feat_var = mx.mean((cls_norm - mx.mean(cls_norm, axis=-1, keepdims=True)) ** 2, axis=-1)

        def normalize(metric: mx.array) -> mx.array:
            min_val = mx.min(metric, axis=1, keepdims=True)
            max_val = mx.max(metric, axis=1, keepdims=True)
            return (metric - min_val) / (max_val - min_val + 1e-8)

        balance = (
            mx.abs(normalize(sim_score) - 0.5)
            + mx.abs(normalize(feat_norm) - 0.5)
            + mx.abs(normalize(feat_var) - 0.5)
        )
        return mx.argmin(balance, axis=1).astype(mx.int32)

    if strategy == "saddle_sim_range":
        sim_max = mx.max(sim_no_diag, axis=-1)
        sim_min = mx.min(sim_no_diag, axis=-1)
        return mx.argmax(sim_max - sim_min, axis=1).astype(mx.int32)

    raise ValueError(
        "unknown reference view selection strategy "
        f"{strategy!r}; expected first, middle, saddle_balanced, or saddle_sim_range"
    )


def _reorder_indices(ref_idx: mx.array, views: int) -> mx.array:
    positions = mx.broadcast_to(mx.arange(views, dtype=mx.int32).reshape(1, views), (ref_idx.shape[0], views))
    ref = ref_idx.reshape(ref_idx.shape[0], 1)
    shifted = mx.where((positions > 0) & (positions <= ref), positions - 1, positions)
    return mx.where(positions == 0, ref, shifted).astype(mx.int32)


def _restore_indices(ref_idx: mx.array, views: int) -> mx.array:
    positions = mx.broadcast_to(mx.arange(views, dtype=mx.int32).reshape(1, views), (ref_idx.shape[0], views))
    ref = ref_idx.reshape(ref_idx.shape[0], 1)
    shifted = mx.where(positions < ref, positions + 1, positions)
    return mx.where(positions == ref, mx.zeros_like(positions), shifted).astype(mx.int32)


def _gather_views(x: mx.array, order: mx.array) -> mx.array:
    index = order.reshape(order.shape[0], order.shape[1], *([1] * (x.ndim - 2)))
    index = mx.broadcast_to(index, x.shape)
    return mx.take_along_axis(x, index, axis=1)


class DA3Attention(nn.Module):
    """DA3 attention with optional per-head q/k LayerNorm and 2D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        rope_frequency: float | None = None,
        qk_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope_frequency = rope_frequency

    def __call__(self, x: mx.array, *, pos: mx.array | None = None) -> mx.array:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.rope_frequency is not None and pos is not None:
            q = apply_da3_rope(q, pos, frequency=self.rope_frequency)
            k = apply_da3_rope(k, pos, frequency=self.rope_frequency)
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, tokens, channels)
        return self.proj(out)


class DA3TransformerBlock(nn.Module):
    """DA3 block wrapper with per-block qk_norm and RoPE flags."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        rope_frequency: float | None = None,
        norm_eps: float = 1e-6,
        qk_norm_eps: float = 1e-5,
        layerscale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = DA3Attention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            rope_frequency=rope_frequency,
            qk_norm_eps=qk_norm_eps,
        )
        self.ls1 = LayerScale(dim, layerscale_init)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MlpFFN(dim, int(dim * mlp_ratio), kind="gelu")
        self.ls2 = LayerScale(dim, layerscale_init)

    def __call__(self, x: mx.array, *, pos: mx.array | None = None) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DA3AnyViewDINOv2(nn.Module):
    """DA3 Small/Base any-view DINOv2 feature extractor."""

    def __init__(self, cfg: DA3AnyViewDINOv2Config) -> None:
        super().__init__()
        if cfg.embed_dim % cfg.num_heads:
            raise ValueError("DA3AnyViewDINOv2 embed_dim must be divisible by num_heads")
        if cfg.head_dim % 4:
            raise ValueError("DA3AnyViewDINOv2 head_dim must be divisible by 4 for DA3 RoPE")
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.depth = cfg.depth
        self.num_heads = cfg.num_heads
        self.patch_size = cfg.patch_size
        self.n_storage = cfg.n_register_tokens

        self.patch_embed = PatchEmbed(cfg.in_chans, cfg.embed_dim, cfg.patch_size)
        self.cls_token = mx.zeros((1, 1, cfg.embed_dim))
        if cfg.alt_start != -1:
            self.camera_token = mx.zeros((1, 2, cfg.embed_dim))
        self.pos_embed = LearnedAbsPosEmb(cfg.embed_dim, cfg.pretrain_grid)
        if self.n_storage > 0:
            self.storage_tokens = mx.zeros((1, self.n_storage, cfg.embed_dim))
        self.blocks = [
            DA3TransformerBlock(
                cfg.embed_dim,
                cfg.num_heads,
                mlp_ratio=cfg.ffn_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qknorm_start != -1 and i >= cfg.qknorm_start,
                rope_frequency=cfg.rope_frequency if cfg.rope_start != -1 and i >= cfg.rope_start else None,
                norm_eps=cfg.layer_norm_eps,
                layerscale_init=cfg.layerscale_init,
            )
            for i in range(cfg.depth)
        ]
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.final_norm_eps)

    def __call__(
        self,
        x: mx.array,
        *,
        intermediate_layers: int | list[int] | tuple[int, ...] | None = None,
        capture_taps: bool = False,
        cam_token: mx.array | None = None,
        reference_view_strategy: RefViewStrategy = "saddle_balanced",
    ) -> BackboneFeatures:
        return self.forward_features(
            x,
            intermediate_layers=intermediate_layers,
            capture_taps=capture_taps,
            cam_token=cam_token,
            reference_view_strategy=reference_view_strategy,
        )

    def _layers_to_take(self, intermediate_layers: int | list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
        if intermediate_layers is None:
            layers = tuple(int(i) for i in self.cfg.out_layers)
        elif isinstance(intermediate_layers, int):
            if intermediate_layers < 0 or intermediate_layers > self.depth:
                raise ValueError(
                    f"intermediate_layers={intermediate_layers} outside valid range 0..{self.depth}"
                )
            layers = tuple(range(self.depth - intermediate_layers, self.depth))
        else:
            layers = tuple(int(i) for i in intermediate_layers)
        invalid = sorted(i for i in set(layers) if i < 0 or i >= self.depth)
        if invalid:
            raise ValueError(f"intermediate layer indices {invalid} outside valid range 0..{self.depth - 1}")
        return layers

    def _prepare_tokens(self, x: mx.array) -> tuple[mx.array, tuple[int, int]]:
        batch, views, channels, height, width = x.shape
        flat = x.reshape(batch * views, channels, height, width)
        patches, grid = self.patch_embed(flat)
        cls = mx.broadcast_to(self.cls_token, (batch * views, 1, self.embed_dim))
        tokens = mx.concatenate([cls, patches], axis=1)
        tokens = tokens + self.pos_embed(grid)
        if self.n_storage > 0:
            storage = mx.broadcast_to(self.storage_tokens, (batch * views, self.n_storage, self.embed_dim))
            tokens = mx.concatenate([tokens[:, :1], storage, tokens[:, 1:]], axis=1)
        return tokens.reshape(batch, views, tokens.shape[1], self.embed_dim), grid

    def _default_camera_tokens(self, batch: int, views: int) -> mx.array:
        if not hasattr(self, "camera_token"):
            raise ValueError("camera tokens require cfg.alt_start != -1")
        ref_token = mx.broadcast_to(self.camera_token[:, :1], (batch, 1, self.embed_dim))
        if views == 1:
            return ref_token
        src_token = mx.broadcast_to(self.camera_token[:, 1:], (batch, views - 1, self.embed_dim))
        return mx.concatenate([ref_token, src_token], axis=1)

    def _camera_tokens(self, batch: int, views: int, cam_token: mx.array | None) -> mx.array:
        if cam_token is None:
            return self._default_camera_tokens(batch, views)
        if cam_token.shape != (batch, views, self.embed_dim):
            raise ValueError(
                "cam_token must have shape "
                f"(B,V,{self.embed_dim}), got {tuple(cam_token.shape)} for B={batch}, V={views}"
            )
        return cam_token

    def _process_attention(
        self,
        x: mx.array,
        block: DA3TransformerBlock,
        *,
        mode: Literal["local", "global"],
        pos: mx.array | None,
    ) -> tuple[mx.array, tuple[int, int, int]]:
        batch, views, tokens, channels = x.shape
        if mode == "local":
            flat = x.reshape(batch * views, tokens, channels)
            flat_pos = None if pos is None else pos.reshape(batch * views, tokens, 2)
            out = block(flat, pos=flat_pos)
            return out.reshape(batch, views, tokens, channels), (batch * views, tokens, channels)
        if mode == "global":
            flat = x.reshape(batch, views * tokens, channels)
            flat_pos = None if pos is None else pos.reshape(batch, views * tokens, 2)
            out = block(flat, pos=flat_pos)
            return out.reshape(batch, views, tokens, channels), (batch, views * tokens, channels)
        raise ValueError(f"invalid DA3 attention mode {mode!r}")

    def _normalize_selected(self, selected: mx.array) -> mx.array:
        if selected.shape[-1] == self.embed_dim:
            return self.norm(selected)
        if selected.shape[-1] == self.embed_dim * 2:
            return mx.concatenate(
                [selected[..., : self.embed_dim], self.norm(selected[..., self.embed_dim :])],
                axis=-1,
            )
        raise ValueError(f"invalid selected DA3 feature width {selected.shape[-1]}")

    def forward_features(
        self,
        x: mx.array,
        *,
        intermediate_layers: int | list[int] | tuple[int, ...] | None = None,
        capture_taps: bool = False,
        cam_token: mx.array | None = None,
        reference_view_strategy: RefViewStrategy = "saddle_balanced",
    ) -> BackboneFeatures:
        if x.ndim != 5:
            raise ValueError(f"DA3AnyViewDINOv2 expects BVCHW input, got shape {tuple(x.shape)}")
        batch, views, channels, height, width = x.shape
        if channels != self.cfg.in_chans:
            raise ValueError(
                f"DA3AnyViewDINOv2 expects {self.cfg.in_chans} channels at axis 2, got shape {tuple(x.shape)}"
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "DA3AnyViewDINOv2 input height/width must be divisible by patch size "
                f"{self.patch_size}, got shape {tuple(x.shape)}"
            )

        layers_to_take = set(self._layers_to_take(intermediate_layers))
        tokens, grid = self._prepare_tokens(x)
        n_prefix = 1 + self.n_storage
        local_pos, global_pos = da3_rope_positions(batch=batch, views=views, grid=grid, n_prefix=n_prefix)
        z = tokens
        local_z = z
        selected: list[tuple[int, mx.array, mx.array]] = []
        taps: dict[str, mx.array | tuple] = {}
        attention_modes: list[str] = []
        attention_input_shapes: list[tuple[int, int, int]] = []
        ref_idx: mx.array | None = None
        reorder_order: mx.array | None = None
        restore_order: mx.array | None = None

        if capture_taps:
            taps["patch_embed"] = tokens
            taps["local_rope_positions"] = local_pos
            taps["global_rope_positions"] = global_pos

        for i, block in enumerate(self.blocks):
            if (
                self.cfg.alt_start != -1
                and i == self.cfg.alt_start - 1
                and views >= self.cfg.ref_selection_threshold
                and cam_token is None
            ):
                ref_idx = select_reference_view(z, strategy=reference_view_strategy)
                reorder_order = _reorder_indices(ref_idx, views)
                restore_order = _restore_indices(ref_idx, views)
                z = _gather_views(z, reorder_order)
                local_z = _gather_views(local_z, reorder_order)
                local_pos = _gather_views(local_pos, reorder_order)
                global_pos = _gather_views(global_pos, reorder_order)
                if capture_taps:
                    taps["reference_indices"] = ref_idx
                    taps["view_reorder"] = reorder_order
                    taps["view_restore"] = restore_order
                    taps["tokens_after_reorder"] = z

            if self.cfg.alt_start != -1 and i == self.cfg.alt_start:
                cam = self._camera_tokens(batch, views, cam_token)
                z = mx.concatenate([cam[:, :, None, :], z[:, :, 1:, :]], axis=2)
                if capture_taps:
                    taps["camera_tokens"] = cam
                    taps["tokens_after_camera_token"] = z

            if self.cfg.alt_start != -1 and i >= self.cfg.alt_start and i % 2 == 1:
                mode: Literal["local", "global"] = "global"
                pos = global_pos if self.cfg.rope_start != -1 and i >= self.cfg.rope_start else None
            else:
                mode = "local"
                pos = local_pos if self.cfg.rope_start != -1 and i >= self.cfg.rope_start else None
            z, input_shape = self._process_attention(z, block, mode=mode, pos=pos)
            if mode == "local":
                local_z = z
            attention_modes.append(mode)
            attention_input_shapes.append(input_shape)

            if capture_taps:
                taps[f"block_{i:02d}"] = z
            if i in layers_to_take:
                out = mx.concatenate([local_z, z], axis=-1) if self.cfg.cat_token else z
                if restore_order is not None:
                    out = _gather_views(out, restore_order)
                selected.append((i, out, self._normalize_selected(out)))
                if capture_taps:
                    taps[f"selected_prenorm_{i:02d}"] = out
                    taps[f"selected_norm_{i:02d}"] = selected[-1][2]

        if not selected:
            out = mx.concatenate([local_z, z], axis=-1) if self.cfg.cat_token else z
            if restore_order is not None:
                out = _gather_views(out, restore_order)
            selected.append((self.depth - 1, out, self._normalize_selected(out)))

        intermediates: list[FeatureMap] = []
        camera_tokens: list[mx.array] = []
        for _, prenorm, normed in selected:
            camera_tokens.append(prenorm[:, :, 0])
            intermediates.append(
                FeatureMap(
                    normed[:, :, n_prefix:],
                    layout=Layout.BSNC,
                    grid=grid,
                    stride=self.patch_size,
                    view_axis=1,
                )
            )

        patch_tokens = intermediates[-1]
        cls_token = camera_tokens[-1]
        extras: dict = {
            "x_prenorm": selected[-1][1],
            "camera_tokens": tuple(camera_tokens),
            "attention_modes": tuple(attention_modes),
            "attention_input_shapes": tuple(attention_input_shapes),
            "anyview_dinov2": {
                "alt_start": self.cfg.alt_start,
                "qknorm_start": self.cfg.qknorm_start,
                "rope_start": self.cfg.rope_start,
                "cat_token": self.cfg.cat_token,
                "out_layers": tuple(self._layers_to_take(intermediate_layers)),
            },
        }
        if ref_idx is not None:
            extras["reference_indices"] = ref_idx
            extras["view_reorder"] = reorder_order
            extras["view_restore"] = restore_order
        if capture_taps:
            taps["attention_modes"] = tuple(attention_modes)
            taps["attention_input_shapes"] = tuple(attention_input_shapes)
            extras["taps"] = taps

        return BackboneFeatures(
            patch_tokens=patch_tokens,
            cls_token=cls_token,
            storage_tokens=None,
            token_layout=TokenLayout.vit(n_storage=self.n_storage),
            intermediates=intermediates,
            extras=extras,
        )
