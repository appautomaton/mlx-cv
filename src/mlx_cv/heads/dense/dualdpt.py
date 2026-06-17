"""Depth Anything 3 DualDPT dense prediction head in MLX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from ...core.features import HeadInput, HeadOutput, Layout
from ...core.registry import register_head
from .dpt import FeatureFusionBlock, Identity, _relu, resize_bilinear_align_corners

__all__ = [
    "DA3DualDPTConfig",
    "DA3DualDPT",
    "build_da3_dualdpt",
    "create_uv_grid",
    "position_grid_to_embed",
]


@dataclass(frozen=True)
class DA3DualDPTConfig:
    dim_in: int
    patch_size: int = 14
    output_dim: int = 2
    activation: str = "exp"
    conf_activation: str = "expp1"
    features: int = 256
    out_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024)
    pos_embed: bool = True
    down_ratio: int = 1
    aux_pyramid_levels: int = 4
    aux_out1_conv_num: int = 5
    head_names: tuple[str, str] = ("depth", "ray")

    @classmethod
    def from_dict(cls, d: dict) -> "DA3DualDPTConfig":
        return cls(
            dim_in=int(d["dim_in"]),
            patch_size=int(d.get("patch_size", 14)),
            output_dim=int(d.get("output_dim", 2)),
            activation=d.get("activation", "exp"),
            conf_activation=d.get("conf_activation", "expp1"),
            features=int(d.get("features", 256)),
            out_channels=tuple(d.get("out_channels", cls.out_channels)),
            pos_embed=bool(d.get("pos_embed", True)),
            down_ratio=int(d.get("down_ratio", 1)),
            aux_pyramid_levels=int(d.get("aux_pyramid_levels", 4)),
            aux_out1_conv_num=int(d.get("aux_out1_conv_num", 5)),
            head_names=tuple(d.get("head_names", ("depth", "ray"))),
        )


def _linspace(start: float, stop: float, steps: int) -> mx.array:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if steps == 1:
        return mx.array([start], dtype=mx.float32)
    t = mx.arange(steps, dtype=mx.float32) / float(steps - 1)
    return start + (stop - start) * t


def create_uv_grid(width: int, height: int, *, aspect_ratio: float | None = None) -> mx.array:
    """Create the normalized DA3 UV grid with shape ``(height, width, 2)``."""

    if width <= 0 or height <= 0:
        raise ValueError(f"UV grid dimensions must be positive, got {(height, width)}")
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag = (aspect_ratio * aspect_ratio + 1.0) ** 0.5
    span_x = aspect_ratio / diag
    span_y = 1.0 / diag
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x = _linspace(left_x, right_x, width)
    y = _linspace(top_y, bottom_y, height)
    uu = mx.broadcast_to(x.reshape(1, width), (height, width))
    vv = mx.broadcast_to(y.reshape(height, 1), (height, width))
    return mx.stack([uu, vv], axis=-1)


def _sincos_pos_embed(embed_dim: int, pos: mx.array, *, omega_0: float = 100.0) -> mx.array:
    if embed_dim % 2:
        raise ValueError(f"sincos positional embedding dimension must be even, got {embed_dim}")
    omega = mx.arange(embed_dim // 2, dtype=mx.float32) / float(embed_dim // 2)
    omega = 1.0 / (omega_0 ** omega)
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    return mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)


def position_grid_to_embed(pos_grid: mx.array, embed_dim: int, *, omega_0: float = 100.0) -> mx.array:
    """Convert a ``(H,W,2)`` UV grid to DA3 sinusoidal embeddings."""

    if pos_grid.ndim != 3 or pos_grid.shape[-1] != 2:
        raise ValueError(f"position grid must have shape (H,W,2), got {tuple(pos_grid.shape)}")
    if embed_dim % 4:
        raise ValueError(f"DA3 2D positional embedding dimension must be divisible by 4, got {embed_dim}")
    height, width = int(pos_grid.shape[0]), int(pos_grid.shape[1])
    flat = pos_grid.reshape(height * width, 2)
    emb_x = _sincos_pos_embed(embed_dim // 2, flat[:, 0], omega_0=omega_0)
    emb_y = _sincos_pos_embed(embed_dim // 2, flat[:, 1], omega_0=omega_0)
    return mx.concatenate([emb_x, emb_y], axis=-1).reshape(height, width, embed_dim)


class DualScratch(nn.Module):
    def __init__(
        self,
        out_channels: Sequence[int],
        features: int,
        *,
        output_dim: int,
        aux_levels: int,
        aux_out1_conv_num: int,
    ) -> None:
        super().__init__()
        self.layer1_rn = nn.Conv2d(out_channels[0], features, 3, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(out_channels[1], features, 3, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(out_channels[2], features, 3, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(out_channels[3], features, 3, padding=1, bias=False)

        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.refinenet1_aux = FeatureFusionBlock(features)
        self.refinenet2_aux = FeatureFusionBlock(features)
        self.refinenet3_aux = FeatureFusionBlock(features)
        self.refinenet4_aux = FeatureFusionBlock(features, has_residual=False)

        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.output_conv2 = [
            nn.Conv2d(features // 2, 32, 3, padding=1),
            Identity(),
            nn.Conv2d(32, output_dim, 1),
        ]
        self.output_conv1_aux = [
            self._make_aux_out1_block(features, aux_out1_conv_num) for _ in range(aux_levels)
        ]
        self.output_conv2_aux = [
            self._make_aux_out2_block(features // 2, use_layer_norm=(i == 0))
            for i in range(aux_levels)
        ]

    def _make_aux_out1_block(self, channels: int, conv_num: int) -> list[nn.Module]:
        if conv_num == 5:
            widths = (channels, channels // 2, channels, channels // 2, channels, channels // 2)
        elif conv_num == 3:
            widths = (channels, channels // 2, channels, channels // 2)
        elif conv_num == 1:
            widths = (channels, channels // 2)
        else:
            raise ValueError(f"aux_out1_conv_num {conv_num} not supported")
        return [nn.Conv2d(widths[i], widths[i + 1], 3, padding=1) for i in range(len(widths) - 1)]

    def _make_aux_out2_block(self, channels: int, *, use_layer_norm: bool) -> list[nn.Module]:
        return [
            nn.Conv2d(channels, 32, 3, padding=1),
            Identity(),
            nn.LayerNorm(32) if use_layer_norm else Identity(),
            Identity(),
            Identity(),
            nn.Conv2d(32, 7, 1),
        ]


class DA3DualDPT(nn.Module):
    """DA3 DualDPT head consuming four ``(B,V,N,C)`` any-view intermediates."""

    def __init__(self, cfg: DA3DualDPTConfig) -> None:
        super().__init__()
        if len(cfg.out_channels) != 4:
            raise ValueError("DA3DualDPT requires exactly four out_channels entries")
        if len(cfg.head_names) != 2:
            raise ValueError("DA3DualDPT requires two head names: main and auxiliary")
        if cfg.output_dim < 2:
            raise ValueError("DA3DualDPT output_dim must include value and confidence channels")
        if cfg.down_ratio <= 0:
            raise ValueError("down_ratio must be positive")
        if not 1 <= cfg.aux_pyramid_levels <= 4:
            raise ValueError("aux_pyramid_levels must be in the range 1..4")
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.activation = cfg.activation
        self.conf_activation = cfg.conf_activation
        self.head_main, self.head_aux = cfg.head_names
        self.norm = nn.LayerNorm(cfg.dim_in)
        self.projects = [nn.Conv2d(cfg.dim_in, oc, 1) for oc in cfg.out_channels]
        self.resize_layers = [
            nn.ConvTranspose2d(cfg.out_channels[0], cfg.out_channels[0], 4, stride=4),
            nn.ConvTranspose2d(cfg.out_channels[1], cfg.out_channels[1], 2, stride=2),
            Identity(),
            nn.Conv2d(cfg.out_channels[3], cfg.out_channels[3], 3, stride=2, padding=1),
        ]
        self.scratch = DualScratch(
            cfg.out_channels,
            cfg.features,
            output_dim=cfg.output_dim,
            aux_levels=cfg.aux_pyramid_levels,
            aux_out1_conv_num=cfg.aux_out1_conv_num,
        )

    def _activate(self, x: mx.array, activation: str) -> mx.array:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return mx.exp(x)
        if act == "expp1":
            return mx.exp(x) + 1
        if act == "expm1":
            return mx.expm1(x)
        if act == "relu":
            return _relu(x)
        if act == "sigmoid":
            return mx.sigmoid(x)
        if act == "softplus":
            return mx.logaddexp(x, mx.zeros_like(x))
        if act == "tanh":
            return mx.tanh(x)
        return x

    def _add_pos_embed(self, x: mx.array, image_size: tuple[int, int], *, ratio: float = 0.1) -> mx.array:
        height, width = int(x.shape[1]), int(x.shape[2])
        image_h, image_w = image_size
        uv = create_uv_grid(width, height, aspect_ratio=float(image_w) / float(image_h))
        pe = position_grid_to_embed(uv, int(x.shape[-1])) * ratio
        return x + pe.reshape(1, height, width, x.shape[-1])

    def _project_stage(
        self,
        x: mx.array,
        grid: tuple[int, int],
        stage_idx: int,
        image_size: tuple[int, int],
    ) -> mx.array:
        batch, views, tokens, channels = x.shape
        ph, pw = grid
        if tokens != ph * pw:
            raise ValueError(f"stage {stage_idx} has {tokens} tokens but grid {grid} has {ph * pw}")
        x = self.norm(x).reshape(batch * views, ph, pw, channels)
        x = self.projects[stage_idx](x)
        if self.cfg.pos_embed:
            x = self._add_pos_embed(x, image_size)
        return self.resize_layers[stage_idx](x)

    def _apply_aux_out1(self, x: mx.array, level_idx: int) -> mx.array:
        for conv in self.scratch.output_conv1_aux[level_idx]:
            x = conv(x)
        return x

    def _apply_aux_out2(self, x: mx.array, level_idx: int) -> mx.array:
        block = self.scratch.output_conv2_aux[level_idx]
        x = block[0](x)
        x = block[2](x)
        x = _relu(x)
        return block[5](x)

    def _fuse(
        self,
        feats: list[mx.array],
        taps: dict[str, mx.array] | None,
    ) -> tuple[mx.array, list[mx.array]]:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[1:3])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[1:3])
        aux_list: list[mx.array] = []
        if self.cfg.aux_pyramid_levels >= 4:
            aux_list.append(aux_out)

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[1:3])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[1:3])
        if self.cfg.aux_pyramid_levels >= 3:
            aux_list.append(aux_out)

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[1:3])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[1:3])
        if self.cfg.aux_pyramid_levels >= 2:
            aux_list.append(aux_out)

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self._apply_aux_out1(aux, i) for i, aux in enumerate(aux_list)]
        if taps is not None:
            taps["fusion_main"] = out
            taps["fusion_aux_final"] = aux_list[-1]
        return out, aux_list

    def __call__(self, inp: HeadInput, *, capture_taps: bool = False) -> HeadOutput:
        features = inp.features
        if len(features.intermediates) != 4:
            raise ValueError(f"DA3DualDPT requires four intermediates, got {len(features.intermediates)}")
        grid = inp.grid or features.grid
        if grid is None:
            raise ValueError("DA3DualDPT requires a patch grid")
        if inp.image_size is None:
            image_size = (int(grid[0] * self.patch_size), int(grid[1] * self.patch_size))
        else:
            image_size = (int(inp.image_size[0]), int(inp.image_size[1]))

        first = features.intermediates[0].data
        if first.ndim != 4:
            raise ValueError(f"DA3DualDPT requires BSNC intermediates, got shape {tuple(first.shape)}")
        batch, views = int(first.shape[0]), int(first.shape[1])
        taps: dict[str, mx.array] | None = {} if capture_taps else None
        resized: list[mx.array] = []
        for i, fm in enumerate(features.intermediates):
            if fm.layout is not Layout.BSNC:
                raise ValueError(f"DA3DualDPT requires BSNC intermediates, got {fm.layout}")
            if fm.data.shape[:2] != (batch, views):
                raise ValueError("DA3DualDPT intermediates must share the same batch/view axes")
            stage = self._project_stage(fm.data, grid, i, image_size)
            resized.append(stage)
            if taps is not None:
                taps[f"projected_{i}"] = stage

        fused_main, aux_pyr = self._fuse(resized, taps)
        h_out = int(image_size[0] / self.cfg.down_ratio)
        w_out = int(image_size[1] / self.cfg.down_ratio)

        fused_main = resize_bilinear_align_corners(fused_main, (h_out, w_out))
        if self.cfg.pos_embed:
            fused_main = self._add_pos_embed(fused_main, image_size)
        main_logits = self.scratch.output_conv2[0](fused_main)
        main_logits = _relu(main_logits)
        main_logits = self.scratch.output_conv2[2](main_logits)

        last_aux = aux_pyr[-1]
        if self.cfg.pos_embed:
            last_aux = self._add_pos_embed(last_aux, image_size)
        aux_logits = self._apply_aux_out2(last_aux, len(aux_pyr) - 1)

        if taps is not None:
            taps["main_logits"] = main_logits
            taps["aux_logits"] = aux_logits

        main_pred = self._activate(main_logits[..., :-1], self.activation)
        main_conf = self._activate(main_logits[..., -1], self.conf_activation)
        aux_pred = self._activate(aux_logits[..., :-1], "linear")
        aux_conf = self._activate(aux_logits[..., -1], self.conf_activation)

        if main_pred.shape[-1] == 1:
            main_pred = main_pred[..., 0]
        data: dict[str, mx.array | dict[str, mx.array]] = {
            self.head_main: main_pred.reshape(batch, views, *main_pred.shape[1:]),
            f"{self.head_main}_conf": main_conf.reshape(batch, views, *main_conf.shape[1:]),
            self.head_aux: aux_pred.reshape(batch, views, *aux_pred.shape[1:]),
            f"{self.head_aux}_conf": aux_conf.reshape(batch, views, *aux_conf.shape[1:]),
        }
        if taps is not None:
            data["taps"] = taps
        return HeadOutput(data=data)


@register_head("da3-dualdpt")
def build_da3_dualdpt(config) -> DA3DualDPT:
    cfg = config if isinstance(config, DA3DualDPTConfig) else DA3DualDPTConfig.from_dict(config)
    return DA3DualDPT(cfg)
