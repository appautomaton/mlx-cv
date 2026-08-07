"""MLX EoMT-DINOv3 universal segmentation model."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ...backbones.vision.dinov3 import DINOv3ViT
from ...core.features import HeadOutput
from ...core.registry import register_model
from ...core.types import Result
from .config import EoMTDINOv3Config

__all__ = ["EoMTDINOv3", "build_eomt_dinov3"]


def _resize_axis_half_pixel(x: mx.array, out_size: int, axis: int) -> mx.array:
    in_size = int(x.shape[axis])
    if in_size == out_size:
        return x
    coordinates = (mx.arange(out_size, dtype=mx.float32) + 0.5) * (in_size / out_size) - 0.5
    lower_raw = mx.floor(coordinates).astype(mx.int32)
    upper_raw = lower_raw + 1
    lower = mx.clip(lower_raw, 0, in_size - 1)
    upper = mx.clip(upper_raw, 0, in_size - 1)
    weight = coordinates - lower_raw.astype(mx.float32)
    left = mx.take(x, lower, axis=axis)
    right = mx.take(x, upper, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = out_size
    weight = weight.reshape(shape).astype(x.dtype)
    return left * (1.0 - weight) + right * weight


def _resize_bilinear_nchw(x: mx.array, size: tuple[int, int]) -> mx.array:
    x = _resize_axis_half_pixel(x, int(size[0]), axis=2)
    return _resize_axis_half_pixel(x, int(size[1]), axis=3)


class EoMTDINOv3ScaleLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            padding=1,
            groups=hidden_size,
            bias=False,
        )
        self.layernorm2d = nn.LayerNorm(hidden_size, eps=1e-6)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        return self.layernorm2d(hidden_states)


class EoMTDINOv3ScaleBlock(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int) -> None:
        super().__init__()
        self.block = [EoMTDINOv3ScaleLayer(hidden_size) for _ in range(num_blocks)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for block in self.block:
            hidden_states = block(hidden_states)
        return hidden_states


class EoMTDINOv3MaskHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = nn.gelu(self.fc2(hidden_states))
        return self.fc3(hidden_states)


class EoMTDINOv3(nn.Module):
    """DINOv3-small with query insertion and mask-classification heads."""

    def __init__(self, cfg: EoMTDINOv3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv3ViT(cfg.backbone)
        self.query = nn.Embedding(cfg.num_queries, cfg.backbone.embed_dim)
        self.upscale_block = EoMTDINOv3ScaleBlock(
            cfg.backbone.embed_dim, cfg.num_upscale_blocks
        )
        self.mask_head = EoMTDINOv3MaskHead(cfg.backbone.embed_dim)
        self.class_predictor = nn.Linear(cfg.backbone.embed_dim, cfg.num_classes + 1)
        # This checkpointed inference control is not a disposable training buffer:
        # the official full package stores -1 to disable intermediate mask attention.
        self.attn_mask_probs = mx.ones((cfg.num_blocks,), dtype=mx.float32)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        config: EoMTDINOv3Config | dict | None = None,
        strict: bool = True,
        revision: str | None = None,
        cache_dir=None,
        local_files_only: bool | None = None,
        token: str | bool | None = None,
    ) -> "EoMTDINOv3":
        """Load an official full package or a converted local model package."""

        from ...hub import resolve_model_package
        from .convert import load_eomt_dinov3_weights

        package = resolve_model_package(
            pretrained_model_name_or_path,
            require_config=config is None,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        if config is None:
            config = _eomt_package_config(package.config)
        cfg = config if isinstance(config, EoMTDINOv3Config) else EoMTDINOv3Config.from_dict(config)
        return load_eomt_dinov3_weights(cls(cfg), package.weights, strict=strict)

    def _predict(
        self,
        hidden_states: mx.array,
        grid: tuple[int, int],
    ) -> tuple[mx.array, mx.array]:
        queries = hidden_states[:, : self.cfg.num_queries]
        class_logits = self.class_predictor(queries)
        prefix = self.cfg.num_queries + 1 + self.cfg.backbone.n_storage_tokens
        patches = hidden_states[:, prefix:]
        batch, _, channels = patches.shape
        patches = patches.reshape(batch, int(grid[0]), int(grid[1]), channels)
        patches = self.upscale_block(patches)
        mask_queries = self.mask_head(queries)
        mask_logits = mx.einsum("bqc,bhwc->bqhw", mask_queries, patches)
        return mask_logits, class_logits

    def _attention_mask(
        self,
        hidden_states: mx.array,
        mask_logits: mx.array,
        grid: tuple[int, int],
    ) -> mx.array:
        batch, tokens, _ = hidden_states.shape
        prefix = self.cfg.num_queries + 1 + self.cfg.backbone.n_storage_tokens
        query_to_patch = _resize_bilinear_nchw(mask_logits, grid).reshape(
            batch, self.cfg.num_queries, -1
        ) > 0
        query_prefix = mx.ones(
            (batch, self.cfg.num_queries, prefix), dtype=mx.bool_
        )
        query_rows = mx.concatenate([query_prefix, query_to_patch], axis=-1)
        other_rows = mx.ones(
            (batch, tokens - self.cfg.num_queries, tokens), dtype=mx.bool_
        )
        return mx.concatenate([query_rows, other_rows], axis=1)

    def __call__(self, pixel_values: mx.array, *, capture_taps: bool = False) -> HeadOutput:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"EoMTDINOv3 expects NCHW input, got shape {tuple(pixel_values.shape)}"
            )
        batch, channels, height, width = pixel_values.shape
        if channels != self.cfg.backbone.in_chans:
            raise ValueError(
                f"EoMTDINOv3 expects {self.cfg.backbone.in_chans} channels, got {channels}"
            )
        if (height, width) != (self.cfg.image_size, self.cfg.image_size):
            raise ValueError(
                "EoMTDINOv3 expects the configured square model input "
                f"{self.cfg.image_size}x{self.cfg.image_size}, got {height}x{width}"
            )

        patches, grid = self.backbone.patch_embed(pixel_values)
        cls = mx.broadcast_to(self.backbone.cls_token, (batch, 1, self.cfg.backbone.embed_dim))
        if self.cfg.backbone.n_storage_tokens:
            storage = mx.broadcast_to(
                self.backbone.storage_tokens,
                (batch, self.cfg.backbone.n_storage_tokens, self.cfg.backbone.embed_dim),
            )
            hidden_states = mx.concatenate([cls, storage, patches], axis=1)
        else:
            hidden_states = mx.concatenate([cls, patches], axis=1)
        rope = self.backbone.position.rope(self.backbone, *grid)
        query_start = self.cfg.backbone.depth - self.cfg.num_blocks
        attention_mask = None
        masks_per_layer: list[mx.array] = []
        classes_per_layer: list[mx.array] = []
        taps: dict[str, mx.array] = {}
        if capture_taps:
            taps["patch_embed"] = hidden_states

        for index, block in enumerate(self.backbone.blocks):
            if index == query_start:
                queries = mx.broadcast_to(
                    self.query.weight,
                    (batch, self.cfg.num_queries, self.cfg.backbone.embed_dim),
                )
                hidden_states = mx.concatenate([queries, hidden_states], axis=1)
                if capture_taps:
                    taps["query_insertion"] = hidden_states

            mask_index = index - query_start
            if index >= query_start and float(self.attn_mask_probs[mask_index].item()) > 0:
                normalized = self.backbone.norm(hidden_states)
                mask_logits, class_logits = self._predict(normalized, grid)
                masks_per_layer.append(mask_logits)
                classes_per_layer.append(class_logits)
                attention_mask = self._attention_mask(hidden_states, mask_logits, grid)
                n_prefix = self.cfg.num_queries + 1 + self.cfg.backbone.n_storage_tokens
                if capture_taps:
                    taps[f"mask_logits_{index:02d}"] = mask_logits
                    taps[f"class_logits_{index:02d}"] = class_logits
                    taps[f"query_patch_attention_mask_{index:02d}"] = attention_mask[
                        :, : self.cfg.num_queries, n_prefix:
                    ]
            else:
                n_prefix = (
                    1 + self.cfg.backbone.n_storage_tokens
                    if index < query_start
                    else self.cfg.num_queries + 1 + self.cfg.backbone.n_storage_tokens
                )

            hidden_states = block(
                hidden_states,
                rope=rope,
                n_prefix=n_prefix,
                attention_mask=attention_mask,
            )
            if capture_taps:
                taps[f"block_{index:02d}"] = hidden_states

        sequence_output = self.backbone.norm(hidden_states)
        mask_logits, class_logits = self._predict(sequence_output, grid)
        masks_per_layer.append(mask_logits)
        classes_per_layer.append(class_logits)
        if capture_taps:
            taps["final_hidden_state"] = sequence_output
            taps["final_mask_logits"] = mask_logits
            taps["final_class_logits"] = class_logits

        data = {
            "masks_queries_logits": mask_logits,
            "class_queries_logits": class_logits,
            "last_hidden_state": sequence_output,
            "masks_queries_logits_per_layer": tuple(masks_per_layer),
            "class_queries_logits_per_layer": tuple(classes_per_layer),
        }
        if capture_taps:
            data["taps"] = taps
        return HeadOutput(data=data)

    def predict(self, image, *, processor=None, **opts) -> Result:
        """Run preprocessing, EoMT inference, and panoptic postprocessing."""

        if processor is None:
            from .processor import EoMTDINOv3Processor

            processor = EoMTDINOv3Processor.from_model_config(self.cfg, **opts)
        elif opts:
            raise ValueError("EoMTDINOv3.predict options require the default processor")
        model_input, context = processor.preprocess(image)
        return processor.postprocess(self(**model_input), context)


@register_model("eomt-dinov3-coco-panoptic-small-640")
def build_eomt_dinov3(config) -> EoMTDINOv3:
    cfg = config if isinstance(config, EoMTDINOv3Config) else EoMTDINOv3Config.from_dict(config)
    return EoMTDINOv3(cfg)


def _eomt_package_config(payload: dict | None) -> EoMTDINOv3Config:
    data = payload or {}
    model_type = data.get("model_type")
    if model_type not in (None, "eomt_dinov3", "eomt-dinov3"):
        raise ValueError(f"EoMT package declares unsupported model_type {model_type!r}")
    variant = data.get("variant")
    if variant not in (None, "coco-panoptic-small-640", "small-640"):
        raise ValueError(f"EoMT package declares unsupported variant {variant!r}")
    return EoMTDINOv3Config.from_dict(data)
