"""SAM 3.1 image and video model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...backbones.vision.sam3 import SAM3ImageBackboneConfig
from ...heads.segmentation import SAM3DecoderConfig
from .text import SAM3TextConfig

__all__ = [
    "SAM3Config",
    "SAM3MultiplexDecoderConfig",
    "SAM3VideoConfig",
    "SAM3VideoMemoryConfig",
    "SAM3VideoTrackerConfig",
]


@dataclass(frozen=True)
class SAM3Config:
    image: SAM3ImageBackboneConfig = field(default_factory=SAM3ImageBackboneConfig)
    text: SAM3TextConfig = field(default_factory=SAM3TextConfig)
    decoder: SAM3DecoderConfig = field(default_factory=SAM3DecoderConfig)

    def __post_init__(self) -> None:
        if self.image.neck_channels != self.decoder.hidden_dim:
            raise ValueError("SAM3 image neck_channels must match decoder hidden_dim")
        if self.image.text_dim != self.text.d_model:
            raise ValueError("SAM3 image text_dim must match text encoder d_model")
        if self.decoder.text_dim != self.text.d_model:
            raise ValueError("SAM3 decoder text_dim must match text encoder d_model")


@dataclass(frozen=True)
class SAM3VideoMemoryConfig:
    """Shape-bearing memory encoder config for SAM3 video inference."""

    hidden_dim: int = 256
    image_size: tuple[int, int] = (1024, 1024)
    feature_grid: tuple[int, int] = (64, 64)
    multiplex_count: int = 16
    condition_as_mask_input: bool = True
    mask_downsample_stride: int = 4
    mask_total_stride: int = 16
    fuser_layers: int = 2

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.multiplex_count, self.mask_downsample_stride, self.mask_total_stride) <= 0:
            raise ValueError("SAM3 video memory dimensions must be positive")
        if self.mask_total_stride % self.mask_downsample_stride != 0:
            raise ValueError("SAM3 video mask_total_stride must be divisible by mask_downsample_stride")
        if min(int(v) for v in self.image_size) <= 0:
            raise ValueError("SAM3 video image_size must be positive")
        if min(int(v) for v in self.feature_grid) <= 0:
            raise ValueError("SAM3 video feature_grid must be positive")

    @property
    def mask_input_channels(self) -> int:
        multiplier = 2 if self.condition_as_mask_input else 1
        return self.multiplex_count * multiplier


@dataclass(frozen=True)
class SAM3MultiplexDecoderConfig:
    """Object Multiplex mask decoder config for bucket-space propagation."""

    hidden_dim: int = 256
    multiplex_count: int = 16
    low_res_mask_size: tuple[int, int] = (256, 256)
    high_res_mask_size: tuple[int, int] = (1024, 1024)
    num_multimask_outputs: int = 0
    pred_obj_scores: bool = True
    decode_mask_with_shared_tokens: bool = False
    decode_mask_attribute_with_shared_tokens: bool = False
    iou_prediction_use_sigmoid: bool = False

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.multiplex_count) <= 0:
            raise ValueError("SAM3 multiplex decoder dimensions must be positive")
        if min(int(v) for v in self.low_res_mask_size) <= 0:
            raise ValueError("SAM3 multiplex decoder low_res_mask_size must be positive")
        if min(int(v) for v in self.high_res_mask_size) <= 0:
            raise ValueError("SAM3 multiplex decoder high_res_mask_size must be positive")
        if self.num_multimask_outputs < 0:
            raise ValueError("SAM3 multiplex decoder num_multimask_outputs must be non-negative")
        if self.decode_mask_with_shared_tokens or self.decode_mask_attribute_with_shared_tokens:
            raise NotImplementedError("shared Object Multiplex token decoding is not ported in MLX yet")


@dataclass(frozen=True)
class SAM3VideoTrackerConfig:
    """Tracker-core config for the SAM3 video Object Multiplex path."""

    hidden_dim: int = 256
    image_size: tuple[int, int] = (1024, 1024)
    feature_grid: tuple[int, int] = (64, 64)
    multiplex_count: int = 16
    num_maskmem: int = 7
    max_obj_ptrs_in_encoder: int = 16
    condition_as_mask_input: bool = True
    condition_as_mask_input_fg: float = 1.0
    condition_as_mask_input_bg: float = 0.0
    apply_sigmoid_to_mask_logits_for_mem_enc: bool = True
    sigmoid_scale_for_mem_enc: float = 2.0
    sigmoid_bias_for_mem_enc: float = -1.0
    use_obj_ptrs_in_encoder: bool = True
    save_image_features: bool = True
    use_maskmem_tpos_v2: bool = True

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.multiplex_count, self.num_maskmem, self.max_obj_ptrs_in_encoder) <= 0:
            raise ValueError("SAM3 video tracker dimensions must be positive")
        if min(int(v) for v in self.image_size) <= 0:
            raise ValueError("SAM3 video tracker image_size must be positive")
        if min(int(v) for v in self.feature_grid) <= 0:
            raise ValueError("SAM3 video tracker feature_grid must be positive")


@dataclass(frozen=True)
class SAM3VideoConfig:
    """SAM3 video Object Multiplex module assembly config."""

    tracker: SAM3VideoTrackerConfig = field(default_factory=SAM3VideoTrackerConfig)
    memory: SAM3VideoMemoryConfig | None = None
    decoder: SAM3MultiplexDecoderConfig | None = None

    def __post_init__(self) -> None:
        memory = self.memory or SAM3VideoMemoryConfig(
            hidden_dim=self.tracker.hidden_dim,
            image_size=self.tracker.image_size,
            feature_grid=self.tracker.feature_grid,
            multiplex_count=self.tracker.multiplex_count,
            condition_as_mask_input=self.tracker.condition_as_mask_input,
        )
        decoder = self.decoder or SAM3MultiplexDecoderConfig(
            hidden_dim=self.tracker.hidden_dim,
            multiplex_count=self.tracker.multiplex_count,
            high_res_mask_size=self.tracker.image_size,
        )
        if memory.hidden_dim != self.tracker.hidden_dim:
            raise ValueError("SAM3 video memory hidden_dim must match tracker hidden_dim")
        if memory.multiplex_count != self.tracker.multiplex_count:
            raise ValueError("SAM3 video memory multiplex_count must match tracker multiplex_count")
        if decoder.hidden_dim != self.tracker.hidden_dim:
            raise ValueError("SAM3 video decoder hidden_dim must match tracker hidden_dim")
        if decoder.multiplex_count != self.tracker.multiplex_count:
            raise ValueError("SAM3 video decoder multiplex_count must match tracker multiplex_count")
        object.__setattr__(self, "memory", memory)
        object.__setattr__(self, "decoder", decoder)

    @classmethod
    def tiny_fixture(cls) -> "SAM3VideoConfig":
        tracker = SAM3VideoTrackerConfig(
            hidden_dim=16,
            image_size=(32, 32),
            feature_grid=(2, 2),
            multiplex_count=2,
            num_maskmem=3,
            max_obj_ptrs_in_encoder=4,
            condition_as_mask_input=True,
            save_image_features=True,
        )
        memory = SAM3VideoMemoryConfig(
            hidden_dim=16,
            image_size=(32, 32),
            feature_grid=(2, 2),
            multiplex_count=2,
            condition_as_mask_input=True,
        )
        decoder = SAM3MultiplexDecoderConfig(
            hidden_dim=16,
            multiplex_count=2,
            low_res_mask_size=(8, 8),
            high_res_mask_size=(32, 32),
            num_multimask_outputs=0,
            pred_obj_scores=True,
        )
        return cls(tracker=tracker, memory=memory, decoder=decoder)
