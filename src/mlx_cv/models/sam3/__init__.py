"""SAM 3.1 image-mode model family."""

from __future__ import annotations

from ...backbones.vision.necks import SAM3FeatureNeck, SAM3FeaturePyramid, SAM3PyramidLevel
from ...backbones.vision.sam3 import SAM3ImageBackbone, SAM3ImageBackboneConfig
from ...heads.segmentation import SAM3DecoderConfig, SAM3ImageDecoder, SAM3MaskDecoder
from .config import (
    SAM3Config,
    SAM3MultiplexDecoderConfig,
    SAM3VideoConfig,
    SAM3VideoMemoryConfig,
    SAM3VideoTrackerConfig,
)
from .convert import (
    convert_sam3_state_dict,
    convert_sam3_video_state_dict,
    inspect_sam3_video_state_dict,
    load_sam3_video_weights,
    load_sam3_weights,
    remap_sam3_key,
    remap_sam3_video_key,
)
from .multiplex_decoder import MLP, SAM3MultiplexDecoderOutput, SAM3MultiplexMaskDecoder
from .multiplex_state import SAM3MultiplexController, SAM3MultiplexState
from .modeling import SAM3FeatureExtractor, SAM3Model
from .processor import SAM3Processor, SAM3ProcessorConfig, SAM3ProcessorContext
from .prompts import SAM3PreparedPrompt, SAM3PromptBundle, normalize_sam3_prompt, prepare_sam3_prompt
from .text import SAM3TextConfig, SAM3TextEncoder, SAM3TextOutput
from .tokenizer import SAM3Tokenizer, bytes_to_unicode, canonicalize_text, default_bpe_path
from .video_memory import (
    SAM3MaskDownSampler,
    SAM3MemoryCXBlock,
    SAM3MemoryEncoder,
    SAM3MemoryEncoderOutput,
    SAM3MemoryFuser,
    SAM3MemoryMaskInput,
    bucket_features_to_object_space,
    build_multiplex_memory_mask_input,
    mask_logits_for_memory,
)
from .video_model import SAM3VideoFrameOutput, SAM3VideoModel
from .video_tracking import SAM3VideoMultiplexTrackerCore, SAM3VideoStageOutput
from .video import (
    SAM3VideoFrameContext,
    SAM3VideoProcessor,
    SAM3VideoProcessorConfig,
    SAM3VideoProcessorContext,
    SAM3VideoPrompt,
    SAM3VideoSessionManager,
    SAM3VideoSessionState,
    SAM3VideoTracker,
)
from .real_video_config import (
    Sam3TrackerMaskDecoderConfig,
    Sam3TrackerPromptEncoderConfig,
    Sam3TrackerVideoConfig,
)
from .real_video_model import (
    Sam3TrackerStageOutput,
    Sam3TrackerVideoModel,
    Sam3VideoModel,
    build_sam3_video_real,
)
from .real_video_association import (
    AssociationResult,
    Sam3AssociationConfig,
    Sam3TrackKeepAlive,
    associate_detections,
    mask_iou,
)
from .real_video_streaming import Sam3VideoFrameResult, Sam3VideoMultiObjectTracker, Sam3VideoSession
from .real_convert import (
    convert_reference_shape,
    load_sam3_video_real_weights,
    remap_sam3_video_real_key,
)

__all__ = [
    "SAM3Config",
    "SAM3DecoderConfig",
    "SAM3FeatureNeck",
    "SAM3FeatureExtractor",
    "SAM3FeaturePyramid",
    "SAM3ImageDecoder",
    "SAM3ImageBackbone",
    "SAM3ImageBackboneConfig",
    "SAM3MaskDecoder",
    "SAM3MaskDownSampler",
    "SAM3MemoryCXBlock",
    "SAM3MemoryEncoder",
    "SAM3MemoryEncoderOutput",
    "SAM3MemoryFuser",
    "SAM3MemoryMaskInput",
    "SAM3Model",
    "SAM3MultiplexController",
    "SAM3MultiplexDecoderConfig",
    "SAM3MultiplexDecoderOutput",
    "SAM3MultiplexMaskDecoder",
    "SAM3MultiplexState",
    "SAM3PreparedPrompt",
    "SAM3PromptBundle",
    "SAM3PyramidLevel",
    "SAM3Processor",
    "SAM3ProcessorConfig",
    "SAM3ProcessorContext",
    "SAM3TextConfig",
    "SAM3TextEncoder",
    "SAM3TextOutput",
    "SAM3Tokenizer",
    "SAM3VideoConfig",
    "SAM3VideoFrameContext",
    "SAM3VideoFrameOutput",
    "SAM3VideoMemoryConfig",
    "SAM3VideoModel",
    "SAM3VideoMultiplexTrackerCore",
    "SAM3VideoProcessor",
    "SAM3VideoProcessorConfig",
    "SAM3VideoProcessorContext",
    "SAM3VideoPrompt",
    "SAM3VideoSessionManager",
    "SAM3VideoSessionState",
    "SAM3VideoStageOutput",
    "SAM3VideoTracker",
    "SAM3VideoTrackerConfig",
    "Sam3TrackerMaskDecoderConfig",
    "Sam3TrackerPromptEncoderConfig",
    "Sam3TrackerStageOutput",
    "Sam3TrackerVideoConfig",
    "Sam3TrackerVideoModel",
    "Sam3VideoFrameResult",
    "Sam3VideoModel",
    "Sam3VideoMultiObjectTracker",
    "Sam3VideoSession",
    "Sam3AssociationConfig",
    "Sam3TrackKeepAlive",
    "AssociationResult",
    "associate_detections",
    "mask_iou",
    "MLP",
    "bytes_to_unicode",
    "bucket_features_to_object_space",
    "build_multiplex_memory_mask_input",
    "build_sam3_video_real",
    "canonicalize_text",
    "convert_reference_shape",
    "convert_sam3_state_dict",
    "convert_sam3_video_state_dict",
    "default_bpe_path",
    "inspect_sam3_video_state_dict",
    "load_sam3_video_real_weights",
    "load_sam3_video_weights",
    "load_sam3_weights",
    "mask_logits_for_memory",
    "normalize_sam3_prompt",
    "prepare_sam3_prompt",
    "remap_sam3_key",
    "remap_sam3_video_key",
    "remap_sam3_video_real_key",
]
