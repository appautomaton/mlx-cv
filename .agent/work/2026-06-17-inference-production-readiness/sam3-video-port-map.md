# SAM3 Video Real-Inference Port Map

Change: `2026-06-17-inference-production-readiness`
Slice: `3: SAM3V-NN-audit`
Status: planning artifact only; no `src/mlx_cv/` behavior change in this slice.

## Boundary

This port is inference-only. The target is the smallest SAM 3.1 video/Object Multiplex path that can produce model-derived masks, track ids, object scores, and multiplex metadata from user-supplied checkpoints.

Included:

- Session and streaming inference through the existing `SAM3VideoSessionManager` and `SAM3VideoTracker` surfaces.
- MLX-native multiplex state, bucket mux/demux, conditional mask-channel memory encoder, tracker/memory attention path, SAM-style interactive path, Object Multiplex propagation mask decoder, per-frame state updates, checkpoint conversion/load, and stable tap capture.
- Tiny random-weight fixtures for shape and control-flow tests, plus later upstream-vs-local numeric taps when a real checkpoint is supplied.

Excluded:

- Training, loss paths, DDP/SPMD behavior, torch compile, CUDA/offload behavior, benchmark evaluator helpers, checkpoint downloads, and any torch/reference imports in `src/mlx_cv/`.
- Point and mask-prompt video interactivity beyond the current explicit rejection, unless a later slice deliberately expands that scope.
- Release parity matrix expansion in this slice. `sam3_video` remains bounded until the final parity-matrix slice decides membership from the implemented comparison truth.

## Reference Surface Map

| Reference surface | Inference role observed | Planned local module/owner | Port notes |
|---|---|---|---|
| `references/sam3/sam3/model_builder.py` | SAM3.1 video entrypoint. `build_sam3_multiplex_video_predictor` is the checkpoint-backed runtime path: it builds `VideoTrackingDynamicMultiplex`/`Sam3VideoTrackingMultiplexDemo`, wraps it in `Sam3MultiplexPredictorWrapper`, builds `Sam3MultiplexDetector`, then assembles `Sam3MultiplexTrackingWithInteractivity` and `Sam3MultiplexVideoPredictor`. | `src/mlx_cv/models/sam3/video_model.py` owns the local assembly boundary; `src/mlx_cv/models/sam3/config.py` owns video/multiplex config defaults. | Slices 4-5 should target this multiplex stack, not the older non-multiplex `Sam3TrackerBase` path. Required defaults to carry forward include `multiplex_count`, `condition_as_mask_input=True`, `apply_sigmoid_to_mask_logits_for_mem_enc=True`, `sigmoid_scale=2.0`, `sigmoid_bias=-1.0`, `use_obj_ptrs_in_encoder=True`, `save_image_features=True`, and `use_maskmem_tpos_v2=True`. |
| `references/sam3/sam3/model/sam3_video_base.py` | Base video orchestrator. Demo inference flows through `_det_track_one_frame`, not `forward`. It runs detector/backbone capture, tracker propagation, detection-track association, memory update, new-object insertion/removal, output assembly, and checkpoint loading. | Keep public API in `src/mlx_cv/models/sam3/video.py`; add an MLX engine owner `src/mlx_cv/models/sam3/video_model.py` for `SAM3VideoModel` and `SAM3VideoFrameOutput`; use existing `src/mlx_cv/core/tracking.py` only for public metadata. | Use as orchestration context only. The SAM3.1 real checkpoint path is the multiplex subclass stack below. Start single-process/single-device and omit distributed all-gather/broadcast, evaluator RLE output, and torch checkpoint load. |
| `references/sam3/sam3/model/sam3_tracker_base.py` | Non-multiplex tracker ancestor. It documents common inference primitives: memory-conditioned feature fusion, SAM heads, object pointer creation, and memory encoding. | Shared helper behavior may live in `src/mlx_cv/models/sam3/video_tracking.py`, but the primary local tracker owner must implement the multiplex signatures from `video_tracking_multiplex.py`. | Do not implement Slice 4 against this object-space-only class alone; that would miss bucket-space tensors and checkpoint shapes. Use it only as a fallback reference for shared math where the multiplex file delegates similar behavior. |
| `references/sam3/sam3/model/multiplex_utils.py` | Defines `MultiplexController` and `MultiplexState`: fixed-size bucket assignments, `_PADDING_NUM`, `_REMOVED_NUM`, `mux`, `demux`, valid masks, available slots, add/remove object behavior, and optional public `object_ids`. | Add `src/mlx_cv/models/sam3/multiplex_state.py` for `SAM3MultiplexController`, `SAM3MultiplexState`, and MLX-native mux/demux matrices; bridge public ids to existing `ObjectMultiplexState` in `src/mlx_cv/core/tracking.py`. | This is a required Slice 4 dependency. Tensor state moves between object/data space `(O, ...)` and multiplex space `(Buc, M, ...)`; fixture and taps must assert both. |
| `references/sam3/sam3/model/video_tracking_multiplex.py` | Primary tracker for SAM3.1. `VideoTrackingMultiplex.track_step(..., multiplex_state=...)` branches between mask-as-output, propagation-only, interaction-only, and propagation-and-interaction. Propagation runs `MultiplexMaskDecoder` in bucket space, demuxes masks/scores/tokens to object space, stores object pointers in mux space, uses conditional mask channels for memory encoding, and demuxes memory features/pos enc when reading them. `VideoTrackingDynamicMultiplex` adds object addition/reconditioning with state mutation. | Add `src/mlx_cv/models/sam3/video_tracking.py` for `SAM3VideoMultiplexTrackerCore`, `SAM3VideoStageOutput`, `track_step`, `_prepare_memory_conditioned_features`, `_encode_new_memory`, `_use_mask_as_output`, `_forward_multiplex_sam_heads`, object pointer mux/demux, and tap capture. | This replaces the earlier object-space-only tracker target. Port inference branches only; keep correction-point sampling and training transition scheduling out unless needed for a later scoped interactivity slice. Preserve the distinction between `pred_masks` object/data space and stored `obj_ptr` mux space. |
| `references/sam3/sam3/model/video_tracking_multiplex_demo.py` | Demo/runtime state manager for the multiplex tracker. It initializes inference state, maps client object ids to model object indices, owns `multiplex_state`, per-object output dictionaries, temp outputs, and consolidated frame outputs. | `src/mlx_cv/models/sam3/video.py` and `src/mlx_cv/models/sam3/video_model.py` share ownership: public sessions stay in `video.py`; model-side inference state and object-index mapping live in `video_model.py`/`video_tracking.py`. | Needed to avoid losing the object id to object index boundary. The local session should maintain public `object_id` while the core uses sequential object indices for mux/demux. |
| `references/sam3/sam3/model/sam3_multiplex_base.py` | Multiplex-aware video orchestrator. Extends planning metadata with `num_buc_per_gpu`, counts buckets after execution, adds new objects into dynamic multiplex states, and can reapply no-object pointers after suppression. | `src/mlx_cv/models/sam3/video_model.py` owns single-device planning/execution; `src/mlx_cv/models/sam3/multiplex_state.py` owns bucket counts and add/remove semantics. | Port the single-device metadata semantics needed for inference: bucket counts, available slots, object addition, object removal, and no-object pointer reapplication if enabled. Omit SPMD/GPU metadata. |
| `references/sam3/sam3/model/sam3_multiplex_tracking.py` | High-level runtime wrapper with interactivity and state extraction. `Sam3MultiplexTrackingWithInteractivity` extracts/demuxes per-object memory, pos enc, and object pointers, then remuxes singleton states. | `src/mlx_cv/models/sam3/video_model.py` owns interactivity-compatible state extraction hooks; initial Slice 5 can expose clear errors for unsupported point/mask refinement while preserving state shape contracts. | Even if point interactivity remains deferred, this file documents how object-space slices are extracted from mux-space tensors. That contract must inform state serialization, removal, and future interactivity hooks. |
| `references/sam3/sam3/model/memory.py` | Memory encoder. In multiplex mode the same `SimpleMaskEncoder` receives bucket-space mask channels from `multiplex_state.mux(mask_for_mem).squeeze(2)`, optionally concatenated with conditional mask-input channels. | Add `src/mlx_cv/models/sam3/video_memory.py` for `SAM3MaskDownSampler`, `SAM3MemoryCXBlock`, `SAM3MemoryFuser`, `SAM3MemoryEncoder`, and a small positional encoding helper. | MLX `Conv2d` kernels need local layout handling consistent with existing converter conventions. Port `multiplex_count` and `input_channel_multiplier` so conditional mask channels match checkpoint shapes. |
| `references/sam3/sam3/model/multiplex_mask_decoder.py` | Object Multiplex propagation decoder. It builds per-bucket/per-slot mask tokens, optional object-score/iou tokens, upscales transformer features, applies hypernetwork MLPs, predicts bucket-space masks, IoU, object scores, and `sam_tokens_out`. | Add `src/mlx_cv/models/sam3/multiplex_decoder.py` for `SAM3MultiplexMaskDecoder` and local `MLP`. `video_tracking.py` calls it only through multiplex-state-aware tracker methods. | Port `forward`, `predict_masks`, dynamic stability fallback, high-res feature fusion, object-score logits, `sam_tokens_out`, and `extra_per_object_embeddings`. Outputs are bucket-space until `multiplex_state.demux` is applied. |

## Local Runtime Ownership

- `src/mlx_cv/models/sam3/video.py`: public session/streaming API, prompt classification, frame preprocessing context, and final `Result`/`VideoResult` mapping. Slice 5 should add a `model`/`engine` injection point and remove the hard dependency on `_deterministic_box`/`_box_mask` for real inference.
- `src/mlx_cv/models/sam3/video_model.py`: per-frame inference coordinator analogous to the single-device subset of `Sam3VideoBase`.
- `src/mlx_cv/models/sam3/multiplex_state.py`: model-side multiplex controller/state, `_PADDING_NUM`/`_REMOVED_NUM`, bucket assignment, mux/demux matrices, valid-object masks, available slots, add/remove object mutation, and object-id bookkeeping.
- `src/mlx_cv/models/sam3/video_tracking.py`: dynamic multiplex tracker core, memory-conditioned features, SAM-style interactive path, bucket-space propagation decoder calls, object pointer mux/demux, conditional mask-channel memory encoding, and state updates.
- `src/mlx_cv/models/sam3/video_memory.py`: multiplex-aware memory encoder components from `memory.py`.
- `src/mlx_cv/models/sam3/multiplex_decoder.py`: Object Multiplex mask decoder from `multiplex_mask_decoder.py`.
- `src/mlx_cv/models/sam3/convert.py`: keep image-mode rejection unchanged; add dedicated later video conversion/load entry points such as `convert_sam3_video_state_dict` and `load_sam3_video_weights`.
- `src/mlx_cv/models/sam3/config.py`: add video config dataclasses only when Slice 4 creates modules. Keep image config compatibility intact.

`src/mlx_cv/core/tracking.py:ObjectMultiplexState` remains the public result/session metadata type. It is not sufficient for model compute because it does not own numeric mux/demux matrices, padding/removed slots, or object-index remapping.

## Minimal Inference Data Flow

1. `SAM3VideoSessionManager.start_session` preprocesses frames and stores frame contexts.
2. `add_prompt` assigns public object ids. The model engine maps those ids to sequential object indices and creates `SAM3MultiplexState` bucket assignments.
3. The neural engine encodes text/geometric prompt evidence and frame image features.
4. For frame `t`, tracker core prepares interactive and propagation feature streams where configured.
5. If `t` is a prompt/conditioning frame, accepted mask/box-derived initialization uses the mask-as-output path and writes object-space masks plus mux-space object pointers. For this bounded port, point/mask prompt refinement remains excluded from the public API unless a later slice expands it.
6. On propagation frames, tracker core builds `memory_conditioned_features` with batch size equal to `num_buckets`, calls `SAM3MultiplexMaskDecoder` in bucket space, demuxes masks/IoU/object-score/token outputs to object space, and stores final object pointers back in mux space.
7. Memory encoder consumes object-space high-res masks, applies sigmoid/scale/bias, muxes them to bucket-space channels, optionally appends conditional mask channels, encodes memory, then demuxes stored `maskmem_features` and `maskmem_pos_enc` when the reference does so.
8. Video runtime maps object-space frame outputs into `Masks`, `Detections`, `Tracks`, `TrackMemoryRecord`, and public `ObjectMultiplexState`, while retaining model-side `SAM3MultiplexState` for subsequent frames.

## Tiny Fixture

Use one shared fixture shape for Slice 4 random-weight tests and Slice 5 wiring tests:

- Frames: `T=3`, RGB, `image_size=(32, 32)`.
- Image batch: `B_img=1`.
- Objects: `O=3` active object ids, public ids `[10, 11, 12]`, model object indices `[0, 1, 2]`.
- Multiplex config: `multiplex_count=M=2`, allowed bucket capacity `2`, `num_buckets=Buc=2`, assignments `[[0, 1], [2, -1]]`; the last slot is padding and must stay zeroed/ignored by demux.
- Hidden width: `C=16`, `num_heads=4`, `num_maskmem=3`, `max_obj_ptrs_in_encoder=4`.
- Propagation top feature grid: object-independent image features `(HW, B_img, C)` expand to bucket batch `(HW, Buc, C)` and `tracker.memory_conditioned_features` `(Buc, C, 2, 2)`.
- Interactive top feature grid: object/data-space path uses `(O, C, 2, 2)` when encoding prompt masks.
- Propagation high-res decoder features: `(Buc, C, 8, 8)` and `(Buc, C, 4, 4)`.
- Bucket-space decoder masks before demux: `(Buc, M, 1, 8, 8)` low-res and `(Buc, M, 1, 32, 32)` high-res.
- Object-space masks after demux: `(O, 1, 8, 8)` low-res and `(O, 1, 32, 32)` high-res.
- Memory encoder mask input before mux: `(O, 1, 32, 32)`.
- Memory encoder mask channels after mux/squeeze: `(Buc, M, 32, 32)`; with `condition_as_mask_input=True`, input channels become `(Buc, 2*M, 32, 32)`.
- Memory encoder output before any demux: `(Buc, C, 2, 2)`; stored object-space memory tap after demux is `(O, C, 2, 2)` if the local implementation follows the reference's storage normalization.
- Prompts: one text prompt (`"person"`), one text prompt (`"bag"`), and one box prompt (`[4, 4, 20, 20]`); no point or public mask-refinement prompts.

The fixture should assert mux/demux round-trip, padding-slot zero/ignore behavior, tensor shapes, deterministic random-seed repeatability, no NaNs/Infs, stable public object ids, stable model object indices, and `claim_level != "local_contract_fixture"` only after Slice 5 wiring exists.

## Comparison Taps

Future taps must be emitted by both reference tooling and local MLX code with names stable enough for `tools/sam3_video_upstream.py` to compare:

| Tap name | Shape in tiny fixture | Source point | Purpose |
|---|---:|---|---|
| `multiplex.assignments` | `[[0, 1], [2, -1]]` | `SAM3MultiplexState` | Exact bucket layout, including padding slot. |
| `multiplex.valid_mask` | `(2, 2)` | `get_valid_object_mask` | Confirms padded slots do not decode as objects. |
| `multiplex.object_ids` | `(3,)` | public-id bridge | Public id to model object-index mapping. |
| `multiplex.mux_probe` | `(2, 2, 1)` | test probe through `mux` | Ensures object/data space to bucket space mapping. |
| `multiplex.demux_probe` | `(3, 1)` | test probe through `demux` | Ensures bucket space to object/data space mapping. |
| `video.det.boxes_xyxy` | `(N, 4)` | after `run_backbone_and_detection` / local detector bridge | Detection-to-track association input. |
| `video.det.mask_logits_low_res` | `(N, 8, 8)` | detection output masks | Association and new-object initialization. |
| `video.det.scores` | `(N,)` | detection logits after sigmoid | New detection threshold and final scores. |
| `tracker.backbone.top_features.image_space` | `(1, 16, 2, 2)` | tracker backbone/FPN bridge before bucket expansion | Confirms image features entering tracker. |
| `tracker.memory_conditioned_features.bucket_space` | `(2, 16, 2, 2)` | output of `_prepare_memory_conditioned_features` | Main bucket-space tracker memory attention comparison. |
| `mask_decoder.masks.bucket_space` | `(2, 2, 1, 8, 8)` | `MultiplexMaskDecoder` before demux | Primary checkpoint-shape decoder tap. |
| `mask_decoder.low_res_masks.object_space` | `(3, 1, 8, 8)` | after `multiplex_state.demux` | Public/object-space mask tap. |
| `mask_decoder.high_res_masks.object_space` | `(3, 1, 32, 32)` | decoder upsample after demux | Final mask source before postprocess. |
| `mask_decoder.iou_pred.bucket_space` | `(2, 2, 1)` | decoder quality head before demux | Score tap in checkpoint runtime shape. |
| `mask_decoder.iou_pred.object_space` | `(3, 1)` | after demux | Score comparison and multimask selection. |
| `mask_decoder.object_score_logits.object_space` | `(3, 1)` | after demux | Object presence and memory gating. |
| `tracker.obj_ptr.object_space` | `(3, 16)` | object pointer projection before storage | Temporal pointer comparison before mux. |
| `tracker.obj_ptr.mux_space` | `(2, 2, 16)` | `current_out["obj_ptr"]` | Stored pointer shape used by memory attention. |
| `memory.mask_for_mem.object_space` | `(3, 1, 32, 32)` | before `multiplex_state.mux` | Confirms sigmoid/scale/bias behavior. |
| `memory.mask_for_mem.mux_space` | `(2, 2, 32, 32)` | after mux/squeeze | Confirms bucket-channel layout. |
| `memory.condition_mask_channels` | `(2, 2, 32, 32)` | conditional object channel build | Confirms `condition_as_mask_input` channels. |
| `memory.encoder_input_channels` | `(2, 4, 32, 32)` | before `SimpleMaskEncoder` | Checkpoint-shape memory encoder input with condition channels. |
| `memory.features.bucket_space` | `(2, 16, 2, 2)` | `SimpleMaskEncoder["vision_features"]` | Raw spatial memory comparison. |
| `memory.features.object_space` | `(3, 16, 2, 2)` | after reference-style demux if stored object-space | Stored spatial memory comparison. |
| `memory.pos_enc.object_space` | `(3, 16, 2, 2)` | stored `vision_pos_enc` after demux if applicable | Stored memory position comparison. |
| `memory.image_features` | `(HW, 1, 16)` | saved propagation image features | Required when `save_image_features=True`. |
| `memory.image_pos_enc` | `(HW, 1, 16)` | saved propagation image pos enc | Required when `save_image_features=True`. |
| `association.new_det_fa_inds` | `(K,)` | after `_associate_det_trk` | Metadata parity for new objects. |
| `association.det_to_track_ids` | JSON/dict | after `_associate_det_trk` | Track continuity parity. |
| `output.track_ids` | `(3,)` | final `Tracks` | Public identity output. |
| `output.masks_bool` | `(3, 32, 32)` | final postprocessed masks | User-visible mask parity. |
| `output.multiplex` | JSON/dict | `ObjectMultiplexState.to_dict()` | Public Object Multiplex metadata parity. |

Comparison tolerances for random-weight local tests are shape-only. Real-checkpoint numeric comparison should start with `atol=1e-4, rtol=1e-4` for direct tensor taps and use stricter exact comparison for ids, bucket assignment, and metadata. If MLX/PyTorch interpolation drift appears, the comparator should report the failing tap and only relax the relevant interpolation-derived mask tap with documented evidence.

## Conversion And Load Surfaces

Slice 5 should add video-specific conversion without making video checkpoints loadable through the image-mode path:

- Keep `convert_sam3_state_dict` rejecting video/tracker/memory/multiplex keys.
- Add `inspect_sam3_video_state_dict` coverage for the concrete key families found in admitted checkpoints.
- Add video converter/load functions with explicit key maps for:
  - `tracker.maskmem_backbone.*` to `video_tracking.memory_encoder.*` / `video_memory.*`, including multiplex mask downsampler channel counts from `multiplex_count` and `condition_as_mask_input`.
  - `tracker.interactive_sam_prompt_encoder.*`, `tracker.interactive_sam_mask_decoder.*`, `tracker.interactive_obj_ptr_proj.*` for prompt/mask-as-output initialization.
  - `tracker.sam_mask_decoder.*` to `multiplex_decoder.*` for bucket-space propagation.
  - `tracker.obj_ptr_proj.*`, `tracker.interactive_obj_ptr_proj.*`, `tracker.obj_ptr_tpos_proj.*`, `tracker.maskmem_tpos_enc`, `tracker.no_mem_embed`, `tracker.no_mem_pos_enc`, `tracker.no_obj_ptr`, `tracker.no_obj_ptr_linear.*`, `tracker.output_valid_embed`, `tracker.output_invalid_embed`, `tracker.no_obj_embed_spatial`, `tracker.obj_cond_embed`, and `tracker.obj_non_cond_embed` where present.
  - `detector.backbone.*` and detector/VL pieces that can reuse the existing SAM3 image feature extractor.
  - `multiplex_controller.*` and Object Multiplex decoder keys to `multiplex_state.*` / `multiplex_decoder.*` where they are learned or config-bearing.
- Apply the existing convolution layout rule consistently: reference PyTorch conv weights are transposed to MLX layout when needed.
- Fail clearly on unknown key families, unsupported checkpoint variants, and shape mismatches.

## Ordered Sub-Slices For Slices 4-5

Slice 4A sub-slice: add video/multiplex config dataclasses and tiny fixture builders under tests only.

- Owner files: `src/mlx_cv/models/sam3/config.py`, new fixture tests.
- Acceptance: construct a 32x32 random-weight video config with hidden width 16, `multiplex_count=2`, `condition_as_mask_input=True`, `save_image_features=True`, and no checkpoint.

Slice 4B sub-slice: port multiplex state.

- Owner files: `src/mlx_cv/models/sam3/multiplex_state.py`.
- Acceptance: `SAM3MultiplexState` creates assignments `[[0, 1], [2, -1]]` for three objects at `multiplex_count=2`; `mux` and `demux` round-trip object-space probes; valid masks, padding slots, add/remove object semantics, and public `object_ids` are covered.

Slice 4C sub-slice: port multiplex-aware memory encoder.

- Owner files: `src/mlx_cv/models/sam3/video_memory.py`.
- Acceptance: `SAM3MemoryEncoder` accepts bucket-space mask channels `(2, 4, 32, 32)` when conditional mask input is enabled, returns bucket-space `memory.features` `(2, 16, 2, 2)`, and supports object-space demux taps `(3, 16, 2, 2)`.

Slice 4D sub-slice: port multiplex mask decoder.

- Owner files: `src/mlx_cv/models/sam3/multiplex_decoder.py`.
- Acceptance: decoder returns bucket-space masks `(2, 2, 1, 8, 8)`, IoU `(2, 2, 1)`, object score logits, and `sam_tokens_out` for `multiplex_count=2`; unsupported token-sharing modes fail explicitly.

Slice 4E sub-slice: port dynamic multiplex tracker core.

- Owner files: `src/mlx_cv/models/sam3/video_tracking.py`.
- Acceptance: `track_step(..., multiplex_state=...)` supports mask-as-output initialization and propagation-only inference on the tiny fixture; it demuxes decoder masks to object space, stores object pointers in mux space, writes conditional mask-channel memory, and reads prior memory in the same shapes as `video_tracking_multiplex.py`.

Slice 4F sub-slice: add local capture taps.

- Owner files: same modules as 4B-4E.
- Acceptance: `capture_taps=True` returns the tap names listed above without changing default outputs.

Slice 5A sub-slice: add video converter/load.

- Owner files: `src/mlx_cv/models/sam3/convert.py`, focused converter tests.
- Acceptance: accepted reference key families map explicitly; image-mode video rejection remains unchanged; unsupported variants and shape mismatches raise clear errors.

Slice 5B sub-slice: wire dynamic multiplex neural engine into video session manager.

- Owner files: `src/mlx_cv/models/sam3/video.py`, `src/mlx_cv/models/sam3/video_model.py`.
- Acceptance: `propagate_in_video` and `SAM3VideoTracker.init/step` can use the neural engine, return model-derived masks, preserve public object ids through model object-index mapping, preserve public multiplex metadata, and report a non-fixture claim level.

Slice 5C sub-slice: update session/tracker/multiplex tests.

- Owner files: video tests only.
- Acceptance: existing public behavior remains, except deterministic fixture assertions are replaced by neural tiny-fixture expectations; point prompt rejection stays intact.

Slice 5D sub-slice: prepare Slice 6 local comparison handoff.

- Owner files: local tap export surfaces and `tools/sam3_video_upstream.py` only if needed for plumbing.
- Acceptance: local capture can emit object-space masks, bucket-space masks, track ids, object scores, object pointer mux-space tensors, memory mux/demux taps, and multiplex metadata for the tiny fixture; required-mode still returns a precise blocker without user-supplied checkpoint/reference runtime.

## Risks To Carry Forward

- The upstream `sam3_video_base.py` path mixes detector, tracker, Object Multiplex, and distributed/demo state. The local port should start with single-device inference and make omitted distributed behavior explicit in errors.
- The SAM3.1 checkpoint/runtime shape is multiplex-first. Implementing against `sam3_tracker_base.py` plus object-space masks alone would create a hybrid tracker that can pass local shape tests but fail real checkpoint key/shape parity.
- `ObjectMultiplexState` is public metadata, not the numerical `MultiplexState`. Slices 4-5 need a model-side mux/demux state with padding and removed-slot semantics.
- Existing local `SAM3MaskDecoder` is an image-mode approximation, not the tracker/multiplex mask decoder. Reusing it directly would hide architecture drift; the tracker/multiplex decoder should be a distinct port.
- Current public tests assert `claim_level: local_contract_fixture`; Slice 5 must intentionally update those assertions when the neural engine replaces deterministic propagation.
- The video checkpoint key map is not converter-only work. It must be paired with module shape tests before any admitted checkpoint is considered meaningfully loadable.
