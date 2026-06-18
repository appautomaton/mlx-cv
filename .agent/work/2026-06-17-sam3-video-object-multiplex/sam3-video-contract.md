# SAM3 Video / Object Multiplex Contract

## Upstream Surfaces

Phase 3 uses these upstream SAM3 names as the local contract boundary:

- `build_sam3_video_predictor` in `references/sam3/sam3/model_builder.py`
- `build_sam3_multiplex_video_predictor` in `references/sam3/sam3/model_builder.py`
- `build_sam3_predictor(version="sam3.1")` as the multiplex entry-point selector
- `Sam3TrackerPredictor` in `references/sam3/sam3/model/sam3_tracking_predictor.py`
- `SimpleMaskEncoder` in `references/sam3/sam3/model/memory.py`
- `MultiplexController` in `references/sam3/sam3/model/multiplex_utils.py`
- `VideoTrackingDynamicMultiplex` in `references/sam3/sam3/model/video_tracking_multiplex.py`
- `start_session`, `add_prompt`, and `propagate_in_video` in `references/sam3/sam3/model/sam3_base_predictor.py`

## Local Boundary

SAM3 Video covers text or concept-prompted video detection and tracking. Sam3Tracker covers visual-prompt video segmentation where the local prompt surface supports it. Object Multiplex covers fixed-capacity multi-object bucket state and batching metadata.

The deterministic local tracker contract proves local frame-sequence processing, memory updates, stable object IDs, and typed per-frame `Result` output. It is not upstream checkpoint parity.

## Checkpoint Gate

Video checkpoint status is tracked only in:

`.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`

Do not register `sam3_video` in:

`.agent/work/2026-06-16-release-parity-hardening/parity-status.json`

That older release-parity matrix is bounded to `da3_multiview`, `locateanything`, `rfdetr`, and `sam3_image`.

The image-mode SAM3 loader must keep rejecting video/tracker/memory/temporal keys. Video checkpoint admission is a separate gate.

