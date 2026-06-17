# SAM 3.1 Video / Object Multiplex

`mlx-cv` now has the Phase 3 local SAM3 video/tracker contract:

- frame-sequence preprocessing from arrays, image paths, or frame directories,
- `start_session`, `add_prompt`, `propagate_in_video`, and `remove_object` session methods,
- text/concept prompts through the SAM3 Video boundary,
- visual box/exemplar prompts through the Sam3Tracker boundary where the local prompt surface supports them,
- per-frame `Result` output with aligned `masks`, `detections.track_ids`, and `tracks`,
- `VideoResult` for ordered frame collections,
- Object Multiplex bucket state with fixed-capacity object assignment metadata,
- a Phase 3-local checkpoint status file at `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`.

## Supported Inputs

The core video processor accepts deterministic frame sequences:

- list or tuple of numpy/PIL/path image frames,
- a numpy array with shape `(T,H,W,C)`,
- a frame directory containing supported image files sorted by filename.

Video-file decoding is intentionally optional. The runtime package does not hard-import OpenCV, Torch, CUDA, xformers, or the upstream reference tree.

## Output Shape

`SAM3VideoSessionManager.propagate_in_video(...)` returns a `VideoResult`.
Each frame is a normal `Result`:

- `Result.masks`: instance masks, one per tracked object,
- `Result.detections.track_ids`: object IDs paired with detections,
- `Result.tracks`: stable object IDs, frame index, scores, labels, and multiplex metadata.

The video-level `metadata` records that this is a `local_contract_fixture`, not upstream parity.

## Object Multiplex

`ObjectMultiplexState` records:

- `bucket_capacity`,
- fixed-capacity `buckets`,
- `object_to_bucket` assignments,
- active object IDs,
- per-frame memory records.

This proves shape/state behavior for multi-object tracking. It does not claim the upstream SAM 3.1 Object Multiplex speedup.

## Checkpoint Gate

The real SAM3 video checkpoint gate is separate from the image-mode SAM3 loader:

- `MLX_CV_SAM3_VIDEO_CHECKPOINT`
- `MLX_CV_SAM3_VIDEO_CONFIG`
- `MLX_CV_SAM3_VIDEO_MODEL_ID`
- `MLX_CV_REQUIRE_SAM3_VIDEO_GATE`

The image-mode loader still rejects video/tracker/memory/temporal keys. The video gate inspects video/tracker/multiplex key families through `inspect_sam3_video_state_dict` and the external gate helper in `tools/sam3_video_upstream.py`.

Current claim level: local deterministic video/tracker/Object Multiplex contract coverage. Upstream SAM3 video parity remains blocked until a usable SAM3 video checkpoint is configured and an upstream-vs-local numeric comparison path exists.

## Verification

Focused local contract:

```bash
UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest \
  tests/test_sam3_video_processor.py \
  tests/test_sam3_video_session.py \
  tests/test_sam3_video_tracking.py \
  tests/test_sam3_object_multiplex.py \
  tests/test_types.py
```

Checkpoint gate:

```bash
UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest \
  tests/test_sam3_video_checkpoint_gate.py \
  tests/test_sam3_convert.py \
  tests/test_runtime_dependency_guards.py
```

Required external gate blocker:

```bash
UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 \
  PYTHONPATH=references/sam3 uv run --extra test pytest \
  tests/test_sam3_video_upstream_parity.py
```
