# SAM 3.1 Video / Object Multiplex

`mlx-cv` now has the faithful SAM3 video/tracker neural path:

- frame-sequence preprocessing from arrays, image paths, or frame directories,
- `start_session`, `add_prompt`, `propagate_in_video`, and `remove_object` session methods,
- box prompts through the Sam3Tracker boundary,
- explicit unsupported blockers for text and exemplar prompts until those detector/exemplar paths are ported,
- per-frame `Result` output with aligned `masks`, `detections.track_ids`, and `tracks`,
- `VideoResult` for ordered frame collections,
- 1797/1797 detector/tracker/neck tensor loading for the real architecture,
- faithful per-frame tracking, memory-bank propagation, and Object Multiplex batching/association,
- Object Multiplex bucket state with fixed-capacity object assignment metadata,
- a historical local-contract status file at `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`.

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

The video-level `metadata` records `claim_level: mlx_neural_forward`. This is real local neural forwarding through the faithful architecture, not upstream parity by itself.

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
- `MLX_CV_SAM3_VIDEO_CACHE_DIR`
- `MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT`
- `MLX_CV_REQUIRE_SAM3_VIDEO_GATE`

The current checkpoint-admission output is:

`.agent/work/2026-06-17-sam3-video-real-checkpoint-admission/sam3-video-checkpoint-status.json`

The official SAM 3.1 video/Object Multiplex source is the gated Hugging Face repo
`facebook/sam3.1`, with `sam3.1_multiplex.pt` and `config.json`. Weights and
configs stay outside git. A cache layout such as
`$MLX_CV_SAM3_VIDEO_CACHE_DIR/facebook--sam3.1/sam3.1_multiplex.pt` plus
`config.json` can be admitted locally, but downloading requires Hugging Face auth
and accepted SAM terms.

The image-mode loader still rejects video/tracker/memory/temporal keys. The video gate inspects video/tracker/multiplex key families through `inspect_sam3_video_state_dict` and the external gate helper in `tools/sam3_video_upstream.py`.

Current claim level: faithful local MLX neural forwarding plus a checkpoint-ready upstream-vs-MLX comparison gate. The release matrix is the canonical claim source; the checkpoint-admission JSON records the live environment/config admission result, and the older Object-Multiplex status is historical local-contract evidence. With no upstream checkpoint/config configured, the live status records `BLOCKED:MLX_CV_SAM3_VIDEO_CHECKPOINT is unset`. If upstream checkpoint/config and a converted local MLX checkpoint are supplied, the gate captures the upstream predictor, runs the local streaming path, and compares masks, boxes, track IDs, object scores, score-probability taps, and stable Object Multiplex metadata. It only reports `UPSTREAM_PASSED` after that real comparison succeeds.

## Verification

Focused local neural path:

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
  MLX_CV_SAM3_VIDEO_CHECKPOINT=/path/to/sam3.1_multiplex.pt \
  MLX_CV_SAM3_VIDEO_CONFIG=/path/to/config.json \
  MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT=/path/to/sam3-video-mlx.npz \
  uv run --extra test pytest \
  tests/test_sam3_video_upstream_parity.py \
  tests/test_sam3_video_checkpoint_gate.py
```
