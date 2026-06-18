# Slice 5 Summary: SAM3V-NN-wire

Status: complete

Files changed:
- `src/mlx_cv/models/sam3/video.py`
- `src/mlx_cv/models/sam3/convert.py`
- `src/mlx_cv/models/sam3/__init__.py`
- `tests/test_sam3_video_tracking.py`
- `tests/test_sam3_video_tracker.py`
- `tests/test_sam3_object_multiplex.py`
- `tests/test_sam3_video_session.py`
- `tests/test_sam3_video_checkpoint_gate.py`

Review:
- Spec reviewer: approved after request-level singular `box` routing was fixed.
- Quality reviewer: approved after fixes for unsupported text/exemplar admission, prompt-frame conditioning, reverse propagation order, duplicate video key mappings, fake-model output provenance, tracker init rollback, and request-level no-mutation behavior.

Verification:
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_tracking.py tests/test_sam3_video_tracker.py tests/test_sam3_object_multiplex.py tests/test_sam3_video_session.py -q`: 22 passed.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_convert.py tests/test_runtime_dependency_guards.py -q`: 19 passed.

Notes:
- The public video path now reports `claim_level: mlx_neural_forward` and derives masks/boxes/scores from `SAM3VideoModel.track_step`.
- Text and exemplar prompts are explicit unsupported blockers until detector/text and exemplar paths are ported.
- The video checkpoint loader is separate from image-mode loading; image-mode video/tracker rejection is preserved.
