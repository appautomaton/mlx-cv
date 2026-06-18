# Slice 6 Summary: SAM3V-CMP

Status: complete

Files changed:
- `tools/sam3_video_upstream.py`
- `tests/test_sam3_video_upstream_parity.py`

Review:
- Spec reviewer: approved.
- Quality reviewer: approved after fixes for normalized SAM3.1 upstream box XYWH and renaming the selected probability tap from logits to `score_probs`.

Verification:
- `python -m py_compile tools/sam3_video_upstream.py tests/test_sam3_video_upstream_parity.py`: passed.
- `git diff --check`: passed.
- `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_runtime_dependency_guards.py -q`: 24 passed.

Notes:
- `evaluate_sam3_video_comparison_gate` now admits the checkpoint/config, preflights upstream reference surfaces, resolves a converted local `.npz`/safetensors checkpoint via `MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT`, captures upstream outputs through `build_sam3_multiplex_video_predictor`, runs the local Slice 5 neural path, and compares frame IDs, track IDs, masks, boxes, scores, score-probability taps, and stable Object Multiplex metadata.
- The upstream capture writes deterministic THWC frames to a temp PNG directory and sends normalized XYWH boxes, matching the real SAM3.1 box route.
- The gate reports precise blockers for missing reference path/surfaces/runtime, missing local checkpoint, reference/local capture failures, comparison component failures, and numeric parity drift; it only reports `UPSTREAM_PASSED` after a successful comparison.
