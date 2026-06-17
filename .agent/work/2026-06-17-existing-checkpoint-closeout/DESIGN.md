# DESIGN: Existing Checkpoint Closeout

Change: `2026-06-17-existing-checkpoint-closeout`

## Status Ownership

Keep `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` as the single release-parity matrix. This change updates only:

- `models.locateanything`
- `models.sam3_image`

Do not add `sam3_video`, DA3 deferred branches, or new model-family rows. `tests/test_runtime_dependency_guards.py::test_release_parity_status_matrix_is_bounded` remains the guard for that boundary.

## Gate Outcome Model

Each gate should return one of these durable outcomes:

- `UPSTREAM_PASSED`: a real checkpoint/reference-vs-MLX comparison ran and met the recorded tolerance.
- `BLOCKED:<reason>`: a concrete prerequisite or component blocks the real comparison.
- `LOCAL_FIXTURE_ONLY`: not expected as the terminal status for these two entries after this closeout unless planning is revisited.

A required gate may pass while the model is blocked only when it asserts the matching `BLOCKED:<reason>` record. A required gate must not skip silently. A checkpoint-present path must not call `pytest.fail("comparison is not implemented")`; if comparison code is absent or unstable, the status is a blocker naming that missing comparison component.

## Checkpoint And Reference Boundary

Checkpoint files stay outside git. Execution should first use explicit env paths when present:

- `MLX_CV_LOCATEANYTHING_CHECKPOINT`
- `MLX_CV_SAM3_IMAGE_CHECKPOINT`

If execution attempts downloads, it must request approval and place artifacts in an out-of-git cache such as `/tmp/mlx-cv-checkpoints/` or `~/.cache/mlx-cv/`. Record source, license, checksum/provenance, and whether the cache contains complete usable weights. The repository may record derived tiny fixtures and status metadata, but not raw upstream weights.

## Runtime Boundary

`src/mlx_cv/` remains MLX-native and import-light. Reference runtimes, Torch, Transformers, Hugging Face clients, and network/download helpers belong in tools or tests only. `references/` can be used through `PYTHONPATH` in opt-in commands, never imported by package runtime.

## Model-Specific Notes

LocateAnything:

- Current `references/LocateAnything-3B/*.safetensors` shards are 135-byte LFS stubs.
- A usable checkpoint is expected to be a directory with `model.safetensors.index.json` and complete shard files, or another explicitly supported full-checkpoint format.
- The first real comparison target is fixed grounding inputs already covered by local fixtures, with decoded boxes/points and stable intermediate taps where upstream exposes them.

SAM 3.1 image-mode:

- `references/sam3/` exists and exposes `build_sam3_image_model` / `build_sam3_predictor` style entry points.
- The image gate must reject video/tracker checkpoints through the existing converter safeguards.
- The first real comparison target is image-mode text plus PCS-style prompts, with masks, paired detections, token/text path evidence, and stable taps where upstream exposes them.
