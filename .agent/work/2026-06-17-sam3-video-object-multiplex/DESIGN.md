# DESIGN: SAM 3.1 Video / Object Multiplex

## Runtime Boundary

Phase 3 adds SAM 3.1 video/tracker capability without importing the upstream PyTorch runtime into `src/mlx_cv`. Reference code under `references/sam3/` is used for contracts, names, request flow, and checkpoint-key families. Runtime code remains MLX/NumPy/Pillow compatible; optional Torch/OpenCV/reference comparisons live in `tools/` or env-gated tests.

The image-mode SAM 3.1 loader stays image-only. Video/tracker checkpoint admission is a separate path so image-mode conversion continues to reject video/tracker keys with clear errors.

Video checkpoint status lives in this change's own status artifact, `.agent/work/2026-06-17-sam3-video-object-multiplex/sam3-video-status.json`. Do not add `sam3_video` to `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`; `tests/test_runtime_dependency_guards.py::test_release_parity_status_matrix_is_bounded` asserts that older matrix is limited to the completed release-parity hardening model set.

## Public Surface

Video output should be a typed collection of per-frame `Result` objects, not ad hoc dictionaries. Each frame result carries `masks`, optional `detections`, and `tracks` with stable IDs. A lightweight video result type can hold frame order, frame indices, and session metadata while preserving the existing per-frame `Result` contract.

The SAM3 video API should preserve the upstream vocabulary where it helps callers:

- `start_session` creates state from a frame sequence.
- `add_prompt` adds a text/concept prompt for SAM3 Video or a visual prompt for Sam3Tracker where supported.
- `propagate_in_video` yields or returns per-frame tracked results.

An optional `handle_request` adapter may mirror the upstream request names, but the local implementation should expose typed Python methods first.

## Internal State

Tracker state should separate four concerns:

- frame storage and per-frame spatial metadata,
- object state with stable IDs and prompt provenance,
- memory records keyed by object ID and frame index,
- Object Multiplex bucket state with fixed-capacity assignment metadata.

The deterministic local tracker can use fixture or adapter-provided masks to exercise memory updates, ID stability, and multiplex grouping. That proves the local typed state and propagation contract. It is not upstream parity and must not be documented as such.

## Checkpoint Admission

The real SAM 3.1 video checkpoint gate is independent from the deterministic tracker fixtures. It should define:

- environment variables for checkpoint/config paths or model ID,
- expected video/tracker/multiplex key families,
- checksum or provenance metadata when a checkpoint is present,
- required-mode behavior that fails with `BLOCKED:<reason>` when prerequisites are missing,
- optional upstream-reference comparison when Torch/reference dependencies and checkpoint access exist.

Default unit tests may assert the recorded blocker and skip heavy reference execution. Required gate commands must not treat missing checkpoint access as a pass.

## Documentation Claim Level

Docs and roadmap status should distinguish three levels:

1. typed local video/tracker state and deterministic fixture behavior,
2. video checkpoint admitted but upstream comparison unavailable,
3. upstream-vs-local SAM 3.1 video gate passed.

Only level 3 may be described as real SAM 3.1 video parity.
