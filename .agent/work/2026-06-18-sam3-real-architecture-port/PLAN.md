# PLAN: Official SAM 3.1 MLX Port

Change: `2026-06-18-sam3-real-architecture-port` — Spec: `SPEC.md`

## Goal

Ship official SAM 3.1 image and multiplex-video inference on MLX Metal from one final-layout BF16 Safetensors checkpoint, then remove SAM 3.0.

## Architecture approach

The official repository and `sam3.1_multiplex.pt` replace the Transformers SAM 3.0 architecture as the sole source of truth. Source conversion is a one-time tool operation; normal runtime loading is strict and conversion-free. The image model consumes the `detector.*` subtree of the same checkpoint the full video model consumes.

## Ordered slice sequence

### Slice 1: Official SAM 3.1 contract and reference spine

**Objective:** Lock the exact 3.1 checkpoint, configuration, preprocessing, public outputs, and deterministic official reference captures.
**Acceptance criteria:** The harness verifies the 1623/1166/457 and dtype inventories, records source/config/input checksums, uses official source rather than Transformers, and returns precise prerequisite failures.
**Verification:** `.venv/bin/python -m pytest -q tests/test_sam31_reference_contract.py`
**Depends on:** none
**Checkpoint after:** none
**Status:** complete
**Evidence:** Added `tools/sam31_reference.py` and `tests/test_sam31_reference_contract.py`; the real local checkpoint passed the exact 1623/1166/457 and 1591-float32/32-complex64 inventory; `.venv/bin/python -m pytest -q tests/test_sam31_reference_contract.py` passed with 7 tests.
**Risks / next:** Official end-to-end capture adapters are exercised with the detector and video APIs in Slices 2 and 3.

### Slice 2: SAM 3.1 detector and image API

**Objective:** Implement the official detector in MLX and expose the canonical image API.
**Acceptance criteria:** All 1166 `detector.*` tensors map by exact name and shape; official preprocessing and image outputs run; component and end-to-end image captures meet the BF16 gate.
**Verification:** `.venv/bin/python -m pytest -q tests/test_sam31_image_model.py tests/test_sam31_image_parity.py`
**Depends on:** Slice 1
**Checkpoint after:** none
**Status:** pending

### Slice 3: SAM 3.1 multiplex tracker and video API

**Objective:** Implement the official multiplex tracker, state machine, and canonical video session API.
**Acceptance criteria:** All 457 `tracker.*` tensors map exactly; prompt, propagation, removal, reset, memory, and dynamic 16-object bucket behavior work; short multi-object captures match official behavior.
**Verification:** `.venv/bin/python -m pytest -q tests/test_sam31_video_model.py tests/test_sam31_multiplex.py`
**Depends on:** Slice 2
**Checkpoint after:** none
**Status:** pending

### Slice 4: Final-layout BF16 Safetensors conversion and direct loading

**Objective:** Convert the official checkpoint once and make strict direct Safetensors loading the only production path.
**Acceptance criteria:** Atomic output contains 1591 BF16 and 32 complex64 tensors plus provenance metadata; clean reload equals the converted state; runtime rejects NPZ/PT, legacy layout, bad metadata, missing/unexpected names, shapes, and dtypes.
**Verification:** `.venv/bin/python -m pytest -q tests/test_sam31_safetensors.py`
**Depends on:** Slice 3
**Checkpoint after:** none
**Status:** pending

### Slice 5: Real Metal BF16 image and video parity

**Objective:** Run the persisted checkpoint through real MLX Metal inference and compare it with official PyTorch captures.
**Acceptance criteria:** Identity fields are exact; masks reach IoU >= 0.98; boxes differ by <= 2 pixels; scores differ by <= 0.02; bucket assignment is exact; no MLX CPU fallback occurs.
**Verification:** `MLX_CV_REQUIRE_SAM31_GATE=1 MLX_CV_SAM31_UPSTREAM=models/sam3-video/upstream/sam3.1_multiplex.pt MLX_CV_SAM31_MLX=models/sam3.1/mlx/sam3.1-multiplex-bf16.safetensors .venv/bin/python -m pytest -q tests/test_sam31_image_parity.py tests/test_sam31_video_parity.py`
**Depends on:** Slice 4
**Checkpoint after:** none
**Status:** pending

### Slice 6: SAM 3.0 removal and release cutover

**Objective:** Remove all obsolete SAM 3.0 runtime, conversion, reference, test, API, and local-weight paths after the real 3.1 gate passes.
**Acceptance criteria:** Only canonical SAM 3.1 APIs remain; NPZ is fixture-only outside the SAM3 production path; old local checkpoints are deleted; both release entries are `UPSTREAM_PASSED`; the complete suite and diff hygiene pass.
**Verification:** `.venv/bin/python -m pytest -q && git diff --check`
**Depends on:** Slice 5
**Checkpoint after:** none
**Status:** pending

## Execution routing and topology

- Direct, serial execution with automatic continuation through all slices.
- Parallel-safe groups: none.
- No human checkpoint; deletion is mechanically gated on successful real image and video parity.
- Large checkpoints remain outside Git.

## Aggregate verification

| Gate | Command |
|---|---|
| Contract | `.venv/bin/python -m pytest -q tests/test_sam31_reference_contract.py` |
| Image | `.venv/bin/python -m pytest -q tests/test_sam31_image_model.py tests/test_sam31_image_parity.py` |
| Video | `.venv/bin/python -m pytest -q tests/test_sam31_video_model.py tests/test_sam31_multiplex.py` |
| Checkpoint | `.venv/bin/python -m pytest -q tests/test_sam31_safetensors.py` |
| Full regression | `.venv/bin/python -m pytest -q` |
| Hygiene | `git diff --check` |
