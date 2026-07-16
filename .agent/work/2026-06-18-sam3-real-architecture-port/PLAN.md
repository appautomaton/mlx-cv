# PLAN: SAM3 Real-Architecture MLX Re-Port (Image + Video)

Change: `2026-06-18-sam3-real-architecture-port` — Stage: execute — Spec: `SPEC.md`

## Goal

Re-implement the real SAM 3 image detector and video tracker in MLX so the real weights load 1:1 and the image and video paths pass numeric parity against the authoritative upstream references (SPEC AC1–AC7).

## Reconciliation Note

Git history is ahead of the original plan ledger. Commits through `ef5aecb` completed the architecture, weight loading, streaming session, and Object-Multiplex association work. The original Slices 7 and 11 combined assembly with external parity; this refresh records their delivered assembly/gate-wiring work as complete and moves the still-unmet real-checkpoint outcomes into Slices 18 and 19.

## Requirement Traceability

| SPEC AC | Satisfying slices | Current state |
|---|---|---|
| AC1 configs | Slice 1 | complete |
| AC2 reference spine | Slice 1 | complete |
| AC3 image subsystems | Slices 2–7 | complete |
| AC4 image gate | Slice 18 | pending external checkpoint run |
| AC5 video subsystems | Slices 8–16 | complete |
| AC6 video gate | Slice 19 | pending external checkpoint run |
| AC7 hygiene | all slices; Slices 17–19 closeout | local suite green; final gates pending |

## Completed Architecture Slices

### Slice 1: Config ingestion + Transformers reference spine
**Objective:** Mirror the real SAM3 configs and provide honest image/video upstream capture gates.
**Acceptance criteria:** Config ingestion round-trips the upstream config; reference capture records deterministic taps or a precise external blocker; runtime sources remain free of Torch/Transformers imports.
**Verification:** `tests/test_sam3_config_ingest.py`, `tests/test_sam3_upstream_hf.py`, and runtime dependency guards.
**Status:** complete
**Evidence:** `ca7fae3`, `dbf7205`; config ingestion and Transformers-native reference capture merged.

### Slice 2: Vision encoder — windowed-RoPE ViT + FPN
**Objective:** Port the faithful vision encoder and FPN.
**Acceptance criteria:** All 538 vision tensors map with exact shapes and the vision path is tap-testable.
**Verification:** `tests/test_sam3_vision_real.py`.
**Status:** complete
**Evidence:** `6251321`.

### Slice 3: Text encoder + projection
**Objective:** Port the CLIP-style text tower and projections.
**Acceptance criteria:** All 391 tensors map with exact shapes and the text path is tap-testable.
**Verification:** `tests/test_sam3_text_real.py`.
**Status:** complete
**Evidence:** `761bc1a`.

### Slice 4: DETR encoder + geometry + scoring
**Objective:** Port the DETR encoder, ROI geometry encoder, and scoring head.
**Acceptance criteria:** All 260 tensors map with exact shapes and the component paths run.
**Verification:** `tests/test_sam3_detr_encoder_real.py`.
**Status:** complete
**Evidence:** `5fa46e1`.

### Slice 5: DETR decoder
**Objective:** Port the six-layer, 200-query decoder.
**Acceptance criteria:** All 247 tensors map with exact shapes and query/box outputs are available to the parity harness.
**Verification:** `tests/test_sam3_detr_decoder_real.py`.
**Status:** complete
**Evidence:** `565cb26`.

### Slice 6: Image mask decoder
**Objective:** Port the FPN pixel mask decoder.
**Acceptance criteria:** All 32 tensors map with exact shapes and mask logits are exposed to the gate.
**Verification:** `tests/test_sam3_mask_decoder_real.py`.
**Status:** complete
**Evidence:** `4144360`.

### Slice 7: Sam3Model assembly + image gate wiring
**Objective:** Assemble the faithful image detector and wire the end-to-end Transformers comparison.
**Acceptance criteria:** All 1468 `detector_model.*` tensors load 1:1; boxes, logits, masks, presence, and semantic outputs are captured by the gate; missing external weights remain an honest blocker.
**Verification:** `tests/test_sam3_upstream_parity.py`, `tests/test_sam3_upstream_hf.py`, and image model tests.
**Status:** complete
**Evidence:** `2dfd484`; the real numeric outcome remains Slice 18.

### Slice 8: Tracker neck + memory encoder
**Objective:** Port the tracker neck and memory encoder.
**Acceptance criteria:** All 62 tensors map with exact shapes and the memory path runs.
**Verification:** `tests/test_sam3_tracker_memory_real.py`.
**Status:** complete
**Evidence:** `fefa215`.

### Slice 9: Memory attention + object pointers
**Objective:** Port memory attention, object-pointer projection, and embeddings.
**Acceptance criteria:** The 112-tensor component set maps with exact shapes and runs through the local tracker path.
**Verification:** `tests/test_sam3_memory_attention_real.py`.
**Status:** complete
**Evidence:** `09e8cd4`.

### Slice 10: Tracker prompt encoder + mask decoder
**Objective:** Port the tracker prompt encoder and SAM2-style mask decoder.
**Acceptance criteria:** All 145 tensors map with exact shapes and the tracker mask path runs.
**Verification:** `tests/test_sam3_tracker_mask_real.py`.
**Status:** complete
**Evidence:** `184b143`.

### Slice 11: Sam3VideoModel assembly + video gate wiring
**Objective:** Assemble detector, tracker, and tracker neck into the faithful video model.
**Acceptance criteria:** All 1797 tensors load 1:1; tracker components are shape-verified; the external comparison path records honest blockers.
**Verification:** video model module tests, converter tests, and `tests/test_sam3_video_upstream_parity.py`.
**Status:** complete
**Evidence:** `7c58452`; the real numeric outcome remains Slice 19.

### Slice 12: Faithful per-frame tracker step
**Objective:** Port the memory-propagation `track_step` used for each frame.
**Acceptance criteria:** The local tracker consumes frame features, prompt/memory state, and emits the expected mask/object-pointer state.
**Verification:** SAM3 real-video tracker-step tests.
**Status:** complete
**Evidence:** `37baf42`.

### Slice 13: Single-object streaming session
**Objective:** Implement memory-bank propagation for a box-prompted object across frames.
**Acceptance criteria:** Session start, prompt admission, propagation, memory updates, and typed frame results work locally.
**Verification:** streaming/session tests.
**Status:** complete
**Evidence:** `898e951`.

### Slice 14: Object-Multiplex batching
**Objective:** Track multiple objects through the faithful batched per-frame path.
**Acceptance criteria:** Multiple active objects are batched without losing object identity or result alignment.
**Verification:** multi-object streaming tests.
**Status:** complete
**Evidence:** `fad464e`.

### Slice 15: Object-Multiplex association
**Objective:** Implement match, spawn, and keep-alive behavior.
**Acceptance criteria:** Detection/tracker association preserves identities, admits new objects, and retains eligible unmatched tracks.
**Verification:** association and multiplex tests.
**Status:** complete
**Evidence:** `656d1e4`.

### Slice 16: Gate repoint + honest SAM3 video blocker
**Objective:** Point the video gate at the faithful streaming path and update the release status without claiming unrun parity.
**Acceptance criteria:** Status says streaming is implemented, architecture/weight loading are not blockers, and the remaining external numeric run is named precisely.
**Verification:** runtime guards, SAM3 video checkpoint tests, and status JSON validation.
**Status:** complete
**Evidence:** `ef5aecb`.

## Remaining Slices

### Slice 17: Reconcile execution ledger and project status
**Objective:** Align the active plan/state, release parity matrix, public status docs, steering, repo map, and forward roadmap with the merged implementation.
**Acceptance criteria:** `current.json` reports this change at `execute`; Slices 1–16 match Git; current-facing files agree on model status and Python/MLX support; the roadmap contains active SAM3 closeout plus pending EoMT-DINOv3; historical work remains intact except the empty orphan directory.
**Verification:** `node .agent/.automaton/scripts/get-context.mjs`; JSON validation; stale-claim searches; focused status tests; full local pytest; path-scoped `git diff --check`.
**Touches:** this plan, release parity matrix, README/current docs, steering/wiki status artifacts, and `current.json` through `sync-status.mjs`.
**Depends on:** Slice 16.
**Checkpoint after:** none.
**Status:** complete
**Evidence:** reconciled the active 19-slice ledger, release matrix, README/current docs, steering, repo map, and forward roadmap; `get-context.mjs` reports `stage: execute` with no diagnostics; JSON validation passed; stale-claim search returned no matches; focused status tests passed with 24 passed and 3 skipped; full local regression passed with 615 passed and 13 skipped.
**Risks / next:** Slices 18 and 19 still require gated upstream checkpoints and converted local MLX checkpoints.

### Slice 18: SAM3 image real-checkpoint parity closeout
**Objective:** Run the faithful MLX image detector against the gated upstream checkpoint and promote `sam3_image` only on measured numeric PASS.
**Acceptance criteria:** Required upstream and converted local checkpoints load; end-to-end taps meet documented tolerances; `sam3_image` becomes `UPSTREAM_PASSED`; failures remain precise blockers.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_IMAGE_GATE=1 MLX_CV_SAM3_IMAGE_CHECKPOINT=/path/to/facebook-sam3 MLX_CV_SAM3_IMAGE_LOCAL_CHECKPOINT=/path/to/sam3-detector-mlx.npz uv run --extra test --with 'transformers>=5.10,<6' --with torch pytest tests/test_sam3_upstream_hf.py -q -k gate_runs_when_required`.
**Depends on:** Slice 17.
**Checkpoint after:** human-action — supply gated upstream and converted local checkpoints when not already available.
**Status:** pending

### Slice 19: SAM3 video real-checkpoint parity closeout
**Objective:** Run the faithful streaming/Object-Multiplex path against the gated SAM3.1 reference and promote `sam3_video` only on measured numeric PASS.
**Acceptance criteria:** Required upstream checkpoint/config and converted local checkpoint load; masks, boxes, identities, scores, and stable taps meet documented tolerances; any preprocessing/reference mismatch is recorded precisely; `sam3_video` becomes `UPSTREAM_PASSED` only on PASS.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_SAM3_VIDEO_GATE=1 MLX_CV_SAM3_VIDEO_CHECKPOINT=/path/to/sam3.1_multiplex.pt MLX_CV_SAM3_VIDEO_CONFIG=/path/to/config.json MLX_CV_SAM3_VIDEO_LOCAL_CHECKPOINT=/path/to/sam3-video-mlx.npz uv run --extra test pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_checkpoint_gate.py -q`.
**Depends on:** Slice 18.
**Checkpoint after:** human-action — supply gated upstream/config and converted local checkpoints when not already available.
**Status:** pending

## Execution Routing

- Direct, serial execution.
- No parallel-safe groups.
- Checkpoints and model weights remain outside Git.
- `parity-status.json` is the canonical release-status matrix; phase-local SAM3 video status files retain their operational/historical roles.

## Aggregate Verification

| Scope | Command |
|---|---|
| Status contracts | `.venv/bin/python -m pytest -q tests/test_runtime_dependency_guards.py tests/test_sam3_video_checkpoint_gate.py tests/test_sam3_upstream_hf.py` |
| Full local regression | `.venv/bin/python -m pytest -q` |
| JSON syntax | `.venv/bin/python -m json.tool <status-file>` |
| Context integrity | `node .agent/.automaton/scripts/get-context.mjs` |
| Diff hygiene | path-scoped `git diff --check` |
