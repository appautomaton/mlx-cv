# PLAN: Real Checkpoint Bootstrap - RF-DETR Nano

Change: `2026-06-16-real-checkpoint-bootstrap-rfdetr-nano` - Stage: plan - Spec: `SPEC.md`

## Goal

Execute `SPEC.md`: finish roadmap Phase 1 by proving RF-DETR Nano real-checkpoint load, upstream/reference comparison, MLX comparison, and truthful `UPSTREAM_PASSED` status.

## Architecture Approach

Keep all network, Torch, checkpoint extraction, and upstream RF-DETR execution in `tools/` and env-gated tests. `src/mlx_cv/` remains runtime-clean and loads only local `.npz`/safetensors-style converted arrays. The real checkpoint audit proved RF-DETR Nano needs upstream-compatible architecture admission, not just converter fixes: windowed DINOv2, P4 C2f projector, two-stage decoder/proposal heads, grouped query slicing, `bbox_reparam`, and exact Nano checkpoint key coverage must land before status promotion. Use the existing release parity status matrix as the current model status source so RF-DETR does not have competing blocker/pass records.

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 3-8, because they cross upstream RF-DETR, checkpoint format inspection, architecture admission, converter behavior, local MLX numerics, and parity.
- Parallel-safe groups: none.
- Checkpoints: none. Network/checkpoint fetch may require sandbox escalation during execution, but it is not a product decision. This phase must not close with a missing-checkpoint blocker or skipped upstream gate.

## Ordered Slice Sequence

### Slice 1: Checkpoint Cache And MD5 Gate

**Objective:** Add an explicit RF-DETR Nano checkpoint preparation path that resolves an out-of-git checkpoint and verifies the expected MD5 before any parity run.

**Acceptance criteria:**
- A tool or helper resolves RF-DETR Nano from `MLX_CV_RFDETR_NANO_CHECKPOINT` or an out-of-git cache root such as `$MLX_CV_CACHE`.
- The helper knows the recorded URL, filename `rf-detr-nano.pth`, and expected MD5 `fb6504cce7fbdc783f7a46991f07639f`.
- If the source URL basename differs from the canonical filename, the helper stores or aliases it as `rf-detr-nano.pth` and verifies the final file bytes.
- Checksum mismatch is a hard failure, not a skip.
- Download behavior is opt-in and does not run in normal CI by accident.
- The helper prints the resolved checkpoint path and MD5 after a successful verification.
- Raw checkpoints and converted weights remain outside git.
- Runtime dependency guards still pass.

**Touches:** `tools/`, `tests/test_rfdetr_checkpoint_gate.py`, `.gitignore` only if a new local cache path needs protection.

**Produces:** Reproducible checkpoint path contract for later slices.

**Verification:** `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_checkpoint_gate.py`

**Status:** complete
**Evidence:** added `tools/rfdetr_checkpoint.py` and `tests/test_rfdetr_checkpoint_gate.py`; `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_checkpoint_gate.py` passed with 12 tests.
**Risks / next:** none.

### Slice 2: Upstream RF-DETR Nano Capture Path

**Objective:** Replace ad hoc reference assumptions with an env-gated upstream RF-DETR Nano runner that captures comparable reference outputs for one fixed input.

**Acceptance criteria:**
- Reference execution is isolated to tools/tests and may use `PYTHONPATH=references/rf-detr/src`.
- The runner uses the verified RF-DETR Nano checkpoint path from Slice 1.
- The fixed input and preprocessing/postprocessing are explicit and deterministic.
- Captured reference data includes final boxes, scores, class IDs, raw logits/boxes when available, and stable taps where reference APIs expose them.
- Any unavailable intermediate tap is recorded as a stable-tap gap rather than invented.
- In normal CI mode without checkpoint configuration, this dedicated test cleanly skips; with `MLX_CV_REQUIRE_RFDETR_GATE=1`, missing checkpoint, bad checksum, missing reference dependency, or skipped capture fails.
- Required-vs-normal behavior uses the shared RF-DETR gate helper so capture, load, and parity tests cannot drift.
- No reference import enters `src/mlx_cv/`.
- The capture verification uses a dedicated `upstream_capture` test target and does not accidentally select the legacy placeholder gate.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `tools/`, `tests/test_rfdetr_upstream_capture.py`, `tests/test_rfdetr_upstream_parity.py`, `src/mlx_cv/parity/fixtures.py` if a fixed real-gate input helper is needed.

**Produces:** Upstream capture function or tool usable by the real parity test.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_capture.py`

**Status:** complete
**Evidence:** added `tools/rfdetr_upstream.py` and `tests/test_rfdetr_upstream_capture.py`; downloaded and verified `/tmp/mlx-cv-checkpoints/rf-detr-nano.pth` with MD5 `fb6504cce7fbdc783f7a46991f07639f`; `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_capture.py` passed with 1 test.
**Risks / next:** reference exposes final raw logits/boxes through public outputs; stable intermediate tap gaps remain documented until parity comparison.

### Slice 3: Real Nano Architecture Contract

**Objective:** Turn the real checkpoint audit into an executable RF-DETR Nano architecture contract before adding MLX modules.

**Acceptance criteria:**
- A tool/test reads the verified checkpoint and records the Nano inference contract: `out_feature_indexes=[3,6,9,12]`, local zero-based layers `(2,5,8,11)`, `projector_scale=['P4']`, `dec_layers=2`, `group_detr=13`, `bbox_reparam=True`, `lite_refpoint_refine=True`, `num_feature_levels=1`, and checkpoint class-head shape.
- The contract names required tensor groups for windowed DINOv2, P4 C2f projector, two-stage encoder proposal heads, decoder self-attention/norm/ref-point head, grouped query slicing, and detection head.
- The contract proves the existing local fixture RF-DETR path is not sufficient for real checkpoint closure.
- Normal no-checkpoint CI skips the real audit cleanly; required mode fails on missing checkpoint or checksum mismatch.
- Runtime dependency guards still prove no Torch/reference import enters `src/mlx_cv/`.

**Execution:** subagent recommended

**Depends on:** Slices 1, 2

**Touches:** `tools/`, `tests/test_rfdetr_real_architecture_contract.py`, `tests/test_runtime_dependency_guards.py`, `src/mlx_cv/models/rfdetr/convert.py` only if a reusable metadata parser belongs there without Torch.

**Produces:** Executable architecture contract for real RF-DETR Nano admission.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_architecture_contract.py tests/test_runtime_dependency_guards.py`

**Status:** complete
**Evidence:** added `tools/rfdetr_real_architecture_contract.py` and `tests/test_rfdetr_real_architecture_contract.py`; expanded `tests/test_runtime_dependency_guards.py`; spec review and quality re-review approved after fixing the contract to serialize `local_fixture_gaps`; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_rfdetr_real_architecture_contract.py tests/test_runtime_dependency_guards.py` passed with 8 passed, 1 skipped; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth uv run pytest tests/test_rfdetr_real_architecture_contract.py tests/test_runtime_dependency_guards.py` passed with 9 passed.
**Risks / next:** none for Slice 3; proceed to Slice 4 windowed DINOv2 and P4 projector admission.

### Slice 4: Windowed DINOv2 And P4 Projector Admission

**Objective:** Add the Nano backbone/projector path required by the checkpoint while preserving existing tiny RF-DETR fixture coverage.

**Acceptance criteria:**
- MLX can instantiate the RF-DETR Nano backbone contract with DINOv2-with-registers small dims, patch size 16, pretrain grid 24, and zero-based output layers `(2,5,8,11)`.
- Windowed DINOv2 behavior required by upstream Nano is represented or explicitly matched for inference; any unsupported training-only behavior remains out of scope.
- The upstream P4 projector path consumes the four DINOv2 feature maps and produces one 256-channel level with the same spatial contract as upstream `projector_scale=['P4']`.
- Checkpoint projector tensor groups under `backbone.0.projector.stages.*` have corresponding MLX parameters or a documented inference-only exclusion with test coverage.
- Existing local/tiny RF-DETR fixture tests still pass unchanged.

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** `src/mlx_cv/backbones/vision/dinov2/`, `src/mlx_cv/backbones/vision/necks/rfdetr.py`, `src/mlx_cv/models/rfdetr/config.py`, `src/mlx_cv/models/rfdetr/modeling.py`, `tests/test_rfdetr_nano_backbone_projector.py`, existing RF-DETR tests.

**Produces:** MLX Nano backbone/projector shape and weight-path admission.

**Verification:** `uv run pytest tests/test_rfdetr_nano_backbone_projector.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py tests/test_runtime_dependency_guards.py`

**Status:** complete
**Evidence:** added RF-DETR Nano DINOv2 config/windowed inference metadata, `RFDETRP4C2fProjector`, `p4_c2f` projector selection, projector-stage converter remapping, and `tests/test_rfdetr_nano_backbone_projector.py`; spec review and final quality re-review approved after fixing projector key remapping and Nano `final_norm_eps=1e-6`; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_rfdetr_nano_backbone_projector.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py tests/test_runtime_dependency_guards.py` passed with 16 tests; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_rfdetr_convert.py` passed with 7 tests.
**Risks / next:** none for Slice 4; proceed to Slice 5 decoder two-stage and grouped-query admission.

### Slice 5: Two-Stage Decoder And Grouped Query Admission

**Objective:** Add the Nano two-stage decoder behavior required by the real checkpoint and localize numerical drift before full checkpoint loading.

**Acceptance criteria:**
- MLX RF-DETR Nano supports decoder self-attention, `ref_point_head`, decoder final norm, two-stage encoder proposal heads, `bbox_reparam=True`, `lite_refpoint_refine=True`, and one feature level.
- Grouped checkpoint query tensors with `group_detr=13` are sliced explicitly for inference without scrambling query order.
- Decoder parameter shapes match the real checkpoint for required inference tensors, including FFN hidden dimension 2048 and cross-attention heads/points implied by checkpoint shapes.
- Unit tests cover deterministic tiny inputs for proposal generation, grouped query slicing, bbox reparameterization, and decoder output/tap ordering.
- Existing unsupported segmentation/PML variant rejection remains intact.

**Execution:** subagent recommended

**Depends on:** Slice 4

**Touches:** `src/mlx_cv/heads/detection/rfdetr.py`, `src/mlx_cv/models/rfdetr/modeling.py`, `src/mlx_cv/models/rfdetr/convert.py`, `tests/test_rfdetr_nano_decoder.py`, `tests/test_rfdetr_convert.py`.

**Produces:** MLX Nano decoder path compatible with real checkpoint tensor shapes.

**Verification:** `uv run pytest tests/test_rfdetr_nano_decoder.py tests/test_rfdetr_convert.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

**Status:** complete
**Evidence:** added Nano decoder config paths for self-attention, `ref_point_head`, decoder final norm, two-stage proposal heads, grouped query slicing, `bbox_reparam`, `lite_refpoint_refine`, and FFN hidden dimension 2048; added decoder/two-stage converter remaps, self-attention in-proj splitting, and grouped-query conversion tests; spec review and final quality re-review approved after fixing cross-attention `query_pos` use and normalizing every stored decoder state; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_rfdetr_nano_decoder.py tests/test_rfdetr_convert.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` passed with 22 tests; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run pytest tests/test_rfdetr_decoder.py tests/test_rfdetr_model.py tests/test_runtime_dependency_guards.py` passed with 12 tests.
**Risks / next:** none for Slice 5; proceed to Slice 6 real checkpoint conversion and load.

### Slice 6: Real Checkpoint Conversion And Load

**Objective:** Convert the real RF-DETR Nano checkpoint into a runtime-clean local representation and load every required inference tensor.

**Acceptance criteria:**
- Torch `.pth` extraction stays in tools/tests outside `src/mlx_cv/`.
- Conversion handles real Nano checkpoint metadata and tensor remapping for all required inference groups from Slices 3-5.
- Unknown or mismatched tensors fail with actionable names unless they are covered by an explicit inference-only exclusion test.
- Converted real weights remain outside git and are loaded from the cache or a user-supplied path.
- A dedicated real-load test reads the verified checkpoint or converted cache output; tiny fixture tests cannot satisfy this slice.
- Normal no-checkpoint CI skips this dedicated test cleanly; required mode fails on missing checkpoint, bad checksum, missing conversion output, or skipped load.

**Execution:** subagent recommended

**Depends on:** Slice 5

**Touches:** `tools/`, `src/mlx_cv/models/rfdetr/convert.py`, `src/mlx_cv/models/rfdetr/config.py`, `tests/test_rfdetr_real_checkpoint_load.py`, `tests/test_rfdetr_convert.py`.

**Produces:** Real RF-DETR Nano checkpoint load path for MLX.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_checkpoint_load.py tests/test_rfdetr_convert.py tests/test_runtime_dependency_guards.py`

**Status:** complete
**Evidence:** added `tools/rfdetr_convert_checkpoint.py`, real Nano `RFDETRConfig.rfdetr_nano()`, HF-style DINOv2 remaps/QKV packing/strict load validation in `src/mlx_cv/models/rfdetr/convert.py`, `tests/test_rfdetr_real_checkpoint_load.py`, and expanded converter tests; spec review and quality review approved; sandbox verification failed before collection with `No Metal device available`, then `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth uv run pytest tests/test_rfdetr_real_checkpoint_load.py tests/test_rfdetr_convert.py tests/test_runtime_dependency_guards.py` passed outside sandbox with 23 tests.
**Risks / next:** real RF-DETR gates require the verified checkpoint and unsandboxed Metal access; proceed to Slice 7 local real-checkpoint forward and taps.

### Slice 7: Local Real-Checkpoint Forward And Taps

**Objective:** Run the loaded MLX RF-DETR Nano model on the fixed input and capture comparable local outputs.

**Acceptance criteria:**
- The local model runs the same fixed image/preprocessed tensor used by upstream capture.
- Local raw logits/boxes, typed detections, and ordered taps are captured with `capture_taps=True`.
- Result postprocessing preserves the upstream class-id semantics for the real COCO checkpoint head.
- Drift diagnosis remains bisectable through ordered taps or named stable-tap gaps.
- Normal no-checkpoint CI skips this dedicated test cleanly; required mode fails on missing checkpoint, conversion, load, or forward failure.

**Execution:** subagent recommended

**Depends on:** Slice 6

**Touches:** `src/mlx_cv/models/rfdetr/processor.py`, `src/mlx_cv/models/rfdetr/modeling.py`, `src/mlx_cv/parity/`, `tests/test_rfdetr_real_forward.py`, existing RF-DETR predict/parity tests.

**Produces:** Local real-checkpoint RF-DETR Nano forward capture.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

**Status:** complete
**Evidence:** added `src/mlx_cv/parity/rfdetr_real.py`, opt-in self-attention tap ordering in `src/mlx_cv/parity/fixtures.py`, and `tests/test_rfdetr_real_forward.py`; spec review requested correction for preprocessing mismatch, then re-review approved after local real capture matched upstream-style tensor preprocessing; quality review approved; sandbox verification failed before collection with `No Metal device available`, then `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth uv run pytest tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` passed outside sandbox with 10 tests; optional-mode outside-sandbox run passed with 9 passed, 1 skipped.
**Risks / next:** real gates require verified checkpoint and unsandboxed Metal access; proceed to Slice 8 upstream-vs-MLX parity gate.

### Slice 8: Real Upstream-vs-MLX Parity Gate

**Objective:** Replace the current success-path `pytest.fail(...)` placeholder with a real RF-DETR Nano upstream-vs-MLX comparison that must pass to close Phase 1.

**Acceptance criteria:**
- `tests/test_rfdetr_upstream_parity.py` no longer contains a success-path placeholder failure.
- With a checksum-matching checkpoint and `MLX_CV_REQUIRE_RFDETR_GATE=1`, the test runs upstream/reference and local MLX on the same fixed input.
- Final boxes, scores, class IDs, raw logits/boxes, and stable taps compare within default tolerance unless a justified model-specific tolerance is recorded.
- Any tolerance looser than `1e-3` returns to planning.
- Drift remains localizable through ordered taps or an explicitly documented stable-tap gap.
- Missing checkpoint may skip only outside the phase-closing verification; the phase-closing command must run with a real checkpoint and pass.
- After RF-DETR is `UPSTREAM_PASSED`, the no-checkpoint branch skips without asserting `BLOCKED:` so normal checkpoint-less CI remains green.
- The successful gate output includes the resolved checkpoint path and MD5.

**Execution:** subagent recommended

**Depends on:** Slices 2, 7

**Touches:** `tests/test_rfdetr_upstream_parity.py`, `tools/`, `src/mlx_cv/parity/`, `tests/fixtures/` only for small derived parity cases.

**Produces:** Passing real RF-DETR Nano upstream parity gate.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

**Status:** complete
**Evidence:** replaced `tests/test_rfdetr_upstream_parity.py` placeholder with a real upstream-vs-MLX checkpoint gate; fixed local CPU parity capture, RF-DETR two-stage near-tie proposal ordering, and RF-DETR no-clip postprocess semantics; added focused decoder/processor regressions; spec review approved after visible checkpoint evidence correction; quality review approved; sandbox MLX verification requires unsandboxed device access, then `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth PYTHONPATH=references/rf-detr/src uv run pytest -q tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py tests/test_rfdetr_nano_decoder.py tests/test_rfdetr_processor.py` printed `RF-DETR Nano checkpoint: path=/tmp/mlx-cv-checkpoints/rf-detr-nano.pth md5=fb6504cce7fbdc783f7a46991f07639f` and passed with 26 tests; `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache MLX_CV_CACHE=/tmp/mlx-cv-empty-rfdetr-cache uv run pytest -q tests/test_rfdetr_upstream_parity.py` passed with 1 passed, 1 skipped.
**Risks / next:** real gates require verified checkpoint and unsandboxed MLX access; proceed to Slice 9 status truthfulness and full regression.

### Slice 9: Status Truthfulness And Full Regression

**Objective:** Promote RF-DETR to real-checkpoint-passed status only after Slice 8 passes, then run the full regression suite.

**Acceptance criteria:**
- `.agent/work/2026-06-16-release-parity-hardening/parity-status.json` records RF-DETR as `UPSTREAM_PASSED` only after the real gate passes.
- LocateAnything and SAM 3.1 image statuses remain blocked and are not silently promoted.
- README, architecture, and steering docs distinguish RF-DETR real-checkpoint parity from remaining local-fixture/blocker states, including current RF-DETR status lines in `README.md`, `docs/ARCHITECTURE.md`, `.agent/steering/PROJECT.md`, and `.agent/steering/REQUIREMENTS.md`.
- With RF-DETR marked `UPSTREAM_PASSED` and no checkpoint env set, normal `uv run pytest` still passes because the env-gated real-checkpoint tests skip without stale `BLOCKED:` assertions.
- Runtime dependency guards pass.
- Full suite passes.
- PLAN.md records command evidence for the checkpoint MD5, real upstream gate, status update, and full regression.

**Depends on:** Slice 8

**Touches:** `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, `README.md`, `.agent/steering/`, `docs/`, `PLAN.md`.

**Produces:** Truthful Phase 1 closure with RF-DETR `UPSTREAM_PASSED`.

**Verification:** `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py && uv run pytest`

## Requirement Traceability

| SPEC acceptance | Satisfying slices |
| --- | --- |
| AC1 Checkpoint acquisition | Slice 1 |
| AC2 Runtime hygiene | Slices 1, 2, 6, 9 |
| AC3 Upstream reference run | Slice 2 |
| AC4 Local real-checkpoint run | Slices 3, 4, 5, 6, 7 |
| AC5 Real parity comparison | Slice 8 |
| AC6 Drift diagnosis | Slices 2, 7, 8 |
| AC7 Status truthfulness | Slices 8, 9 |
| AC8 Unsupported variants | Slices 5, 6, 9 |
| AC9 Regression | Slice 9 |

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Checkpoint/cache guard | `uv run pytest tests/test_runtime_dependency_guards.py tests/test_rfdetr_checkpoint_gate.py` |
| Reference capture | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_capture.py` |
| Architecture contract | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_architecture_contract.py tests/test_runtime_dependency_guards.py` |
| Backbone/projector admission | `uv run pytest tests/test_rfdetr_nano_backbone_projector.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py tests/test_runtime_dependency_guards.py` |
| Decoder admission | `uv run pytest tests/test_rfdetr_nano_decoder.py tests/test_rfdetr_convert.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| Local checkpoint load | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_checkpoint_load.py tests/test_rfdetr_convert.py tests/test_runtime_dependency_guards.py` |
| Local real forward | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> uv run pytest tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| Real RF-DETR parity | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py tests/test_rfdetr_real_forward.py tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| Full regression | `MLX_CV_REQUIRE_RFDETR_GATE=1 MLX_CV_RFDETR_NANO_CHECKPOINT=<verified-rfdetr-nano.pth> PYTHONPATH=references/rf-detr/src uv run pytest tests/test_rfdetr_upstream_parity.py && uv run pytest` |

## Risks

- Downloading the checkpoint may require explicit network approval during execution.
- The reference package may require optional dependencies not present in the base project environment.
- The real checkpoint already exposed architecture gaps not covered by tiny fixtures; the corrected plan explicitly admits those before conversion/load.
- Reference APIs may not expose every desired tap; final output parity remains mandatory, and tap gaps must be named.
- The status matrix currently lives under the prior parity-hardening change; this plan intentionally updates it as the current user-facing status source rather than creating a competing status file.
- The RF-DETR upstream test must intentionally diverge from the LocateAnything/SAM blocker-gate shape after RF-DETR passes: checkpoint-less normal CI skips are allowed, but required-gate verification fails when the checkpoint is absent.
- The capture, real-load, and parity tests must share one gate helper so normal no-checkpoint CI skips stay green while required phase-closing runs fail on missing prerequisites.
- The previous Claude engineering review approved the converter/load plan before the real checkpoint audit exposed missing architecture; the corrected Slices 3-9 need a fresh engineering review before execution resumes.

## Engineering Review

- Reviewer: Claude Code Opus 4.8, max effort, read-only plan mode, session `80eeb71e-2198-446f-9073-5c67bc598c0f`.
- Initial verdict: `NEEDS_CORRECTION`.
- Accepted corrections: require a non-skipped real gate with checkpoint path/MD5 evidence before status promotion; make the RF-DETR no-checkpoint branch compatible with future `UPSTREAM_PASSED` normal CI; replace broad `-k` selectors with dedicated checkpoint, upstream-capture, and real-load test targets; make Slice 3 prove real checkpoint load instead of relying on tiny-fixture tests.
- Final re-review verdict: `APPROVED_WITH_RISKS`.
- Incorporated final review guidance: require shared gate semantics for upstream capture, local real-load, and parity tests; make new env-gated tests cleanly skip in normal no-checkpoint CI but fail in required phase-closing mode; map AC7 to both Slice 4 and Slice 5; preserve canonical `rf-detr-nano.pth` naming even when the download URL basename differs.
- Superseding correction: execution of the approved plan discovered that real RF-DETR Nano checkpoint loading is architecture-gated, not converter-gated. Slices 3-9 above supersede the old reviewed Slices 3-5 and should receive a fresh engineering review before execution resumes.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The corrected plan sequences the real checkpoint work through an executable contract before architecture admission, conversion/load, forward capture, parity, and status promotion, which matches the discovered failure mode.
- Concern: Slices 4 and 5 carry the highest implementation risk because windowed DINOv2, the P4 C2f projector, two-stage decoder behavior, and grouped query slicing all touch core model numerics before any real checkpoint can load.
- Action: Proceed with Slice 3 first and require its contract test to name every inference tensor group that Slice 4 through Slice 6 must either consume or explicitly exclude before continuing.
- Verified: Canonical SPEC/DESIGN/PLAN, roadmap wording, Automaton context, RF-DETR config/modeling/converter/decoder/projector code, checkpoint/upstream tools, and existing RF-DETR tests were checked.
