# SPEC: Inference Production Readiness For Remaining Model Families

Change: `2026-06-17-inference-production-readiness` - Stage: frame - Source: this session's inference-gap audit, `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`, prior SAM3 image/video and LocateAnything changes, `.agent/wiki/LEARNINGS.md`, and `docs/ARCHITECTURE.md`.

## Bounded Goal

Bring the three not-yet-production-ready model families — LocateAnything grounding, SAM 3.1 image segmentation, and SAM 3.1 video tracking — to *checkpoint-ready* status: real MLX neural inference plus a complete, honest upstream-vs-MLX parity harness, so that supplying a real checkpoint yields a truthful PASS or a precise blocker, never a fake pass.

## Broader Intent

Finish `mlx-cv` as a completely-MLX, inference-only, production library. DA3 depth (multi-view) and RF-DETR detection are already `UPSTREAM_PASSED`. This change closes the remaining distance for the other surfaces so all four families share one truth bar: real neural inference verified against the upstream reference within documented tolerances. The work that can be done without redistributable weights (real inference code, comparison harnesses, honest gates, tiny fixtures) is done here; the external checkpoints remain a user-supplied dependency, not an excuse for an unverified claim.

## Target User / Stakeholder

Library users who load real weights and need outputs they can trust (verified parity, not "runs without error"), and maintainers who need the parity matrix to state the literal truth about each family.

## Work Scale And Shape

- Scale: capability-sized; three model surfaces, two of which need only verification tooling and one (SAM3 video) needs a real neural-path port plus tooling.
- Shape: **parity / gap-matrix**. Reference source per family, gap matrix with gap-IDs, target conformance within documented tolerances, and gap-ID-anchored verification. SAM3 video additionally carries a **model-port** sub-shape (replace the deterministic fixture with the real memory-encoder/tracker-decoder neural path).
- Selected lenses: engineering, runtime, product.

## Required Outcome

A truthful, checkpoint-ready state for each remaining family. Verified `UPSTREAM_PASSED` is gated on user-supplied external weights (run outside the sandbox, mirroring the DA3 `PASSED_OUTSIDE_SANDBOX` pattern); everything reachable without those weights is delivered and committed.

### Gap matrix

| Gap ID | Family | Current state | Required closure | Blocked-only-on |
|---|---|---|---|---|
| LA-CMP | LocateAnything | Real neural inference complete; `tools/locateanything_upstream.py` comparison gate is a stub that records "comparison component missing" | Implement upstream torch-reference capture + local capture + numeric comparison for decoded boxes/points and stable taps (DA3-tool pattern); honest component-specific blocker when capture/tap is unavailable | External LocateAnything-3B checkpoint (local files are 135-byte LFS stubs) |
| SAM3I-CMP | SAM3 image | Real neural inference complete; `tools/sam3_image_upstream.py` comparison gate is a stub | Implement stable image tap capture (upstream + local) + numeric comparison for masks, paired detections, and token/text evidence; honest blocker when missing | External SAM 3.1 image checkpoint (env unset) |
| SAM3V-NN | SAM3 video | Local propagation is a deterministic fixture (`claim_level: local_contract_fixture`; `_propagate_frame` synthesizes boxes/masks, no neural net) | Port the real SAM 3.1 video neural path (memory encoder + tracker/mask decoder) so propagation produces model-derived masks; keep the streaming `SAM3VideoTracker` and session API on top | Real `facebook/sam3.1` checkpoint + (port effort) |
| SAM3V-CMP | SAM3 video | `tools/sam3_video_upstream.py` has admission/blocker taxonomy but no real local outputs to compare | Complete upstream-vs-MLX comparison over the real neural outputs from SAM3V-NN; honest blocker when missing | Real `facebook/sam3.1` checkpoint |

### Cross-cutting outcomes
- Every gate, run in required-mode without a usable checkpoint, returns a **precise component-specific blocker**, never a skip-as-pass or a fake pass.
- The release parity matrix becomes literally true: LA-CMP and SAM3I-CMP move from "comparison-component-missing" to "blocked only on external checkpoint"; SAM3 video's matrix membership is decided explicitly (added once SAM3V-NN+SAM3V-CMP exist, or kept out with a stated reason).
- Package runtime (`src/mlx_cv/`) stays MLX-native and import-light; all torch/reference code stays in `tools/` and `tests/`.

## Constraints And Risks

- **No redistributable weights.** Real checkpoints are user-supplied and out-of-git; verified PASS runs outside the sandbox. This change cannot itself produce an `UPSTREAM_PASSED` flip without those weights — it produces everything else and an honest gate.
- **Architecture-mismatch risk (precedent-backed).** Per `LEARNINGS.md`, the RF-DETR Nano real checkpoint required real architecture work (windowed DINOv2, C2f projector, two-stage proposals), not just a converter. LocateAnything and SAM3 image may likewise reveal local-vs-real architecture gaps when a real checkpoint is admitted; the comparison harness must localize such drift to a gap-ID, and the plan must allow architecture corrections, not assume converter-only.
- **SAM3 video port is the largest item** and depends on the reference memory/tracker design under `references/sam3/`; it is real model-port work, not tooling. Planning should sequence it after (or parallel to) the two comparison-harness gaps.
- **MoonViT bicubic convention** (`LEARNINGS.md`): LocateAnything parity depends on PyTorch's border-clamped bicubic weights; reference capture must preserve that.
- Comparison harnesses must be unit-testable without real weights (tiny fixtures / mocked reference), keeping committed tests light; the heavy real-checkpoint run stays env-gated.
- Do not weaken the existing SAM3 image-vs-video checkpoint rejection while adding the video neural path.

## Acceptance Criteria

1. **LA-CMP harness exists and is honest.** `tools/locateanything_upstream.py` implements real upstream capture + local capture + numeric comparison for decoded boxes/points and at least one stable tap; required-mode without a checkpoint yields a precise blocker; a committed test exercises the comparison logic on a tiny fixture/mock (no real weights) and the honest-blocker default path.
2. **SAM3I-CMP harness exists and is honest.** `tools/sam3_image_upstream.py` implements stable image tap capture + comparison for masks, paired detections, and token/text evidence; required-mode without a checkpoint yields a precise blocker; committed tiny-fixture/mock test plus honest-blocker default test.
3. **SAM3V-NN real inference.** SAM 3.1 video propagation produces model-derived masks from the real neural memory/tracker path (not `_deterministic_box`/`_box_mask`); `claim_level` for the real path is no longer `local_contract_fixture`; a tiny-fixture forward test asserts model-derived outputs; the streaming `SAM3VideoTracker` and session API still pass.
4. **SAM3V-CMP harness exists and is honest.** `tools/sam3_video_upstream.py` compares real local neural outputs against upstream within documented tolerances; required-mode without a checkpoint yields a precise component-specific blocker.
5. **Honest parity matrix.** `parity-status.json` is updated so each family's status string states the literal remaining blocker; any matrix-membership change (e.g., adding `sam3_video`) is explicit and justified. No family is claimed `UPSTREAM_PASSED` without a real reference comparison having passed.
6. **Runtime hygiene preserved.** No torch/transformers/huggingface_hub imports in `src/mlx_cv/`; reference/torch confined to `tools/`+`tests/`; `test_runtime_dependency_guards.py` passes; full `uv run pytest` suite green; `git diff --check` clean.
7. **Checkpoint-ready commands documented.** Each family has a documented required-mode command shape (`MLX_CV_REQUIRE_*_GATE=1` + env + tool/test) that a user with weights runs out-of-sandbox to obtain a truthful PASS or precise blocker.

## Scope Coverage Decisions

- **Included:** LA-CMP, SAM3I-CMP, SAM3V-NN, SAM3V-CMP; honest gates; parity-matrix truthfulness; light mocked/fixture tests; documented checkpoint-ready commands.
- **Assumption (not a blocker):** verified `UPSTREAM_PASSED` for any family requires the user to supply a real checkpoint and reference runtime; achieving the flip is out-of-sandbox and not a committed-CI outcome. If the user can supply a checkpoint path for any family, that family's real PASS can be attempted during execution.
- **Deferred / not in scope:** new model families (EoMT/DEIMv2/Sapiens2 — owned by the Phase 3 decision and its follow-on brief); visualization (`Result.draw`); SAM3/LocateAnything point-prompt (PVS) support (PCS/text/geometric scope is intentional); training/loss paths.
- **Needs-decision (record, not blocking):** whether `sam3_video` joins the release parity matrix is decided during execution based on whether SAM3V-NN+SAM3V-CMP reach a stable comparison; default is to add it with an honest blocker status.

## Source Evidence

- Audit + per-surface state: this session's findings; `parity-status.json` (`da3_multiview`/`rfdetr` PASSED, `locateanything`/`sam3_image` BLOCKED with stated missing components).
- Working comparison pattern to mirror: `tools/da3_upstream.py` (real torch capture + tap comparison), `tools/rfdetr_upstream.py`, `tests/test_da3_upstream_parity.py`.
- Stub gates to complete: `tools/locateanything_upstream.py:evaluate_locateanything_comparison_gate`, `tools/sam3_image_upstream.py:evaluate_sam3_image_comparison_gate`, `tools/sam3_video_upstream.py`.
- Deterministic video fixture to replace: `src/mlx_cv/models/sam3/video.py:_propagate_frame` / `_deterministic_box` / `_box_mask`; reference design under `references/sam3/` (`sam3_tracker_base.py`, `sam3_video_base.py`, `memory.py:SimpleMaskEncoder`).
- Real neural inference already present (no port needed): `SAM3Model.__call__`, `LocateAnythingModel.predict`/`pbd_generate`.
- Feasibility/risk facts: `.agent/wiki/LEARNINGS.md` (RF-DETR real-checkpoint architecture work; MoonViT bicubic convention).
- Reference checkpoints: SAM3.1 video `facebook/sam3.1` (`sam3.1_multiplex.pt`); LocateAnything-3B (`nvidia/LocateAnything-3B`, local shards are LFS stubs); SAM3.1 image checkpoint env `MLX_CV_SAM3_IMAGE_CHECKPOINT`.

## Anti-Goals

- Do not commit, convert-and-commit, or redistribute upstream weights.
- Do not claim `UPSTREAM_PASSED` from skipped tests, synthetic/deterministic fixtures, or local-only comparisons.
- Do not add torch/transformers/Triton/CUDA or `references/` imports to `src/mlx_cv/` runtime.
- Do not weaken SAM3 image-mode rejection of video/tracker checkpoints when adding the video neural path.
- Do not implement new model families, visualization, point-prompt (PVS) support, or training/loss in this change.
- Do not narrow the change to only the two easy comparison harnesses and silently drop the SAM3 video neural path.
