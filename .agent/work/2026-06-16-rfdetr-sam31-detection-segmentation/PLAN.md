# PLAN: RF-DETR + SAM 3.1 Detection And Segmentation

Change: `2026-06-16-rfdetr-sam31-detection-segmentation` - Stage: plan - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal

Execute `SPEC.md`: land RF-DETR image detection and SAM 3.1 image-mode segmentation as MLX-native, parity-tested paths returning `Result.detections` and `Result.masks` in original-image coordinates.

## Architecture Approach

See `DESIGN.md`. Execution should share only the small primitives both paths need: mask/result handling, prompt normalization, sampling/deformable-attention ops, and reusable neck/head pieces when parity-safe. RF-DETR and SAM 3.1 remain separate concrete model families until tests prove a shared abstraction is safe.

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Subagent recommended: Slices 2, 4, 8, 9, and 11 because they cross reference parity, model assembly, or fixture drift diagnosis.
- Parallel-safe groups: none by default. RF-DETR and SAM 3.1 both may touch shared fixture registries, `tests/fixtures/`, result/prompt/geometry tests, and reusable neck modules. Any parallel execution requires an explicit coordinator partition for those shared files; Slice 12 waits for both chains.
- Checkpoints: none. If reference fixture minting fails, record the blocker and return to planning instead of claiming parity.

## Ordered Slice Sequence

### Slice 1: Shared Mask, Prompt, And Geometry Contracts

**Objective:** Finish the result/prompt/geometry surface needed by both detection and image segmentation without widening `core/` beyond the spec.

**Acceptance criteria:**
- `Result.to_dict()` includes `masks` with shape, kind, and labels when present.
- `Masks` validates instance-mask label length when labels are supplied.
- Existing prompt dataclasses remain the shared prompt contract; this slice only adds missing validation or serialization needed by RF-DETR/SAM 3.1.
- Existing `SpatialTransform.invert_mask` behavior is covered on non-square resize and letterbox cases.
- Existing detection, depth, LocateAnything, and core import-light tests still pass.

**Touches:** `src/mlx_cv/core/types.py`, `src/mlx_cv/core/geometry.py`, `src/mlx_cv/prompts/`, `tests/test_types.py`, `tests/test_geometry.py`, `tests/test_prompts.py`, `tests/test_runtime_dependency_guards.py`.

**Produces:** Shared output/prompt guardrails for RF-DETR and SAM 3.1 processors.

**Verification:** `uv run pytest tests/test_types.py tests/test_geometry.py tests/test_prompts.py tests/test_runtime_dependency_guards.py tests/test_qwen2_integration_guards.py`

**Status:** complete
**Evidence:** changed `src/mlx_cv/core/types.py`, `tests/test_types.py`, `tests/test_geometry.py`, and `tests/test_runtime_dependency_guards.py`; `uv run pytest tests/test_types.py tests/test_geometry.py tests/test_prompts.py tests/test_runtime_dependency_guards.py tests/test_qwen2_integration_guards.py` passed with 34 tests.
**Risks / next:** none.

### Slice 2: MLX Sampling And Deformable Attention Primitives

**Objective:** Add the MLX sampling/deformable-attention primitive RF-DETR needs, with focused reference parity before any detector uses it.

**Acceptance criteria:**
- MLX sampling handles normalized coordinates, multi-level feature maps, attention weights, and out-of-range behavior matching the RF-DETR reference fixture.
- `MSDeformAttn`-equivalent output matches a tiny PyTorch reference case within documented tolerance.
- Shape and dtype errors fail clearly.
- The primitive is isolated under `ops/` or a head/model-local ops module, not `core/`.

**Touches:** `src/mlx_cv/ops/`, `tests/test_grid_sample.py`, `tests/test_ms_deform_attn.py`, `tools/mint_rfdetr_fixture.py`, `src/mlx_cv/parity/fixtures.py`.

**Produces:** Deformable attention parity fixture and reusable MLX op coverage.

**Verification:** `uv run pytest tests/test_grid_sample.py tests/test_ms_deform_attn.py tests/test_ops_boxes.py`

**Status:** complete
**Evidence:** changed `src/mlx_cv/ops/sampling.py`, `src/mlx_cv/ops/deformable.py`, `src/mlx_cv/ops/__init__.py`, `src/mlx_cv/parity/fixtures.py`, `tools/mint_rfdetr_fixture.py`, `tests/test_grid_sample.py`, and `tests/test_ms_deform_attn.py`; `uv run pytest tests/test_grid_sample.py tests/test_ms_deform_attn.py tests/test_ops_boxes.py` passed with 9 tests, and `uv run pytest tests/test_runtime_dependency_guards.py` passed with 3 tests.
**Risks / next:** full RF-DETR fixture minting remains for Slice 6; this slice covers the reusable deformable-attention primitive and tiny fixed reference case.

### Slice 3: RF-DETR DINOv2 Adapter And Multi-Scale Projector

**Objective:** Build RF-DETR's DINOv2 feature adapter and multi-scale projector on top of the existing DINOv2 backbone.

**Acceptance criteria:**
- RF-DETR config constructs the correct DINOv2-backed feature path without using DINOv3.
- Multi-scale projector outputs the reference level count, shapes, strides, and mask/positional metadata for a tiny config.
- Existing DINOv2 forward/parity tests still pass unchanged.
- The code path is model-owned or neck-owned with clear module boundaries.

**Touches:** `src/mlx_cv/backbones/vision/necks/`, `src/mlx_cv/models/rfdetr/`, `tests/test_rfdetr_neck.py`, `tests/test_rfdetr_model.py`, `tests/test_dinov2_forward.py`.

**Produces:** RF-DETR-ready multi-scale feature path.

**Verification:** `uv run pytest tests/test_rfdetr_neck.py tests/test_rfdetr_model.py tests/test_dinov2_forward.py`

**Depends on:** Slice 2

### Slice 4: RF-DETR Query Decoder, Detection Head, And Model Assembly

**Objective:** Assemble the RF-DETR detection compute graph from projected features through decoder logits and boxes.

**Acceptance criteria:**
- Transformer encoder/decoder pieces produce hidden states, reference points, class logits, and normalized boxes with reference-compatible shapes.
- Detection head returns model-space logits/boxes in `HeadOutput` or the RF-DETR model output contract.
- Deformable attention is wired through the decoder path and covered by tests that fail if it is bypassed.
- A tiny fixed-seed RF-DETR model forwards from image tensor/features to raw detection outputs.

**Execution:** subagent recommended.

**Touches:** `src/mlx_cv/heads/detection/`, `src/mlx_cv/models/rfdetr/modeling.py`, `tests/test_rfdetr_decoder.py`, `tests/test_rfdetr_model.py`.

**Produces:** RF-DETR raw detection model path.

**Verification:** `uv run pytest tests/test_rfdetr_decoder.py tests/test_rfdetr_model.py tests/test_ms_deform_attn.py`

**Depends on:** Slice 3

### Slice 5: RF-DETR Conversion, Processor, And Postprocess

**Objective:** Add RF-DETR conversion/load and user-facing preprocessing/postprocessing into `Result.detections`.

**Acceptance criteria:**
- Conversion maps reference RF-DETR keys explicitly and rejects unsupported segmentation-checkpoint variants with a clear error.
- Tiny converted weights load into the local RF-DETR model.
- Processor preprocess records `SpatialTransform`, normalizes/resizes images consistently with the reference, and builds model inputs.
- Postprocess converts normalized boxes to original-image `xyxy`, filters/top-k scores/classes consistently with the reference, and returns `Result.detections`.

**Touches:** `src/mlx_cv/models/rfdetr/{convert.py,processor.py,__init__.py}`, `tests/test_rfdetr_convert.py`, `tests/test_rfdetr_processor.py`, `tests/test_types.py`.

**Produces:** RF-DETR local load and `predict` surface.

**Verification:** `uv run pytest tests/test_rfdetr_convert.py tests/test_rfdetr_processor.py tests/test_types.py`

**Depends on:** Slice 4

### Slice 6: RF-DETR Reference Fixture And Detection Parity

**Objective:** Commit a tiny RF-DETR reference fixture and prove local detection parity end to end.

**Acceptance criteria:**
- Mint tool writes tiny RF-DETR fixture and weights from the reference path without adding runtime dependencies.
- Fixture taps cover projector features, deformable attention output, decoder hidden/logits, final boxes/scores/classes.
- `bisect` returns `None` for the local RF-DETR path.
- `predict` returns typed `Result.detections` for a fixed image.
- Status docs do not claim RF-DETR shipped/reference-proven unless this reference parity passes.

**Execution:** subagent recommended.

**Touches:** `tools/mint_rfdetr_fixture.py`, `tests/fixtures/`, `src/mlx_cv/parity/fixtures.py`, `tests/test_rfdetr_parity.py`, `tests/test_rfdetr_predict.py`, docs/status files as needed.

**Produces:** RF-DETR detection parity proof.

**Verification:** `uv run pytest tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py`

**Depends on:** Slice 5

### Slice 7: SAM 3.1 Runtime-Light Text And PCS Prompt Path

**Objective:** Add local SAM 3.1 prompt input handling for text and PCS box-exemplar geometry prompts without runtime PyTorch/Transformers dependencies.

**Acceptance criteria:**
- Local prompt normalization accepts `TextPrompt`, `BoxPrompt`, and equivalent dict/string inputs for image-mode grounding.
- `ExemplarPrompt` is accepted only when it maps to the fixture-backed image-mode box-exemplar grounding path; otherwise it fails clearly and remains deferred.
- `PointPrompt` is rejected for the PCS grounding path unless a separate SAM1-style interactive fixture is added in a later plan revision.
- SAM 3.1 tokenizer is implemented as deterministic CLIP-style BPE using committed real or reduced BPE assets, with a canonical string-to-token-id fixture.
- SAM 3.1 text encoder path can be constructed from typed config plus tiny weights and is tested with the tokenizer output.
- PCS box-exemplar prompt encoders cover box coordinates and labels in model space.
- The path explicitly rejects video/tracker prompt state and unsupported mask prompt state.
- No new hard runtime dependency is added for `torch`, `transformers`, `triton`, CUDA packages, or reference tokenizer helpers such as `ftfy`/`iopath`; if exact Unicode parity requires extra packages, stop and return to planning before adding them.
- Root/package import guards remain green.

**Touches:** `src/mlx_cv/models/sam3/`, `src/mlx_cv/heads/segmentation/`, `tests/fixtures/`, `tests/test_sam3_tokenizer.py`, `tests/test_sam3_prompts.py`, `tests/test_runtime_dependency_guards.py`, `tests/test_qwen2_integration_guards.py`.

**Produces:** SAM 3.1 prompt/text encoding surface.

**Verification:** `uv run pytest tests/test_sam3_tokenizer.py tests/test_sam3_prompts.py tests/test_runtime_dependency_guards.py tests/test_qwen2_integration_guards.py`

**Depends on:** Slice 1

### Slice 8: SAM 3.1 Image/VL Backbone And Neck

**Objective:** Port SAM 3.1 image-mode ViT/VL backbone and neck outputs needed by the image predictor.

**Acceptance criteria:**
- Tiny SAM 3.1 image/VL backbone config constructs without tracker/video modules.
- Neck outputs match reference level count, shapes, and dtype for a fixed tiny input.
- Text/VL fusion is represented when needed for text prompt parity; geometry-only shortcuts are not used for text prompt tests.
- Runtime imports stay MLX/native and package-root imports stay light.

**Execution:** subagent recommended.

**Touches:** `src/mlx_cv/backbones/vision/`, `src/mlx_cv/models/sam3/`, `tests/test_sam3_backbone.py`, `tests/test_sam3_neck.py`.

**Produces:** SAM 3.1 image/VL feature path.

**Verification:** `uv run pytest tests/test_sam3_backbone.py tests/test_sam3_neck.py tests/test_qwen2_integration_guards.py`

**Depends on:** Slice 7

### Slice 9: SAM 3.1 Decoder, Mask Decoder, And Model Assembly

**Objective:** Assemble SAM 3.1 image-mode decoding from image/prompt features to mask logits.

**Acceptance criteria:**
- Decoder and mask decoder produce reference-compatible mask logits, object scores, labels, and optional boxes for a tiny config.
- The model path is image-mode only and has no dependency on tracker memory or video state.
- Model construction and fixed-seed forward tests cover text and PCS box-exemplar prompt variants.
- Unsupported interactive point/click, mask-prompt, or video features fail clearly if not implemented.

**Execution:** subagent recommended.

**Touches:** `src/mlx_cv/heads/segmentation/`, `src/mlx_cv/models/sam3/modeling.py`, `tests/test_sam3_decoder.py`, `tests/test_sam3_model.py`.

**Produces:** SAM 3.1 raw image segmentation model path.

**Verification:** `uv run pytest tests/test_sam3_decoder.py tests/test_sam3_model.py tests/test_sam3_prompts.py`

**Depends on:** Slice 8

### Slice 10: SAM 3.1 Conversion, Processor, And Mask Postprocess

**Objective:** Add SAM 3.1 conversion/load and user-facing preprocessing/postprocessing into `Result.masks` and grounding detections when emitted.

**Acceptance criteria:**
- Conversion maps reference SAM 3.1 image-mode keys explicitly and ignores/rejects video/tracker keys by variant with clear errors.
- Tiny converted weights load into the local SAM 3.1 model.
- Processor preprocess records `SpatialTransform`, prepares image tensors, and maps original-space box/exemplar prompts into model space.
- Postprocess thresholds/resizes/inverts masks through `SpatialTransform.invert_mask`, preserving instance mask shape and labels when available.
- Paired mask/object scores remain on `Result.detections.scores` when grounding boxes are emitted unless a concrete parity blocker proves `Masks` needs a minimal extension.
- Postprocess carries grounding boxes/scores/classes into `Result.detections` when the SAM 3.1 image API emits boxes.
- Non-square image tests prove original-image mask resolution.

**Touches:** `src/mlx_cv/models/sam3/{convert.py,processor.py,__init__.py}`, `tests/test_sam3_convert.py`, `tests/test_sam3_processor.py`, `tests/test_geometry.py`, `tests/test_types.py`.

**Produces:** SAM 3.1 local load and `predict` surface.

**Verification:** `uv run pytest tests/test_sam3_convert.py tests/test_sam3_processor.py tests/test_geometry.py tests/test_types.py`

**Depends on:** Slice 9

### Slice 11: SAM 3.1 Reference Fixture And Image Segmentation Parity

**Objective:** Commit a tiny SAM 3.1 image-mode reference fixture and prove local text and PCS box-exemplar prompt mask parity.

**Acceptance criteria:**
- Mint tool writes tiny SAM 3.1 image-mode fixture and weights from the reference path without adding runtime dependencies.
- Fixture taps cover image/VL features, token ids/text embeddings, PCS box-exemplar prompt embeddings, decoder/mask logits, and final masks/boxes/scores.
- Text prompt and PCS box-exemplar prompt cases are covered by focused parity tests.
- If the top-level reference forward path cannot expose stable taps, the mint tool captures submethod-level taps and documents the tap points in the fixture metadata.
- `bisect` returns `None` for the local SAM 3.1 path.
- `predict` returns typed `Result.masks` at original-image resolution and typed `Result.detections` when grounding boxes are emitted for fixed inputs.

**Execution:** subagent recommended.

**Touches:** `tools/mint_sam3_fixture.py`, `tests/fixtures/`, `src/mlx_cv/parity/fixtures.py`, `tests/test_sam3_parity.py`, `tests/test_sam3_predict.py`.

**Produces:** SAM 3.1 image segmentation parity proof.

**Verification:** `uv run pytest tests/test_sam3_parity.py tests/test_sam3_predict.py`

**Depends on:** Slice 10

### Slice 12: Final Status, Package Guards, And Full Regression

**Objective:** Close the final roadmap phase only after RF-DETR and SAM 3.1 parity and public paths are verified.

**Acceptance criteria:**
- README, roadmap, and architecture docs report RF-DETR/SAM 3.1 status truthfully.
- License notes for RF-DETR and SAM 3.1 weights are surfaced without gating code inclusion.
- `pyproject.toml` has no runtime `torch`, `transformers`, `triton`, or CUDA-only dependency.
- A committed guard test rejects `torch`, `transformers`, `triton`, and CUDA-only runtime dependencies.
- SAM 3.1 tokenizer/text code does not import PyTorch or reference tokenizer helper packages at runtime.
- Package-root and `core/` import guards stay MLX-free.
- Full test suite passes.

**Touches:** `README.md`, `.agent/steering/ROADMAP.md`, `docs/`, `tests/test_runtime_dependency_guards.py`, guard tests, package exports.

**Produces:** Final phase closure evidence.

**Verification:** `uv run pytest`

**Depends on:** Slices 6 and 11

## Requirement Traceability

| SPEC acceptance | Satisfying slices |
| --- | --- |
| AC1 RF-DETR construction | Slices 3, 4 |
| AC2 RF-DETR conversion/load | Slice 5 |
| AC3 RF-DETR deformable attention | Slice 2 |
| AC4 RF-DETR predict/path parity | Slices 5, 6 |
| AC5 SAM 3.1 construction | Slices 7, 8, 9 |
| AC6 SAM 3.1 conversion/load | Slice 10 |
| AC7 SAM 3.1 prompt handling | Slices 7, 10, 11 |
| AC8 SAM 3.1 mask output | Slices 10, 11 |
| AC9 Shared infrastructure | Slices 1, 2, 3, 7 |
| AC10 Parity gates and full suite | Slices 6, 11, 12 |
| AC11 Status truthfulness | Slices 6, 11, 12 |

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Shared contracts | `uv run pytest tests/test_types.py tests/test_geometry.py tests/test_prompts.py tests/test_runtime_dependency_guards.py tests/test_qwen2_integration_guards.py` |
| Sampling/deformable attention | `uv run pytest tests/test_grid_sample.py tests/test_ms_deform_attn.py tests/test_ops_boxes.py` |
| RF-DETR model path | `uv run pytest tests/test_rfdetr_neck.py tests/test_rfdetr_decoder.py tests/test_rfdetr_model.py tests/test_rfdetr_convert.py tests/test_rfdetr_processor.py` |
| RF-DETR parity | `uv run pytest tests/test_rfdetr_parity.py tests/test_rfdetr_predict.py` |
| SAM 3.1 model path | `uv run pytest tests/test_sam3_tokenizer.py tests/test_sam3_prompts.py tests/test_sam3_backbone.py tests/test_sam3_neck.py tests/test_sam3_decoder.py tests/test_sam3_model.py tests/test_sam3_convert.py tests/test_sam3_processor.py tests/test_runtime_dependency_guards.py` |
| SAM 3.1 parity | `uv run pytest tests/test_sam3_parity.py tests/test_sam3_predict.py` |
| Final suite | `uv run pytest` |
| Dependency guard | `uv run pytest tests/test_runtime_dependency_guards.py` plus `uv run python -c "from pathlib import Path; s=Path('pyproject.toml').read_text(); assert all(x not in s for x in ('torch', 'transformers', 'triton', 'cuda'))"` |
| Core import guard | `uv run python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"` |

## Risks

- The phase is large. The plan keeps one active change because the roadmap asks for one final phase, but execution should keep slice evidence strict and not blur RF-DETR completion with SAM completion.
- RF-DETR and SAM 3.1 are not parallel-safe by default because they share fixtures, type/geometry tests, and possible neck utilities. Keep serial execution unless shared write ownership is explicitly partitioned.
- Deformable attention can pass shape tests while drifting numerically. Slice 2 must land fixture parity before RF-DETR model assembly relies on it.
- SAM 3.1 text prompts require the VL/text path and deterministic BPE tokenization. A geometry-only SAM port, or a tokenizer without string-to-token parity, would not satisfy the SPEC.
- SAM 3.1 has separate PCS grounding and interactive point/click prompt paths. This plan targets text and PCS box-exemplar grounding; interactive point/click support remains deferred unless a dedicated fixture is added.
- SAM 3.1 can emit scores with boxes and masks, but `Masks` has no score field today. Keep scores on paired `Result.detections` unless a parity blocker forces a minimal typed extension.
- SAM 3.1 reference code includes video/tracker paths near image-mode code. Execution must reject tracker/memory imports instead of accidentally pulling them into runtime.
- RF-DETR upstream has segmentation variants. Conversion must identify and reject those variants unless a detection checkpoint is being loaded.
- If either reference mint path is blocked by missing local reference dependencies, the correct outcome is a recorded parity blocker and truthful status wording, not a local-only completion claim.

## Review: Engineering

- Reviewer: Claude Code Opus 4.8, max effort, read-only plan mode, session `dc4882f7-d81d-4471-8d92-1125f11edd33`.
- Initial verdict: approved with risks pending plan corrections.
- Accepted corrections: make SAM 3.1 tokenizer/BPE assets explicit; narrow geometry scope to PCS box-exemplar grounding; add committed dependency guards for `triton`/CUDA and tokenizer imports; remove default parallel execution; clarify Slice 1 only fills missing mask/result gaps; carry SAM grounding boxes into `Result.detections`.
- Final re-review verdict: approved with risks, no required plan corrections remaining.
- Independent follow-up: tightened conditional `ExemplarPrompt` handling and kept SAM scores on paired `Result.detections.scores` unless a parity blocker proves `Masks` needs a minimal extension; Claude's final delta review preserved the approved-with-risk verdict.
- Action: proceed to execution with the remaining risks treated as execution guidance, not planning blockers.
