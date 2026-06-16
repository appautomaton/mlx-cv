# SPEC: RF-DETR + SAM 3.1 Detection And Segmentation

Change: `2026-06-16-rfdetr-sam31-detection-segmentation` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 5, `docs/BUILDING-BLOCKS.md`, `docs/ARCHITECTURE.md`, `references/rf-detr/`, `references/sam3/`, completed phases 1-4.

## Bounded Goal

Complete the final roadmap phase by landing RF-DETR image detection and SAM 3.1 image-mode segmentation as MLX-native, parity-tested model paths that return typed `Result.detections` and `Result.masks` in original-image coordinates.

## Broader Intent

Close the current MVP roadmap after LocateAnything by proving the remaining high-value output pillars: dense open-image detection through RF-DETR and promptable image segmentation through SAM 3.1. The phase should also harden the shared detection/segmentation building blocks that future models can reuse.

## Target User

Library users who want a single Apple-Silicon-native CV surface for detection and segmentation: load weights, pass an image and optional prompt, receive a typed `Result` without importing PyTorch at runtime.

## Work Scale And Shape

- Scale: final capability phase with two model families and shared infrastructure.
- Shape: parity-driven model ports plus shared ops/heads/processors.
- Selected lenses: product, engineering, runtime.

## Required Outcome

- RF-DETR detection path:
  - DINOv2-backed RF-DETR model assembly using the existing DINOv2 backbone contract where possible.
  - Multi-scale projector/neck, DETR query decoder, detection head, postprocess, conversion/load, and processor.
  - Deformable attention or a correctness-equivalent MLX implementation with focused parity tests.
  - User-facing `predict` path returning `Result(detections=...)` with boxes mapped back through `SpatialTransform`.
- SAM 3.1 image segmentation path:
  - Image-mode SAM 3.1 model assembly for ViT/VL backbone, neck, prompt/text encoding, decoder, mask decoder, conversion/load, and processor.
  - Text prompt and geometric prompt coverage for image segmentation; optional visual exemplar support only if the reference path can be bounded and fixture-tested in this change.
  - User-facing `predict` path returning `Result(masks=...)`, and detections when the reference image API yields boxes alongside masks.
- Shared detection/segmentation infrastructure:
  - Reusable homes for multi-scale neck/projector, query decoder pieces, deformable/grid sampling, mask resize/inversion, and model-space-to-original-space postprocess.
  - Runtime remains MLX/native and import-light: no hard `torch` or `transformers` dependency in package runtime.
- Parity and fixtures:
  - Tiny reference fixtures for RF-DETR and SAM 3.1 that cover enough intermediate taps to localize drift.
  - Full test suite remains green, with focused tests for shared ops, conversion/load, processor transforms, and public predict paths.

## Constraints And Risks

- `core/` must stay MLX-free. New tensor compute belongs under model/backbone/head/ops modules that already allow MLX.
- The hard invariant still applies: a shipped model path needs reference parity. If a full reference fixture cannot be minted in this runtime, the phase must record that as a deferred hardening gap and avoid shipped-model claims.
- RF-DETR is DINOv2-based, not DINOv3-based. Do not rewire it to DINOv3 because the repo already corrected that roadmap point.
- SAM 3.1 video/tracking memory is explicitly out of scope for this phase. Image-mode segmentation must not require tracker state.
- RF-DETR instance-segmentation variants are out of scope unless needed only as reference evidence. The RF-DETR output pillar for this phase is detection; the segmentation pillar is SAM 3.1 image-mode masks.
- SAM 3.1 text prompt support requires the reference VL/text path, not only geometric prompts.
- Deformable attention and mask coordinate inversion are high-risk because they can pass shape tests while drifting numerically or spatially.
- Keep weight-license behavior consistent with existing policy: surface RF-DETR and SAM 3.1 weight licenses, do not redistribute weights, and do not gate code inclusion on the weight license.

## Source Evidence

- Roadmap Phase 5 requires RF-DETR detection plus SAM 3.1 image segmentation, with video/tracking deferred: `.agent/steering/ROADMAP.md` lines 55-63 and 65-72.
- Building-block inventory identifies the shared gaps: multi-scale neck/projector, query-decoder family, geometry ops, prompt encoders, and mask support: `docs/BUILDING-BLOCKS.md` lines 21-35 and 56-66.
- RF-DETR reference anchors: `references/rf-detr/src/rfdetr/models/lwdetr.py:83`, `backbone/projector.py:162`, `transformer.py:135`, `transformer.py:350`, `ops/modules/ms_deform_attn.py:35`, `postprocess.py:19`.
- SAM 3.1 reference anchors: `references/sam3/sam3/model/sam3_image.py:34`, `sam3_image.py:167`, `vitdet.py:743`, `necks.py:15`, `decoder.py:192`, `geometry_encoders.py:83`, `geometry_encoders.py:404`, `geometry_encoders.py:470`, `vl_combiner.py:19`, `text_encoder_ve.py:255`, `tokenizer_ve.py:130`.
- Current output surface already has `Result.detections` and `Result.masks`, but mask JSON/export behavior may need extension: `src/mlx_cv/core/types.py` lines 27-93 and 135-202.

## Acceptance Criteria

1. RF-DETR construction: a local RF-DETR detection model can be constructed from a typed config, reusing existing DINOv2 components where compatible and isolating RF-DETR-specific projector/decoder/head code under clear module ownership.
2. RF-DETR conversion/load: reference state dict keys map explicitly into the local parameter tree; unsupported checkpoint variants fail with clear errors; tiny converted weights load into the local model.
3. RF-DETR deformable attention: the MLX implementation matches a reference fixture for representative sampling locations, attention weights, multi-level shapes, and output values within a documented tolerance.
4. RF-DETR predict path: `preprocess -> model -> postprocess` returns `Result.detections` with original-image `xyxy` boxes, scores, class ids, and labels; fixed-input reference parity covers final detections and at least one intermediate decoder/logit tap.
5. SAM 3.1 construction: a local image-mode SAM 3.1 model can be constructed from a typed config with image/VL backbone, neck, text/geometric prompt encoding, decoder, and mask decoder surfaces.
6. SAM 3.1 conversion/load: reference state dict keys map explicitly; tiny converted weights load; missing optional video/tracker keys are intentionally ignored or rejected by variant with clear errors.
7. SAM 3.1 prompt handling: processor support covers text prompts and geometric prompts, expands/encodes prompts through local runtime-light APIs, and does not make PyTorch or Transformers a runtime import.
8. SAM 3.1 mask output: `preprocess -> model -> postprocess` returns `Result.masks` in original-image resolution with labels/scores when available; mask resize/inversion is deterministic and tested on non-square images.
9. Shared infrastructure: reusable ops/heads/necks added for this phase have focused unit tests and do not regress DINOv2, DA3, LocateAnything, or core import-light guards.
10. Parity gates: RF-DETR and SAM 3.1 each have committed tiny fixture artifacts, mint tools, bisectable taps, and focused parity tests. The full `uv run pytest` suite passes.
11. Status truthfulness: README/roadmap/docs describe RF-DETR and SAM 3.1 as reference-proven only after the actual reference parity gates pass; any local-only fixture correction is called out explicitly.

## Scope Coverage Decisions

- Included: RF-DETR detection, SAM 3.1 image-mode segmentation, shared detection/segmentation ops and heads required by those two paths, conversion/load, processors, predict paths, parity fixtures, status docs.
- Deferred: SAM 3.1 video, tracker memory, object multiplex tracking, RF-DETR segmentation model variants, EoMT, Sapiens2, DEIMv2, YOLO26, visualization/UI, model download ergonomics beyond local load helpers.
- Assumption: this is the final roadmap phase for the current MVP, but planning may still split implementation into ordered slices inside this one change.

## Anti-Goals

- Do not start SAM video/tracking or build a tracker/memory subsystem in this change.
- Do not implement RF-DETR instance segmentation just because RF-DETR has segmentation variants upstream.
- Do not replace the existing DINOv2/DA3/LocateAnything implementations with reference copies.
- Do not add PyTorch, Transformers, Triton, CUDA, or other reference-runtime dependencies to package runtime.
- Do not claim shipped-model parity from local synthetic fixtures alone.
- Do not widen `Result` into task-specific subclasses; use the existing optional-field surface unless a concrete parity blocker proves a minimal extension is required.
