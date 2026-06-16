# DESIGN: RF-DETR + SAM 3.1 Detection And Segmentation

Change: `2026-06-16-rfdetr-sam31-detection-segmentation` - Spec: `SPEC.md`

## Architecture Approach

Keep this phase as two model-family ports sharing small, proven infrastructure:

- Shared MLX ops and typed contracts live in reusable homes only when both model paths need them.
- RF-DETR owns RF-DETR-specific DINOv2 feature adaptation, multi-scale projector choices, DETR decoder wiring, detection head, conversion, and processor.
- SAM 3.1 owns SAM-specific ViTDet/VL/text/prompt/mask decoder wiring, conversion, and processor.
- `core/` remains numpy-only. MLX compute stays under `backbones/`, `heads/`, `ops/`, and `models/`.

Do not force RF-DETR and SAM 3.1 into one abstract query-decoder class unless an implementation slice proves shared code removes real duplication. They both use query-decoder patterns, but the reference semantics differ enough that premature unification would hide parity bugs.

## Module Boundaries

Planned homes:

- `src/mlx_cv/ops/sampling.py` and/or `src/mlx_cv/ops/deformable.py`: MLX sampling/deformable attention primitives.
- `src/mlx_cv/backbones/vision/necks/`: reusable multi-scale neck/projector components when the RF-DETR/SAM implementations share them.
- `src/mlx_cv/heads/detection/`: RF-DETR query decoder, detection head, and postprocess helpers.
- `src/mlx_cv/heads/segmentation/`: SAM 3.1 prompt/mask decoder helpers that are useful outside the concrete model wrapper.
- `src/mlx_cv/models/rfdetr/`: config, model assembly, conversion, processor, and public prediction path.
- `src/mlx_cv/models/sam3/`: config, image-mode model assembly, conversion, processor, tokenizer/text prompt helpers, and public prediction path.

Core output changes must be minimal: `Result.masks` already exists, but JSON/export behavior and mask-shape validation can be completed if needed by SAM 3.1 postprocess.

## RF-DETR Path

Reference anchors:

- `references/rf-detr/src/rfdetr/models/lwdetr.py:83`
- `references/rf-detr/src/rfdetr/models/backbone/projector.py:162`
- `references/rf-detr/src/rfdetr/models/transformer.py:135`
- `references/rf-detr/src/rfdetr/models/transformer.py:350`
- `references/rf-detr/src/rfdetr/models/ops/modules/ms_deform_attn.py:35`
- `references/rf-detr/src/rfdetr/models/postprocess.py:19`

Implementation direction:

1. Reuse the existing DINOv2 backbone and conversion machinery where the reference checkpoint layout matches.
2. Add RF-DETR's multi-scale projector and query decoder as RF-DETR-owned code first.
3. Add a deformable attention primitive with focused fixture parity before relying on end-to-end detection parity.
4. Postprocess into `Result.detections` in original-image `xyxy` coordinates with scores, class ids, and labels.

RF-DETR segmentation variants remain out of scope; segmentation for this phase is SAM 3.1 image-mode.

## SAM 3.1 Path

Reference anchors:

- `references/sam3/sam3/model/sam3_image.py:34`
- `references/sam3/sam3/model/sam3_image.py:167`
- `references/sam3/sam3/model/vitdet.py:743`
- `references/sam3/sam3/model/necks.py:15`
- `references/sam3/sam3/model/decoder.py:192`
- `references/sam3/sam3/model/geometry_encoders.py:83`
- `references/sam3/sam3/model/vl_combiner.py:19`
- `references/sam3/sam3/model/text_encoder_ve.py:255`
- `references/sam3/sam3/model/tokenizer_ve.py:130`

Implementation direction:

1. Implement image-mode only: no tracker state, memory bank, propagation, or video predictor.
2. Support text prompts and PCS grounding box-exemplar prompts through runtime-light local APIs.
3. Use SAM 3.1's VL/text path for text prompt parity; do not silently reduce the scope to geometry-only SAM.
4. Treat SAM1-style interactive point/click prompting as deferred unless a later slice adds a separate reference fixture for that path.
5. Postprocess masks through `SpatialTransform.invert_mask` to original-image resolution, and carry reference grounding boxes into `Result.detections` when the image API emits them.

Visual exemplar support is limited to image-mode box-exemplar grounding and should land only if it can be fixture-tested without widening into video/object-multiplex tracking. The existing `ExemplarPrompt` type may be accepted only when execution can map it to that fixture-proven path; otherwise it must fail clearly instead of silently becoming a different prompt mode.
The tokenizer path must be explicit: SAM 3.1's reference tokenizer is CLIP-style BPE backed by a merge file, not a seed-weight artifact. The local runtime should use committed BPE assets or a fixture-backed tiny BPE asset, and must not add hard runtime imports for `torch`, `transformers`, `triton`, CUDA packages, or hidden reference tokenizer dependencies.

## Parity Strategy

Each model path needs a tiny reference fixture with enough taps to localize drift:

- RF-DETR: DINOv2/projector feature taps, deformable attention output, decoder hidden/logits, final boxes/scores/classes.
- SAM 3.1: image/VL feature taps, token ids/text embeddings, PCS box-exemplar prompt embeddings, decoder/mask logits, final masks, boxes, labels, and scores when available.

Reference minting may use throwaway reference dependencies out of band. Runtime package dependencies must not gain `torch`, `transformers`, `triton`, or CUDA-specific packages.

If a reference fixture cannot be minted in this runtime, the plan must not mark the model path as shipped/reference-proven. It must record the blocker and keep status wording honest.
