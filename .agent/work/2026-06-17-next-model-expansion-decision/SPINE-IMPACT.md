# Spine Impact: Next Model Expansion Decision

Change: `2026-06-17-next-model-expansion-decision`

## Existing Spine Surfaces Used For This Comparison

| Surface | Current evidence |
|---|---|
| Unified output | `src/mlx_cv/core/types.py:Result` with `Detections`, `Masks`, `Keypoints`, `DepthMap`, `Embedding`, `Tracks`, DA3 multi-view depth, and camera geometry. |
| DINOv3 reuse | `src/mlx_cv/backbones/vision/dinov3/modeling.py` binds DINOv3 to shared `ViTBackbone` and exposes parity-oriented taps. |
| Detection reuse | `src/mlx_cv/heads/detection/rfdetr.py` has query decoder machinery and `ms_deform_attn_core` usage. |
| Segmentation reuse | `src/mlx_cv/heads/segmentation/sam3.py` has prompt-conditioned mask projection over image features, but not EoMT query-token injection. |
| Dense reuse | `src/mlx_cv/heads/dense/dpt.py` and `dualdpt.py` cover depth-style dense maps, not Sapiens2 normals/pointmaps/matting result contracts. |
| Transform discipline | `src/mlx_cv/transforms/` exposes resize/letterbox/normalize primitives and keeps spatial inversion in `core.geometry`. |

This phase does not change those surfaces. Any `Result` widening named below is future-scope for the selected-family implementation or a later roadmap phase.

## DEIMv2

| Impact area | Analysis |
|---|---|
| `Result` impact | Low. DEIMv2 produces boxes, scores, and class labels that fit `Result.detections` / `Detections` without adding new fields. |
| Processor/transform impact | Moderate. The DINOv3-S COCO config uses 640 x 640 resize and ImageNet normalization; postprocess must map normalized boxes back to original image size. Existing `Resize`, `Letterbox`, and detection coordinate code are relevant, but DEIMv2's COCO postprocessor needs a dedicated processor. |
| Backbone reuse | Medium. It reuses the existing DINOv3 family conceptually, but the selected S path goes through `DINOv3STAs`, a Spatial Tuning Adapter that turns DINOv3 features into multi-scale detector features. That adapter is not in `src/mlx_cv/backbones/vision/necks/`. |
| Neck/head reuse | Medium. `src/mlx_cv/heads/detection/rfdetr.py` already has DETR-style query machinery and multi-scale deformable attention primitives, but DEIMv2 has its own hybrid encoder, DEIM transformer, dense O2O conventions, and postprocessor. Reuse is architectural, not drop-in. |
| Missing ops / blocks | STA detail fusion, DEIM hybrid encoder, exact decoder layers, possible deformable or multi-scale sampling differences, and COCO postprocess exactness. |
| Converter/load complexity | Moderate to high. The target HF mixin model is reachable as a model ID, but the DINOv3-backed S config also references a separate distilled tiny DINOv3 file at `./ckpts/vitt_distill.pt`; the converter must reconcile HF state dict layout, reference `engine.*` names, and local DINOv3/adapter names. |
| Local fixture shape | One RGB image at 640 x 640, reference taps for STA multi-scale features, encoder outputs, decoder logits/boxes, postprocessed boxes/scores/labels, and a small COCO-label mapping check. |
| Import-light risk | Low for the package if reference imports stay in `tools/` or env-gated tests. Medium for tests because the official reference stack imports PyTorch and its local `engine` package. |

### DEIMv2 Summary

DEIMv2 is the cleanest `Result` fit because detection is already modeled. Its drawback is roadmap overlap: RF-DETR already covers a real checkpoint detection lane, while DEIMv2 would mainly deepen the same output pillar and add adapter/deformable complexity before expanding segmentation or human-centric outputs.

## EoMT-DINOv3

| Impact area | Analysis |
|---|---|
| `Result` impact | Low to moderate. EoMT outputs class logits and mask logits that fit `Result.masks` for semantic/instance/panoptic masks. Panoptic segment metadata may need a future metadata convention, but no broad new top-level field is required for a first gate. |
| Processor/transform impact | Moderate. EoMT DINOv3 panoptic uses fixed image sizing in config and model-zoo variants at 640 or 1280. The follow-on gate can start with a single 640 image, normalized tensor input, and postprocess to mask logits or class-index masks before full panoptic serialization. |
| Backbone reuse | High. EoMT directly builds on a ViT/DINOv3-style encoder. The current `DINOv3ViT` already carries token order, RoPE, and capture taps, though EoMT must inject query tokens into the final ViT blocks and run the final blocks with optional attention masks. |
| Neck/head reuse | Medium. Existing segmentation code has mask projection ideas from SAM3, but EoMT's `ScaleBlock`, query embedding, late-block query insertion, and `einsum("bqc, bchw -> bqhw", ...)` mask head are distinct enough to deserve a small `heads/segmentation/eomt.py` or model-local head. |
| Missing ops / blocks | Query-token insertion into final ViT blocks, optional masked attention over query/patch tokens, `ScaleBlock` upscaler, class head, mask head, and stable final/per-layer mask/class tap capture. It does not appear to require deformable attention. |
| Converter/load complexity | Moderate. The local DINOv3 model-zoo file says EoMT DINOv3 weights are deltas relative to original DINOv3 weights. A credible converter must compose or validate those deltas against a separately admitted DINOv3 base checkpoint. |
| Local fixture shape | One RGB image at 640 x 640, base DINOv3 checkpoint or precise blocker, EoMT delta checkpoint, final mask logits `B,Q,H,W`, class logits `B,Q,C+1`, and optional per-layer logits for tap parity. |
| Import-light risk | Low if the reference stack remains isolated. The local implementation can be narrow: DINOv3 backbone reuse plus a small segmentation head. Runtime dependency risk comes mainly from the env-gated reference comparison path, not package imports. |

### EoMT-DINOv3 Summary

EoMT-DINOv3 gives the best spine leverage for the next expansion: it exercises the already-built DINOv3 backbone in a new output pillar, adds a compact mask head, and avoids the broad `Result` widening of Sapiens2. Its central risk is honest checkpoint handling because the DINOv3 weights are deltas.

## Sapiens2

| Impact area | Analysis |
|---|---|
| `Result` impact | High. Body-part segmentation can fit `Result.masks`, and pose can fit `Result.keypoints`, but normals, pointmaps, albedo, and matting are not represented as first-class typed fields. A pretrain-only gate would fit `Result.embedding` but would not prove a user-visible task. |
| Processor/transform impact | High. The reference models are trained at 1024 x 768 `(H, W)` and use human-centric task preprocessing. Pose additionally needs person boxes from a detector. A task gate needs image resize/normalization plus body-part class mapping or task-specific dense-map postprocessing. |
| Backbone reuse | Medium. Sapiens2 is ViT-shaped, but it brings grouped-query attention, SwiGLU FFN, large-resolution handling, multiple model scales, and task-specific heads. The current shared `ViTBackbone` is a useful starting point, not a near drop-in. |
| Neck/head reuse | Medium. `src/mlx_cv/heads/dense/` is depth-oriented and `heads/segmentation/` is SAM3-oriented. Sapiens2 needs new dense/seg/pose heads or adapters around reference task heads. |
| Missing ops / blocks | Sapiens2 backbone variants, GroupedQueryAttention, SwiGLUFFN exactness, dense head variants, pose heatmap head, matting/pointmap/normal result typing, and possibly detector orchestration for top-down pose. |
| Converter/load complexity | High. Checkpoints are safetensors and numerous, but model scale starts at 0.1B pretrain and 0.4B task checkpoints. Task heads, config files, license gates, and large image sizes make the first honest gate heavier than DEIMv2 or EoMT. |
| Local fixture shape | For body-part segmentation: one human RGB image at 1024 x 768, class logits or label map, final `Masks` payload, and stable backbone/head taps. For fallback pretrain: one synthetic/image tensor and dense feature tap, but that is not a complete user-visible task gate. |
| Import-light risk | Medium to high. The reference code has a broad task tree and license-sensitive use cases. Future work must isolate reference imports to tools/tests and avoid adding PyTorch/safetensors import requirements to package import paths. |

### Sapiens2 Summary

Sapiens2 has the highest user-facing breadth, but it is not the cleanest next step. Its license restrictions, scale, task spread, and `Result` widening risk make it better as a later dedicated phase after the segmentation lane proves the DINOv3+mask-head pattern.

## Cross-Candidate Spine Judgment

| Candidate | Spine leverage | Result churn | First-gate complexity | Main reason to select or defer |
|---|---|---|---|---|
| DEIMv2 | Medium | Low | Medium-high | Select only if checkpoint simplicity outweighs detection-pillar overlap. |
| EoMT-DINOv3 | High | Low-medium | Medium | Strongest balance: reuses DINOv3 and opens segmentation/panoptic masks with a small head. |
| Sapiens2 | Medium | High | High | Defer until the project is ready for human-centric license notes, larger checkpoints, and possible `Result` widening. |
