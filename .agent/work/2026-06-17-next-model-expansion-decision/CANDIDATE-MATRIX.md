# Candidate Matrix: Next Model Expansion Decision

Change: `2026-06-17-next-model-expansion-decision`

## Scope

This matrix covers exactly the three Phase 3 candidates:

| Candidate | Lane | Local reference | Current public source |
|---|---|---|---|
| DEIMv2 | DINOv3-backed real-time detection | `references/DEIMv2/` | https://github.com/Intellindust-AI-Lab/DEIMv2 |
| EoMT-DINOv3 | encoder-only segmentation / panoptic / instance / semantic | `references/eomt/` | https://github.com/tue-mps/eomt |
| Sapiens2 | human-centric pose / body-part masks / dense maps / matting | `references/sapiens2/` | https://github.com/facebookresearch/sapiens2 |

No other candidate is in scope. YOLO26 remains watchlist-only and RT-DETRv4 remains dropped by the roadmap.

## Evidence Labels

- `confirmed-against-current-source`: confirmed against a current public source during this phase, or against a current official repository page that links the checkpoint.
- `local-reference-only`: present in the vendored reference/docs, but not independently confirmed from a current public page during this phase.
- `external-access-required`: requires external acceptance, auth, DINOv3 base access, or out-of-git checkpoint download before a real gate can run.

## DEIMv2

### Source And Status

| Field | Evidence |
|---|---|
| Current source URL | https://github.com/Intellindust-AI-Lab/DEIMv2 |
| Public status | Current official repo page advertises "Real-Time Object Detection Meets DINOv3", a 2025-09-26 DEIMv2 release, 2025-11-03 Hugging Face model upload, and later 2026 updates. |
| Local reference | `references/DEIMv2/README.md`, `references/DEIMv2/configs/deimv2/deimv2_dinov3_s_coco.yml`, `references/DEIMv2/engine/backbone/dinov3_adapter.py` |
| Reference entry shape | Hugging Face mixin wrapper around `DINOv3STAs`, `HybridEncoder`, `DEIMTransformer`, and `PostProcessor`. |
| License/access | Repository badge and local `LICENSE` indicate Apache-2.0 code. DINOv3 or distilled backbone weights remain external artifacts; DINOv3-backed S/M/L/X paths are not pure Apache weight redistributions. |

### Checkpoint And Cache Inventory

| Model/checkpoint ID | Source status | Config/source metadata | Expected cache layout | Notes |
|---|---|---|---|---|
| `Intellindust/DEIMv2_DINOv3_S_COCO` | `confirmed-against-current-source` from current official model-zoo link; direct HF page text was not captured in this run | `references/DEIMv2/configs/deimv2/deimv2_dinov3_s_coco.yml`; `DINOv3STAs.name=vit_tiny`; `weights_path=./ckpts/vitt_distill.pt`; COCO AP 50.9 in local/current README | `/tmp/mlx-cv-checkpoints/deimv2/Intellindust/DEIMv2_DINOv3_S_COCO/` plus a separate distilled DINOv3 tiny checkpoint file | Best DINOv3-backed DEIMv2 gate candidate, but it needs STA, hybrid encoder, DETR decoder, postprocess, and likely deformable/multi-scale op coverage. |
| `Intellindust/DEIMv2_HGNetv2_FEMTO_COCO` | `confirmed-against-current-source` from current public HF model-card search and local/current model zoo | `references/DEIMv2/configs/deimv2/deimv2_hgnetv2_femto_coco.yml`; HGNetv2/LiteEncoder path | `/tmp/mlx-cv-checkpoints/deimv2/Intellindust/DEIMv2_HGNetv2_FEMTO_COCO/` | Smaller checkpoint path but not the approved DINOv3-backed candidate lane; useful fallback only if DINOv3 S is blocked. |

### Runtime Dependencies And Fit

- Reference runtime: PyTorch, Hugging Face Hub mixin, official `engine.*` modules, and COCO-style preprocessing/postprocessing.
- Existing mlx-cv reuse: `src/mlx_cv/backbones/vision/dinov3/`, `src/mlx_cv/backbones/vision/necks/`, `src/mlx_cv/heads/detection/rfdetr.py`, `src/mlx_cv/core/types.py:Detections`.
- Major missing local pieces: DINOv3 STA adapter, multi-scale spatial prior, DEIM/DFINE decoder details, and deformable-attention or equivalent multi-scale sampling if the reference path depends on it.

### First Gate Target

| Gate field | DEIMv2 target |
|---|---|
| First gate | Admit `Intellindust/DEIMv2_DINOv3_S_COCO` as a real-checkpoint status gate, without adding a release-parity row yet. |
| Model ID / checkpoint | `Intellindust/DEIMv2_DINOv3_S_COCO`; companion distilled DINOv3 tiny file from `DINOv3STAs.weights_path=./ckpts/vitt_distill.pt`. |
| Config/source | `references/DEIMv2/configs/deimv2/deimv2_dinov3_s_coco.yml`; reference wrapper in the official README; components `DINOv3STAs`, `HybridEncoder`, `DEIMTransformer`, `PostProcessor`. |
| Env/cache shape | `MLX_CV_DEIMV2_CHECKPOINT`, `MLX_CV_DEIMV2_DINOV3_BACKBONE`, `MLX_CV_DEIMV2_CONFIG`, `MLX_CV_REQUIRE_DEIMV2_GATE=1`; cache under `/tmp/mlx-cv-checkpoints/deimv2/`. |
| Reference entry point | Official README Hugging Face mixin sample plus `references/DEIMv2/engine/`; future gate tool should import reference code only in an env-gated tool/test. |
| Expected output/taps | DINOv3STA multi-scale features, encoder outputs, decoder logits/boxes, postprocessed COCO boxes/scores/labels mapped to `Result.detections`. |
| Blocker taxonomy | `external_checkpoint_missing`, `external_backbone_missing`, `unsupported_checkpoint_format`, `local_sta_missing`, `local_deformable_or_sampling_missing`, `converter_missing`, `comparison_tap_missing`. |

## EoMT-DINOv3

### Source And Status

| Field | Evidence |
|---|---|
| Current source URL | https://github.com/tue-mps/eomt |
| Public status | Current official repo page identifies EoMT as CVPR 2025 Highlight code and includes DINOv3 support with COCO panoptic, COCO instance, and ADE20K semantic model-zoo entries. |
| Local reference | `references/eomt/README.md`, `references/eomt/model_zoo/dinov3.md`, `references/eomt/models/eomt.py`, `references/eomt/configs/dinov3/` |
| Reference entry shape | Encoder-only ViT path: learned query tokens are inserted into late ViT blocks; final outputs are mask logits and class logits. |
| License/access | MIT code. DINOv3 EoMT weights are deltas relative to original DINOv3 weights, so DINOv3 base access is required before a real local gate can run. |

### Checkpoint And Cache Inventory

| Model/checkpoint ID | Source status | Config/source metadata | Expected cache layout | Notes |
|---|---|---|---|---|
| `tue-mps/coco_panoptic_eomt_small_640_dinov3` / `pytorch_model.bin` | `confirmed-against-current-source` from current HF model listing and local/current DINOv3 model zoo | `references/eomt/configs/dinov3/coco/panoptic/eomt_small_640_2x.yaml`; COCO panoptic 640; local zoo reports 47.2 PQ | `/tmp/mlx-cv-checkpoints/eomt-dinov3/coco_panoptic_eomt_small_640_dinov3/pytorch_model.bin` plus a separate DINOv3 base cache | Smallest credible DINOv3 panoptic gate. Needs delta-weight composition against DINOv3 before conversion/comparison. |
| `tue-mps/ade_semantic_eomt_large_512_dinov3` / `pytorch_model.bin` | `confirmed-against-current-source` from local/current DINOv3 model zoo; HF model listing confirms DINOv3 EoMT family | `references/eomt/configs/dinov3/ade20k/semantic/eomt_large_512.yaml`; ADE20K semantic 512; local zoo reports 59.5 mIoU | `/tmp/mlx-cv-checkpoints/eomt-dinov3/ade_semantic_eomt_large_512_dinov3/pytorch_model.bin` plus DINOv3 base cache | Higher-value semantic benchmark but larger first gate and less aligned with smallest admission. |

### Runtime Dependencies And Fit

- Reference runtime: PyTorch 3.13-era environment, timm/transformer backbone conventions, PyTorch Lightning stack for training/validation scripts, and optional notebook inference.
- Existing mlx-cv reuse: `src/mlx_cv/backbones/vision/dinov3/`, `src/mlx_cv/heads/segmentation/`, `src/mlx_cv/core/types.py:Masks`, `src/mlx_cv/transforms/`.
- Major missing local pieces: EoMT query-token injection into final ViT blocks, `ScaleBlock` upscaler, mask/class heads, DINOv3 delta-weight composition, and stable mask/class tap capture.

### First Gate Target

| Gate field | EoMT-DINOv3 target |
|---|---|
| First gate | Admit `tue-mps/coco_panoptic_eomt_small_640_dinov3` as a real-checkpoint status gate, with exact blocker reporting if DINOv3 base access or delta-weight composition is unavailable. |
| Model ID / checkpoint | `tue-mps/coco_panoptic_eomt_small_640_dinov3`, file `pytorch_model.bin`; separate original DINOv3 base checkpoint required because EoMT DINOv3 weights are deltas. |
| Config/source | `references/eomt/configs/dinov3/coco/panoptic/eomt_small_640_2x.yaml`; reference module `references/eomt/models/eomt.py:EoMT`; model zoo `references/eomt/model_zoo/dinov3.md`. |
| Env/cache shape | `MLX_CV_EOMT_DINOV3_CHECKPOINT`, `MLX_CV_EOMT_DINOV3_BASE_CHECKPOINT`, `MLX_CV_EOMT_DINOV3_CONFIG`, `MLX_CV_REQUIRE_EOMT_DINOV3_GATE=1`; cache under `/tmp/mlx-cv-checkpoints/eomt-dinov3/`. |
| Reference entry point | `references/eomt/models/eomt.py:EoMT` and validation/inference paths from `references/eomt/main.py` / `references/eomt/inference.ipynb`. |
| Expected output/taps | Per-layer mask logits and class logits from `EoMT.forward`; final mask/class tensors mapped to `Result.masks` with semantic/panoptic metadata deferred to the follow-on spec. |
| Blocker taxonomy | `external_checkpoint_missing`, `dinov3_base_access_missing`, `delta_weight_composition_missing`, `local_query_token_path_missing`, `local_scale_block_missing`, `converter_missing`, `comparison_tap_missing`. |

## Sapiens2

### Source And Status

| Field | Evidence |
|---|---|
| Current source URL | https://github.com/facebookresearch/sapiens2 |
| Public status | Current official repo page identifies Sapiens2 as ICLR 2026, records initial release on 2026-04-24 for pose/body-part segmentation/surface normals/pointmaps, and a 2026-05-15 matting release. |
| Local reference | `references/sapiens2/README.md`, `references/sapiens2/docs/MODEL_ZOO.md`, `references/sapiens2/LICENSE.md`, `references/sapiens2/sapiens/backbones/sapiens2.py` |
| Reference entry shape | High-resolution ViT backbone with standalone `Sapiens2`; task heads live under pose/dense paths for pose heatmaps, segmentation, normals, pointmaps, and matting. |
| License/access | Custom Sapiens2 license. It includes surveillance, biometric processing, re-identification, deepfake, sensitive-information, professional-practice, and trade-control restrictions; every future gate needs explicit license notes. |

### Checkpoint And Cache Inventory

| Model/checkpoint ID | Source status | Config/source metadata | Expected cache layout | Notes |
|---|---|---|---|---|
| `facebook/sapiens2-pretrain-0.1b` / `sapiens2_0.1b_pretrain.safetensors` | `local-reference-only` from `MODEL_ZOO.md`; current repo links the Sapiens2 HF collection but this direct page was not text-captured | Standalone backbone gate; no user-visible task head unless treated as `Embedding` | `/tmp/mlx-cv-checkpoints/sapiens2/pretrain/sapiens2_0.1b_pretrain.safetensors` | Smallest backbone admission candidate, but weak as a user-facing Result gate. |
| `facebook/sapiens2-seg-0.4b` / `sapiens2_0.4b_seg.safetensors` | `local-reference-only` from `MODEL_ZOO.md`; current HF search confirmed the Sapiens2 task-family pages but not this exact direct page | Body-part segmentation task; expected `Masks(kind="semantic")` or class-index mask output | `/tmp/mlx-cv-checkpoints/sapiens2/seg/sapiens2_0.4b_seg.safetensors` | Smallest user-visible segmentation gate if license and runtime scale are acceptable. |
| `facebook/sapiens2-normal-0.4b` / `sapiens2_0.4b_normal.safetensors` | `confirmed-against-current-source` from current public HF model-card search and local model zoo | Surface-normal task; 1024 x 768 inference; 0.398B backbone | `/tmp/mlx-cv-checkpoints/sapiens2/normal/sapiens2_0.4b_normal.safetensors` | Direct dense-map gate, but it requires future `Result` widening for normals before it is a clean public surface. |

### Runtime Dependencies And Fit

- Reference runtime: Python >=3.12, PyTorch >=2.7, `safetensors`, and task-specific demo scripts; pose additionally requires a person detector.
- Existing mlx-cv reuse: `src/mlx_cv/backbones/vision/vit.py`, `src/mlx_cv/heads/dense/`, `src/mlx_cv/heads/segmentation/`, `src/mlx_cv/core/types.py:Masks`, `Keypoints`, and `Embedding`.
- Major missing local pieces: Sapiens2 backbone variants, grouped-query attention, SwiGLU FFN exactness, large-resolution positional handling, pose/dense heads, and potentially new `Result` fields for normals, pointmaps, and matting.

### First Gate Target

| Gate field | Sapiens2 target |
|---|---|
| First gate | Prefer a task-visible body-part segmentation admission gate, `facebook/sapiens2-seg-0.4b`, if the direct HF page and license acceptance are confirmed; otherwise fall back to a pretrain-only backbone status gate and mark user-visible task parity blocked. |
| Model ID / checkpoint | Primary: `facebook/sapiens2-seg-0.4b`, file `sapiens2_0.4b_seg.safetensors`; fallback: `facebook/sapiens2-pretrain-0.1b`, file `sapiens2_0.1b_pretrain.safetensors`. |
| Config/source | `references/sapiens2/sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py`; backbone `references/sapiens2/sapiens/backbones/sapiens2.py`; segmentation head `references/sapiens2/sapiens/dense/src/models/heads/seg_head.py`. |
| Env/cache shape | `MLX_CV_SAPIENS2_CHECKPOINT`, `MLX_CV_SAPIENS2_TASK=seg`, `MLX_CV_SAPIENS2_CONFIG`, `MLX_CV_REQUIRE_SAPIENS2_GATE=1`; cache under `/tmp/mlx-cv-checkpoints/sapiens2/seg/` or `/tmp/mlx-cv-checkpoints/sapiens2/pretrain/`. |
| Reference entry point | `references/sapiens2/sapiens/dense/scripts/demo/seg.sh`, dense segmentation estimator/head paths, and standalone backbone quick-start for pretrain-only fallback. |
| Expected output/taps | Body-part class logits or final label map to `Result.masks`; fallback backbone features to `Result.embedding` only if task gate is blocked. |
| Blocker taxonomy | `license_acceptance_required`, `external_checkpoint_missing`, `unsupported_safetensors_layout`, `local_sapiens2_backbone_missing`, `local_gqa_or_swiglu_missing`, `local_task_head_missing`, `result_surface_widening_required`, `comparison_tap_missing`. |

## Release Matrix Boundary

Phase 3 is a decision phase. None of these first gates expands `.agent/work/2026-06-16-release-parity-hardening/parity-status.json`; a selected-family implementation must create its own status artifact first and earn any future release-matrix row through a separate verified change.
