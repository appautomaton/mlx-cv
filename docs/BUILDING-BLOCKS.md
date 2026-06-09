# mlx-cv — Building Blocks & Spine Gaps

> **Status: foundation design — verified June 2026.** Derived by decomposing 10 real reference
> implementations (cloned under `references/`, git-ignored) against the `v0.0.2` spine. Cross-read by
> Codex (111 file reads across the corpus + `src/mlx_cv`) and spot-verified against the MLX reference
> (the official PyTorch DINOv3, `references/dinov3`) and the current `src/mlx_cv` contracts. This is the source of truth for *what
> the modular foundation needs*; the build sequence lives in `.agent/steering/ROADMAP.md`.

## Why this exists

The product is the **foundation**, not any single model. Each model stress-tests a different spine
contract. Decomposing the real reference code tells us (a) which small, reusable **building blocks**
recur — the things we build once and reuse — and (b) where the current spine is too narrow to hold
them. Build the shared blocks once → adding the Nth model is mostly `convert.py` + `processor.py`.

Corpus (`references/README.md`): DINOv3 · RF-DETR · Depth Anything V3 · SAM 3.1 · EoMT · Sapiens2 ·
DEIMv2 · LocateAnything (PyTorch + `mlx-vlm` port).

## Part 1 — Build-once blocks (ranked by reuse)

| # | Block | What it is | Used by | Home |
|---|-------|-----------|---------|------|
| 1 | **ViT backbone contract** | patch-embed + cls/register/storage tokens + intermediate-layer extraction + output-layout adapter | ~all (DINOv3, RF-DETR, DA3, SAM, EoMT, Sapiens2, LocateAnything-vision) | `backbones/vision` |
| 2 | **Parameterized Transformer block** | norm + attn + FFN + LayerScale + DropPath; MLP/SwiGLU, RMS/LayerNorm choices | all ViTs + Qwen2 LLM | `core/layers` |
| 3 | **2D positional-encoding suite** | learned-abs-interp / sine / RoPE / complex-RoPE / rel-bias / windowed-RoPE | all (variant-heavy) | `backbones/vision/position` + `ops/position` |
| 4 | **Attention ops family** | SDPA core + packed/separate qkv + GQA/MQA + window/global policy + block masks + KV-cache | all (NOT drop-in) | `ops/attention` |
| 5 | **Multi-scale neck/projector** | ViT features → FPN/pyramid (conv up/down, STA/detail fusion) | RF-DETR, SAM, DEIMv2, DA3 | `backbones/vision/necks` |
| 6 | **Dense-map head family** | DPT fusion + generic deconv/PixelShuffle map heads | DA3, Sapiens2, EoMT, RF-seg | `heads/dense` |
| 7 | **Query-decoder family** | DETR/SAM learned queries + refpoints + iterative box-refine + query-mask + optional text/prompt cross-attn | RF-DETR, SAM, DINO-det | `heads/detection` · `heads/segmentation` |
| 8 | **Geometry ops package** | box convert/IoU/GIoU/NMS, mask IoU/mask-to-box, grid_sample, RoIAlign, camera pose/quaternion | all det/seg/depth | `ops/` |
| 9 | **Processor + SpatialTransform v2** | resize/letterbox/normalize/patchify + composable inverse for boxes/points/**masks/depth/heatmaps** | all | `core` + `transforms` |
| 10 | **VLM bridge** | patch-merge + vision→LLM projector + image-token scatter + Qwen2 cache/block-mask | LocateAnything (+ future VLMs) | `backbones/llm` + `backbones/vlm` |
| 11 | **PBD / grounding decoder** | token grammar + coordinate-token decode + label association + norm→pixel | LocateAnything | `heads/grounding` + `ops/coord` |
| 12 | **Tracker / memory subsystem** | video state object + mask-memory encoder + object pointers + det↔track association | SAM video | `core/tracking` + `heads/tracking` |
| 13 | **Weight convert / `sanitize`** *(cross-cutting plumbing)* | key-remap + layout fixes: reference `state_dict` → MLX param tree | every model (load path) | `hub/` + `models/*/convert.py` |

**Top 5 = the modular core** (cover the most models) — build these first.

**Design directive:** attention (#4) and positional encoding (#3) are *variant-heavy across models* —
build each as a **parameterized family** (selectable norm / FFN / attention variant / posenc), not a
single implementation. The official `references/dinov3` blocks (`SelfAttention` + pre-norm block)
are the template — they already parameterize norm / attn / FFN / LayerScale.

## Part 2 — Spine contract gaps (what `v0.0.2` is missing)

| Contract (`src/mlx_cv/...`) | Current | Needs | Stressed by | Priority |
|---|---|---|---|---|
| `core/base.py:VisionBackbone.__call__(x)->list` | image → list | richer feature contract: token dicts / multi-view `(feat,aux)` / packed tokens + grid metadata | every MVP model | 🔴 highest |
| `core/geometry.py:SpatialTransform` | points + boxes only | dense inversion for masks / depth / heatmaps (+ multi-view camera metadata) | DA3, SAM, EoMT | 🔴 high |
| `core/base.py:Head.__call__(feats)` | feats only | also query state / image size / prompt / memory / logits | RF-DETR, DA3, SAM | 🔴 high |
| `core/base.py:LanguageBackbone` | embeds→hidden + embed() | KV-cache, RoPE position ids, block/Magi masks, image-token scatter | LocateAnything (anchor) | 🔴 high |
| `ops/` | basic box ops | deformable-attn, mask IoU / mask-to-box, grid_sample, RoIAlign, GIoU, camera/quaternion | RF-DETR (deform), SAM (mask) | 🟠 medium |
| `core/types.py:Result` — DA3 fields | `DepthMap` has no confidence/camera | `depth_conf` + camera pose/intrinsics | **Depth Anything V3 (MVP, Phase 3)** | 🔴 high (MVP) |
| `core/types.py:Result` — human fields | no normals/albedo/pointmap/matting | those fields | Sapiens2 | 🟡 low (deferred) |
| `core/base.py:Tracker.init/step` | minimal | memory bank, per-object state, feature cache, add/remove/update, det↔track association | SAM **video** | 🟡 low (image seg works without it) |
| `prompts/` | dataclasses only | an encoder contract (mutable prompt sequences: boxes/points/masks/labels) | SAM | 🟡 low (with prompt/video) |

**Good news:** `core/registry.py` direction is right — `BACKBONES` already separates `vision`/`llm`
kinds; just extend with feature-contract / config-schema, no rewrite.

> **Concrete `BackboneFeatures` schema (makes the #1 gap implementable):** the feature contract must
> carry — token/map layout (`B,H,W,C` | `B,N,C` | packed `L,C` | multi-view `B,S,N,C`), strides / grid
> shape, cls/register/storage token offsets, valid mask, view axis, dtype. "Richer feature contract"
> alone is too vague to build against.

## Part 3 — Per-model component map (entry points under `references/`)

- **DINOv3** — `dinov3/dinov3/models/vision_transformer.py:DinoVisionTransformer`; layers `patch_embed/attention/block/ffn_layers/rms_norm/layer_scale`. Eval heads exist: `eval/depth/.../dpt_head.py:DPTHead`, `eval/detection/.../detr.py:PlainDETR`, `eval/segmentation/...:Mask2FormerHead`.
- **RF-DETR** — `rf-detr/src/rfdetr/models/lwdetr.py:LWDETR` (DINOv2 backbone — *not* v3); `backbone/projector.py:MultiScaleProjector`; `transformer.py:Transformer/TransformerDecoder`; `ops/modules/ms_deform_attn.py:MSDeformAttn`; `postprocess.py:PostProcess`.
- **Depth Anything V3** — `Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net`; multi-view DINOv2 w/ local+global attn + camera-token injection; heads `dpt.py:DPT`, `dualdpt.py:DualDPT`, `gsdpt.py:GSDPT`; `cam_enc.py`/`cam_dec.py`; `gs_adapter.py:GaussianAdapter`.
- **SAM 3.1** — `sam3/sam3/model/vitdet.py:ViT`; `necks.py:Sam3DualViTDetNeck`; `encoder.py`/`decoder.py:TransformerDecoder`; prompts `geometry_encoders.py:{Prompt,SequenceGeometryEncoder,MaskEncoder}`; video `memory.py:SimpleMaskEncoder`, `sam3_tracker_base.py:Sam3TrackerBase`, `sam3_video_base.py:Sam3VideoBase`; `box_ops.py`. **Text/VL path (required for text prompts, not just geometry):** `vl_combiner.py:SAM3VLBackbone`, `text_encoder_ve.py:VETextEncoder`, `tokenizer_ve.py:SimpleTokenizer`; prompt fusion `sam3_image.py:Sam3Image._encode_prompt` (text + geometric + optional `visual_prompt_embed`).
- **EoMT** — `eomt/models/eomt.py:EoMT` (injects query tokens into final ViT blocks; einsum mask head); `vit.py:ViT`; `scale_block.py:ScaleBlock`.
- **Sapiens2** — `sapiens2/sapiens/backbones/sapiens2.py:Sapiens2` (+ `Tokenizer`, `GroupedQueryAttention`, `SwiGLUFFN`); dense `dense/src/models/core/*_estimator.py` → heads `SegHead/NormalHead/AlbedoHead/MattingHead/PointmapHead`; pose `pose/.../pose_heatmap_head.py:PoseHeatmapHead`.
- **DEIMv2** — `DEIMv2/engine/backbone/dinov3/vision_transformer.py:DinoVisionTransformer`; `dinov3_adapter.py:DINOv3STAs` + `SpatialPriorModulev2` (DINOv3 → c2/c3/c4 multi-scale).
- **DINOv3 (MLX target)** — port our own from official PyTorch `references/dinov3` (`dinov3/dinov3/models/vision_transformer.py` + layers). No trusted MLX reference exists (low-star `mlx-image` rejected); this is a net-new port.
- **LocateAnything (MLX)** — `mlx-vlm/mlx_vlm/models/locateanything/{vision.py(MoonViT,Rope2DPosEmb,patch_merger),locateanything.py(projector,get_input_embeddings),language.py(Qwen2),pbd.py(PBDDecoder)}`.
- **LocateAnything (PyTorch ref)** — `LocateAnything-3B/{modeling_locateanything.py,modeling_vit.py(MoonVit*),modeling_qwen2.py,generate_utils.py(decode_bbox_avg,handle_pattern)}`.

## Part 4 — MLX status (drives effort)

- **Already in MLX** (have a trusted reference): LocateAnything → `mlx-vlm` (Blaizzy, ~5k★).
- **Net-new MLX ports** (mlx-cv's typed + parity-tested value-add): DINOv3 · RF-DETR · Depth Anything V3 · SAM 3.1 · EoMT · Sapiens2 · DEIMv2. (DINOv3's only MLX prior art was the low-star `mlx-image`, rejected as an oracle — we port from official PyTorch.)

**Weight-license notes (surfaced per §14, not gating):** Depth Anything V3 is **per-checkpoint** —
DA3-BASE Apache-2.0, DA3-LARGE/GIANT CC-BY-NC-4.0 (do *not* treat as uniformly permissive); DINOv3 Meta
commercial license; SAM 3.1 SAM license; RF-DETR Apache-2.0 (XL/2XL PML-1.0); LocateAnything-3B NVIDIA
non-commercial.
