# mlx-cv — Architecture

> **Status: design / pre-implementation.** This document is the architectural blueprint for
> `mlx-cv`. It defines the contracts and module boundaries *before* any model code is written,
> so that every model we add later slots into a stable spine instead of reshaping it. Code does
> not exist yet; this is the thing we build against.

---

## 1. Purpose & positioning

`mlx-cv` is the **classical computer-vision perception layer for Apple Silicon**, built natively
on [MLX](https://github.com/ml-explore/mlx).

The MLX ecosystem today is split:

- **`mlx-vlm`** is healthy and actively maintained, and already owns *generative* vision-language
  — captioning, VQA, and VLM-based grounding (Qwen3-VL, Molmo, PaliGemma 2, LocateAnything).
- **Classical perception** — depth, pose, detection, segmentation, tracking — is a patchwork of
  one-person hobby ports with no unified API, inconsistent pre/post-processing, and no parity
  guarantees. Even Apple's own flagship CV models (Depth Pro, FastVLM) are not on MLX.

> **Mission.** `mlx-vlm` owns generative vision-language. **`mlx-cv` owns classical perception —
> depth, pose, detection, segmentation, tracking — behind one consistent API, plus the
> post-processing glue that turns model outputs (including `mlx-vlm` grounding text) into real,
> typed boxes / masks / keypoints / depth maps.**

We *interoperate* with `mlx-vlm` rather than compete with it (see the grounding glue in §7).

A condensed view of the 2026 model landscape that motivates this scope — including which models
are worth porting, which are already done, and which are license-blocked — is in
[Appendix A](#appendix-a--2026-cv-landscape-condensed).

---

## 2. Design principles

1. **One spine, many plug-ins.** A task-agnostic core; each task (depth, detection, …) is added
   as a plug-in, never by editing the core. (Open/Closed.)
2. **Compute is separate from orchestration.** `nn.Module`s are pure, trainable compute graphs
   with no I/O. Pre/post-processing and file/hub access live in separate objects.
3. **Coordinates are sacred.** Every spatial result can be mapped losslessly back to the original
   image. The spine enforces this; individual models cannot get it wrong silently.
4. **One output type for everything.** All tasks return the same `Result` container, so `.draw()`,
   `.to_coco()`, and serialization are uniform.
5. **Trust by parity.** Every model ships golden fixtures and is gated in CI against the reference
   implementation. This is the differentiator over hobby ports.
6. **Reuse backbones.** Encoders (ViT, DINOv3, AIMv2, SigLIP, Hiera, CSPNeXt) are ported once and
   shared across heads/tasks.
7. **License-aware by default.** Prefer Apache/MIT, flag custom/non-commercial licenses explicitly
   (see §14).
8. **Forward-compatible.** The type system and module boundaries reserve space for batching, video,
   quantization, custom kernels, and training without redesign (§13).

---

## 3. The unifying insight

Strip away the task labels and depth, detection, segmentation/tracking, and pose are the **same
pipeline**:

> **image(s) → preprocess (with an invertible spatial transform) → shared backbone features →
> task head → postprocess (map back to original coords) → typed `Result` (draw / serialize).**

The **only** things that differ per task are:

| Axis | Varies by task? | Mechanism |
|---|---|---|
| **Head** | yes | pluggable decoder (`heads/`) |
| **Result field** | yes | one of the optional fields on `Result` |
| **Prompt** | sometimes | opt-in `Prompt` mixin |
| **Temporal state** | sometimes | opt-in `Tracker` mixin |
| Preprocess / backbone / postprocess discipline / weights / hub / viz | **no** | the stable spine |

So the architecture is a **task-agnostic spine** plus the four pluggable contracts
`{Head, Result, Prompt, State}`.

---

## 4. Data flow

```
        ┌──────────────────────── mlx-cv spine (task-agnostic, stable) ────────────────────────┐
inputs ─►Image/Batch ─► Transforms ─► Backbone ─►[ Head ]─► Postprocess ─► Result ─► viz/COCO/JSON
(path,     (+orig       (→tensor +    (shared    (PLUGGABLE  (uses ctx →    (typed,
 PIL, np,   size)        SpatialCtx)   encoder    per task)   orig coords)   unified)
 mlx,                                  registry)
 video)
        └──────────────────────────────────────────────────────────────────────────────────────┘
   weights ▲ hub.from_pretrained → WeightConverter → quantize → load
   prompts ▲ Text / Points / Boxes / Exemplars   (promptable models only)
   state   ▲ Tracker memory (video / VOS)    trust ▲ parity-fixture tests (every model)
```

Pluggable = `{Head, Result, Prompt, State}`. Stable spine = everything else.

---

## 5. Core contracts

Five abstractions are the whole game. Signatures below are **illustrative**, not final.

### 5.1 `Result` — the output lingua franca

One container with optional, composable fields, so a panoptic+depth model and a plain detector
share the same surface:

```python
class Result:
    detections: Detections | None   # boxes (xyxy), scores, labels, optional track_ids
    masks:      Masks | None         # instance / semantic / panoptic
    keypoints:  Keypoints | None     # skeleton + per-point confidence
    depth:      DepthMap | None      # metric|relative flag, units, focal length
    embedding:  Embedding | None     # feature vectors / feature maps
    tracks:     Tracks | None        # temporal identities (video)
    image_size: tuple[int, int]      # original (H, W); every field references this frame

    def draw(self, image=None, **opts) -> Image: ...
    def to_coco(self) -> dict: ...
    def save(self, path) -> None: ...
```

Why one container with optional fields (not a class hierarchy): real models emit multiple
modalities at once (Sapiens → pose + normals + depth; panoptic → masks + labels). A flat,
composable container keeps `.draw()` / `.to_coco()` uniform and avoids combinatorial subclasses.

Field types (`Detections`, `Masks`, `Keypoints`, `DepthMap`, …) live in `core/types.py`, each with
numpy / `supervision` / COCO interop.

### 5.2 `SpatialTransform` — invertible coordinate context

The single most common bug in ad-hoc ports is mapping outputs back through resize / letterbox /
pad. We make that structural:

```python
tensor, ctx = transform(image)          # ctx records scale, pad, crop, orig_size
...                                       # model runs in model-input space
result = head.postprocess(raw, ctx)     # ctx.invert(coords) → original-image space
```

Preprocess **always** returns `(tensor, ctx)`. Postprocess **must** consume `ctx`. The spine wires
this; a model author does not hand-roll coordinate math.

### 5.3 `Backbone` registry — port once, reuse everywhere

Depth, detection, segmentation, and pose all sit on a small set of encoders. A backbone is just
`input → multi-scale features`, registered by name:

```python
@register_backbone("dinov3-l")
class DINOv3(nn.Module):
    def __call__(self, x) -> list[FeatureMap]: ...
```

Porting DINOv3 once unlocks EoMT (segmentation) *and* DEIMv2 (detection) *and* depth heads.
Initial registry targets: `vit`, `dinov3`, `aimv2`, `siglip2`, `hiera`, `cspnext`, `hgnetv2`,
`convnext`.

### 5.4 `Predictor` / `Processor` / `Module` — compute vs orchestration

Three roles, deliberately separated:

- **`Module`** (`nn.Module`): pure compute graph, no I/O. Stays trainable and composable.
- **`Processor`**: owns pre- and post-processing (and prompt encoding). Testable in isolation.
- **`Predictor`**: wires `Processor → Module → Processor` into `predict()`; the user-facing object.

```python
class Predictor:
    task: Task                       # DEPTH | DETECTION | SEGMENTATION | POSE | ...
    def predict(self, inputs, *, prompt=None, **opts) -> Result: ...
```

This boundary is what lets us add training (`mlx-cv[train]`) later without touching inference, and
lets researchers use raw `Module`s directly (API Tier 3, §9).

### 5.5 `Prompt` & `Tracker` — the two optional axes

Both are **opt-in mixins**, so a plain depth model never pays for machinery it doesn't use.

```python
# Promptable models accept a Prompt; others ignore it.
Prompt = TextPrompt | PointPrompt | BoxPrompt | ExemplarPrompt

# Temporal models implement the stateful protocol.
class Tracker:
    def init(self, frame, prompt) -> Result: ...
    def step(self, frame) -> Result: ...      # carries memory across frames
```

---

## 6. Proof it generalizes — every pillar on one spine

| Pillar | Backbone (reused) | Head | Prompt | `Result` field | State |
|---|---|---|---|---|---|
| **Depth** | ViT / DINOv2 | DPT decoder | — | `depth` | — |
| **Detection** | HGNetv2 / ViT | DETR / YOLO | — | `detections` | — |
| **Open-vocab detection** | ViT + text enc | OWL head | `Text` | `detections` | — |
| **Segmentation** | DINOv3 / Hiera | EoMT / mask decoder | (pts / box / text) | `masks` | — |
| **Tracking / VOS** | Hiera (SAM2) | mask decoder + memory | `Box` / `Point` | `masks` + `tracks` | ✅ memory |
| **Pose** | ViT / CSPNeXt | keypoint head | — | `keypoints` | — |
| **Grounding glue** | *(via mlx-vlm)* | text→struct parser | `Text` | `detections` / `keypoints` | — |

Same spine, same `Result`, same `.draw()`. The last row is the ecosystem play: `mlx-cv` parses
`mlx-vlm` grounding output into our typed results rather than reimplementing the VLM.

---

## 7. Package layout

```
src/mlx_cv/
  core/        types.py (Result, Detections, Masks, Keypoints, DepthMap, Tracks, Embedding)
               geometry.py (SpatialTransform)   image.py   registry.py
  transforms/  resize, letterbox, normalize, patchify   → (tensor, ctx)
  ops/         nms, box_decode, mask_ops, coord_map      (pure mlx; custom kernels live here)
  backbones/   vit/ dinov3/ aimv2/ siglip2/ hiera/ cspnext/ hgnetv2/ convnext/   ← registered, shared
  heads/       dpt/ detr/ owl/ eomt/ keypoint/                                    ← reusable decoders
  models/      <family>/  config.py  modeling.py  processor.py  convert.py        ← one folder per model
  prompts/     text, points, boxes, exemplars
  pipelines/   compose (detect→track, detect→segment), video, trackers (bytetrack, samurai)
  hub/         from_pretrained, download/cache, quantize (4/8-bit), dtype policy
  parity/      golden-fixture contract + bisect harness   ← trust, first-class
  viz/         annotators
```

Adding a model touches **one folder** (`models/<family>/`, plus maybe a new `heads/` or
`backbones/` entry) and one registry line — never the spine.

---

## 8. Model lifecycle

**Load** (`hub.from_pretrained`): resolve name → config + processor + module → download/cache
weights from HF Hub → `WeightConverter` remaps the reference `state_dict` to the MLX param tree →
optional quantization (4/8-bit) and dtype policy → load.

**Predict**: `Predictor.predict()` runs `Processor.preprocess → Module.__call__ →
Processor.postprocess`, returning a `Result` in original-image coordinates.

**Convert** (`models/<family>/convert.py`): declarative key-remap rules from the PyTorch /
safetensors reference to MLX, kept next to the model it serves. Handles conv-layout differences,
fused/unfused params, naming, etc.

---

## 9. Three-tier API

```python
# Tier 1 — high level (mlx-lm shaped): the 90% path
m = mlx_cv.load("owlv2-base")
r = m.predict("photo.jpg", prompt="a cat")     # → Result
r.draw().save("out.png")

# Tier 2 — compose: swap a backbone, reuse a head
m = Detector(backbone="dinov3-l", head=OwlHead(...), processor=...)

# Tier 3 — raw mlx.nn modules for research / training
feats  = DINOv3()(x)
logits = OwlHead()(feats)
```

Tier 1 serves users; Tier 2 serves integrators; Tier 3 serves researchers and the future training
path. All three return / operate on the same core types.

---

## 10. Extensibility model

- **Registries** map names → builders for models, backbones, and heads
  (`@register_model("owlv2-base")`).
- **Third-party plugins** register via Python entry points (group `mlx_cv.models`) — extend
  `mlx-cv` without forking it.
- **Adding a model — checklist:**
  1. `models/<family>/config.py` — config dataclass.
  2. `models/<family>/modeling.py` — `nn.Module` (reusing a `backbones/` encoder + a `heads/` decoder where possible).
  3. `models/<family>/processor.py` — pre/post using `transforms/` + `ops/` + a `SpatialTransform`.
  4. `models/<family>/convert.py` — weight remap rules.
  5. Register the name; add golden fixtures under `parity/`.

If a step requires editing the spine, that's a signal the spine is missing an abstraction — fix the
spine, not the model.

---

## 11. Trust & parity strategy

Parity is a **first-class architectural element**, not an afterthought:

- Each model ships **golden fixtures**: reference outputs *and* selected intermediate activations on
  fixed inputs.
- The `parity/` harness asserts MLX output matches the reference within tolerance, and can **bisect**
  via intermediate taps to localize a divergence to a specific module.
- CI **gates releases** on parity. A model that drifts from its reference does not ship.

This is precisely what scattered hobby ports lack and is the basis for `mlx-cv` being trustworthy.

---

## 12. Testing & CI

- **Unit**: spine contracts (coordinate round-trips through `SpatialTransform`, NMS, box/mask decode,
  `Result` serialization).
- **Parity**: per-model fixture tests (§11).
- **Smoke**: `load → predict → draw` for each registered model on a tiny input.
- The existing publish workflow (`.github/workflows/workflow.yml`, Node-24 actions, OIDC trusted
  publishing) is extended with a separate **test** job that runs unit + parity on PR/push, keeping
  `id-token: write` scoped to the publish job only.

---

## 13. Forward-looking reservations (designed for, not built yet)

- **Batching & video streaming** — `ImageBatch` and a `VideoSource` abstraction exist in the type
  system from day one; trackers consume frames from it.
- **Quantization & dtype policy** — a `hub` concern applied uniformly across models (4/8-bit,
  bf16/fp16/fp32), not per-model ad hoc.
- **Custom Metal kernels** — e.g. deformable attention (needed for the D-FINE / DETR detector
  family) lives as a pluggable op in `ops/`, written once and shared.
- **Training / fine-tuning** — because modules stay pure `nn.Module`, a future `mlx-cv[train]` extra
  adds losses/optimizers/datasets without touching inference.
- **Per-family optional deps** — `pip install mlx-cv[depth]`, `[detection]`, `[segmentation]`,
  `[pose]` keep the base install light; heavy/optional deps are scoped to the family that needs them.
- **Stable result schema + versioning** — `Result` and `to_coco()` output are treated as a public
  contract with explicit versioning.

---

## 14. Licensing posture

`mlx-cv` itself is **MIT**. Per the 2026 survey (Appendix A), license is a first-class selection
criterion for what we port and how we ship it:

- **Prefer** Apache-2.0 / MIT models — default, shippable, redistributable.
- **Flag custom-but-commercial** licenses (SAM, DINOv3, Depth Pro, Sapiens2) explicitly in each
  model's card; weights are user-fetched from the original source, with attribution and any
  propagation terms surfaced.
- **Mark non-commercial** models (LocateAnything, Depth Anything *large* variants, Metric3D) and
  **copyleft** ones (YOLO-World GPL, YOLOE/Ultralytics AGPL) clearly; do not present them as
  drop-in commercial defaults.
- **Never** ship redistributed weights in a way that violates the upstream license; conversion is a
  local/user step.

---

## 15. Roadmap / phasing (deliberately un-rushed)

1. **Lock this architecture.** ← *this document.* No model code.
2. **Scaffold the spine (`v0.0.2`)**: `core` types, `geometry`, `registry`, `parity` contract,
   `transforms` / `ops` interfaces — abstract bases + tests, **zero** model implementations. This is
   a meaningful release that also validates the Node-24 CI.
3. **Drive one vertical end-to-end** through the spine to pressure-test the contracts. Cleanest first
   probes (permissive license + tractable architecture + real gap):
   - **Depth** — Depth Anything V2-Small (Apache) and/or Depth Pro (Apple); single-output, cleanest.
   - **Open-vocab detection** — OWLv2 (Apache, simplest OVD architecture).
4. **Expand by reuse** — each subsequent model should mostly reuse an existing backbone + head, with
   only conversion + processor new.

---

## Appendix A — 2026 CV landscape (condensed)

Synthesized from a June 2026 survey across detection, open-vocab grounding, segmentation/tracking,
backbones, depth/pose, and the existing MLX ecosystem. Used to choose `mlx-cv`'s scope and targets.

### Best MLX port opportunities (SOTA × license × gap × effort)

| Capability → Model | Signal | License | On MLX? | Effort |
|---|---|---|---|---|
| Depth → **Depth Pro** (Apple) | sharp zero-shot *metric* depth, 2.25MP ~0.3s | Apple custom (commercial OK, review) | ❌ | Med |
| Depth → **Depth Anything V2-S / V3 (Apache variants)** | current depth SOTA | Apache-2.0 | ❌ | Easy |
| Open-vocab det → **OWLv2** | 44.6 LVIS-rare zero-shot | Apache-2.0 | ❌ | Easy–Med |
| Panoptic/semantic → **EoMT** (DINOv2) | 58.9 PQ / 59.5 mIoU, 4× faster than Mask2Former | MIT code | ❌ | Easy |
| Pose → **ViTPose++ / RTMPose** | COCO-SOTA / 75.8 AP @430fps | Apache-2.0 | ❌ | Easy–Med |
| Tracking → **SAMURAI** on SAM2-MLX | +7.1% LaSOT, training-free | Apache-2.0 | ❌ (base exists) | Easy |
| Grounding/referring → **MM-Grounding-DINO** | 50.6 COCO / 41.4 LVIS zero-shot, open weights+training | Apache-2.0 | ❌ | Med |
| Detection → **D-FINE / DEIMv2-light** | 55.8–59.3 AP / tiny 0.49M | Apache-2.0 | ❌ | Med |

**Shared backbones to port once and reuse:** AIMv2 (Apple ships an official MLX backend), DINOv3
(community MLX impl exists), SigLIP 2 (Apache).

### Skip / already-covered

- **Unportable — no weights released:** Grounding DINO 1.5/1.6, DINO-X, T-Rex2 (API-only).
- **Already well-served on MLX:** YOLO26 & RF-DETR (native ports), SAM 2.1 (Apache port), SAM 3
  *image* (`mlx-community/sam3-image`), VLM grounding (`mlx-vlm`).
- **License-blocked for commercial defaults:** LocateAnything (NVIDIA non-commercial; already
  ported), Depth Anything *large* variants & Metric3D (NC), YOLO-World/YOLOE (GPL/AGPL).

### Key sources

- MLX ecosystem: <https://github.com/ml-explore/mlx-examples> · <https://github.com/Blaizzy/mlx-vlm> · <https://github.com/riccardomusmeci/mlx-image> · <https://huggingface.co/mlx-community>
- SAM 3 / 3.1: <https://ai.meta.com/blog/segment-anything-model-3/> · <https://github.com/facebookresearch/sam3> · MLX: <https://huggingface.co/mlx-community/sam3-image> · <https://github.com/avbiswas/sam2-mlx>
- Detection: <https://github.com/Peterande/D-FINE> · <https://github.com/Intellindust-AI-Lab/DEIMv2> · <https://github.com/roboflow/rf-detr> · <https://docs.ultralytics.com/models/yolo26>
- Open-vocab / grounding: <https://huggingface.co/docs/transformers/model_doc/owlv2> · <https://huggingface.co/docs/transformers/model_doc/mm-grounding-dino> · <https://nvidia.github.io/> LocateAnything <https://huggingface.co/nvidia/LocateAnything-3B>
- Segmentation: <https://github.com/tue-mps/eomt> · <https://yangchris11.github.io/samurai/> · <https://github.com/facebookresearch/EdgeTAM>
- Backbones / depth / pose: <https://github.com/apple/ml-aim> · <https://github.com/facebookresearch/dinov3> · <https://huggingface.co/blog/siglip2> · <https://github.com/apple/ml-depth-pro> · <https://github.com/ByteDance-Seed/Depth-Anything-3> · <https://github.com/open-mmlab/mmpose>
