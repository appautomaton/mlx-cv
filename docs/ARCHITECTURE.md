# mlx-cv — Architecture

> **Status: design / pre-implementation.** This document is the architectural blueprint for
> `mlx-cv`. It defines the contracts and module boundaries *before* any model code is written,
> so that every model we add later slots into a stable spine instead of reshaping it. Code does
> not exist yet; this is the thing we build against.

---

## 1. Purpose & positioning

`mlx-cv` is the **MLX-native computer-vision inference library for Apple Silicon**, built on
[MLX](https://github.com/ml-explore/mlx). It is an **inference-only pipeline**: load weights, run,
get typed results.

> **Mission.** A single, consistent, trustworthy way to run **current-generation (2025+) SOTA**
> vision models on Apple Silicon — detection, segmentation, depth, pose, tracking, and
> text-prompted grounding — turning raw model outputs into typed boxes / masks / keypoints / depth
> maps. Clean MIT code, parity-tested against the reference, extensible by design.

Scope decisions are made on their own merits — is it the best current model, is it portable, does it
fit the spine. We do not scope around what any other library is or isn't doing, and the code is
**weight-agnostic**: it can load weights of any license (see §14).

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
7. **Inference-only & weight-agnostic.** The code is MIT and load-and-run only; it can load weights
   of any license. A model's *weight* license is the end-user's compliance concern, never a
   constraint on what the code supports (see §14).
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
| **Grounding** (LocateAnything) | MoonViT + LLM | parallel-box decoder | `Text` | `detections` / `keypoints` | — |

Same spine, same `Result`, same `.draw()`. The last row is **grounding**: a text prompt → a VLM
(vision encoder + LLM + parallel-box decoder) → typed `Detections` / `Keypoints`. It is the anchor
vertical (LocateAnything, §15) and proves the spine accommodates **LLM-backed** models, not just
CNN/ViT heads — the `Module` may be a full VLM; the `Processor` owns tokenization and box parsing.

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
  pipelines/   compose (detect→track, detect→segment), video, trackers
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
- **Custom Metal kernels** — e.g. deformable attention (needed for the DEIMv2 / RT-DETR detector
  family) lives as a pluggable op in `ops/`, written once and shared.
- **Training / fine-tuning** — out of scope for now (`mlx-cv` is inference-only), but because
  modules stay pure `nn.Module`, the door stays open to a future `mlx-cv[train]` extra without
  reworking inference.
- **Per-family optional deps** — `pip install mlx-cv[depth]`, `[detection]`, `[segmentation]`,
  `[pose]` keep the base install light; heavy/optional deps are scoped to the family that needs them.
- **Stable result schema + versioning** — `Result` and `to_coco()` output are treated as a public
  contract with explicit versioning.

---

## 14. Licensing posture

**Code and weights are separate.** `mlx-cv`'s code is **MIT** and is an inference-only pipeline; it
loads weights, it does not relicense or redistribute them.

- License is **not** a gating criterion for what models the code supports. If a model is current,
  portable, and fits the spine, we can support it — regardless of its weight license.
- Weights are **fetched by the user** from the original source (e.g. HF Hub); conversion is a local
  step. Complying with a weight's license (commercial, non-commercial, attribution) is the
  **end-user's** responsibility.
- We **surface** each model's weight license in its model card so users can make an informed call —
  e.g. LocateAnything's weights are NVIDIA non-commercial; that's a note for users, not a reason to
  withhold the inference code.

---

## 15. Roadmap / phasing (deliberately un-rushed)

1. **Lock this architecture.** ← *this document.* No model code.
2. **Scaffold the spine (`v0.0.2`)**: `core` types, `geometry`, `registry`, `parity` contract,
   `transforms` / `ops` interfaces — abstract bases + tests, **zero** model implementations. This is
   a meaningful release that also validates the Node-24 CI.
3. **Anchor vertical — LocateAnything (grounding).** Drive it end-to-end through the spine: text
   prompt → VLM (MoonViT + LLM + parallel-box decoder) → typed `Detections` / `Keypoints` →
   `.draw()` / COCO. This was the original goal and is the hardest, highest-signal probe — if the
   spine carries an LLM-backed grounding model cleanly, it carries everything.
4. **Expand by reuse** — subsequent current-gen models (Depth Anything V3, EoMT-DINOv3, Sapiens2,
   DEIMv2 / RT-DETRv4) mostly reuse an existing backbone + head, with only conversion + processor
   new.

---

## Appendix A — current-generation (2025+) target set

Synthesized from a June 2026 survey. **Scope rule: current-generation SOTA only — 2025+.** Anything
older (OWLv2 '23, ViTPose '22, RTMPose '23, MM-Grounding-DINO '24, Depth Anything V2 '24, D-FINE
'24, SAMURAI '24, EVA-02 '23, and the late-2024 Apple models Depth Pro / AIMv2) is intentionally
dropped. Weight licenses are surfaced per §14, never used to gate inclusion.

### Anchor

| Capability → Model | When | Signal | Weight license | MLX status |
|---|---|---|---|---|
| **Grounding → LocateAnything-3B** | 2026.05 | strongest open-weight grounding / detection / pointing / GUI / OCR-localization; parallel-box decoding | NVIDIA non-commercial (weights only) | only an **unmerged mlx-vlm PR** + community 4/8-bit weights — no clean first-class path yet |

### Target set (current-gen, portable, fits the spine)

| Capability → Model | When | Signal | Weight license | Effort |
|---|---|---|---|---|
| Depth → **Depth Anything V3** (Apache variants) | 2025.11 | current depth SOTA | Apache (S/B/Metric-L/Mono-L) | Easy–Med |
| Segmentation/panoptic → **EoMT-DINOv3** | 2025 | 58.9 PQ / 59.5 mIoU, 4× faster than Mask2Former | MIT code / DINOv3 backbone | Easy |
| Human → **Sapiens2** | 2026.04 | SOTA human pose / normals / depth / part-seg | custom | Med |
| Detection → **DEIMv2** / **RT-DETRv4** | 2025.09 / .10 | 56–58 AP; DEIMv2 light down to 0.49M | Apache / CC-BY | Med (needs deformable-attn op) |
| Detection (real-time flagship) → **RF-DETR** | ICLR'26 | first real-time >60 mAP | Apache (N–L) | Med (partial port exists) |
| Tracking/video → **SAM 3.1 video / Object-Multiplex** | 2026.03 | text-promptable detect + segment + track in video | SAM license | Hard (nominal mlx-vlm port unvalidated) |

**Shared backbones (all 2025), port once → reuse:** DINOv3 (2025.08), SigLIP 2 (2025.02),
Perception Encoder (2025), C-RADIOv3 (2025).

### Not targets

- **Unportable — no weights released:** Grounding DINO 1.5/1.6, DINO-X, T-Rex-Omni (API-only).
- **Old generation (pre-2025) — dropped per scope rule:** OWLv2, MM-Grounding-DINO, ViTPose++,
  RTMPose, Depth Anything V2, D-FINE, SAMURAI, EVA-02, Depth Pro, AIMv2.
- **Copyleft** (a note for users, not an exclusion): YOLO-World (GPL), YOLOE / Ultralytics (AGPL).

### Key sources

- Anchor — LocateAnything: <https://huggingface.co/nvidia/LocateAnything-3B> · MLX weights: <https://huggingface.co/mlx-community/LocateAnything-3B-4bit>
- Depth: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Segmentation: <https://github.com/tue-mps/eomt> · SAM 3.1: <https://ai.meta.com/blog/segment-anything-model-3/> · <https://github.com/facebookresearch/sam3>
- Detection: <https://github.com/Intellindust-AI-Lab/DEIMv2> · <https://github.com/RT-DETRs/RT-DETRv4> · <https://github.com/roboflow/rf-detr>
- Human: <https://huggingface.co/facebook/sapiens2>
- Backbones: <https://github.com/facebookresearch/dinov3> · <https://huggingface.co/blog/siglip2> · <https://github.com/facebookresearch/perception_models> · <https://huggingface.co/nvidia/C-RADIOv3-H>
- MLX: <https://github.com/ml-explore/mlx>
