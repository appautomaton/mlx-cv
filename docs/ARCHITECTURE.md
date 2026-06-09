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
    keypoints:  Keypoints | None     # named skeleton + per-point confidence (pose)
    points:     Points | None        # sparse 2D localization points — pointing / GUI (no skeleton)
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

Field types (`Detections`, `Masks`, `Keypoints`, `Points`, `DepthMap`, …) live in `core/types.py`, each with
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

**Two backbone kinds.** Vision backbones satisfy the signature above (`input → multi-scale
features`). LLM-backed models (the grounding anchor) also need a reusable **language/decoder
backbone** with a different contract (`tokens / embeds → hidden states`, plus a decode loop) — so the
registry hosts both kinds, under `backbones/vision/` and `backbones/llm/`, with port-once-reuse
applying to each (§16 ports MoonViT *and* Qwen2.5 this way).

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
| **Grounding** (LocateAnything) | MoonViT + Qwen2.5 LLM | parallel-box decoder (token-space) | `Text` | `detections` / `points` | — |

Same spine, same `Result`, same `.draw()`. The last row is **grounding**: a text prompt → a VLM
(vision encoder + LLM + parallel-box decoder) → typed `Detections` / `Points` (this model emits no
per-box score, so `Detections.scores` is empty). It is the anchor vertical (LocateAnything, §15) and
proves the spine accommodates **LLM-backed** models, not just CNN/ViT heads — the `Module` may be a
full VLM; the `Processor` owns tokenization and box parsing. **Full part-by-part mapping in §16.**

---

## 7. Package layout

```
src/mlx_cv/
  core/        types.py (Result, Detections, Masks, Keypoints, Points, DepthMap, Tracks, Embedding)
               geometry.py (SpatialTransform)   image.py   registry.py
  transforms/  resize, letterbox, normalize, patchify   → (tensor, ctx)
  ops/         nms, box_decode, mask_ops, coord_map      (pure mlx; custom kernels live here)
  backbones/   vision/ vit/ dinov3/ aimv2/ siglip2/ hiera/ cspnext/ hgnetv2/ convnext/ moonvit/  ← vision encoders → features
               llm/    qwen2/ …                                                                  ← LLM decoders (VLM-backed)
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
- CI runs unit + parity tests via a separate workflow (`.github/workflows/test.yml`, Node-24
  actions) on push / PR; it has **no** `id-token` permission, so OIDC trusted publishing stays
  isolated to the release-only `workflow.yml` (where `id-token: write` lives alone).

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

> ⚠️ **Superseded by `.agent/steering/ROADMAP.md`** (verified June 2026). The concrete, evidence-backed
> sequence is now foundation-first: spine-contract hardening → build-once blocks → anchor
> (LocateAnything) → models. The phasing *intent* below still holds; the live plan is the roadmap.

1. **Lock this architecture.** ← *this document.* No model code.
2. **Scaffold the spine (`v0.0.2`)**: `core` types, `geometry`, `registry`, `parity` contract,
   `transforms` / `ops` interfaces — abstract bases + tests, **zero** model implementations. This is
   a meaningful release that also validates the Node-24 CI.
3. **Anchor vertical — LocateAnything (grounding).** Drive it end-to-end through the spine: text
   prompt → VLM (MoonViT + Qwen2.5 + parallel-box decoder) → typed `Detections` / `Points` →
   `.draw()` / COCO. This was the original goal and is the hardest, highest-signal probe — if the
   spine carries an LLM-backed grounding model cleanly, it carries everything. The full part-by-part
   mapping, weight-conversion, and parity plan is now written up in §16.
4. **Expand by reuse** — subsequent current-gen models (Depth Anything V3, EoMT-DINOv3, Sapiens2,
   DEIMv2 / RT-DETRv4) mostly reuse an existing backbone + head, with only conversion + processor
   new.

---

## 16. Anchor case study — LocateAnything-3B → the spine

> Porting the anchor *on paper first* pressure-tests §1–§15 against a real, LLM-backed model before
> any code. Facts below are verified against the model card, the paper, and the merged MLX reference
> (sources at the end). The few **spine refinements** this forced are flagged ⟢ and already folded
> into the sections above.

### 16.1 Model at a glance (verified)

- **Composition:** MoonViT-SO-400M encoder (1152-d, 27 layers, patch-14, 2×2 token merge) → **MLP
  projector** (LayerNorm 4608 → Linear → GELU → Linear → 2048) → **Qwen2.5-3B-Instruct** decoder
  (2048-d, 36 layers, GQA 16/2, vocab 152 681). ~7.66 GB bf16, two safetensors shards.
- **Output is a token stream, not tensors.** A box is the vocab sequence
  `<ref>label</ref><box><x1><y1><x2><y2></box>`; a point is `<box><x><y></box>`. Each coordinate is
  an integer in **[0, 1000]** carried by a dedicated token (`value = token_id − 151677`).
- **Parallel Box Decoding (PBD):** generation in fixed **6-token blocks** (Semantic / Box / Negative
  / End), three modes from one checkpoint — **Fast** (parallel multi-token), **Slow** (plain
  autoregressive), **Hybrid** (default; Fast with per-block fallback to Slow). All agree on clean
  inputs.
- **Emits:** boxes + points + labels. **No masks, no keypoints, no per-box scores.**
- **Prompts:** text only. No visual-exemplar / point / box *inputs*.
- **Weights:** NVIDIA non-commercial (Qwen2.5 = Qwen Research License, MoonViT = MIT) — a §14
  user-facing note, not a gate.

### 16.2 The parts → the spine

| Model part | mlx-cv home | Contract | Reused by |
|---|---|---|---|
| MoonViT-SO-400M | `backbones/vision/moonvit/` ⟢ | vision backbone (`image → features`); native-res, 2D-RoPE, per-image block attn | Kimi-VL-family / MoonViT models |
| Qwen2.5-3B-Instruct | `backbones/llm/qwen2/` ⟢ | **language backbone** (`embeds → hidden states` + decode) — a *new backbone kind* | every Qwen2.5-backed VLM |
| MLP projector | `models/locateanything/modeling.py` | model-local connector (vision-dim → LLM-dim) | pattern generalizes on the 2nd VLM |
| PBD generation | decode strategy beside the LLM backbone | reusable parallel / autoregressive decode loop | future generative-grounding models |
| box/point token → coord | `ops/box_decode` + `Processor.postprocess` | pure token→geometry, then `ctx.invert` | any token-space localizer |

⟢ **Spine refinement:** `backbones/` now hosts **two kinds** — vision encoders (`→ FeatureMap[]`)
and LLM decoders (`→ hidden states` + decode loop). There is **no `heads/` entry** for grounding:
the "head" is the LLM + PBD, so the decoder lives in the backbone + a decode strategy and the
*parsing* is postprocess — the spine carrying an LLM-backed model exactly as §6 promised.

### 16.3 End-to-end flow (through the stable spine)

```
"Locate all the cats."  +  image
   │ Processor.preprocess
   ▼
 MoonViT dynamic resize (bicubic → patch/merge multiple; mean/std 0.5) → (pixels, ctx)
 chat template; expand <image-N> → <img> + <IMG_CONTEXT>×N + </img>   (N = gh·gw / 4)
   │ Module.__call__  (the full VLM)
   ▼
 MoonViT → projector → scatter image features into Qwen2.5 embeds at <IMG_CONTEXT>
   │ → PBD generate (Fast | Slow | Hybrid) → token stream
   ▼ Processor.postprocess
 parse <ref>…</ref><box>…</box> → coords(token_id−151677) → ctx.invert([0,1000] → pixels)
   ▼
 Result(detections=…, points=…, image_size=orig)  →  .draw() / .to_coco()
```

- **`SpatialTransform` is the linchpin.** The model speaks a normalized **[0,1000]** frame over a
  dynamically-resized, patch-padded image; `ctx` records that resize/pad so `invert` lands boxes on
  the *original* pixels — the "coordinates are sacred" contract (§5.2) doing the exact job ad-hoc
  ports fumble.
- **Prompt:** only `TextPrompt` is exercised; other `Prompt` variants stay dormant (opt-in, §5.5).
- **Scores:** `Detections.scores` is `None` here — optional fields already allow it; nothing bends.

### 16.4 Weight conversion (`models/locateanything/convert.py`)

Source = HF reference keys (`nvidia/LocateAnything-3B`); target = our backbone tree. Declarative
rules (verified against the merged MLX `sanitize()`):

| Reference key | mlx-cv target | Note |
|---|---|---|
| `language_model.lm_head.weight` | *(drop)* | tied to `embed_tokens` |
| `vision_model.encoder.*` | MoonViT backbone | strip `encoder.` |
| `vision_model.*` | MoonViT backbone | patch-embed conv, pos-emb, final norm |
| `language_model.model.*` | Qwen2 backbone | shape unchanged |
| `mlp1.0 / .1 / .3` | projector `layer_norm / linear_1 / linear_2` | index 2 = GELU (no params) |

**No transposes, no QKV split in the rules** — MoonViT's fused `wqkv` and conv `patch_embed.proj`
are mirrored inside the module definitions (HF layout), not in `convert` (the §8 discipline).
Conversion is proven by parity (§16.6), not by eyeball.

### 16.5 Quantization (validates the per-module policy, §13)

`hub.quantize` **must** support per-module bit overrides — not hypothetical: pure 4-bit *breaks*
this model because the tied coordinate-token embedding degrades. The community recipe (a usable
reference) keeps **embeddings at 8-bit** and selected `v_proj` / `down_proj` at 8-bit, the rest at
4-bit (group 64, affine). Our dtype/quant policy carries a per-path map and can ingest that recipe.

### 16.6 Parity & trust (§11 applied)

- **Reference truth:** PyTorch `nvidia/LocateAnything-3B` (transformers). Mint golden fixtures from a
  fixed image+prompt: reference boxes/points/labels **plus** intermediate taps — MoonViT patch-embed
  out, MoonViT final hidden, projector out, LLM layer-0 hidden, logits at box-token positions.
- **Bisect:** if final boxes drift, the first diverging tap localizes the fault to
  vision / projector / LLM / decode.
- **Fast oracle:** the merged mlx-vlm port (same framework) should be ~bit-identical — a cheap
  pre-gate before the heavier PyTorch comparison. We *read from* it; we don't *depend on* it.
- **Decode invariants:** deterministic / greedy decode; assert **Fast ≡ Slow ≡ Hybrid** on clean
  inputs; boxes within a few px of reference after `invert`.

### 16.7 Risks (carry into implementation)

- **MoonViT is the hard part** — native / variable resolution, 2D-RoPE, per-image block attention,
  2×2 merge: the highest-risk module to match numerically.
- The non-causal block-attention mask PBD uses during parallel decode must be reproduced exactly.
- Confirm the [0,1000] frame is relative to the *resized* image vs. the padded grid, so `ctx.invert`
  accounts for pad (resolve when fixtures exist).
- Max detections is bounded by token budget (≤ 8192 new tokens), not a fixed slot count — fine for
  `Result`, worth a documented limit.

**Sources:** model <https://huggingface.co/nvidia/LocateAnything-3B> · paper
<https://arxiv.org/abs/2605.27365> · merged MLX reference
<https://github.com/Blaizzy/mlx-vlm/pull/1242> · MLX weights
<https://huggingface.co/mlx-community/LocateAnything-3B-4bit>.

---

## Appendix A — current-generation (2025+) target set

> ⚠️ **Superseded (verified June 2026).** The candidate set below was verified, narrowed, and
> reconciled against 10 cloned reference implementations — see **`docs/BUILDING-BLOCKS.md`** (foundation
> + per-model evidence) and **`.agent/steering/ROADMAP.md`** (build sequence). Corrections from
> verification: RF-DETR is built on **DINOv2** (not DINOv3); **RT-DETRv4 is dropped** (redundant + least
> popular); **SAM 3.1 Object Multiplex (2026.03)** is the confirmed tracking pick; **YOLO26** is a
> popular watchlist item (AGPL). MVP = LocateAnything-3B · DINOv3 · RF-DETR · Depth Anything V3 · SAM
> 3.1. (The `arxiv.org/abs/2605.27365` id cited below is an unverified placeholder; the model itself is
> verified via the HF card + cloned `references/`.)
>
> Per-checkpoint license correction: Depth Anything V3 weights are **not** uniformly Apache — DA3-BASE
> is Apache-2.0; DA3-LARGE/GIANT are CC-BY-NC-4.0.

Synthesized from a June 2026 survey. **Scope rule: current-generation SOTA only — 2025+.** Anything
older (OWLv2 '23, ViTPose '22, RTMPose '23, MM-Grounding-DINO '24, Depth Anything V2 '24, D-FINE
'24, SAMURAI '24, EVA-02 '23, and the late-2024 Apple models Depth Pro / AIMv2) is intentionally
dropped. Weight licenses are surfaced per §14, never used to gate inclusion.

### Anchor

| Capability → Model | When | Signal | Weight license | MLX status |
|---|---|---|---|---|
| **Grounding → LocateAnything-3B** | 2026.05 | strongest open-weight grounding / detection / pointing / GUI / OCR-localization; parallel-box decoding | NVIDIA non-commercial (weights only) | MLX reference exists (**merged mlx-vlm PR #1242**, 2026-06-03) + community bf16/4/8-bit weights; **no typed-CV path yet** — mlx-cv provides the first-class, parity-tested one (§16) |

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

- Anchor — LocateAnything: model <https://huggingface.co/nvidia/LocateAnything-3B> · paper <https://arxiv.org/abs/2605.27365> · MLX reference (merged) <https://github.com/Blaizzy/mlx-vlm/pull/1242> · MLX weights <https://huggingface.co/mlx-community/LocateAnything-3B-4bit>
- Depth: <https://github.com/ByteDance-Seed/Depth-Anything-3>
- Segmentation: <https://github.com/tue-mps/eomt> · SAM 3.1: <https://ai.meta.com/blog/segment-anything-model-3/> · <https://github.com/facebookresearch/sam3>
- Detection: <https://github.com/Intellindust-AI-Lab/DEIMv2> · <https://github.com/RT-DETRs/RT-DETRv4> · <https://github.com/roboflow/rf-detr>
- Human: <https://huggingface.co/facebook/sapiens2>
- Backbones: <https://github.com/facebookresearch/dinov3> · <https://huggingface.co/blog/siglip2> · <https://github.com/facebookresearch/perception_models> · <https://huggingface.co/nvidia/C-RADIOv3-H>
- MLX: <https://github.com/ml-explore/mlx>
