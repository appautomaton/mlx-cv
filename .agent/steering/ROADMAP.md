# Roadmap

Foundation-first, but **contract-proof, not paper-first**: Phase 1 proves the core spine contracts on
one real model before fleshing them out. Verified June 2026 against 10 reference impls (`references/`)
+ two Codex passes (decompose + skeptical review); full evidence in `docs/BUILDING-BLOCKS.md`. Phases 1–5
are done for the committed local fixture gates. Remaining hardening items are explicitly deferred:
LocateAnything upstream full-checkpoint parity, RF-DETR full upstream checkpoint parity, SAM 3.1 upstream
image reference parity, and SAM 3.1 video/tracker memory.

## Phase 1: Contract-proof slice + parity harness

- status: done
- change: `2026-06-09-spine-contract-proof`
- objective: Define the *minimal* versions of the 🔴 spine contracts AND prove them on one real model — not abstractly.
- why now: Hardening contracts purely on paper risks getting the shapes wrong; a thin slice validated against a real backbone de-risks everything downstream (the chicken-and-egg risk).
- likely outputs: a concrete `BackboneFeatures`/`FeatureMap` schema (token/map layout, strides, grid shape, cls/register offsets, valid mask, view axis, dtype); `SpatialTransform` v2 (deterministic dense mask/depth/heatmap resampling); `HeadInput`/`HeadOutput` shapes; a golden-fixture schema + one tiny fixture; **a DINOv3 forward pass on MLX matching a committed golden fixture (minted from the official PyTorch DINOv3) within tolerance** as the proof.
- evidence: `docs/BUILDING-BLOCKS.md` Parts 1(#1) & 2; `references/dinov3` (official PyTorch DINOv3); `src/mlx_cv/core/{base,geometry,registry}.py`
- exit signal: DINOv3 forward-parity passes through the fixture harness with no `core/` edit needed for it.

## Phase 2: Build-once core blocks (parameterized families)

- status: done
- change: `2026-06-14-build-once-core-blocks`
- scope: **lean** — extract only the blocks the Phase-1 DINOv3 port already proves; build only variants with a consumer now; wire (don't implement) the rest.
- objective: Extract DINOv3's inline blocks into reusable parameterized families on the now-verified contracts, re-express DINOv3 on them with **zero forward-parity regression**, and prove generalization by instantiating a second real ViT config (**DINOv2**) with **no new block code**.
- why now: They cover the most models; everything downstream composes from them. Letting the imminent second consumer (DINOv2, reused by DA3/RF-DETR) drive the extraction keeps it contract-proof, not paper-first.
- likely outputs: ViT backbone family; parameterized Transformer block (selectable norm/FFN/LayerScale; LayerNorm + GELU-MLP + LayerScale implemented); 2D positional-encoding suite (**2D-RoPE + learned-abs-interp** — the two real variants); attention ops family (**SDPA + packed-qkv only**); reusable weight-convert / `sanitize` engine (DINOv3 `convert.py` is the seed); DINOv3 re-expressed on the families; **DINOv2 as a structural second config** (instantiate + forward shapes; full convert/parity deferred).
- deferred to consuming phase: GQA/MQA + KV-cache (Phase 4 Qwen2); window/global attention + block masks (later backbones); multi-scale neck/projector (Phase 3 DA3 / Phase 5 RF-DETR); SwiGLU/RMSNorm bodies (Phase 4); full DINOv2 convert + parity (Phase 3).
- home correction: the mlx block families live in an **mlx-allowed** dir (e.g. a new `backbones/layers/`), **not** `core/layers` — `core/` stays mlx-free (Phase-1 invariant). `docs/BUILDING-BLOCKS.md` #2's `core/layers` label is superseded.
- evidence: `docs/BUILDING-BLOCKS.md` Part 1 (#1–5, #13); `.agent/work/2026-06-14-build-once-core-blocks/SPEC.md`; Phase-1 `src/mlx_cv/backbones/vision/dinov3/`; DINOv2 ref `references/rf-detr/.../backbone/dinov2*`.
- exit signal: DINOv3 forward-parity still passes **unchanged**; DINOv2 instantiates + forwards via the shared families with no new block code; `core/` mlx-free; full `pytest` green.

## Phase 3: Depth Anything V3 (first full task model)

- status: done
- change: `2026-06-15-depth-anything-v3-monocular`
- objective: First end-to-end task model — DINOv2 backbone + DPT dense head → `Result.depth` + `depth_conf` (monocular).
- why now: Lowest-difficulty full model (Codex ranking); proves the vision spine + dense head + the load/convert path *before* the heavy VLM.
- likely outputs: DPT dense-head family; DA3 model + convert + processor; **`Result` gains `depth_conf`** (monocular depth + confidence); depth parity fixture. Surface DA3's **per-checkpoint** weight license (BASE = Apache-2.0; LARGE/GIANT = CC-BY-NC-4.0).
- evidence: `docs/BUILDING-BLOCKS.md` Parts 1(#6) & 3; `references/Depth-Anything-3/`, `references/dinov3/`
- exit signal: DA3 depth + confidence parity within tolerance on a fixed image.
- deferred: camera pose/intrinsics + multi-view (`cam_enc`/`cam_dec`, `DualDPT`, GS) — out of monocular scope; revisit as a separate change if a consumer needs pose.

## Phase 4: LocateAnything-3B — full VLM anchor

- status: done
- change: `2026-06-15-locateanything-vlm-integration`
- objective: Complete the high-signal LLM-backed probe: MoonViT + Qwen2.5 + PBD → typed `Detections`/`Points`.
- completed framed changes: `2026-06-15-locateanything-qwen2-backbone` — Qwen2.5 LLM backbone with GQA, KV-cache, RMSNorm, SwiGLU, block masks, convert/load, and tiny reference parity; `2026-06-15-locateanything-moonvit-backbone` — MoonViT-SO-400M vision backbone with packed-patch input, per-image block attention, convert/load, and tiny reference parity.
- completed integration slice: `2026-06-15-locateanything-vlm-integration` — projector, image-token scatter, full `LocateAnythingModel`, processor, PBD generation, local integration fixture, `predict`, and typed `Result` path. Upstream full-checkpoint reference parity remains a later hub/reference-environment hardening item.
- why now: Hardest, highest-signal (ARCHITECTURE §15) — but sequenced **after** a concrete vision path exists (Phase 1 DINOv3 + Phase 3 DA3), so the heavy VLM hardening isn't built on an unproven vision spine.
- likely outputs: VLM bridge (projector + image-token scatter); processor/modeling complete; PBD generate; tokenizer-backed `predict → Result`; deterministic local integration parity. Upstream full-checkpoint parity vs `references/mlx-vlm` + `references/LocateAnything-3B` is deferred to a later hardening gate before any shipped-model claim.
- evidence: `src/mlx_cv/models/locateanything/`; `docs/BUILDING-BLOCKS.md` Part 1 (#10–11)
- exit signal: local `preprocess → pbd_generate → postprocess` returns typed boxes/points in original-image coordinates, with deterministic integration taps passing. Full reference boxes/points parity remains an explicit deferred gate.

## Phase 5: RF-DETR + SAM 3.1 — detection and segmentation round

- status: done
- change: `2026-06-16-rfdetr-sam31-detection-segmentation`
- objective: One next-round task-model expansion that lands RF-DETR detection and SAM 3.1 image segmentation on the shared spine.
- why now: After the VLM anchor, these cover the remaining high-value output pillars and exercise the next shared blocks: deformable attention, query decoding, multi-scale necks, prompt/text encoders, and mask decoding.
- outputs: RF-DETR deformable-attention MLX op, DETR query-decoder head, multi-scale neck, model, convert, processor, `predict`, and committed detector fixture; SAM 3.1 image-mode VL backbone, text encoder/tokenizer, prompt encoder for text + PCS box/exemplar geometry, mask decoder, processor, `predict`, and committed text/PCS image fixtures. Video/tracking remains deferred.
- evidence: `docs/BUILDING-BLOCKS.md` Parts 1 (#5,#7,#8) & 3; `references/rf-detr/`; `references/sam3/`
- exit signal: RF-DETR `detections` and SAM 3.1 image-mode `masks`/paired grounding detections return typed original-image results on fixed inputs; deformable-attention and prompt/mask critical paths are unit-tested. Caveat: RF-DETR detector and SAM 3.1 image fixtures are committed local MLX tiny-oracle fixtures; full upstream checkpoint/reference parity remains a separate hardening gate before a release claim.

## Deferred or Not Now

- **DEIMv2** — DINOv3-native detector; phase-2 backbone-reuse unification once DINOv3 + DETR head exist.
- **EoMT** — closed-set semantic/panoptic seg; add only when dense class-labeled output is needed (SAM 3.1 ≠ panoptic).
- **Sapiens2** — human pose/normals/albedo/pointmap; needs new `Result` fields (`normals`/`albedo`/`pointmap`/`matting`). Add only if human pose becomes a first-class pillar.
- **SAM 3.1 video / Tracker memory** — the streaming memory-bank subsystem (block #12); after image segmentation lands.
- **RT-DETRv4** — dropped: least popular (507★) and redundant with RF-DETR / DEIMv2.
- **YOLO26** — watchlist only: very popular + spans det/seg/pose, but AGPL (copyleft) and not flagship-accuracy.
