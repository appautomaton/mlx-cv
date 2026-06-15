# Roadmap

Foundation-first, but **contract-proof, not paper-first**: Phase 1 proves the core spine contracts on
one real model before fleshing them out. Verified June 2026 against 10 reference impls (`references/`)
+ two Codex passes (decompose + skeptical review); full evidence in `docs/BUILDING-BLOCKS.md`. Phases 1–3
done; Phases 4–6 pending.

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

- status: pending
- change:
- objective: The high-signal LLM-backed probe: MoonViT + Qwen2.5 + PBD → typed `Detections`/`Points`. (Stage-1 config/convert/decode already exist.)
- recommended first framed change: Qwen2.5 LLM backbone — GQA + KV-cache + RMSNorm + SwiGLU + block mask (the LLM bodies Phase 2 explicitly deferred here).
- why now: Hardest, highest-signal (ARCHITECTURE §15) — but sequenced **after** a concrete vision path exists (Phase 1 DINOv3 + Phase 3 DA3), so the heavy VLM hardening isn't built on an unproven vision spine.
- likely outputs: VLM bridge (projector + image-token scatter); Qwen2 LLM backbone w/ KV-cache + block mask; PBD generate; processor/modeling complete; parity vs `references/mlx-vlm` + `references/LocateAnything-3B`.
- evidence: `src/mlx_cv/models/locateanything/`; `docs/BUILDING-BLOCKS.md` Part 1 (#10–11)
- exit signal: `load → predict → Result` matches reference boxes/points after `invert`.

## Phase 5: RF-DETR (detection)

- status: pending
- change:
- objective: Detection pillar via RF-DETR (most popular, Apache, real-time; DINOv2 backbone).
- why now: Exercises the deformable-attn op + query decoder + multi-scale neck (blocks #5/#7/#8).
- likely outputs: deformable-attention MLX op; DETR query-decoder head; multi-scale neck; RF-DETR model + convert + processor; detection parity.
- evidence: `docs/BUILDING-BLOCKS.md` Parts 1 (#5,#7,#8) & 3; `references/rf-detr/`
- exit signal: RF-DETR `detections` parity within tolerance; deformable-attn op unit-tested.

## Phase 6: SAM 3.1 (promptable image segmentation)

- status: pending
- change:
- objective: Segmentation pillar via SAM 3.1 image PCS (text + exemplar → masks); video/tracking deferred.
- why now: Hardest port (Codex ranking); image path is tractable before the video memory subsystem.
- likely outputs: SAM VL backbone + text encoder + tokenizer (`vl_combiner.py`/`text_encoder_ve.py`/`tokenizer_ve.py`); prompt encoder concatenating **text + geometric + optional visual-exemplar embed** (`Sam3Image._encode_prompt`) — exemplar needs a processor contract for the visual path, not just a geometry prompt encoder; mask decoder; `masks` parity.
- evidence: `docs/BUILDING-BLOCKS.md` Parts 1 (#7) & 3; `references/sam3/`
- exit signal: SAM 3.1 image-mode `masks` parity within tolerance on a fixed text prompt.

## Deferred or Not Now

- **DEIMv2** — DINOv3-native detector; phase-2 backbone-reuse unification once DINOv3 + DETR head exist.
- **EoMT** — closed-set semantic/panoptic seg; add only when dense class-labeled output is needed (SAM 3.1 ≠ panoptic).
- **Sapiens2** — human pose/normals/albedo/pointmap; needs new `Result` fields (`normals`/`albedo`/`pointmap`/`matting`). Add only if human pose becomes a first-class pillar.
- **SAM 3.1 video / Tracker memory** — the streaming memory-bank subsystem (block #12); after image segmentation lands.
- **RT-DETRv4** — dropped: least popular (507★) and redundant with RF-DETR / DEIMv2.
- **YOLO26** — watchlist only: very popular + spans det/seg/pose, but AGPL (copyleft) and not flagship-accuracy.
