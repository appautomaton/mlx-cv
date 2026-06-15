# SPEC: Build-once core blocks (lean)

Change: `2026-06-14-build-once-core-blocks` · Stage: frame · Source: this request + conversation, `.agent/steering/ROADMAP.md` Phase 2, `docs/BUILDING-BLOCKS.md` Part 1 (#1–5, #13), Phase-1 `src/mlx_cv/backbones/vision/dinov3/`

## Bounded goal
Extract the blocks the Phase-1 DINOv3 MLX port already proves into reusable, **parameterized** families, re-express DINOv3 on them with **zero forward-parity regression**, and prove generalization by instantiating a **second real ViT config (DINOv2)** from those families with **no new block code** — `core/` staying mlx-free throughout.

## Broader intent
Phase 2 of the foundation-first roadmap: turn the one-off DINOv3 port into the shared, parameterized foundation that downstream models (DA3, RF-DETR, SAM, LocateAnything) compose from, so the Nth model is mostly `convert.py` + `processor.py`. **Lean discipline:** build only what has a consumer now (DINOv3 + the imminent DINOv2); wire — but do not implement — variant slots whose only consumers arrive in later phases.

## Work scale / shape / lenses
- Scale: capability (Phase 2 of a roadmap). Shape: **refactor** (primary — extract + generalize) with a **parity-regression guard** and a **structural second-config proof**.
- Lenses: **engineering** (primary — block decomposition, parameterization, no-regression), **product** (light — ergonomics for future model authors adding the Nth backbone).

## Constraints & risks (that change implementation)
- **Behavior-preserving for DINOv3.** Extraction must not change DINOv3 numerics: the Phase-1 forward-parity fixture (≤ 2e-6 per tap on the CPU stream; `assert_parity` atol 1e-4) must still pass. This is the regression proof and gates the refactor.
- **`core/` stays mlx-free** (numpy + typing only; regex + `sys.modules` import smoke, as Phase 1). The mlx block families are `nn.Module`s → they **cannot live in `core/`**. ⟢ This supersedes `docs/BUILDING-BLOCKS.md` #2's `core/layers` home label — to be corrected as part of this change. Home must be mlx-allowed (`backbones/` is the existing mlx zone; `ops/` is currently numpy-only). Exact directory layout is a plan-stage decision.
- **No `core/` edit forced by a block** (the Phase-1 no-core-edit invariant continues; registration stays decorator-driven, `core/registry.py` untouched).
- **Families must be parameterized, not DINOv3-hardcoded.** DINOv3 settings = 2D-RoPE, no LayerScale, GELU-MLP, LayerNorm, packed-qkv, manual-softmax SDPA. DINOv2 (second config) flips **posenc → learned-absolute-with-interpolation** and **LayerScale → on** — so those two axes get a *real* second variant now. Norm/FFN stay selectable **slots** with only their current-consumer variant (LayerNorm, GELU-MLP) implemented.
- **DINOv2 this phase is structural only:** config + instantiate + `forward_features` with correct shapes/token-order via the shared families, no new block code. Full DINOv2 weight-convert + numerical parity is **deferred to its consuming phase (Phase 3 DA3)**. DINOv2 reference for shapes/config: `references/rf-detr/.../backbone/dinov2.py` + `dinov2_configs/*.json` (use the plain, non-windowed variant).
- Full `pytest` (currently **71**) must stay green; new family + DINOv2 tests add to it.

## Required outcome

**Structural change** (extract into mlx-allowed homes; DINOv3 re-expressed to consume them):
- **ViT backbone contract/family** — patch-embed + cls/register/storage tokens + intermediate-layer extraction + output-layout adapter → `BackboneFeatures`, driven by a config; DINOv3 re-expressed as config + thin assembly.
- **Parameterized Transformer block** — norm + attn + FFN (+ optional LayerScale, DropPath), with selectable norm / FFN / LayerScale; LayerNorm + GELU-MLP + LayerScale(on/off) implemented.
- **2D positional-encoding suite** — 2D-RoPE (DINOv3) **and** learned-absolute-with-interpolation (DINOv2); no other posenc variants.
- **Attention ops family** — SDPA core + packed-qkv only.
- **Weight-convert / `sanitize` engine** — the key-remap + layout-fix machinery factored so each model's `convert.py` is declarative rules over a shared engine (DINOv3 `convert.py` is the seed).
- **DINOv2** — second config + instantiation assembling only the shared families.

**Behavioral invariants:**
- DINOv3 forward-parity vs the Phase-1 committed fixture still passes within tolerance (behavior-preserving refactor).
- `core/` imports no mlx; `core/registry.py` and `core/` types untouched by the extraction.
- Adding DINOv2 requires **no new block-level module** and **no spine edit** (the generalization proof).
- Full `pytest` stays green.

## Acceptance criteria
1. DINOv3 forward-parity test (`tests/test_dinov3_parity.py`, Phase-1 fixture) passes **unchanged** after the refactor.
2. DINOv3 `modeling.py` no longer defines bespoke Block / Attention / RoPE; a structural diff shows that code **moved into the shared families** (imported, not duplicated).
3. DINOv2 instantiates from its config and runs `forward_features` on a fixed input returning `BackboneFeatures` with correct shapes + token order `[cls, storage…, patch…]`, **using only the shared families** — verified by a structure check that the DINOv2 folder defines no block/attention/posenc module of its own.
4. Each extracted family has a unit test exercising the variants it now has consumers for: block (LayerScale on/off), posenc (RoPE **and** learned-abs-interp), attention (SDPA + packed-qkv), convert engine (a remap-rule round-trip).
5. `core/` mlx-free (regex for `import mlx` / `from mlx` + `sys.modules` import smoke); `core/registry.py` untouched.
6. Full `pytest` green (71 prior + new).

## Anti-goals
- **GQA / MQA, KV-cache, decode position-ids** — Phase 4 (Qwen2 LLM backbone).
- **Window/global attention policy + block/Magi masks** — later backbones that need them.
- **Multi-scale neck / projector (FPN/pyramid)** — Phase 3 (DA3) / Phase 5 (RF-DETR).
- **Dense-map heads, query-decoder heads** — Phases 3/5/6.
- **Full DINOv2 weight-conversion + numerical parity** — deferred to Phase 3 (its first consumer); this phase proves structural instantiation only.
- **SwiGLU / RMSNorm bodies** — wire the selectable slot, but do not implement+test a variant with no Phase-2 consumer (arrives with Qwen2).
- **Any task model** (DA3 / RF-DETR / SAM / LocateAnything) — later phases.

## Scope coverage
- **Included:** 5 families (ViT contract, parameterized block, posenc [RoPE + learned-abs-interp], attention [SDPA + packed-qkv], convert/sanitize engine); DINOv3 re-expressed; DINOv2 structural second config.
- **Deferred (with reason):** full DINOv2 convert/parity → Phase 3 consumer; GQA/cache/window/neck/dense-heads/query-decoder → consuming phases; SwiGLU/RMSNorm bodies → Phase 4 Qwen2.
- **Anti-goals:** as above.
- **Needs-decision (resolved):** second config = **DINOv2** (user, 2026-06-14); route = Phase 2 **lean** (user, 2026-06-14).

## Assumptions
- DINOv2 architecture (dims/depth/heads/pos-emb/LayerScale) is sourced from `references/rf-detr/.../backbone/dinov2*` + config JSONs; structural shapes only (no weights/parity this phase).
- DINOv3 parity tolerance unchanged from Phase 1 (CPU stream, atol 1e-4).
- "No new block code" is verifiable structurally (the DINOv2 folder contains only config + a thin assembly importing the shared families).
- mlx block-family home is an mlx-allowed directory (likely a new `backbones/layers/` + posenc/attention placement); exact layout settled in plan/eng-review.
