# PLAN: Build-once core blocks (lean)

Change: `2026-06-14-build-once-core-blocks` · Stage: plan · Spec: `SPEC.md` · Design: `DESIGN.md` (this dir)

## Goal
Extract DINOv3's inline blocks into reusable parameterized families under `backbones/layers/` + `backbones/vision/vit.py` (+ `hub/convert.py`), re-express DINOv3 on them with **zero forward-parity regression**, and prove generalization by a structural **DINOv2** second config that uses only the shared families. Full contract: `SPEC.md`; layout + posenc-strategy + parity method: `DESIGN.md`.

## Architecture approach
See `DESIGN.md`. Key invariants threaded through every slice: **`core/` imports no mlx** (families live in `backbones/`/`hub/`, behind `[mlx]`); **`core/registry.py` untouched** (decorator registration only); **DINOv3 forward-parity stays green after every slice** (the extraction is behavior-preserving — `tests/test_dinov3_parity.py` is the continuous gate, ordered taps localize drift). Extraction is bottom-up (leaves → block → assembly → convert → second config) so each slice keeps parity green independently.

## Ordered slice sequence

### Slice 1: Leaf families — attention + posenc(RoPE) + MLP + patch-embed
**Objective:** Extract DINOv3's `Attention` (packed-qkv + manual-softmax SDPA, optional rope hook), rope helpers (`RoPE2D`), `Mlp` (GELU; SwiGLU slot), and `PatchEmbed` into `backbones/layers/`; DINOv3 imports them with behavior unchanged.
**Acceptance criteria:**
- `backbones/layers/{attention,position,mlp,patch_embed}.py` exist; `Attention.__call__(x, rope=None, n_prefix=...)` applies rope only when given.
- `dinov3/modeling.py` no longer defines `DINOv3Attention` / `DINOv3Mlp` / `DINOv3PatchEmbed` / `_rope_*` — it imports them.
- DINOv3 forward-parity unchanged (all taps ≤ tol).
**Verification:** `.venv/bin/python -m pytest tests/test_dinov3_parity.py tests/test_dinov3_forward.py tests/test_layers.py -q`
**Touches:** `backbones/layers/` (new), `backbones/vision/dinov3/modeling.py`, `tests/test_layers.py` (new)
**Status:** complete
**Evidence:** added `backbones/layers/{__init__,attention,position,mlp,patch_embed}.py`; `dinov3/modeling.py` now imports `Attention`/`MlpFFN`/`PatchEmbed` + rope helpers and the inline `DINOv3Attention`/`DINOv3Mlp`/`DINOv3PatchEmbed`/`_rope_*` are gone; submodule param paths unchanged (`attn.qkv`, `mlp.fc1`, `patch_embed.proj`, top-level `periods`) so weights still load. `pytest test_dinov3_parity test_dinov3_forward test_layers -q` → **14 passed** (DINOv3 parity unchanged; 8 new family tests).
**Risks / next:** none.

### Slice 2: Parameterized Transformer block
**Objective:** Extract `DINOv3Block` → `layers/block.py:TransformerBlock` — pre-norm, selectable norm (LayerNorm; RMSNorm slot) / FFN (GELU-MLP; SwiGLU slot) / **LayerScale (on/off)**; DINOv3 wires LayerNorm + GELU-MLP + LayerScale off.
**Acceptance criteria:**
- `TransformerBlock` composes the Slice-1 attention + MLP; LayerScale off ⇒ exact identity (no residual scaling), on ⇒ per-channel scale params.
- `dinov3/modeling.py` no longer defines `DINOv3Block`.
- DINOv3 forward-parity unchanged.
**Verification:** `.venv/bin/python -m pytest tests/test_dinov3_parity.py tests/test_layers.py -q`
**Touches:** `backbones/layers/block.py` (new), `backbones/vision/dinov3/modeling.py`, `tests/test_layers.py`
**Depends on:** Slice 1
**Status:** complete
**Evidence:** added `backbones/layers/block.py` (`TransformerBlock` + `LayerScale`; norm/ffn/layerscale selectable, RMSNorm/SwiGLU slots raise); `DINOv3Block` removed — `DINOv3ViT` now builds `TransformerBlock(..., layerscale=False)`. LayerScale-off creates **no** `ls`/`gamma` params, so the block param tree is byte-identical → `pytest test_dinov3_parity test_layers -q` → **16 passed** (DINOv3 parity unchanged; block tests cover ls on/off identity + RMSNorm slot).
**Risks / next:** none.

### Slice 3: ViT backbone family (the assembly) + DINOv3 re-expressed
**Objective:** Add `backbones/vision/vit.py:ViTBackbone` — the shared assembly with a `RoPEStrategy` / `AbsPosStrategy` hook, using the **unified token-assembly order** (`DESIGN.md` §PositionStrategy, corrected per eng-review B2: `[cls]` → abs-pos on `[cls,patch]` if abs → **insert storage/register after cls** → rope on patch in-block → final norm → split → `BackboneFeatures`, `capture_taps` preserved); re-express `DINOv3ViT` as config + `RoPEStrategy` over it. The abs branch is built but unexercised until Slice 5, so the assembly needs **no** change there.
**Acceptance criteria:**
- `DINOv3ViT` carries no bespoke assembly logic beyond config wiring; `build_dinov3` still registers `"dinov3"` (registry untouched).
- `capture_taps` emits the **same tap names + order** as the committed fixture (`patch_embed`, `rope_sincos`, `block_NN`, `norm`, `cls/storage/patch`).
- DINOv3 forward-parity unchanged (every tap ≤ tol; `bisect` finds no drift); `core/` mlx-free.
**Verification:** `.venv/bin/python -m pytest tests/test_dinov3_parity.py tests/test_dinov3_forward.py -q && ! grep -rqE "^[[:space:]]*(import mlx|from mlx)" src/mlx_cv/core`
**Execution:** subagent recommended
**Touches:** `backbones/vision/vit.py` (new), `backbones/vision/dinov3/modeling.py`
**Depends on:** Slice 2

### Slice 4: Convert / `sanitize` rule engine + DINOv3 convert re-expressed
**Objective:** Add `hub/convert.py` — declarative `Rename`/`Transpose`/`Drop` rules applied over a `state_dict` → `[(mlx_path, mx.array)]`; re-express DINOv3's three fixes (drop `mask_token`, rename `rope_embed.periods→periods`, transpose `patch_embed.proj.weight`) as rules.
**Acceptance criteria:**
- `dinov3/convert.py` expresses its mapping as rules over the shared engine; `load_dinov3_weights` still loads the minted weights.
- DINOv3 forward-parity unchanged (parity test loads via this path).
- A convert-engine unit test covers a rename + transpose + drop round-trip.
**Verification:** `.venv/bin/python -m pytest tests/test_dinov3_parity.py tests/test_convert.py -q`
**Touches:** `hub/convert.py` (new), `backbones/vision/dinov3/convert.py`, `tests/test_convert.py` (new)
**Depends on:** Slice 3 (module paths stable)

### Slice 5: DINOv2 structural second config (the generalization proof)
**Objective:** Add `LearnedAbsPosEmb` (bicubic-interp) to `layers/position.py`, and `backbones/vision/dinov2/{config,modeling}.py` — a thin assembly = `ViTBackbone(config, posenc=AbsPosStrategy, layerscale=on, rope=off)` registered as `"dinov2"`, using **only** the shared families. Config from `references/rf-detr/.../dinov2_configs` (with-registers-small: dim 384, depth 12, heads 6, registers 4, patch 14).
**Acceptance criteria:**
- DINOv2 instantiates from config (random init) and `forward_features` on a fixed input returns `BackboneFeatures` with correct shapes + token order `[cls, register×4, patch×N]`; appears in `BACKBONES`.
- **Registers receive no pos-emb (eng-review B2):** the `LearnedAbsPosEmb` table covers `[cls, patch]` only (width `1 + Npatch`, excluding register slots), and registers are inserted **after** the abs-pos add — asserted by a test on the pos-table width + final token order, matching the reference (`dinov2_with_windowed_attn.py:425–459`).
- **DINOv3 parity unchanged (eng-review B1):** this slice edits shared `layers/position.py`, so `tests/test_dinov3_parity.py` must still pass.
- **No new block code:** `backbones/vision/dinov2/**/*.py` defines no `*Attention`/`*Block`/`*Mlp`/rope/posenc class, AND `dinov2/modeling.py` imports the families from `backbones/layers`/`backbones/vision/vit` (grep is the tripwire; the import-assertion + small diff is the binding proof — `DESIGN.md` §R1).
- `core/` mlx-free; full `pytest` green.
**Verification:** `.venv/bin/python -m pytest tests/test_dinov3_parity.py tests/test_dinov2_forward.py -q && ! grep -rqE "class .*(Attention|Block|Mlp)|def .*rope" src/mlx_cv/backbones/vision/dinov2/**/*.py && .venv/bin/python -c "import sys, mlx_cv.core; assert not any(m=='mlx' or m.startswith('mlx.') for m in sys.modules)"`
**Touches:** `backbones/layers/position.py`, `backbones/vision/dinov2/` (new), `tests/test_dinov2_forward.py` (new)
**Depends on:** Slice 3

### Slice 6: Docs + final gate
**Objective:** Correct `docs/BUILDING-BLOCKS.md` block-home labels to the actual mlx-allowed homes (#2 `core/layers` → `backbones/layers/`; note posenc/#3 + convert/#13 homes), and run the full close-out gate.
**Acceptance criteria:**
- BUILDING-BLOCKS Part 1 homes match the shipped layout; the `core/layers` label is gone.
- Full `pytest` green (71 prior + new); `core/` mlx-free (regex + `sys.modules` smoke); `core/registry.py` untouched since scaffold.
**Verification:** `.venv/bin/python -m pytest -q && ! grep -rqE "^[[:space:]]*(import mlx|from mlx)" src/mlx_cv/core && .venv/bin/python -c "import sys, mlx_cv.core; assert not any(m=='mlx' or m.startswith('mlx.') for m in sys.modules)" && git diff --quiet 526fd4f -- src/mlx_cv/core/registry.py && echo "registry untouched"`
**Depends on:** Slice 5

## Execution routing and topology
- Default: **continue** through all slices after each verification passes; execution windows are context batches, not stops.
- Subagent: **Slice 3** (recommended — the assembly refactor is the broad, parity-critical one).
- Checkpoints: **none** — the one structural decision (mlx-family home) is settled in `DESIGN.md` + the `auto-eng-review` gate before execute; nothing mid-stream needs a human.
- **Parallel-safe groups: none.** Slices are serial (block needs attention; assembly needs block; convert + dinov2 need stable module paths). Slice 4 is logically independent of the nn extraction but kept serial to avoid path churn.

## Acceptance-criteria coverage (SPEC § Acceptance → slice)
| SPEC criterion | Slice(s) |
|---|---|
| 1. DINOv3 parity passes unchanged | 1, 2, 3, 4 (continuous gate) |
| 2. DINOv3 block code moved, not duplicated | 1, 2, 3 |
| 3. DINOv2 forward via shared families, correct shapes | 5 |
| 4. Each family unit-tested (block/posenc/attention/convert) | 1, 2, 4 |
| 5. `core/` mlx-free; `core/registry.py` untouched | 3, 5, 6 |
| 6. Full `pytest` green | 5, 6 |

## Risks (carry into execution)
- **Slice 3 is the parity risk** — reordering ops in the shared assembly can drift DINOv3 numerics; the ordered taps + `bisect` localize it, CPU-stream comparison as in Phase 1.
- **Convert engine has one consumer this phase** (DINOv3) — minimal by design; full multi-model generalization is Phase 3 (deliberate, per SPEC lean scope).
- **DINOv2 pos-emb interpolation** must match grid sizing for the fixed test input; structural-only (no parity), so tolerance is shape-correctness, not numerics.
- **[R1] "No new block code" grep is gameable** (renamed classes / copied logic slip through) — paired with the import-assertion + small-diff eyeball (Slice 5, `DESIGN.md` §R1).
- **[R2] Convert engine generality is unproven** — DINOv2/DA3 will later need separate-q/k/v→packed-qkv handling the DINOv3 seed lacks; keep the engine minimal and do not claim multi-model conversion generality until Phase 3.

## Review: Engineering

- Reviewer: **Codex `gpt-5.5`**, reasoning `xhigh`, `--sandbox read-only` (independent cross-model review; 38 read-only commands, sources inspected incl. `modeling.py`/`convert.py`/`features.py`/`parity/`/`tests/`/DINOv2 ref).
- Verdict: **approved_with_risks** (prior `needs_correction` superseded after both blockers were corrected in this PLAN + `DESIGN.md`).
- Blockers (resolved):
  - **[B1]** Slice 5 edits shared `layers/position.py` but originally skipped the DINOv3 parity gate, violating the "parity after every slice" invariant. → **Fixed:** `tests/test_dinov3_parity.py` added to Slice 5 verification + an explicit acceptance criterion.
  - **[B2]** `AbsPosStrategy` under-specified for register tokens — the DINOv2-with-registers reference adds pos to `[cls, patch]` then inserts registers (registers get **no** pos), but the plan prepended `[cls, storage…]` before posenc; shape-only tests would mask it. → **Fixed:** `DESIGN.md` §PositionStrategy now specifies the unified token-assembly order (registers inserted after abs-pos) and Slice 5 asserts pos-table width excludes registers + token order, against `dinov2_with_windowed_attn.py:425–459`.
- Carried risks: **[R1]** gameable grep, **[R2]** convert-engine generality — both in Risks above.
- Strengths (reviewer): DINOv3 parity coverage is strong where run (CPU stream, final + ordered taps + bisect cover RoPE/qkv-order/GELU/LayerNorm-eps/prefix-skip/conv-layout); slice order sane (leaves → block → assembly → convert → second config); mlx-free-core well protected by `backbones/`/`hub/` placement + regex + import-smoke. Non-blocking, adopted: keep `hub/convert.py` (docs name `hub/` as the convert home); scope grep checks to `*.py`.
