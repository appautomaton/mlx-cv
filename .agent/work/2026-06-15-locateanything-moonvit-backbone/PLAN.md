# PLAN: LocateAnything MoonViT Vision Backbone

Change: `2026-06-15-locateanything-moonvit-backbone` - Stage: plan - Source: `SPEC.md`, `DESIGN.md`, `references/LocateAnything-3B/modeling_vit.py`, `references/LocateAnything-3B/image_processing_locateanything.py`, `references/mlx-vlm/mlx_vlm/models/locateanything/vision.py`

## Goal
Implement the MoonViT-SO-400M vision backbone slice from [SPEC.md](/Users/ac/dev/ai/mlx-cv/.agent/work/2026-06-15-locateanything-moonvit-backbone/SPEC.md:1), with packed-patch input, per-image block attention, 2D RoPE, 2x2 patch merge, convert/load support, and tiny reference parity.

## Architecture Approach
Use a standalone MoonViT packed-patch backbone, not `ViTBackbone`. The shared ViT assembly assumes ordinary image batches plus cls/storage token layouts; MoonViT consumes already-patchified packed image patches, isolates attention by `grid_hws`, and returns merged per-image token tensors for a later projector slice. [DESIGN.md](/Users/ac/dev/ai/mlx-cv/.agent/work/2026-06-15-locateanything-moonvit-backbone/DESIGN.md:1) is normative for shape contracts, module layout, and reference-matching decisions.

## Execution Routing and Topology
- Default route: direct execution by the orchestrator.
- Parallel-safe groups: none. The slices are intentionally serial because later parity and conversion tests depend on earlier module paths and tap names.
- Checkpoints: none. Continue through all slices after each slice verification passes.
- Subagent route: optional read-only review only; implementation remains direct unless execution uncovers independent, disjoint test generation work.

## Ordered Slice Sequence

### Slice 1: Canonical Config And Import Boundary

**Objective:** Move MoonViT configuration ownership into the backbone package while preserving LocateAnything config behavior and mlx-free imports.

**Acceptance criteria:**
- `src/mlx_cv/backbones/vision/moonvit/config.py` defines canonical `MoonViTConfig` with LocateAnything-3B defaults and compatibility properties (`embed_dim`, `depth`, `num_heads`, `head_dim`, `spatial_merge_size`, `from_dict`).
- `src/mlx_cv/models/locateanything/config.py` imports/reuses the canonical MoonViT config without importing `mlx`.
- Package exports are updated without registering a partially implemented builder too early.
- Existing LocateAnything config tests still pass, and import guards still prove `core` and config-only LocateAnything imports do not import `mlx`.

**Verification:** `uv run pytest tests/test_la_config.py tests/test_qwen2_config.py tests/test_qwen2_integration_guards.py`

**Touches:** `src/mlx_cv/backbones/vision/moonvit/config.py`, `src/mlx_cv/backbones/vision/moonvit/__init__.py`, `src/mlx_cv/models/locateanything/config.py`, `tests/test_la_config.py`, `tests/test_qwen2_config.py`, `tests/test_qwen2_integration_guards.py`

**Produces:** canonical MoonViT config and preserved import boundary.

**Status:** complete
**Evidence:** Added canonical `src/mlx_cv/backbones/vision/moonvit/config.py`; updated MoonViT package root and LocateAnything config reuse; `uv run pytest tests/test_la_config.py tests/test_qwen2_config.py tests/test_qwen2_integration_guards.py` passed 10 tests.
**Risks / next:** none.

### Slice 2: Packed Primitives

**Objective:** Implement and unit-test MoonViT's packed-patch primitives before building the full encoder.

**Acceptance criteria:**
- Shape helpers validate `grid_hws`, calculate per-image lengths, and build `cu_seqlens`.
- `MoonViTPatchEmbed` accepts processor-style NCHW packed patches and transposes internally for MLX Conv2d.
- `Learnable2DInterpPosEmb` matches the reference same-grid path and has an interpolation path covered by tests.
- `Rope2DPosEmb` and `apply_rope` match the reference complex layout for multiple grid shapes.
- `make_block_attention_mask` prevents cross-image visibility and handles single-image/multi-image packed sequences.
- `patch_merger` matches reference 2x2 order and output shape `(merged_tokens, 4 * hidden_size)`, not the merged mlx-vlm port's intermediate `(merged_tokens, 4, hidden_size)` shape, and fails clearly on non-divisible grids or length/grid mismatches.

**Verification:** `uv run pytest tests/test_moonvit_primitives.py`

**Touches:** `src/mlx_cv/backbones/vision/moonvit/modeling.py`, optional local interpolation helper under `src/mlx_cv/backbones/vision/moonvit/`, `tests/test_moonvit_primitives.py`

**Produces:** tested MoonViT leaves and packing utilities.

### Slice 3: Encoder, BackBone, And Registry

**Objective:** Assemble the MoonViT encoder and registered backbone on top of the packed primitives.

**Acceptance criteria:**
- `MoonViTMLP`, `MoonViTEncoderLayer`, and `MoonViTBackbone` match reference residual order, fused `wqkv`, attention/output bias usage, LayerNorm eps/defaults, and final LayerNorm behavior.
- `wqkv` and `wo` are direct encoder-layer attributes, or the implementation generates explicit declarative per-layer rename rules; the plan does not allow imperative string-surgery conversion.
- Forward accepts `(pixel_values, grid_hws)`, returns a list of merged per-image tensors, and can optionally capture ordered taps (`patch_embed`, `rope_freqs_cis`, first/all block outputs, `norm`, `merged_*`) for parity bisect.
- `build_moonvit_so400m` registers `moonvit-so400m` as a vision backbone when the MLX modeling surface is imported.
- Registration and shape tests pass without touching projector, processor, full LocateAnything model, or Qwen2.

**Verification:** `uv run pytest tests/test_moonvit_model.py tests/test_registry.py`

**Touches:** `src/mlx_cv/backbones/vision/moonvit/modeling.py`, `src/mlx_cv/backbones/vision/moonvit/__init__.py`, `tests/test_moonvit_model.py`, `tests/test_registry.py`

**Produces:** executable MoonViT backbone with registry entry and deterministic local tests.

### Slice 4: Reference Fixture And Tap Schema

**Objective:** Mint and commit the tiny PyTorch MoonViT fixture with deterministic inputs, weights, and ordered taps.

**Acceptance criteria:**
- `src/mlx_cv/parity/fixtures.py` exports `MOONVIT_FIXTURE_CONFIG`, `moonvit_fixed_inputs`, and `moonvit_tap_order`.
- `tools/mint_moonvit_fixture.py` imports `references/LocateAnything-3B/modeling_vit.py` out of band, forces fp32 SDPA behavior, asserts the reference blocks are using `attn_implementation == "sdpa"`, uses at least two image grids, and writes both `tests/fixtures/moonvit_tiny_fixture.npz` and `tests/fixtures/moonvit_tiny_fixture_weights.npz` atomically.
- `MOONVIT_FIXTURE_CONFIG` records `attn_implementation="sdpa"` so the reference eager path cannot be selected accidentally.
- The tiny fixture config keeps `head_dim % 4 == 0` and uses grids that cover both position-embedding branches: one grid equal to the fixture's `init_pos_emb_height/init_pos_emb_width`, and one differing grid for interpolation.
- Fixture taps include enough forward-ordered evidence to localize drift: patch embed, RoPE frequencies, normalized block-attention visibility with a consistent saved shape, first/all encoder blocks, final norm, and indexed merged outputs (`merged_00`, `merged_01`, ...) so ragged per-image outputs round-trip through the `.npz` parity harness.
- Runtime package dependencies remain unchanged; `torch` and `transformers` are mint-only.

**Verification:** `uv run --with torch --with transformers python tools/mint_moonvit_fixture.py` and `uv run pytest tests/test_moonvit_fixture.py`

**Touches:** `src/mlx_cv/parity/__init__.py`, `src/mlx_cv/parity/fixtures.py`, `tools/mint_moonvit_fixture.py`, `tests/fixtures/moonvit_tiny_fixture.npz`, `tests/fixtures/moonvit_tiny_fixture_weights.npz`, `tests/test_moonvit_fixture.py`

**Produces:** committed MoonViT tiny fixture, weights fixture, and schema tests.

### Slice 5: Convert, Loaded Parity, And Full Guard Sweep

**Objective:** Load PyTorch-style MoonViT weights through conversion rules and prove final MLX parity plus integration boundaries.

**Acceptance criteria:**
- `src/mlx_cv/backbones/vision/moonvit/convert.py` maps reference state dict keys into the local module tree through shared `hub.convert` rules where practical.
- Conversion handles conv weight layout, drops metadata-only tensors, preserves fused `wqkv`, and does not hide imperative key surgery in tests.
- Conversion targets standalone MoonViT fixture keys for this slice; full `vision_model.*` or `vision_tower.*` namespace bridging remains a later full-VLM concern and should be solved with prefix rules, not by changing this slice's standalone proof.
- Loaded MLX MoonViT outputs and ordered taps match the committed fixture within tolerance on the CPU stream; `bisect` returns `None`.
- Focused import/dependency guards still pass, including no runtime `torch` or `transformers` in `pyproject.toml`.
- Full test suite passes after the MoonViT slice.

**Verification:** `uv run pytest tests/test_moonvit_convert.py tests/test_moonvit_parity.py tests/test_qwen2_integration_guards.py` and `uv run pytest`

**Touches:** `src/mlx_cv/backbones/vision/moonvit/convert.py`, `src/mlx_cv/backbones/vision/moonvit/__init__.py`, `tests/test_moonvit_convert.py`, `tests/test_moonvit_parity.py`, any guard tests needed to keep import/dependency boundaries explicit.

**Produces:** conversion/load support, loaded-weight parity proof, and final focused/full verification.

## Aggregate Verification Commands

| Slice | Command |
| --- | --- |
| 1 | `uv run pytest tests/test_la_config.py tests/test_qwen2_config.py tests/test_qwen2_integration_guards.py` |
| 2 | `uv run pytest tests/test_moonvit_primitives.py` |
| 3 | `uv run pytest tests/test_moonvit_model.py tests/test_registry.py` |
| 4 | `uv run --with torch --with transformers python tools/mint_moonvit_fixture.py` and `uv run pytest tests/test_moonvit_fixture.py` |
| 5 | `uv run pytest tests/test_moonvit_convert.py tests/test_moonvit_parity.py tests/test_qwen2_integration_guards.py` and `uv run pytest` |

## Risks For Engineering Review
- Interpolation may need a local bicubic helper if `nn.Upsample(mode="cubic")` is not close enough to PyTorch `F.interpolate(..., mode="bicubic")`.
- The PyTorch reference uses `PytorchGELUTanh()`; use tanh-approx GELU and verify the block/MLP tap so erf/`none` cannot slip in.
- MLX SDPA boolean mask semantics must be proven for packed multi-image visibility, not assumed, and the fixture must be minted through the reference SDPA path rather than eager.
- The public input contract is NCHW packed patches from the processor; the merged mlx-vlm port internally uses NHWC at the call site.
- Slice 4 needs a mint environment with `torch` and `transformers`; these must not become runtime dependencies.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: All three blocking corrections landed cleanly - the fixture is pinned to and asserts the SDPA path (with the eager-isolation rationale), moonvit/__init__ is mlx-free via the Qwen2 lazy pattern, and Slice 1's gate now runs test_qwen2_config.py, alongside corrected merger shape, flat wqkv/wo, and GELU/interpolation guidance.
- Concern: The remaining risk is irreducible numerical parity that only running the fixture settles - bicubic interpolation exactness and MLX SDPA boolean-mask semantics - both now gated by dedicated taps but unproven in-repo.
- Action: Proceed; during Slice 2/3 confirm the interpolation tap and the MLX bool-mask attention match the fixture before the final assertion, falling back to a ported pure-MLX bicubic if nn.Upsample drifts.
- Verified: Re-read SPEC/DESIGN/PLAN and confirmed all three blocking fixes plus the (N, 4*D) merger shape, flat wqkv/wo + declarative-convert constraint, tanh-GELU guidance, and indexed merged tap keys.
