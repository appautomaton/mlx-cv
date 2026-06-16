# PLAN: LocateAnything VLM Integration

Change: `2026-06-15-locateanything-vlm-integration` - Stage: plan - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal

Finish Phase 4 by integrating verified MoonViT and Qwen2 into a LocateAnything VLM with processor, PBD generation, conversion, and end-to-end typed `Result` parity.

## Architecture Approach

Keep LocateAnything as the multimodal owner: Qwen2 remains a reusable LLM backbone, MoonViT remains a packed-patch vision backbone, and `models/locateanything/` owns projector, image-token scatter, PBD loop, processor, predictor-style user path, and full model conversion. Runtime dependencies stay limited to existing project dependencies; reference minting remains throwaway-only.

## Execution Routing And Topology

- Default execution: direct, serial, continue through all slices when verification passes.
- Parallel-safe groups: none. Slices share LocateAnything integration surfaces and should land in dependency order.
- Checkpoints: none. Phase 4 remains owned until Slice 6 verification passes.

## Ordered Slice Sequence

### Slice 1: Model Assembly And Image Scatter

**Objective:** Replace the LocateAnything model stub with the assembled MLX VLM shell.

**Acceptance criteria:**
- `LocateAnythingProjector` matches the reference `LayerNorm(4 * vision_hidden) -> Linear(text_hidden) -> GELU -> Linear(text_hidden)` shape.
- `LocateAnythingModel` constructs `MoonViTBackbone`, `Qwen2ForCausalLM`, projector, and stores `image_token_index`.
- `get_input_embeddings` supports no-image, live MoonViT features, and cached image features.
- Scatter replaces exactly the image-token positions and raises on feature/token count mismatch.
- Forward delegates into Qwen2 using prepared `inputs_embeds`.

**Verification:** `uv run pytest tests/test_la_model.py tests/test_la_config.py tests/test_qwen2_integration_guards.py`

**Touches:** `src/mlx_cv/models/locateanything/modeling.py`, `src/mlx_cv/models/locateanything/__init__.py`, `tests/test_la_model.py`

**Produces:** working LocateAnything MLX model shell with deterministic unit tests.

**Status:** complete
**Evidence:** Added `LocateAnythingModel`, `LocateAnythingProjector`, lazy exports, and `tests/test_la_model.py`; `uv run pytest tests/test_la_model.py tests/test_la_config.py tests/test_qwen2_integration_guards.py` passed 14 tests.
**Risks / next:** PBD cache/position behavior remains for Slice 2.

### Slice 2: PBD Generation Loop

**Objective:** Port the local PBD generation path on top of Qwen2 cache and mask semantics.

**Acceptance criteria:**
- Local PBD helpers expose token-id extraction, block sampling, pattern handling, MTP prefill, MTP step, AR fallback, and hybrid switching.
- The implementation adapts to local Qwen2 tuple outputs and `Qwen2KVCache`.
- It fails clearly when `text_config.block_size` does not match `n_future_tokens`.
- Deterministic tests cover legal boxes, points, empty boxes, ref text, MTP-to-AR fallback, and AR-to-MTP switch-back.

**Verification:** `uv run pytest tests/test_la_pbd.py tests/test_qwen2_masks.py tests/test_qwen2_cache.py`

**Touches:** `src/mlx_cv/models/locateanything/pbd.py`, `src/mlx_cv/models/locateanything/modeling.py`, `tests/test_la_pbd.py`

**Produces:** `LocateAnythingModel.pbd_generate` with covered PBD behavior.

**Status:** complete
**Evidence:** Added local `pbd.py`, wired `LocateAnythingModel.pbd_generate`, added `Qwen2KVCache.trim`, and covered token sampling/pattern transitions in `tests/test_la_pbd.py`; `uv run pytest tests/test_la_pbd.py tests/test_qwen2_masks.py tests/test_qwen2_cache.py` passed 16 tests.
**Risks / next:** Processor must map normalized PBD coords back to original pixels without runtime tokenizer dependency.

### Slice 3: Runtime-Light Processor And Result Mapping

**Objective:** Implement image/prompt preprocessing and grounding-token postprocess without adding Transformers as a runtime dependency.

**Acceptance criteria:**
- Image preprocess resizes to patch/merge multiples, enforces RoPE limits, normalizes with mean/std `0.5`, patchifies to packed NCHW patches, emits `image_grid_hws`, and records `SpatialTransform`.
- Prompt preprocessing expands each `<image-N>` placeholder into `<img> + <IMG_CONTEXT> * merged_token_count + </img>`.
- Tokenization works through a supplied tokenizer protocol or fake tokenizer in tests; no hard `transformers` import.
- Postprocess parses generated ids, decodes labels, converts normalized `[0, 1000]` coords to model pixels, inverts to original image pixels, and returns `Result(detections=..., points=...)`.
- Placeholder/image count mismatch and missing tokenizer cases fail clearly.

**Verification:** `uv run pytest tests/test_la_processor.py tests/test_la_decode.py tests/test_geometry.py`

**Touches:** `src/mlx_cv/models/locateanything/processor.py`, `src/mlx_cv/models/locateanything/__init__.py`, `tests/test_la_processor.py`

**Produces:** isolated processor coverage and typed result mapping.

**Status:** complete
**Evidence:** Implemented runtime-light `LocateAnythingProcessor`, processor config/context, tokenizer protocol support, placeholder expansion, patchification, and `Result` postprocess; `uv run pytest tests/test_la_processor.py tests/test_la_decode.py tests/test_geometry.py` passed 20 tests.
**Risks / next:** Conversion/load must now compose full-model keys without broadening runtime dependencies.

### Slice 4: Conversion And Loading Composition

**Objective:** Make full LocateAnything weight conversion/load compose the model, MoonViT, Qwen2, and projector rules explicitly.

**Acceptance criteria:**
- Model-level conversion maps `vision_model.*`, `language_model.*`, and `mlp1.*` into the local module tree.
- MoonViT conv/layout and Qwen2 key rules remain delegated to their existing conversion helpers where needed.
- Tied `language_model.lm_head.weight` and metadata-only tensors are dropped intentionally.
- Tiny converted weights can be loaded into a tiny `LocateAnythingModel` and produce stable projector/scatter outputs.
- Existing no-runtime-`torch`/`transformers` guard remains green.

**Verification:** `uv run pytest tests/test_la_convert.py tests/test_moonvit_convert.py tests/test_qwen2_convert.py tests/test_qwen2_integration_guards.py`

**Touches:** `src/mlx_cv/models/locateanything/convert.py`, optional load helper in `modeling.py`, `tests/test_la_convert.py`

**Produces:** explicit full-model conversion/load support.

**Status:** complete
**Evidence:** Reworked full LocateAnything conversion to delegate MoonViT and Qwen2 rules, added `load_locateanything_weights`, and tightened tied-head/layout tests; `uv run pytest tests/test_la_convert.py tests/test_moonvit_convert.py tests/test_qwen2_convert.py tests/test_qwen2_integration_guards.py` passed 16 tests.
**Risks / next:** Tiny integration fixture must prove projector/scatter/PBD/result behavior together.

### Slice 5: Tiny Integration Fixture And Parity

**Objective:** Add a tiny end-to-end LocateAnything fixture that localizes integration drift across projector, scatter, PBD, and parsed result.

**Acceptance criteria:**
- `src/mlx_cv/parity/fixtures.py` exports a LocateAnything tiny config/input schema.
- A mint tool writes a tiny integration fixture out-of-band without adding runtime dependencies.
- Fixture taps include projector output, scattered embeddings, PBD block logits or sampled tokens, generated ids, and parsed boxes/points.
- MLX parity test loads the fixture, runs the local integration path, and `bisect` returns `None`.
- The fixture uses existing tiny MoonViT/Qwen2 dimensions or equivalently small configs, not production checkpoint dimensions.

**Verification:** `uv run pytest tests/test_la_integration_fixture.py tests/test_la_parity.py`

**Touches:** `src/mlx_cv/parity/fixtures.py`, `tools/mint_locateanything_fixture.py`, `tests/fixtures/locateanything_tiny_fixture*.npz`, `tests/test_la_integration_fixture.py`, `tests/test_la_parity.py`

**Produces:** committed tiny LocateAnything integration parity fixture.

**Plan correction:** `references/mlx-vlm` is not importable in the current runtime without extra package dependencies (`requests` is missing at import entry). Slice 5 will mint a deterministic local integration fixture for projector/scatter/PBD/result drift localization; upstream reference parity remains a later full-checkpoint/hub environment concern.

**Status:** complete
**Evidence:** Added `LOCATEANYTHING_FIXTURE_CONFIG`, deterministic fixed inputs/tap order, `tools/mint_locateanything_fixture.py`, committed tiny fixture/weights, and local integration parity tests; `uv run python tools/mint_locateanything_fixture.py` minted 0.06 MB/0.05 MB fixture artifacts; `uv run pytest tests/test_la_integration_fixture.py tests/test_la_parity.py` passed 3 tests.
**Risks / next:** Public predict path must make the integration usable without claiming upstream full-checkpoint parity.

### Slice 6: Public Surface, Predict Path, And Full Guard Sweep

**Objective:** Close Phase 4 with a user-facing `predict` path and full regression proof.

**Acceptance criteria:**
- Public exports expose `LocateAnythingModel`, `LocateAnythingProjector`, `LocateAnythingProcessor`, and PBD helpers lazily enough to preserve package import behavior.
- A user-facing prediction wrapper or model method covers `preprocess -> pbd_generate -> postprocess`.
- End-to-end tests prove a fixed image/prompt returns typed `Result` with boxes/points in original-image coordinates.
- README or module docstring status no longer describes LocateAnything model/processor as unimplemented.
- Full suite passes, and Phase 4 remains the only active roadmap phase until verify marks it done.

**Verification:** `uv run pytest tests/test_la_predict.py tests/test_qwen2_integration_guards.py && uv run pytest`

**Touches:** `src/mlx_cv/models/locateanything/__init__.py`, `src/mlx_cv/models/locateanything/modeling.py`, `src/mlx_cv/models/locateanything/processor.py`, focused docs/docstrings, `tests/test_la_predict.py`

**Produces:** Phase 4 completion surface and full regression evidence.

## Aggregate Verification Commands

| Slice | Command |
| --- | --- |
| 1 | `uv run pytest tests/test_la_model.py tests/test_la_config.py tests/test_qwen2_integration_guards.py` |
| 2 | `uv run pytest tests/test_la_pbd.py tests/test_qwen2_masks.py tests/test_qwen2_cache.py` |
| 3 | `uv run pytest tests/test_la_processor.py tests/test_la_decode.py tests/test_geometry.py` |
| 4 | `uv run pytest tests/test_la_convert.py tests/test_moonvit_convert.py tests/test_qwen2_convert.py tests/test_qwen2_integration_guards.py` |
| 5 | `uv run pytest tests/test_la_integration_fixture.py tests/test_la_parity.py` |
| 6 | `uv run pytest tests/test_la_predict.py tests/test_qwen2_integration_guards.py && uv run pytest` |

## Risks

- PBD generation depends on precise cache trim/position-id behavior; Slice 2 must prove this before Slice 5 parity.
- Processor coordinate inversion can look correct while remaining off by resize scale; Slice 3 must test non-square resized images.
- The local Qwen2 output is a tuple, while references use output objects; all integration code must use the local tuple contract.
- Runtime dependency guards are non-negotiable; tokenizer support must stay protocol-based.
