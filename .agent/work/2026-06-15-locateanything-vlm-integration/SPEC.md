# SPEC: LocateAnything VLM Integration

Change: `2026-06-15-locateanything-vlm-integration` - Stage: frame - Source: `.agent/steering/ROADMAP.md` Phase 4, completed Qwen2 and MoonViT work artifacts, `src/mlx_cv/models/locateanything/`, `references/mlx-vlm/mlx_vlm/models/locateanything/`, `references/LocateAnything-3B/`

## Bounded Goal

Finish Phase 4 by assembling the verified MoonViT and Qwen2.5 components into a working LocateAnything VLM that can preprocess an image/prompt, scatter projected image features into the language stream, run PBD generation, and return typed `Detections`/`Points` in original-image coordinates with fixture-backed parity.

## Broader Intent

Phase 4 is owned until full LocateAnything is end-to-end verified. The earlier Qwen2 and MoonViT changes are completed components inside this phase, not stopping points.

## Work Scale And Shape

- Scale: one capability completion slice for Phase 4.
- Shape: parity-driven model integration plus processor/predictor wiring.
- Selected lenses: product, engineering, runtime.

## Required Outcome

- `src/mlx_cv/models/locateanything/modeling.py` exposes the assembled MLX model: `vision_tower`, `multi_modal_projector`, `language_model`, image-token embedding scatter, cache creation, forward, and `pbd_generate`.
- `src/mlx_cv/models/locateanything/processor.py` implements a runtime-light processor that patchifies images for MoonViT, expands image placeholders into the expected image token run, maps generated grounding tokens back through `decode.py`, and returns `Result` with `Detections` and `Points`.
- LocateAnything conversion/load support composes the existing MoonViT and Qwen2 conversion rules, maps projector weights, drops tied metadata-only tensors, and keeps all rules explicit.
- A tiny end-to-end fixture proves the integration path against the reference shape and semantics without requiring production checkpoint memory.
- Public package exports and focused guard tests make the Phase 4 completion discoverable while preserving `core/` mlx-free and no runtime `torch`/`transformers`.

## Constraints And Risks

- Do not add `torch` or `transformers` to runtime dependencies. Reference minting may use throwaway `uv --with ...` commands, as prior parity fixtures do.
- The processor may depend on a tokenizer object/protocol supplied by the caller or tests, but must not make Transformers a hard runtime import.
- Preserve the already verified Qwen2 and MoonViT contracts. Integration should adapt to those modules, not reopen their internals unless a verified incompatibility is found.
- Image-token scatter must fail clearly if the number of image token positions does not match projected image features.
- PBD generation must use the Qwen2 block-mask/cache semantics verified in the Qwen2 slice.
- Spatial postprocess must return original-image pixel coordinates, not normalized `[0, 1000]` coordinates.
- Full production weight loading and real Hugging Face download ergonomics are valuable, but this change is complete only when the local tiny integration parity and user-facing `predict` path are verified.

## Acceptance Criteria

1. Model assembly: constructing `LocateAnythingModel` from `LocateAnythingConfig` creates MoonViT, Qwen2, and projector modules with the reference projector shape `LayerNorm(4 * vision_hidden) -> Linear(text_hidden) -> GELU -> Linear(text_hidden)`.
2. Image scatter: `get_input_embeddings` embeds text ids, projects MoonViT merged features, replaces exactly the `image_token_index` positions, supports cached image features, and raises a clear error on feature/token count mismatch.
3. Forward/PBD: model forward delegates to Qwen2 with either token ids or prepared embeddings, `make_cache` returns one cache per decoder layer, and `pbd_generate` follows the reference MTP/AR hybrid block flow.
4. Processor preprocess: images are resized to patch/merge multiples, normalized with mean/std `0.5`, patchified into packed NCHW patches, and paired with `image_grid_hws`; prompt image placeholders expand to the exact merged-token count.
5. Processor postprocess: generated grounding token ids parse through `parse_grounding_tokens`, labels decode through the supplied tokenizer interface, normalized coords map back to original-image pixels, and output is a `Result` with `Detections` and/or `Points`.
6. Conversion/load: full LocateAnything state-dict conversion composes model-level, MoonViT, and Qwen2 rules without imperative hidden key surgery; tied LM-head weights are dropped; projector keys map to local modules.
7. Fixture parity: a tiny reference fixture covers projector output, scattered embeddings, PBD block sampling, and final parsed boxes/points enough to localize drift.
8. Integration guards: package exports, model construction, processor behavior, conversion, PBD decode, and no-runtime-`torch`/`transformers` guards pass; full `uv run pytest` passes.

## Anti-Goals

- Do not begin Phase 5 RF-DETR or SAM work.
- Do not replace the verified Qwen2 or MoonViT implementations with reference copies.
- Do not add a Transformers runtime dependency just to tokenize test prompts.
- Do not claim Phase 4 complete from backbone parity alone; completion requires end-to-end LocateAnything result parity.
- Do not implement SAM video/tracking, RF-DETR, or unrelated result fields in this change.
