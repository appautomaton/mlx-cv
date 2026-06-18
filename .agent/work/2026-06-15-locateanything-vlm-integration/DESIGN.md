# DESIGN: LocateAnything VLM Integration

Change: `2026-06-15-locateanything-vlm-integration`

## Architecture

The integration should be a thin assembly over the already verified components:

- `MoonViTBackbone` owns packed-patch vision encoding and returns per-image merged features.
- `Qwen2ForCausalLM` owns language decoding, MAGI-style block masks, logits, and KV cache behavior.
- `LocateAnythingModel` owns only cross-modal assembly: projector, image-token scatter, forward delegation, cache factory, and PBD generation.
- `LocateAnythingProcessor` owns image preprocessing, prompt expansion, and grounding-token postprocess into `Result`.

Do not push LocateAnything image insertion into Qwen2. The Qwen2 slice explicitly left visual feature insertion to the VLM assembly, and keeping it there avoids coupling the reusable LLM backbone to one multimodal contract.

## Model Contract

`LocateAnythingModel(config)` contains:

- `vision_tower: MoonViTBackbone`
- `language_model: Qwen2ForCausalLM`
- `multi_modal_projector: LayerNorm(4 * vision_hidden) -> Linear(text_hidden) -> GELU -> Linear(text_hidden)`
- `image_token_index`

`get_input_embeddings(input_ids, pixel_values=None, image_grid_hws=None, cached_image_features=None, image_token_id=None)` returns an embedding tensor. With image features, it:

1. Embeds `input_ids` through `language_model.get_input_embeddings()`.
2. Gets merged MoonViT features or uses `cached_image_features`.
3. Projects them through the multimodal projector.
4. Replaces flattened positions where `input_ids == image_token_id`.
5. Raises if projected feature count differs from image-token count.

`__call__` passes prepared embeddings into `Qwen2ForCausalLM(inputs_embeds=...)` and returns that tuple unchanged. Tests should document tuple order rather than inventing a new output type.

## PBD Contract

The PBD loop should be ported from `references/mlx-vlm/mlx_vlm/models/locateanything/pbd.py` into the local package, adapted to the local Qwen2 tuple outputs:

- index `out[0]` as logits
- pass `past_key_values` using the local `Qwen2KVCache`
- keep the same block-size assumption and fail clearly if `text_config.block_size != n_future_tokens`

## Processor Contract

The processor remains runtime-light:

- PIL and numpy are allowed through current runtime dependencies.
- MLX arrays are produced at the model boundary because LocateAnything execution is an MLX feature.
- A tokenizer object is supplied by the caller/tests and must expose the small interface used here: encode/call for tokenization, `decode` or `batch_decode`, and image-token id lookup.
- No hard `transformers` import.

Preprocess creates a context object that includes the `SpatialTransform`, original image size, model image size, image grid, and prompt metadata. Postprocess consumes generated token ids plus that context and returns `Result`.

Coordinates from grounding tokens are normalized `[0, 1000]`; convert them first into model-space pixels using the resized model dimensions, then invert through `SpatialTransform` into original pixels.

## Verification Strategy

Use focused tests before full parity:

1. Pure model construction/scatter/projector tests with tiny configs.
2. PBD sampling tests with deterministic fake logits/model behavior.
3. Processor preprocess/postprocess tests using a fake tokenizer.
4. Conversion tests for model-level, MoonViT, Qwen2, and projector keys.
5. Tiny end-to-end integration fixture/test proving `preprocess -> model/pbd or cached generation -> postprocess -> Result`.

The final gate is `uv run pytest`, while reference-mint commands stay out-of-band and must not add runtime dependencies.
