# DESIGN: LocateAnything Qwen2.5 LLM Backbone

## Architecture Approach

Implement Qwen2 as an MLX language-backbone package:

`Qwen2Config -> Qwen2ForCausalLM(Qwen2Model(embed -> layers -> norm), tied logits) -> KVCache + block masks -> fixture parity`

The source of truth is:

- `references/LocateAnything-3B/configuration_qwen2.py`
- `references/LocateAnything-3B/modeling_qwen2.py`
- `references/LocateAnything-3B/mask_sdpa_utils.py`
- existing `src/mlx_cv/models/locateanything/config.py`
- existing `src/mlx_cv/models/locateanything/convert.py`

Excluded here: MoonViT, multimodal projector, visual feature scatter, tokenizer/chat template, PBD generation, FlashAttention, MagiAttention, and training losses.

## Decisions

1. **Canonical config location.** Create `backbones/llm/qwen2/config.py` as the canonical Qwen2 config home. `models/locateanything/config.py` imports or aliases that dataclass from the config submodule, not from the package root, so Stage-1 LocateAnything imports stay mlx-free. The dataclass keeps current LocateAnything defaults and adds model-body fields needed by the reference path: `hidden_act`, `attention_dropout`, `use_cache=False`, `pad_token_id`, `text_mask_token_id`, and output flags where needed. It also records the supported local attention implementation as SDPA/manual additive masks; Magi remains a reference/default value outside this change. `LocateAnythingConfig.text_mask_token_id` must either be the same single source or be asserted equal to `text_config.text_mask_token_id` so mask dispatch does not split across two token-id truths.

2. **Module shape and parameter paths.** Implement both `Qwen2Model` and `Qwen2ForCausalLM`. The wrapper has `model` and a tied-logit projection using `model.embed_tokens.weight`, so `language_model.model.*` reference keys still map naturally when the later full VLM owns a `language_model` submodule. `language_model.lm_head.weight` remains dropped because it is tied.

3. **Low-level leaves.** Qwen2 gets its own LLM leaves rather than forcing them into the ViT leaf family: `Qwen2RMSNorm`, `Qwen2MLP` (SwiGLU), `Qwen2RotaryEmbedding` or equivalent RoPE helpers, and `repeat_kv`. These live under `backbones/llm/qwen2/` because they carry decoder-specific shapes and cache semantics. Bias layout is part of parity: `q_proj`, `k_proj`, and `v_proj` have bias; `o_proj`, `gate_proj`, `up_proj`, and `down_proj` do not; RMSNorm has only a weight.

4. **Attention path.** Implement a deterministic manual SDPA path using additive 4D masks shaped `(B, 1, Q, K)`. GQA is handled by repeating KV heads after cache update, matching the eager/reference-SDPA behavior. Dropout is configured but no-op in inference tests. Fixture minting must force the reference to SDPA/fp32; the reference `config.json` Magi path uses range dictionaries and optional CUDA-only dependencies, so it is not a comparable oracle for this local path.

5. **Position IDs and RoPE.** The RoPE implementation follows the reference half-rotation layout. Full forward defaults to monotonic `position_ids`; explicit `position_ids` are accepted so block-diff fixture cases can reset position counts. Cache decode uses the caller-supplied position ID or derives it from the cache offset.

6. **Cache.** Use a simple append-only per-layer cache for this change. It stores un-repeated key/value tensors shaped `(B, kv_heads, T, head_dim)` and returns the concatenated sequence before GQA repetition. Rotating/sliding/quantized caches are deferred.

7. **Block masks and dispatch.** Port `find_prefix_seq_length_by_pe`, `create_block_diff_mask_by_pe_4d`, `update_causal_mask_with_pad_non_visible_2d`, and `update_causal_mask_for_one_gen_window_2d` into a local helper module. `find_pred_pos_from_input_ids` stays deferred because it is only used by the reference training/loss path, which is an anti-goal here. Also reimplement the base 4D causal additive mask shape/semantics that the reference gets from `transformers._prepare_4d_causal_attention_mask`; final parity validates it against fixture-captured masks by visibility, not by depending on a live transformers private utility at test time. Slice 3 tests are self-contained and use hand-constructed expected arrays or local numpy recomputation, not torch-generated fixtures that are minted later. Model forward wires only inference branches for this change: no-cache inference mask plus `update_causal_mask_with_pad_non_visible_2d`, AR early return for one-token/non-mask decode, and cache generation-window mask plus `update_causal_mask_for_one_gen_window_2d`. The training block-diff mask remains helper-only coverage until training is in scope. Tests compare fixed numpy/MLX masks by masked visibility, treating both `-inf` and dtype `finfo.min` as masked because the reference mixes transformer base-mask values with LocateAnything helper `-inf` values.

8. **Fixture parity.** Add `tools/mint_qwen2_fixture.py` before the first parity test for out-of-band torch/reference execution. The mint host needs torch plus a compatible transformers version because the local reference imports private transformers APIs such as `_prepare_4d_causal_attention_mask`; the tool records torch/transformers versions and seed in fixture provenance. The committed activations fixture and weights fixture must be minted atomically from one seeded reference instance so local no-cache parity and later loaded-weight parity use identical weights. The tiny config uses the same architecture axes: multiple layers, `num_attention_heads > num_key_value_heads`, non-default `rope_theta`, `block_size`, `causal_attn=False`, `use_cache=False` by default, `batch_size=1`, `attention_mask=None`, and fixed position IDs that exercise a prefix drop. The mint tool must force SDPA and fp32, include a no-cache SDLM input that ends with `text_mask_token_id` so the non-AR no-cache mask helper runs, include an AR cache-step case whose last token is not `text_mask_token_id` so one-token decode can equal the full-sequence suffix, and keep SDLM generation-window masking as a separate mask fixture. The fixture includes weights, input IDs, masks, no-cache hidden/logits, AR cache-step hidden/logits, SDLM mask helper outputs, and evidence that tied logits make dropping `lm_head.weight` safe.

9. **Dependency boundary.** Runtime code never imports torch/transformers. The mint tool may import the reference and torch, but tests consume only `.npz` artifacts. `models/locateanything` keeps exporting Stage-1 mlx-free config/convert/decode names.

## Public Surface

Planned exports from `mlx_cv.backbones.llm.qwen2`:

- `Qwen2Config`
- `Qwen2Model`
- `Qwen2ForCausalLM`
- `Qwen2KVCache` / cache creation helper
- `convert_qwen2_state_dict`
- `load_qwen2_weights`
- `build_qwen2`

The registry key should be `"qwen2.5-3b"` with `kind="llm"`. Tests should assert it is listed only under the LLM kind.

Import discipline: `mlx_cv.backbones.llm.qwen2.config` must stay mlx-free, and Python executes `qwen2/__init__.py` before any `qwen2.config` import. Therefore the package root `__init__.py` must also stay mlx-free and may only re-export `Qwen2Config` eagerly. `Qwen2Model`, `Qwen2ForCausalLM`, `build_qwen2`, cache, and convert symbols are imported through submodules or lazy access, and concrete registry side effects occur only when `mlx_cv.backbones.llm.qwen2.modeling` is imported.

Registration ownership: the concrete `build_qwen2` builder is introduced once the model exists, not as a placeholder. Importing `mlx_cv.backbones.llm.qwen2.modeling` registers the builder; importing `mlx_cv.backbones.llm.qwen2` or `mlx_cv.backbones.llm.qwen2.config` must not import `mlx` or register the builder.

## Data and Taps

Fixtures:

- `tests/fixtures/qwen2_tiny_fixture.npz`
- `tests/fixtures/qwen2_tiny_fixture_weights.npz`

Suggested tap order:

1. `embed_tokens`
2. `layer_00.input_layernorm`, `layer_00.self_attn`, `layer_00.post_attention_layernorm`, `layer_00.mlp`
3. repeat per tiny layer
4. `norm`
5. `logits`
6. `cache_step.logits`

`bisect` support is useful but not mandatory unless parity drift appears during execution; the fixture should still store enough named arrays to localize failures.

## Verification Notes

Run Qwen2 parity on a deterministic CPU path and keep the fixture tiny enough to commit. Final close-out should include focused Qwen2 tests, existing LocateAnything Stage-1 tests, the full suite, a pyproject dependency guard, and a core/import mlx-free smoke.
