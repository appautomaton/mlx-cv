# SPEC: LocateAnything Qwen2.5 LLM Backbone

Change: `2026-06-15-locateanything-qwen2-backbone` - Stage: frame - Source: `ROADMAP.md` Phase 4, Claude Opus read-only roadmap review, existing LocateAnything Stage-1 code, `references/LocateAnything-3B/`

## Bounded Goal
Implement the LocateAnything Qwen2.5 language backbone in MLX - including RMSNorm, SwiGLU, RoPE, GQA, tied logits, append-only KV-cache, and LocateAnything block-mask support - with fixed-seed tiny reference parity, while preserving the existing mlx-free Stage-1 LocateAnything config/convert/decode surface.

## Broader Intent
This is the first framed Phase 4 change. It closes the language-model building blocks Phase 2 deferred and gives later LocateAnything slices a real decoder foundation before the model expands into MoonViT, projector/image-token scatter, tokenizer, and PBD generation.

## Work Scale / Shape / Lenses
- Scale: capability (Phase 4 foundation for LocateAnything).
- Shape: mixed - feature implementation plus parity closure against the local reference.
- Lenses: engineering (primary: model correctness, cache/mask semantics, dependency boundaries), runtime (secondary: cache and mask behavior), product (light: future model author ergonomics).

## Constraints & Risks
- `core/` stays mlx-free. Qwen2 code lives under `src/mlx_cv/backbones/llm/qwen2/`, and `models/locateanything` must remain importable without `mlx`.
- Runtime package dependencies stay clean: no `torch` or `transformers` in `pyproject.toml`; fixture minting may import the reference out of band from `tools/` and requires a mint host with `torch` plus a compatible `transformers` version that still exposes the private Qwen2 reference APIs used by `references/LocateAnything-3B/modeling_qwen2.py`.
- Preserve existing LocateAnything Stage-1 behavior: config defaults, token IDs, `convert_state_dict`, and PBD token parser tests must keep passing.
- Parameter paths should stay compatible with reference keys under `language_model.model.*`; the local LM head is tied to embeddings, matching the existing converter drop of `language_model.lm_head.weight`.
- Custom block-mask semantics are required, but FlashAttention, MagiAttention, training losses, and full PBD generation are out of scope.
- Reference `config.json` defaults to `_attn_implementation="magi"` and `torch_dtype="bfloat16"`, but the local parity oracle for this change must force the reference into an SDPA/fp32 path so masks, cache, and tolerances are comparable on this host.
- The tiny parity fixture is the proof target; real 3B weights may be loadable later but are not required or committed here.

## Required Outcome

**Structural change**
- `src/mlx_cv/backbones/llm/qwen2/` gains a canonical config, model implementation, cache helper, mask helper, convert/load rules, and exports.
- `models/locateanything/config.py` reuses or re-exports the canonical Qwen2 config so defaults cannot drift between the model package and the backbone package.
- The registered LLM builder constructs a Qwen2 language model with `embed_tokens`, decoder layers, final RMSNorm, and tied-logit projection.
- Qwen2 attention supports GQA (`num_attention_heads != num_key_value_heads`), additive 4D masks, position IDs, and append-only KV-cache.
- LocateAnything block-mask helpers are ported with self-contained deterministic tests before torch-oracle fixtures exist, then no-cache/cache inference mask dispatch is wired through model forward once fixtures are minted.
- `tools/mint_qwen2_fixture.py` mints the tiny reference activations fixture and tiny weights fixture atomically from the same seeded SDPA/fp32 reference instance before the first local parity assertion, recording seed plus torch/transformers provenance; later tests load those fixtures through local convert rules.

**Behavioral invariants**
- `import mlx_cv.models.locateanything` and `import mlx_cv.core` do not import `mlx`.
- Existing LocateAnything Stage-1 tests (`config`, `convert`, `decode`) stay green.
- The full package still has no runtime `torch` or `transformers` dependency.
- Qwen2 accepts both `input_ids` and `inputs_embeds`; later VLM assembly can scatter image features into embeddings without redesigning the decoder.

**Parity target**
- Reference oracle: `references/LocateAnything-3B/modeling_qwen2.py` using a tiny config forced to SDPA/fp32, fixed input IDs, fixed position IDs, deterministic weights, `batch_size=1`, `attention_mask=None`, and selected additive masks.
- Conformance: MLX Qwen2 hidden states, tied logits, no-cache forward, and one-step cache suffix match the fixture within tolerance. Mask helper outputs match reference-generated masks exactly or within dtype-compatible equality for `0/-inf` additive masks.

## Acceptance Criteria
1. Canonical Qwen2 config defaults match LocateAnything-3B values (`hidden_size=2048`, `layers=36`, `heads=16`, `kv_heads=2`, `intermediate=11008`, `vocab=152681`, `rope_theta=1000000`, `block_size=6`, `causal_attn=False`, `use_cache=False`) while pinning the supported local attention implementation to SDPA/manual additive masks instead of Magi, and are reused by `LocateAnythingConfig`.
2. RMSNorm, SwiGLU MLP, Qwen-style RoPE, GQA attention, and the Qwen2 projection bias layout (`q/k/v` bias, `o/gate/up/down` no bias, RMSNorm weight only) have focused unit tests against reference formulas, hand-constructed arrays, or local numpy recomputation before the torch-oracle fixture is minted.
3. Qwen2 model no-cache forward returns hidden states and tied logits matching the tiny reference fixture, whose no-cache input ends with `text_mask_token_id` and contains a mask token so the non-AR no-cache mask dispatch is actually exercised.
4. Append-only KV-cache one-token AR decode matches the corresponding full-sequence suffix for hidden states/logits, with correct RoPE position IDs and mask width; SDLM generation-window cache masking is covered as a separate mask fixture, not by the AR equality assertion.
5. LocateAnything block-mask helpers and model-forward mask dispatch reproduce the reference mask behavior for fixed `position_ids`, `input_ids`, `text_mask_token_id`, `block_size`, `causal_attn`, no-cache inference, and cache generation-window cases; helper tests before fixture minting use independent expected arrays, and raw mask comparisons treat both `-inf` and dtype `finfo.min` as masked.
6. Convert/load rules map tiny reference Qwen2 weights into the local MLX module tree without imperative key surgery, and the dropped tied `lm_head` remains intentional and is proven by fixture or conversion tests.
7. The Qwen2 builder registers in `BACKBONES` with `kind="llm"` when the MLX modeling surface is imported; package-root/config-only imports and Stage-1 LocateAnything imports remain mlx-free.
8. Full focused tests pass; `core/` is mlx-free by regex and import smoke; `pyproject.toml` contains no `torch` or `transformers`.

## Anti-Goals
- MoonViT implementation, multimodal projector, image-token scatter, full `LocateAnythingModel`, tokenizer/chat template, PBD generation loop, or end-to-end grounding outputs.
- FlashAttention, MagiAttention, sliding-window/rotating cache, quantized cache, LoRA, training loss reporting, or optimizer/training support.
- Downloading or committing real 3B checkpoint weights.
- Any DA3 hardening, camera pose/intrinsics, multi-view depth, or unrelated roadmap cleanup.

## Scope Coverage
- Included / Deferred / Anti-goals: per `INTAKE.md` and adopted here.
- Needs-decision: none for this frame. The chosen first Phase 4 slice is Qwen2.5 LLM backbone; later Phase 4 slices can frame MoonViT/projector/PBD independently.

## Assumptions
- The local reference clone is the normative source for LocateAnything Qwen2 behavior.
- Tiny fixed-seed fixture parity is sufficient proof for this change, consistent with prior DINOv3 and DA3 phases.
- Tolerance can start at fp32 `atol=1e-4` and tighten if MLX/reference drift permits.
- The registered backbone may expose both low-level `Qwen2Model` hidden states and a tied-logit wrapper, but generation policy remains deferred.
