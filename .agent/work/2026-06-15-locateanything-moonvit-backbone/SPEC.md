# SPEC: LocateAnything MoonViT Vision Backbone

Change: `2026-06-15-locateanything-moonvit-backbone` - Stage: frame - Source: `ROADMAP.md` Phase 4, Qwen2 verified state, `docs/ARCHITECTURE.md` Section 16, `references/LocateAnything-3B/modeling_vit.py`, `references/LocateAnything-3B/image_processing_locateanything.py`, and `references/mlx-vlm/mlx_vlm/models/locateanything/vision.py`

## Bounded Goal
Implement the LocateAnything MoonViT-SO-400M vision backbone in MLX - including packed patch input, learnable 2D interpolated position embeddings, complex 2D RoPE, per-image block attention, fused `wqkv` encoder layers, 2x2 patch merge, convert/load rules, and tiny reference parity - while leaving projector, image-token scatter, processor, PBD generation, and the full `LocateAnythingModel` for later Phase 4 slices.

## Broader Intent
This is the next bounded Phase 4 dependency after the verified Qwen2.5 backbone. It closes the highest-risk vision module called out in `docs/ARCHITECTURE.md` before assembling MoonViT + projector + Qwen2.5 into a full grounding VLM.

## Roadmap Reality Check
- Before this frame, state was `verified` for `2026-06-15-locateanything-qwen2-backbone`; the Qwen2 sub-change is complete.
- Phase 4 remains `pending` because the full VLM anchor is not complete.
- The roadmap now records MoonViT-SO-400M as the current framed change, excluding projector/image-token scatter, PBD generation, full model, and processor work.
- No roadmap rewrite is needed for this frame. Per the local roadmap contract, this frame does not activate or create a roadmap phase; it creates the next bounded spec under the existing Phase 4.

## Work Scale / Shape / Lenses
- Scale: capability slice inside Phase 4.
- Shape: parity-first model implementation plus conversion/load support.
- Lenses: engineering (primary: numerical parity, packing/mask correctness, conversion paths), runtime (secondary: variable-resolution token counts and MLX attention behavior), product (light: future VLM assembly ergonomics).

## Constraints & Risks
- `core/` stays mlx-free. MoonViT runtime code lives under `src/mlx_cv/backbones/vision/moonvit/`; `models/locateanything` config-only imports must remain importable without `mlx`.
- Runtime package dependencies stay clean: no `torch` or `transformers` in `pyproject.toml`. Fixture minting may import the local PyTorch reference out of band from `tools/`.
- The input contract is packed patches, not ordinary batched images: `image_processing_locateanything.py` emits `pixel_values` shaped as concatenated `(num_patches, C, patch_size, patch_size)` and `image_grid_hws` shaped `(num_images, 2)`.
- Patch grids must remain compatible with the reference processor: images are resized to patch grids below 512 and dimensions are rounded to multiples of `merge_kernel_size * patch_size`, so grid height/width are divisible by the 2x2 merger.
- Attention must preserve per-image isolation through cumulative sequence lengths or an equivalent boolean block mask. Tokens from different images must never attend to each other in the packed sequence.
- Parameter paths should stay compatible with reference keys under `vision_model.*` and the merged mlx-vlm `VisionModel.sanitize()` behavior. Conv weights require PyTorch `(O, I, kH, kW)` to MLX `(O, kH, kW, I)` handling when loaded from PyTorch-style fixtures.
- `docs/ARCHITECTURE.md` says MoonViT is the hard part of LocateAnything because native variable resolution, 2D RoPE, per-image block attention, and 2x2 merge all have to match numerically.
- Tiny fixture parity is the proof target. Real 400M weights may be loadable later but are not required or committed here.

## Required Outcome

**Structural change**
- `src/mlx_cv/backbones/vision/moonvit/` gains a canonical config, modeling implementation, packing/mask helpers, convert/load rules, and exports.
- `models/locateanything/config.py` reuses or re-exports the canonical MoonViT config so LocateAnything and the backbone cannot drift.
- The registered vision builder constructs a MoonViT backbone with patch embed, learnable 2D interpolated position embedding, encoder blocks, final LayerNorm, and 2x2 patch merger.
- The implementation accepts packed patch tensors plus `grid_hws`, returns one merged token tensor per image, and preserves reference token ordering through patch embed and merge.
- `tools/mint_moonvit_fixture.py` mints a fixed-seed tiny PyTorch reference fixture and matching tiny weights fixture atomically, including intermediate taps that can localize drift: patch-embed output, RoPE frequencies, block-attention visibility, first encoder layer output, final norm output, and merged outputs.
- Convert/load rules map tiny reference MoonViT weights into the local MLX module tree through the shared conversion engine where practical, including conv weight transpose and any reference-to-local path renames.

**Behavioral invariants**
- `import mlx_cv.core` and `import mlx_cv.models.locateanything` do not import `mlx`.
- Existing LocateAnything config/convert/decode and Qwen2 tests stay green.
- The package still has no runtime `torch` or `transformers` dependency.
- Multi-image packed inputs preserve image boundaries in attention and output splitting.
- A later VLM assembly slice can concatenate merged image features into the projector without redefining MoonViT output shape.

**Parity target**
- Reference oracle: `references/LocateAnything-3B/modeling_vit.py` using a tiny config forced to fp32 SDPA behavior on this host, with deterministic weights and packed patch inputs covering at least two image grids. Do not mint through the reference eager path: it adds a boolean mask to attention logits without `-inf`, so multi-image packed tokens are not isolated.
- Fast oracle: `references/mlx-vlm/mlx_vlm/models/locateanything/vision.py` may be read as a same-framework sanity check, but the local implementation must not depend on it at runtime.
- Conformance: MLX MoonViT patch embed, RoPE frequencies, block attention visibility, encoder output, final norm output, merged per-image outputs, and loaded-weight forward all match the tiny reference fixture within tolerance.

## Acceptance Criteria
1. Canonical MoonViT config defaults match LocateAnything-3B values (`hidden_size=1152`, `num_hidden_layers=27`, `num_attention_heads=16`, `intermediate_size=4304`, `patch_size=14`, `num_channels=3`, `init_pos_emb_height=64`, `init_pos_emb_width=64`, `merge_kernel_size=(2, 2)`) and are reused by `LocateAnythingConfig`.
2. Packed patch input is implemented and tested from the processor contract: `(sum_patches, C, patch_size, patch_size)` plus `grid_hws` maps to `(sum_patches, hidden_size)` in reference token order.
3. Learnable 2D interpolated position embeddings match the PyTorch reference for same-size and interpolated grids, including concatenation across multiple images.
4. Complex 2D RoPE precompute, slicing, concatenation, and `apply_rope` match the reference for multiple grid shapes and validate unsupported shapes/dimensions.
5. Per-image block attention mask or `cu_seqlens` behavior prevents cross-image attention and matches reference visibility for packed multi-image inputs.
6. MoonViT encoder layer math matches the reference for fused `wqkv`, attention bias, LayerNorm eps/defaults, GELU-tanh MLP, residual order, and final LayerNorm.
7. The 2x2 patch merger returns one tensor per image with the same order and shape as the reference, and rejects or clearly errors on grids not divisible by the merge kernel.
8. Convert/load rules map tiny reference weights into the local MLX module tree, including conv weight layout handling, and loaded MoonViT output parity passes against the fixture.
9. The MoonViT builder registers in `BACKBONES` with `kind="vision"` when the MLX modeling surface is imported; config-only/package-root imports remain mlx-free.
10. Focused and full tests pass; `core/` remains mlx-free by regex/import smoke; `pyproject.toml` contains no `torch` or `transformers`.

## Anti-Goals
- Multimodal projector, image-token scatter, full `LocateAnythingModel`, tokenizer/chat template, LocateAnything image processor, prompt formatting, PBD generation, or end-to-end grounding outputs.
- Qwen2 changes, cache/mask changes in the language backbone, or any DA3/RF-DETR/SAM work.
- FlashAttention parity, MagiAttention, training losses, optimizer/training support, quantization, or LoRA.
- Downloading or committing real LocateAnything-3B or MoonViT-SO-400M checkpoint weights.
- Roadmap rewrite or Phase 4 status promotion.

## Scope Coverage
- Included: MoonViT vision backbone implementation, packed patch/token contracts, conversion/load, tiny fixtures, parity tests, registration, and import/dependency guards.
- Deferred: projector/image-token scatter, processor, PBD, full VLM model assembly, user-facing `predict`, real checkpoint smoke, and quantization.
- Needs-decision: none for this frame. The next plan should resolve exact file/module names and fixture dimensions from the references before execution.

## Assumptions
- The local PyTorch reference is the normative source for MoonViT behavior.
- The merged mlx-vlm implementation is useful evidence but not a dependency.
- Tiny fixed-seed fixture parity is sufficient proof for this change, consistent with prior DINOv3, DA3, and Qwen2 phases.
- Tolerance can start at fp32 `atol=1e-4` and tighten or loosen only with fixture-backed evidence.
