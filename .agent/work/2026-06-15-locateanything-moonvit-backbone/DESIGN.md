# DESIGN: LocateAnything MoonViT Backbone

Change: `2026-06-15-locateanything-moonvit-backbone`

## Architecture Approach

MoonViT should be a standalone packed-patch backbone under `src/mlx_cv/backbones/vision/moonvit/`, not a subclass of the shared `ViTBackbone`.

Reason: the existing shared ViT family starts from ordinary image batches, creates `[cls, storage, patch]` token layouts, and returns `BackboneFeatures`. MoonViT starts from the LocateAnything processor contract: already-patchified packed inputs shaped `(sum_patches, C, patch_size, patch_size)`, one `grid_hws` row per image, no cls/register tokens, per-image block attention over a packed sequence, and a final 2x2 patch merge that returns one tensor per image for the later projector.

The implementation should reuse local patterns where they fit: dataclass config, registry decorator, shared `hub.convert` rules, parity fixture helpers, and import/dependency guards. The packed attention, complex 2D RoPE, reference interpolation, and patch merger stay MoonViT-specific because their shapes and semantics do not match the current build-once ViT leaves.

## Module Layout

- `src/mlx_cv/backbones/vision/moonvit/config.py`
  - canonical `MoonViTConfig`
  - defaults match LocateAnything-3B
  - `from_dict`, `head_dim`, `embed_dim`, `depth`, `num_heads`, and `spatial_merge_size` compatibility properties
- `src/mlx_cv/backbones/vision/moonvit/modeling.py`
  - packed patch shape helpers
  - `Learnable2DInterpPosEmb`
  - `Rope2DPosEmb`, `apply_rope`
  - `make_block_attention_mask`, `cu_seqlens_from_grid_hws`
  - `MoonViTPatchEmbed`, attention math helpers, `MoonViTMLP`, `MoonViTEncoderLayer`, `MoonViTBackbone`
  - `patch_merger`
  - `build_moonvit_so400m` registered as `moonvit-so400m`, `kind="vision"`
- `src/mlx_cv/backbones/vision/moonvit/convert.py`
  - conversion rules for PyTorch reference weights
  - conv transpose `(O, I, kH, kW) -> (O, kH, kW, I)`
  - reference path remaps into the local module tree
- `src/mlx_cv/backbones/vision/moonvit/__init__.py`
  - mlx-free config exports at package root
  - lazy MLX exports via `__getattr__`, matching the Qwen2 package pattern
  - registration happens only when `moonvit.modeling` is imported
- `src/mlx_cv/parity/fixtures.py`
  - `MOONVIT_FIXTURE_CONFIG`
  - `moonvit_fixed_inputs`
  - `moonvit_tap_order`
- `tools/mint_moonvit_fixture.py`
  - out-of-band PyTorch fixture minting only
- `tests/test_moonvit_*.py`
  - focused unit, convert, fixture, parity, and guard tests

## Data Flow

1. Input is packed PyTorch/processor layout:
   - `pixel_values`: `(sum(grid_h * grid_w), C, patch_size, patch_size)`
   - `grid_hws`: `(num_images, 2)`
2. `MoonViTPatchEmbed` transposes packed patches from NCHW to NHWC for `mlx.nn.Conv2d`, applies one full-patch conv per patch, flattens to `(sum_patches, hidden_size)`, then adds per-image learnable 2D interpolated position embeddings.
3. `Rope2DPosEmb` builds complex cis values for each image grid, concatenated in packed sequence order.
4. Attention computes fused `wqkv`, applies complex 2D RoPE to q/k, and runs SDPA or manual attention with per-image block visibility. Cross-image attention must be impossible.
5. Encoder layers use reference residual order:
   - `x = x + attention(norm0(x), cu_seqlens, rope)`
   - `x = x + mlp(norm1(x))`
6. Final LayerNorm produces packed tokens.
7. `patch_merger` splits packed tokens by `grid_hws`, reshapes each image grid into 2x2 windows, flattens each window to `4 * hidden_size`, and returns a list of per-image tensors shaped `(merged_tokens, 4 * hidden_size)`. This follows the PyTorch reference, not the merged mlx-vlm port's `(merged_tokens, 4, hidden_size)` intermediate shape.

## Reference-Matching Decisions

- Input layout: local public backbone accepts NCHW packed patches from the PyTorch processor contract. Internally it transposes for MLX Conv2d. Do not silently adopt the mlx-vlm NHWC call convention as the public contract.
- Interpolation: the PyTorch reference uses bicubic `F.interpolate` over `(1, D, H, W)`. Gate this with a dedicated CPU-capable interpolation tap; default to a small local MLX bicubic helper if `nn.Upsample(mode="cubic")` is not fixture-close.
- Activation: the PyTorch reference uses `PytorchGELUTanh()` for MoonViT. Use MLX tanh-approx GELU (`approx="tanh"` or the equivalent `approx="precise"` on this host), never erf/`none`, and verify on the MLP or block tap.
- Attention mask: prefer the reference-compatible boolean block mask for multi-image packed inputs. Single-image inputs may pass `None` only if output parity proves MLX SDPA semantics match.
- Parameter tree: choose local paths that make standalone MoonViT loading clean. Keep `wqkv` and `wo` as direct encoder-layer attributes unless the plan also generates explicit declarative per-layer rename rules; do not rely on imperative string surgery and do not reshape or split fused `wqkv`.

## Verification Strategy

Early slices use independent deterministic unit tests for shapes, masks, RoPE, merge order, residual order, and import boundaries. The fixture slice mints PyTorch taps and weights from the local reference. The final slice proves the loaded MLX model matches fixture outputs and that bisect reports no first-drift tap.

Risk is concentrated in interpolation, GELU variant, boolean mask semantics in MLX SDPA, and packed patch layout. Each has a dedicated test before the final parity assertion.
