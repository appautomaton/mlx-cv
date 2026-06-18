# DESIGN: Depth Anything V3 Monocular

## Architecture Approach

Implement the monocular DA3 path as:

`processor -> DINOv2ViT -> DPTHead -> DepthAnythingV3Model -> processor.postprocess -> Result(depth=DepthMap(...))`

The source of truth is the local reference clone:

- `references/Depth-Anything-3/src/depth_anything_3/model/da3.py`
- `references/Depth-Anything-3/src/depth_anything_3/model/dinov2/`
- `references/Depth-Anything-3/src/depth_anything_3/model/dpt.py`
- `references/Depth-Anything-3/src/depth_anything_3/configs/da3mono-large.yaml`

Excluded from this design: camera encoders/decoders, multi-view attention, `DualDPT`, `GSDPT`, Gaussian export, metric alignment, and checkpoint downloads.

## Decisions

1. **DINOv2 feature API.** Extend `DINOv2ViT.forward_features` through the existing `ViTBackbone` path with selected intermediate layers. Selected block outputs are normalized with the final norm, stripped of cls/register tokens, and stored in `BackboneFeatures.intermediates` as ordered `FeatureMap(layout=BNC)` entries. This mirrors DA3 reference `get_intermediate_layers`. The DA3 mono fixture config must use zero register tokens, disabled rope/alt/cat-token branches, packed qkv, and a reference-matched final norm eps or a locked tap-delta test proving the eps difference is tolerated.

2. **Convert engine.** Keep `hub.convert` declarative, but add the minimum rule types needed by DA3 weights: prefix rename for wrapper paths such as `pretrained.`, exact key rename for `pos_embed -> pos_embed.table`, and path-specific tensor layout rules. PyTorch `Conv2d` weights map OIHW to MLX OHWI with `(0, 2, 3, 1)`. PyTorch `ConvTranspose2d` weights map IOHW to MLX OHWI with `(1, 2, 3, 0)`. Do not add split-q/k/v packing unless a fixture proves split keys exist; DA3 vendored DINOv2 mono and the local attention path both use packed qkv.

3. **DPT head.** Add `heads/dense/dpt.py` as a parameterized MLX DPT family. It consumes exactly four ViT intermediates, reshapes patch tokens to the patch grid, applies per-stage 1x1 projections, DA3 resize stages `x4, x2, identity, stride2`, top-down fusion, output upsample, and activations. The fixture config pins `use_sky_head=False`, `pos_embed=False`, `down_ratio=1`, `norm_type="idt"`, and `output_dim=2`. The head supports `output_dim=1` and `output_dim=2`; `output_dim=2` splits depth and confidence. MLX bilinear resize must match reference `align_corners=True`.

4. **Confidence and sky-head mismatch in reference configs.** Stock `da3mono-large.yaml` uses `DPT(output_dim=1)` and omits `use_sky_head`, whose reference default is `True`. That path has no `depth_conf` and may emit `sky`, after which `DepthAnything3Net` can alter `depth` through mono sky postprocessing. The committed tiny parity fixture will use the same DPT code path with `output_dim=2` and `use_sky_head=False`; this produces a synthetic confidence channel for the library proof without claiming stock DA3Mono-Large checkpoint confidence. The processor still accepts missing confidence and maps it to `DepthMap.depth_conf=None`.

5. **Processor dependency boundary.** Do not add OpenCV. The parity image is square, patch-divisible, and uses `process_res` equal to the image size, so reference preprocessing performs no resize for the golden forward. The tiny DINOv2 fixture also sets `pretrain_grid == runtime_grid` to avoid the known position-embedding interpolation divergence between reference bicubic with offset and the current MLX cubic path. General processor tests cover aspect-preserving resize, ImageNet normalization, and `SpatialTransform.invert_depth`/`invert_dense` for depth and confidence.

6. **Runtime dependencies.** `torch`, `transformers`, and the DA3 reference remain out-of-band. Fixture minting lives under `tools/` and may import torch/reference code; `src/mlx_cv/` and `pyproject.toml` must not.

## Data and Taps

Fixtures:

- `tests/fixtures/dinov2_da3_tiny_fixture.npz`
- `tests/fixtures/dinov2_da3_tiny_fixture_weights.npz`
- `tests/fixtures/da3_monocular_tiny_fixture.npz`
- `tests/fixtures/da3_monocular_tiny_fixture_weights.npz`

Tap order for DA3 parity:

1. DINOv2: `patch_embed`, selected `block_##`, final selected intermediates.
2. DPT: projected stages, resized stages, fusion stages, output logits.
3. Outputs: `depth`, `depth_conf`.

`bisect` must report the first differing tap in that order.

## Verification Notes

Run parity on the MLX CPU stream to match the torch CPU oracle. Full-size pretrained checkpoints are not committed; the tiny fixed-seed fixture proves the same code path under the pinned mono fixture config above. Real checkpoint licensing is documented in the model card, but checkpoint download/load is not a gate for this change.
