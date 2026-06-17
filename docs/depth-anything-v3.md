# Depth Anything V3

`mlx-cv` implements the monocular DA3 path and the DA3-SMALL any-view
multi-view depth/camera path without bundling DA3 checkpoints. The package path
depends on MLX and NumPy/Pillow only; torch and the upstream DA3 reference are
used solely by out-of-band fixture, conversion, and parity tools under `tools/`.

Current supported DA3 surface:

- Monocular DINOv2 + DPT tiny parity fixture.
- DA3-SMALL multi-view preprocessing for fixed still-image sets.
- Strict local load of converted DA3-SMALL MLX weights from an out-of-git cache.
- Multi-view depth, confidence, and camera extrinsics/intrinsics through the
  shared `Result` depth/camera fields.
- Required upstream-vs-local parity tooling for fixed three-view inputs:
  `tests/test_da3_upstream_parity.py` compares depth, confidence, extrinsics,
  intrinsics, and selected aux taps with explicit tolerances, while
  `tools/da3_demo.py` writes `/tmp/mlx-cv-da3-demo/` depth PNGs plus camera and
  parity JSON summaries.

Deferred or unsupported DA3 branches:

- Streaming/video input.
- `NestedDepthAnything3Net` metric scaling.
- Metric-only and mono-large presets outside the selected DA3-SMALL any-view
  contract.
- 3DGS/Gaussian splatting heads, adapters, and exports.

The required upstream-vs-MLX DA3 parity gate needs an MLX Metal device and is
env-gated. In the completed pass it ran outside the managed sandbox against the
local DA3-SMALL checkpoint and wrote `/tmp/mlx-cv-da3-demo/` evidence.

Checkpoint licensing differs by upstream weight family. DA3-SMALL and DA3-BASE
weights are published under Apache-2.0, while LARGE/GIANT weights are published
under CC-BY-NC-4.0. Users must fetch and use checkpoints under the applicable
upstream license.
