# Depth Anything V3

`mlx-cv` implements the monocular DA3 path without bundling DA3 checkpoints.
The package path depends on MLX and NumPy/Pillow only; torch and the upstream DA3
reference are used solely by out-of-band fixture minting tools under `tools/`.

Checkpoint licensing differs by upstream weight family. DA3 BASE weights are
published under Apache-2.0, while LARGE/GIANT weights are published under
CC-BY-NC-4.0. Users must fetch and use checkpoints under the applicable upstream
license.
