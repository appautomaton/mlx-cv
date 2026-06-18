# DESIGN: DA3 Multi-View Checkpoint Gate

Change: `2026-06-16-depth-anything-v3-multiview-checkpoint`

## Architecture Approach

Keep the production runtime MLX-native and treat upstream DA3, Torch, OpenCV, torchvision, and Hugging Face Hub as test/tool dependencies only. The runtime surface extends the existing DA3 package from one-image monocular depth to any-view image-set geometry; the checkpoint resolver, upstream capture, and parity comparison live outside `src/mlx_cv` except for reusable converted-weight loaders.

## Checkpoint Boundary

- Primary model id: `depth-anything/DA3-SMALL`.
- Fallback model id: `depth-anything/DA3-BASE`.
- Required files: `config.json`, `model.safetensors`, and provenance metadata.
- Expected cache shape: out-of-git cache under `$MLX_CV_CACHE`, `~/.cache/mlx-cv/da3/`, or explicit env paths.
- Suggested env:
  - `MLX_CV_DA3_MODEL_ID`
  - `MLX_CV_DA3_CHECKPOINT`
  - `MLX_CV_DA3_CONFIG`
  - `MLX_CV_REQUIRE_DA3_GATE=1`

Normal tests may skip real checkpoint gates. Required mode must fail on missing checkpoint, missing config, missing reference deps, skipped capture, or provenance mismatch.

## Runtime Contract

The local API should remain compatible with current monocular DA3 while adding an explicit multi-view path:

- Input: list of still images, optionally with per-view extrinsics `(V,4,4)` or `(V,3,4)` and intrinsics `(V,3,3)`.
- Model tensor: `(B,V,3,H,W)` after preprocessing, with view order preserved.
- Output depth/confidence: view-ordered arrays in original-image coordinates when all processed views share an output size; otherwise an explicit per-view representation is required before execution proceeds.
- Output contract: add `depth_views: list[DepthMap] | None` and `camera_geometry: CameraGeometry | None` to `Result`, preserving existing `depth: DepthMap | None` and `DepthMap.__post_init__` behavior for single-view users.
- Camera geometry: typed extrinsics/intrinsics plus convention and view count. The default DA3 convention is final `w2c` extrinsics after upstream's `affine_inverse(c2w)` step.

Do not overload video/tracker concepts for this phase. Multi-view is an image-set axis.

## Model Admission

DA3 Small/Base is not just the existing monocular DINOv2 + DPT path. The real architecture contract must admit:

- DA3 DINOv2 `vits` or `vitb` with `out_layers=[5,7,9,11]`.
- `alt_start=4`, `qknorm_start=4`, `rope_start=4`, and `cat_token=True`.
- Per-block conditional `qk_norm`: LayerNorm on Q and K per head for blocks at or after `qknorm_start`.
- Per-block conditional DA3 2D RoPE: upstream `RotaryPositionEmbedding2D` with `frequency=100`, distinct from the existing DINOv3 axial RoPE implementation.
- Alternating local/global attention from `alt_start`: even blocks operate per-view as `(B*V,N,C)`, and odd blocks operate cross-view as `(B,V*N,C)`.
- `camera_token` parameter `(1,2,embed_dim)` injected at `alt_start`, replacing cls tokens with reference/source camera tokens.
- Reference-view selection and reorder/restore behavior, exercised with three views chosen to force a non-first reference view when feasible.
- `cat_token=True` output concatenation: local and global features are concatenated so the head sees `embed_dim*2`; only the second half receives final norm.
- Multi-view feature layout `(B,V,N,C)`, including the view token/reference-view behavior needed by upstream.
- `DualDPT` depth/confidence/ray head. This is a new module, not a configuration of the existing `DPTHead`: it has independent main and auxiliary fusion chains, UV positional embeddings, 2-channel depth/confidence output, and 7-channel ray/ray-confidence auxiliary output.
- `CameraEnc` for pose-conditioned inputs and `CameraDec` for predicted extrinsics/intrinsics.
- Camera utilities: pose encoding/decoding, scalar-last quaternion conversion, FOV/intrinsics conversion, and affine inversion.

Unsupported branches such as Gaussian splatting should be rejected by the loader unless explicitly required by the selected checkpoint for depth/camera inference.

## Parity Strategy

Use one fixed tiny three-view input across upstream and local MLX. Choose the views to force non-first reference-view selection when feasible so reorder/restore logic is exercised. Capture enough taps to localize drift before comparing final postprocessed outputs:

- preprocessing output shape and normalization metadata
- tokens after preprocessing and before the first block
- tokens around reference-view selection and camera-token injection at `alt_start`
- selected DINOv2 intermediate features, including `cat_token` outputs before head input
- DualDPT main depth/confidence logits and auxiliary ray/ray-confidence logits
- camera decoder 9D pose encoding before geometry conversion
- final depth/confidence/extrinsics/intrinsics

The parity gate should run under deterministic settings as much as upstream allows, prefer CPU/float32 for reproducibility when feasible, and print checkpoint evidence in required mode. Tolerances must be explicit and justified by measured drift.

## Demo Evidence

The final slice should produce a local artifact outside git, for example:

- `/tmp/mlx-cv-da3-demo/da3-small-depth-view-*.png`
- `/tmp/mlx-cv-da3-demo/da3-small-camera-summary.json`
- `/tmp/mlx-cv-da3-demo/da3-small-parity-summary.json`

This is not a replacement for parity, but it makes the delivered result inspectable.
