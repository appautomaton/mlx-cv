# Depth Anything V3

`mlx-cv` implements two MLX-native Depth Anything V3 paths:

| Path | Model surface | Output |
|---|---|---|
| Monocular | `DepthAnythingV3Monocular` with DINOv2 and a DPT head | one shared `Result.depth` |
| DA3-SMALL any-view | `DepthAnythingV3MultiView` with AnyView DINOv2, DualDPT, and camera decoding | `Result.depth_views` plus confidence and camera geometry |

Both paths are inference-only. Checkpoints are external artifacts and are never
bundled with the Python package.

## Runtime contract

The monocular model accepts one image. The any-view model accepts a fixed set of
still images and may optionally receive paired camera extrinsics and intrinsics
for pose-conditioned inference.

`DA3Processor` owns resize/normalization and maps dense outputs back to each
original view. Multi-view camera extrinsics use the `w2c` convention exposed by
`CameraGeometry`.

Model construction is explicit through `DA3MonocularConfig` or
`DA3MultiViewConfig`. Both model classes now expose package-native
`from_pretrained(...)` constructors and are available through
`mlx_cv.load(...)`. Converted `.npz` and Safetensors weights are supported;
strict parameter coverage is the package-loading default.

## Verification

- The monocular DINOv2 + DPT path has deterministic fixture coverage.
- The DA3-SMALL any-view path passed a real upstream/local comparison for depth,
  confidence, extrinsics, intrinsics, and selected intermediate taps.
- The real gate covers a fixed three-view input, a two-image set, and a
  three-frame set sampled from video.

The durable status and measured tolerance record is
`.agent/work/2026-06-16-release-parity-hardening/parity-status.json`. Generated
captures, converted checkpoints, caches, and other new test artifacts belong
under a system temporary root or another external artifact store, never in Git.
Existing ignored reference checkouts under `references/` remain part of the
local test setup.

## Deliberate limits

- No streaming video API.
- No `NestedDepthAnything3Net` metric-scaling path.
- No metric-only or larger preset promise beyond the admitted DA3-SMALL path.
- No Gaussian-splatting heads, adapters, training, or export pipeline.
