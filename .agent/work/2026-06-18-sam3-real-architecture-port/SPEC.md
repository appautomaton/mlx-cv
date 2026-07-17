# SPEC: Official SAM 3.1 MLX Port (Image + Multiplex Video)

Change: `2026-06-18-sam3-real-architecture-port`

## Objective

Replace every SAM 3.0/reduced/Transformers-derived runtime path with one faithful MLX implementation of the official SAM 3.1 architecture. Convert the official `sam3.1_multiplex.pt` checkpoint once into final-layout BF16 Safetensors, run image and video inference on MLX Metal, and promote both release gates only after measured parity against the official PyTorch source.

## Source of truth

- Architecture and inference behavior: `references/sam3` official repository.
- Source checkpoint: `models/sam3-video/upstream/sam3.1_multiplex.pt` plus its `config.json`.
- Checkpoint contract: 1623 tensors: 1166 `detector.*`, 457 `tracker.*`; 1591 float32 tensors and 32 complex64 RoPE tables.
- Reference inference: official `build_sam3_image_model` and SAM 3.1 multiplex video builder/predictor. Transformers SAM 3.0 is not authoritative for this change.

## Runtime contract

- MLX-native inference only; no Torch, Transformers, or official-source imports under `src/mlx_cv/`.
- Production execution is Metal with BF16 learned/real weights. The 32 complex RoPE tables remain complex64.
- A single final checkpoint serves both image and video:
  `models/sam3.1/mlx/sam3.1-multiplex-bf16.safetensors`.
- Normal loading is direct `mx.load()` into final MLX names/layouts. Runtime loading performs no NumPy materialization, key remapping, convolution transpose, or source-checkpoint conversion.
- Runtime loaders reject `.pt`, `.npz`, wrong metadata, wrong tensor names/shapes, and wrong dtypes.

## Public API

- Canonical image API: `SAM3Model`, `SAM3Processor`, `load_sam3_weights`.
- Canonical video API: `SAM3VideoModel`, `SAM3VideoSession`, `load_sam3_video_weights`.
- Versionless `SAM3*` names always mean the latest supported official SAM 3.1 implementation.
- Duplicate `Sam3*`, reduced-model, and `real_*` public exports are removed after parity cutover; there is no SAM 3.0 compatibility alias.

## Acceptance criteria

1. The source checkpoint inventory is exact and the reference harness records deterministic image and short-video captures with source/config/input checksums.
2. The image model covers all 1166 detector tensors with exact name/shape mapping and runs the official image preprocessing and output contract.
3. The video model covers all 457 tracker tensors and implements the official 1008px, stride-14, seven-memory, dynamic 16-object multiplex behavior.
4. The converter writes one atomically published Safetensors file with final MLX layouts, 1591 BF16 tensors, 32 complex64 tensors, and verified provenance metadata; a clean reload is tensor-identical to the converted in-memory state.
5. Metal BF16 parity passes: image/video object and frame identity exact; mask IoU >= 0.98; box error <= 2 pixels; score error <= 0.02; multiplex bucket assignment exact.
6. Only after both real parity gates pass, old 1797-tensor SAM 3.0 code, HF gates, NPZ production loading, duplicate APIs, and local SAM 3.0 weights are deleted. Full tests and diff hygiene pass, and both release entries become `UPSTREAM_PASSED`.

## Out of scope

- Training, loss functions, matchers, dataset pipelines, CUDA/FA3 optimization, and the SAM3 Agent/MLLM wrapper.
- Maintaining or improving SAM 3.0.
- Committing multi-gigabyte model weights to Git.

## Anti-goals

- Do not retrofit SAM 3.1 weights into the incompatible 1797-tensor SAM 3.0 tree.
- Do not keep a production NPZ fallback or perform conversion during normal loading.
- Do not loosen parity into a structural-only or synthetic pass.

## Execution constraints

- Execute serially and continue automatically between verified slices.
- Never claim parity from structural checks or synthetic fixtures.
- Old code and weights remain available until the final real-checkpoint gate passes, providing rollback during migration.
