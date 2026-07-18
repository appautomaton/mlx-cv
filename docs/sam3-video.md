# SAM 3.1 Image and Object Multiplex

SAM 3.1 is the only supported SAM runtime. Image and video inference share one
final-layout BF16 checkpoint:

`models/sam3.1/mlx/sam3.1-multiplex-bf16.safetensors`

The runtime loads it directly with `mx.load()` and strictly validates metadata,
names, shapes, and BF16 dtypes. PT/NPZ conversion is an offline tool operation;
normal inference does not remap names, transpose tensors, or materialize NumPy
checkpoint arrays.

## Public API

- Image: `SAM3Model`, `SAM3Processor`, `load_sam3_weights`
- Video: `SAM3VideoModel`, `SAM3VideoSession`, `load_sam3_video_weights`

`SAM3VideoSession` supports frame arrays, image paths, and frame directories;
box, point, and mask prompts; forward/reverse propagation; object removal; reset;
seven-memory temporal attention; and dynamic 16-object multiplex buckets.

## Verified contract

- Source: official `sam3.1_multiplex.pt`, 1623 tensors
- Final checkpoint: 1963 BF16 parameters (1506 detector + 457 tracker)
- Image: mask IoU 0.999618, box error 0.1626px, score error 0.001305
- Multiplex decoder: mask IoU 0.99215
- Official MPS captures passed for interactive decoding, memory encoding, and memory attention
- Real two-frame propagation completed on MLX Metal

Run the persisted real gate with:

```bash
MLX_CV_REQUIRE_SAM31_GATE=1 \
MLX_CV_SAM31_MLX=models/sam3.1/mlx/sam3.1-multiplex-bf16.safetensors \
.venv/bin/python -m pytest -q \
  tests/test_sam31_image_parity.py \
  tests/test_sam31_video_parity.py
```
