---
library_name: mlx
pipeline_tag: mask-generation
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/sam3/blob/main/LICENSE
base_model:
- facebook/sam3.1
tags:
- mlx
- apple-silicon
- image-segmentation
- video-object-segmentation
- bfloat16
---

# SAM 3.1 Multiplex BF16 for MLX

Final-layout BF16 detector and Object Multiplex tracker weights for running Meta SAM 3.1 with [`mlx-cv`](https://github.com/appautomaton/mlx-cv) on Apple Silicon. BF16 is reduced precision, not integer quantization.

```bash
pip install "mlx-cv[mlx,hub]==0.0.3"
```

```python
from mlx_cv.models.sam3 import SAM3Processor, SAM3VideoSession

image_model = SAM3Processor.from_pretrained("sam3.1")
prediction = image_model.predict(image, "person")

video = SAM3VideoSession.from_pretrained("sam3.1")
```

## Verification

The strict 1963-tensor BF16 checkpoint loads directly into MLX with no runtime PyTorch conversion. The persisted Metal image gate reached mask IoU 0.999618, maximum box error 0.1626 px, and score error 0.001305 against the official reference. The multiplex decoder mask IoU was 0.99215, with official MPS component checks and real two-frame MLX propagation also passing.

## Distribution note

Meta's source Hugging Face repository requires users to accept access terms. This derivative MLX checkpoint is distributed publicly by App Automaton under the bundled SAM License; downloading or using it constitutes acceptance of those terms. Review the complete `LICENSE` before use.

## Limitations

- Inference only, on MLX-supported Apple Silicon systems.
- Segmentation and tracking can fail on ambiguous prompts, occlusion, tiny objects, or domain shift.
- Apple Metal and the official PyTorch/MPS implementation do not share identical kernels; the measured parity thresholds account for bounded numerical disparity.
- Memory and latency depend on frame size, sequence length, object count, and machine.

## Links

- [mlx-cv source](https://github.com/appautomaton/mlx-cv)
- [App Automaton](https://appautomaton.github.io/)
- [Official SAM 3 code](https://github.com/facebookresearch/sam3)
- [Upstream checkpoint](https://huggingface.co/facebook/sam3.1)
