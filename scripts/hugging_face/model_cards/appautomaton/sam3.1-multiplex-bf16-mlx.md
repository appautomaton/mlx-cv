---
library_name: mlx
pipeline_tag: mask-generation
license: other
license_name: sam-license
license_link: LICENSE
base_model:
- facebook/sam3.1
tags:
- mlx
- apple-silicon
- image-segmentation
- video-object-segmentation
- object-tracking
- bfloat16
---

# SAM 3.1 Multiplex — MLX (bf16)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--cv-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-cv)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.renocrypt.com/mlx-cv/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)

MLX-native bf16 conversion of [Meta SAM 3.1](https://huggingface.co/facebook/sam3.1) for text-prompted image segmentation and stateful Object Multiplex video tracking on Apple Silicon. The package contains the complete detector and tracker layout used by [`mlx-cv`](https://github.com/appautomaton/mlx-cv), with no PyTorch conversion at inference time.

## Model Details

- Developed by: [App Automaton](https://appautomaton.renocrypt.com)
- Upstream model: [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)
- Tasks: text-prompted image segmentation and prompt-driven video object tracking
- Architecture: detector, mask decoder, memory encoder, memory attention, and Object Multiplex tracker
- Precision: bf16 checkpoint weights; this is reduced precision, not integer quantization
- Runtime: MLX on Apple Silicon

## Contents

| File | Purpose | Format |
| --- | --- | --- |
| `model.safetensors` | 1,963 final-layout detector and tracker tensors | bf16 Safetensors |
| `config.json` | SAM 3.1 architecture configuration | JSON |
| `bpe_simple_vocab_16e6.txt.gz` | Text prompt tokenizer vocabulary | gzip text |
| `manifest.json` | File sizes and SHA-256 hashes | JSON |

## How to Get Started

```bash
pip install "mlx-cv[mlx,hub]"
hf download appautomaton/sam3.1-multiplex-bf16-mlx \
  --local-dir models/sam3_1_multiplex/mlx-bf16
```

```python
import mlx_cv

image_model = mlx_cv.load(
    "sam3.1-image",
    "models/sam3_1_multiplex/mlx-bf16",
)
segments = image_model.predict(image, "person")

video_model = mlx_cv.load(
    "sam3.1-video",
    "models/sam3_1_multiplex/mlx-bf16",
)
```

The exact remote repository ID can also be used in place of the local path:

```python
image_model = mlx_cv.load(
    "sam3.1-image",
    "appautomaton/sam3.1-multiplex-bf16-mlx",
)
```

## Verification

The strict 1,963-tensor checkpoint loads directly into MLX. The recorded Metal image comparison reached mask IoU 0.999618, maximum box error 0.1626 px, and score error 0.001305. The Object Multiplex decoder reached mask IoU 0.99215; component captures and real two-frame MLX propagation also passed.

## Status and Limitations

- Inference only on MLX-supported Apple Silicon systems.
- Image mode is text-prompted; point, box, and mask prompts belong to the video session.
- Segmentation and tracking can fail under ambiguous prompts, occlusion, tiny objects, or domain shift.
- Memory and latency depend on frame size, sequence length, and tracked object count.
- No quantized checkpoint is included in this bf16 package.

## Links

- [mlx-cv source](https://github.com/appautomaton/mlx-cv)
- [Project page](https://appautomaton.renocrypt.com/mlx-cv/)
- [Official SAM 3 code](https://github.com/facebookresearch/sam3)
- [Upstream checkpoint](https://huggingface.co/facebook/sam3.1)
- [App Automaton on Hugging Face](https://huggingface.co/appautomaton)

## License

The weights retain the bundled SAM License. Review `LICENSE` before use. `mlx-cv` code is MIT licensed separately.
