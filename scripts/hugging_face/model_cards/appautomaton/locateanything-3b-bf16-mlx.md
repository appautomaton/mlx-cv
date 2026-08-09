---
library_name: mlx
pipeline_tag: image-text-to-text
license: other
license_name: nvidia-license
license_link: LICENSE
base_model:
- nvidia/LocateAnything-3B
tags:
- mlx
- apple-silicon
- visual-grounding
- object-detection
- bfloat16
---

# LocateAnything-3B — MLX (bf16)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--cv-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-cv)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.renocrypt.com/mlx-cv/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)

MLX-native bf16 conversion of [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) for text-prompted visual grounding on Apple Silicon. It returns boxes, points, and labels through [`mlx-cv`](https://github.com/appautomaton/mlx-cv), with no PyTorch or upstream model code at inference time. Weights ship as a final-layout `.safetensors` checkpoint.

## Model Details

- Developed by: [App Automaton](https://appautomaton.renocrypt.com)
- Upstream model: [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)
- Task: text-prompted visual grounding with boxes and points
- Architecture: MoonViT vision tower, MLP projector, Qwen2.5 decoder, and Parallel Box Decoding
- Precision: bf16 checkpoint weights; this is reduced precision, not integer quantization
- Runtime: MLX on Apple Silicon

## Contents

| File | Purpose | Format |
| --- | --- | --- |
| `model.safetensors` | 769 final-layout model tensors | bf16 Safetensors |
| `config.json` | MLX architecture and grounding-token contract | JSON |
| `tokenizer.json` | Self-contained runtime tokenizer | JSON |
| tokenizer and processor assets | Vocabulary, merges, special tokens, and image preprocessing | JSON/text |
| `manifest.json` | File sizes and SHA-256 hashes | JSON |

## How to Get Started

```bash
pip install "mlx-cv[mlx,hub]"
hf download appautomaton/locateanything-3b-bf16-mlx \
  --local-dir models/locateanything_3b/mlx-bf16
```

```python
import mlx_cv

pipeline = mlx_cv.load(
    "locateanything-3b",
    "models/locateanything_3b/mlx-bf16",
)
result = pipeline.predict(image, "find every traffic sign")
```

The exact remote repository ID can also be passed in place of the local path:

```python
pipeline = mlx_cv.load(
    "locateanything-3b",
    "appautomaton/locateanything-3b-bf16-mlx",
)
```

## Verification

The original fp32 MLX conversion passed the real upstream parameter, geometry, selected-tap, and generated-token comparison. The final bf16 package preserved generated tokens and output geometry on four sequential real-image regression cases.

## Status and Limitations

- Inference only on MLX-supported Apple Silicon systems.
- The current package processes one image at a time.
- Visual grounding can omit, repeat, or mislabel objects.
- Memory and latency vary with image resolution, prompt, and output density.
- No quantized checkpoint is included in this bf16 package.

## Links

- [mlx-cv source](https://github.com/appautomaton/mlx-cv)
- [Project page](https://appautomaton.renocrypt.com/mlx-cv/)
- [Upstream model](https://huggingface.co/nvidia/LocateAnything-3B)
- [App Automaton on Hugging Face](https://huggingface.co/appautomaton)

## License

The weights retain the bundled NVIDIA License and are restricted to academic and non-profit research purposes. Commercial use is not permitted except as described by that license. `mlx-cv` code is MIT licensed separately.
