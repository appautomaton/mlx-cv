---
library_name: mlx
pipeline_tag: image-text-to-text
license: other
license_name: nvidia-license
license_link: https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE
base_model:
- nvidia/LocateAnything-3B
tags:
- mlx
- apple-silicon
- visual-grounding
- object-detection
- bfloat16
---

# LocateAnything-3B BF16 for MLX

Final-layout BF16 weights for running [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) with [`mlx-cv`](https://github.com/appautomaton/mlx-cv) on Apple Silicon. BF16 is reduced precision, not integer quantization.

```bash
pip install "mlx-cv[mlx,hub]==0.0.3"
```

```python
from mlx_cv.models.locateanything import LocateAnythingPipeline

pipeline = LocateAnythingPipeline.from_pretrained("locateanything-3b-bf16")
result = pipeline.predict(image, "find every traffic sign")
```

## Verification and performance

The MLX FP32 port first passed the upstream parameter and selected-tap parity gate. The BF16 package then preserved generated tokens and output geometry on four sequential real-image checks (desktop, street signs, document, and webpage). Local peak-memory observations ranged from roughly 9.8 GB to 52.3 GB depending on image and output complexity; these are machine-specific measurements, not requirements or guarantees.

One desktop multi-category prompt repeatedly emitted a monitor category. This known behavior is recorded as a model/output limitation rather than hidden by post-processing.

## Limitations

- Inference only, on MLX-supported Apple Silicon systems.
- Visual grounding output can omit, repeat, or mislabel objects; validate it for consequential uses.
- Latency and memory vary substantially with image resolution, prompt, and requested output density.
- This conversion does not change the upstream acceptable-use or license restrictions.

## License

The weights retain the bundled NVIDIA License and are restricted to academic and non-profit research purposes. Commercial use is not permitted except as described by that license. `mlx-cv` code is MIT licensed separately.

## Links

- [mlx-cv source](https://github.com/appautomaton/mlx-cv)
- [App Automaton](https://appautomaton.github.io/)
- [Upstream model](https://huggingface.co/nvidia/LocateAnything-3B)
