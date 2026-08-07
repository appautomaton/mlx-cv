# SAM 3.1 image and Object Multiplex

SAM 3.1 is the repository's supported SAM runtime. Image segmentation and
stateful video propagation share one final-layout, 1963-parameter BF16
checkpoint package.

## Public surface

| Mode | Entry point | Current output |
|---|---|---|
| Text-prompted image segmentation | `SAM3Processor.from_pretrained(...)` | shared `Result` |
| Prompted video propagation | `SAM3VideoSession.from_pretrained(...)` | `VideoResult` |

Image results populate `Result.detections`, `Result.masks`, and query-index
metadata. Each video frame populates `Result.masks` and `Result.tracks`, with
multiplex bucket assignments in metadata. `SAM3ImagePrediction` and
`SAM3VideoFrameResult` remain import aliases to `Result` during the pre-alpha
transition.

The video session accepts frame arrays, image paths, or frame directories. It
supports box, point, and mask prompts; forward and reverse propagation; object
removal; reset; seven-memory temporal attention; and dynamic 16-object Object
Multiplex buckets.

## Checkpoint contract

The normal runtime strictly validates Safetensors metadata, parameter names,
shapes, and BF16 dtypes. Source-format PyTorch conversion is an offline tool
operation; inference does not remap parameter names or materialize NumPy copies
of the checkpoint.

A self-contained local package supplies `model.safetensors` and the BPE asset
needed by image text prompting:

```python
from mlx_cv.models.sam3 import SAM3Processor, SAM3VideoSession

package = "/path/to/sam31-package"
image_processor = SAM3Processor.from_pretrained(package)
video_session = SAM3VideoSession.from_pretrained(package)
```

Loading a direct checkpoint file is also supported; image mode then requires an
explicit `bpe_path`. See [Local model packages](model-packages.md) for the staged
directory contract.

## Verification

- Official source inventory: 1623 tensors.
- Final MLX package: 1506 detector parameters plus 457 tracker parameters.
- Image comparison: mask IoU 0.999618, maximum box error 0.1626 px, score error
  0.001305.
- Object Multiplex decoder comparison: mask IoU 0.99215.
- Interactive decoding, memory encoding, memory attention, and real two-frame
  propagation were exercised on Apple GPU runtimes.

The canonical evidence is recorded in
`.agent/work/2026-06-16-release-parity-hardening/parity-status.json`. Reference
checkouts used by existing tests remain under the ignored `references/`
workspace. The SAM checkout is official upstream source plus any local
Apple-platform compatibility patches needed by its reference harness; durable
comparisons must identify both the upstream revision and that local diff. New
checkpoints, pytest state, captures, and logs should use a system temporary root
or another explicit artifact store.
