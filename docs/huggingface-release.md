# Hugging Face release workflow

`mlx-cv` keeps multi-gigabyte weights outside Git and publishes two flat BF16 runtime packages under the `appautomaton` organization:

| Target | Repository |
|---|---|
| `locateanything-3b-bf16` | `appautomaton/locateanything-3b-bf16-mlx` |
| `sam3.1-multiplex-bf16` | `appautomaton/sam3.1-multiplex-bf16-mlx` |

BF16 is reduced precision and is not described as integer quantization. Each package contains `model.safetensors`, normalized runtime assets, the App Automaton card as `README.md`, the upstream model license, and a SHA256 `manifest.json`. Source NPZ/PT files, reference code, training artifacts, and media are excluded.

## Local preparation

The registry-backed command is the only publishing entry point:

```bash
python tools/huggingface_release.py list
python tools/huggingface_release.py stage --all
python tools/huggingface_release.py verify --all
```

Staging is sequential and atomic under the ignored `.release/huggingface/` directory. Verification rejects undeclared files, symlinks, incorrect hashes, missing licenses, invalid card front matter, and incompatible Safetensors metadata.

The LocateAnything final checkpoint is minted once from the verified BF16 file while recording the FP32 source checksum:

```bash
python tools/convert_locateanything_checkpoint.py \
  models/locateanything/mlx/locateanything-3b-bf16.safetensors \
  models/locateanything/mlx/locateanything-3b-bf16.strict.safetensors \
  --source models/locateanything/mlx/locateanything-3b.npz
```

Normal runtime loading never performs this conversion.

## Publication order and checkpoint

Do not create or upload remote repositories until a maintainer explicitly authorizes the public release and confirms Hugging Face authentication.

1. Build and check the 0.0.3 wheel and sdist locally.
2. Publish GitHub release `v0.0.3`; the trusted-publishing workflow publishes PyPI.
3. Verify `mlx-cv==0.0.3` from PyPI in a clean environment.
4. Upload LocateAnything and verify it remotely.
5. Upload SAM 3.1 and verify it remotely.

```bash
python tools/huggingface_release.py upload locateanything-3b-bf16
python tools/huggingface_release.py verify-remote locateanything-3b-bf16 --fresh-cache
python tools/huggingface_release.py upload sam3.1-multiplex-bf16
python tools/huggingface_release.py verify-remote sam3.1-multiplex-bf16 --fresh-cache
```

The uploader refuses an existing repository by default. `--resume` is allowed only for the exact repository configured in the registry. It uses one upload worker and never deletes remote files or changes repository visibility.
