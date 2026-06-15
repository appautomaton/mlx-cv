"""Mint DA3 monocular golden fixtures (OUT-OF-BAND — needs a torch env).

This script may import torch and the DA3 reference clone; `mlx_cv` never does.
Slice 2 mints the DA3-style DINOv2 fixture. Later slices extend this same script
with DPT and end-to-end DA3 fixture minting.

Usage in a throwaway environment:

    uv venv /tmp/da3-mint --python 3.13
    uv pip install --python /tmp/da3-mint/bin/python torch numpy einops addict omegaconf -e .
    PYTHONPATH=references/Depth-Anything-3/src /tmp/da3-mint/bin/python tools/mint_da3_fixture.py --target dinov2
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "references" / "Depth-Anything-3" / "src"))

from mlx_cv.parity import (  # noqa: E402
    DINOV2_DA3_FIXTURE_CONFIG,
    ParityCase,
    dinov2_da3_fixed_input,
    dinov2_da3_tap_order,
    save_case,
)

FIXTURE_DIR = REPO / "tests" / "fixtures"


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def build_dinov2_reference(cfg: dict):
    from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer

    return DinoVisionTransformer(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["ffn_ratio"],
        num_register_tokens=cfg["n_register_tokens"],
        init_values=1.0,
        alt_start=-1,
        qknorm_start=-1,
        rope_start=-1,
        cat_token=False,
    )


def capture_dinov2(model, x: torch.Tensor, cfg: dict) -> tuple[dict, dict]:
    """Run the DA3 DINOv2 path and capture ordered taps."""
    layers = cfg["intermediate_layers"]
    taps: dict[str, np.ndarray] = {}
    with torch.no_grad():
        B, S, _, H, W = x.shape
        z = model.prepare_tokens_with_masks(x)
        taps["patch_embed"] = _np(z[:, 0])
        selected: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(model.blocks):
            z = model.process_attention(z, blk, "local", pos=None)
            taps[f"block_{i:02d}"] = _np(z[:, 0])
            if i in layers:
                selected[i] = z

        intermediate_arrays = []
        for i in layers:
            patch_tokens = model.norm(selected[i])[:, :, 1 + model.num_register_tokens :, :]
            patch_tokens = patch_tokens[:, 0]
            taps[f"intermediate_{i:02d}"] = _np(patch_tokens)
            intermediate_arrays.append(_np(patch_tokens))

        z_norm = model.norm(z)
        taps["norm"] = _np(z_norm[:, 0])
        taps["cls"] = _np(z_norm[:, 0, 0])
        taps["patch"] = _np(z_norm[:, 0, 1 + model.num_register_tokens :])

        # Cross-check against the public DA3 DINOv2 entrypoint.
        ref_outputs, _ = model.get_intermediate_layers(x, layers)
        ref_intermediates = np.stack([_np(out[0][:, 0]) for out in ref_outputs])
        assert np.allclose(np.stack(intermediate_arrays), ref_intermediates, atol=1e-6)

    assert list(taps.keys()) == dinov2_da3_tap_order(
        depth=cfg["depth"], intermediate_layers=layers
    ), list(taps.keys())
    expected = {
        "x_norm_clstoken": taps["cls"],
        "x_norm_patchtokens": taps["patch"],
        "intermediates": np.stack(intermediate_arrays),
    }
    return expected, taps


def mint_dinov2(seed: int) -> None:
    cfg = DINOV2_DA3_FIXTURE_CONFIG
    torch.manual_seed(seed)
    model = build_dinov2_reference(cfg)
    model.eval()

    x_np = dinov2_da3_fixed_input(seed=seed, img_size=cfg["img_size"])
    x = torch.from_numpy(x_np[:, None])  # DA3 DINOv2 expects B,S,C,H,W.
    expected, taps = capture_dinov2(model, x, cfg)

    case = ParityCase(name=cfg["name"], inputs={"x": x_np}, expected=expected, taps=taps)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{cfg['name']}.npz"
    weights_path = FIXTURE_DIR / f"{cfg['name']}_weights.npz"
    save_case(case, fixture_path)
    weights = {k: _np(v) for k, v in model.state_dict().items()}
    np.savez(weights_path, **weights)

    print(f"target=dinov2 cfg={cfg['name']} layers={cfg['intermediate_layers']}")
    print(f"  fixture -> {fixture_path} ({fixture_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  weights -> {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print("  expected shapes: " + ", ".join(f"{k}{v.shape}" for k, v in expected.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["dinov2"], default="dinov2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.target == "dinov2":
        mint_dinov2(args.seed)


if __name__ == "__main__":
    main()
