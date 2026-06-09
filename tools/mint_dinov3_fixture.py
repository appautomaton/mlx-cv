"""Mint the DINOv3 golden parity fixture (OUT-OF-BAND — needs a torch env).

This is **not** a package dependency. `torch` and the official DINOv3 reference
are never imported by `mlx_cv`; they live only here, run once to produce a
committed fixture, and the library compares against that fixture forever after.

Phase-1 oracle: **fixed-seed → MLX** structural/implementation parity. We seed the
official PyTorch DINOv3, run `forward_features` while capturing ordered intermediate
taps, then export the exact weights so the MLX port (Slice 4) can load them and
reproduce the forward bit-for-bit within tolerance (Slice 6).

Usage (throwaway env; torch never enters pyproject):

    uv venv /tmp/dinov3-mint --python 3.13
    uv pip install --python /tmp/dinov3-mint/bin/python torch numpy einops -e .
    PYTHONPATH=references/dinov3 /tmp/dinov3-mint/bin/python tools/mint_dinov3_fixture.py

`--variant fixture` (default) mints the small committed config; `--variant vit_small`
mints full DINOv3 ViT-S/16 (weights ~88 MB — do not commit).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "references" / "dinov3"))

from mlx_cv.parity import (  # noqa: E402  (after sys.path / torch env setup)
    DINOV3_FIXTURE_CONFIG,
    DINOV3_VARIANT,
    ParityCase,
    dinov3_fixed_input,
    dinov3_tap_order,
    save_case,
)

FIXTURE_DIR = REPO / "tests" / "fixtures"


def build_model(cfg: dict):
    import dinov3.models.vision_transformer as vt

    model = vt.DinoVisionTransformer(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        ffn_ratio=cfg.get("ffn_ratio", 4.0),
        n_storage_tokens=cfg["n_storage_tokens"],
        norm_layer="layernorm",            # LayerNorm(eps=1e-6)
        ffn_layer="mlp",                   # plain Mlp + GELU (ViT-S default)
        layerscale_init=None,              # -> ls1/ls2 = Identity
        mask_k_bias=False,
        pos_embed_rope_base=cfg.get("pos_embed_rope_base", 100.0),
        pos_embed_rope_dtype=cfg.get("pos_embed_rope_dtype", "fp32"),
    )
    return model


def capture(model, x: torch.Tensor, n_storage: int, depth: int) -> tuple[dict, dict]:
    """Run forward_features faithfully, capturing ordered taps + the output dict."""
    taps: dict[str, np.ndarray] = {}
    with torch.no_grad():
        tokens, (H, W) = model.prepare_tokens_with_masks(x, None)   # (B, 1+R+P, C)
        taps["patch_embed"] = tokens.clone()
        sin, cos = model.rope_embed(H=H, W=W)
        taps["rope_sincos"] = torch.stack([sin, cos])               # (2, HW, D_head)
        xb = tokens
        for i, blk in enumerate(model.blocks):
            xb = blk(xb, model.rope_embed(H=H, W=W))
            taps[f"block_{i:02d}"] = xb.clone()
        x_norm = model.norm(xb)
        taps["norm"] = x_norm.clone()
        taps["cls"] = x_norm[:, 0]
        taps["storage"] = x_norm[:, 1 : 1 + n_storage]
        taps["patch"] = x_norm[:, 1 + n_storage :]

        ref = model.forward_features(x)   # cross-check against the real entrypoint

    # forward order must match the committed schema
    assert list(taps.keys()) == dinov3_tap_order(depth=depth), list(taps.keys())
    # the manual split must equal forward_features (proves the manual path is faithful)
    for k_manual, k_ref in [("cls", "x_norm_clstoken"), ("storage", "x_storage_tokens"),
                            ("patch", "x_norm_patchtokens")]:
        assert torch.allclose(taps[k_manual], ref[k_ref], atol=1e-6), k_manual

    expected = {
        "x_norm_clstoken": ref["x_norm_clstoken"].numpy(),
        "x_storage_tokens": ref["x_storage_tokens"].numpy(),
        "x_norm_patchtokens": ref["x_norm_patchtokens"].numpy(),
    }
    taps_np = {k: v.numpy().astype(np.float32) for k, v in taps.items()}
    return expected, taps_np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fixture", "vit_small"], default="fixture")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = DINOV3_FIXTURE_CONFIG if args.variant == "fixture" else DINOV3_VARIANT
    name = cfg["name"]
    torch.manual_seed(args.seed)

    model = build_model(cfg)
    model.init_weights()      # deterministic given the seed
    model.eval()

    x_np = dinov3_fixed_input(seed=args.seed, img_size=cfg["img_size"])
    x = torch.from_numpy(x_np)

    expected, taps = capture(model, x, cfg["n_storage_tokens"], cfg["depth"])
    case = ParityCase(name=name, inputs={"x": x_np}, expected=expected, taps=taps)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{name}.npz"
    weights_path = FIXTURE_DIR / f"{name}_weights.npz"
    save_case(case, fixture_path)
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(weights_path, **weights)

    fsize = fixture_path.stat().st_size / 1e6
    wsize = weights_path.stat().st_size / 1e6
    print(f"variant={args.variant} cfg={name} tokens={1 + cfg['n_storage_tokens'] + (cfg['img_size'] // cfg['patch_size']) ** 2}")
    print(f"  fixture -> {fixture_path}  ({fsize:.2f} MB; {len(taps)} taps)")
    print(f"  weights -> {weights_path}  ({wsize:.2f} MB; {len(weights)} tensors)")
    print(f"  expected shapes: " + ", ".join(f"{k}{v.shape}" for k, v in expected.items()))


if __name__ == "__main__":
    main()
