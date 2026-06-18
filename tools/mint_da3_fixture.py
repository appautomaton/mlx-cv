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
    DA3_MONOCULAR_FIXTURE_CONFIG,
    DINOV2_DA3_FIXTURE_CONFIG,
    ParityCase,
    da3_monocular_tap_order,
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


def _bhwc(x: torch.Tensor) -> np.ndarray:
    return _np(x.permute(0, 2, 3, 1))


def build_dpt_reference(cfg: dict):
    from depth_anything_3.model.dpt import DPT

    return DPT(
        dim_in=cfg["dim_in"],
        patch_size=cfg["patch_size"],
        output_dim=cfg["output_dim"],
        activation=cfg["activation"],
        conf_activation=cfg["conf_activation"],
        features=cfg["features"],
        out_channels=cfg["out_channels"],
        pos_embed=cfg["pos_embed"],
        down_ratio=cfg["down_ratio"],
        head_name=cfg["head_name"],
        use_sky_head=cfg["use_sky_head"],
        norm_type=cfg["norm_type"],
    )


def capture_dpt(dpt, feats, H: int, W: int, cfg: dict) -> tuple[dict, dict]:
    taps: dict[str, np.ndarray] = {}
    with torch.no_grad():
        B, S, N, C = feats[0][0].shape
        flat_feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        ph, pw = H // cfg["patch_size"], W // cfg["patch_size"]
        resized = []
        for stage_idx, take_idx in enumerate(dpt.intermediate_layer_idx):
            x = flat_feats[take_idx]
            x = dpt.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B * S, C, ph, pw)
            x = dpt.projects[stage_idx](x)
            x = dpt.resize_layers[stage_idx](x)
            resized.append(x)
            taps[f"dpt.projected_{stage_idx}"] = _bhwc(x)

        l1, l2, l3, l4 = resized
        l1_rn = dpt.scratch.layer1_rn(l1)
        l2_rn = dpt.scratch.layer2_rn(l2)
        l3_rn = dpt.scratch.layer3_rn(l3)
        l4_rn = dpt.scratch.layer4_rn(l4)
        out = dpt.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        taps["dpt.fusion_4"] = _bhwc(out)
        out = dpt.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        taps["dpt.fusion_3"] = _bhwc(out)
        out = dpt.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        taps["dpt.fusion_2"] = _bhwc(out)
        out = dpt.scratch.refinenet1(out, l1_rn)
        taps["dpt.fusion_1"] = _bhwc(out)
        fused = dpt.scratch.output_conv1(out)
        h_out = int(ph * cfg["patch_size"] / cfg["down_ratio"])
        w_out = int(pw * cfg["patch_size"] / cfg["down_ratio"])
        from depth_anything_3.model.utils.head_utils import custom_interpolate

        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        logits = dpt.scratch.output_conv2(fused)
        taps["dpt.output_logits"] = _bhwc(logits)

        raw = dpt(feats, H, W, patch_start_idx=0, chunk_size=None)
        depth = _np(raw["depth"])
        conf = _np(raw["depth_conf"])
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if conf.ndim == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]
    return {"depth": depth[0], "depth_conf": conf[0]}, taps


def mint_da3(seed: int) -> None:
    cfg = DA3_MONOCULAR_FIXTURE_CONFIG
    dino_cfg = cfg["dinov2"]
    dpt_cfg = cfg["dpt"]
    torch.manual_seed(seed)
    backbone = build_dinov2_reference(dino_cfg)
    head = build_dpt_reference(dpt_cfg)
    backbone.eval()
    head.eval()

    x_np = dinov2_da3_fixed_input(seed=seed, img_size=dino_cfg["img_size"])
    x = torch.from_numpy(x_np[:, None])
    dino_expected, dino_taps = capture_dinov2(backbone, x, dino_cfg)
    with torch.no_grad():
        feats, _ = backbone.get_intermediate_layers(x, dino_cfg["intermediate_layers"])
    expected, dpt_taps = capture_dpt(head, feats, dino_cfg["img_size"], dino_cfg["img_size"], dpt_cfg)
    taps = {f"dinov2.{k}": v for k, v in dino_taps.items()}
    taps.update(dpt_taps)
    assert list(taps.keys()) == da3_monocular_tap_order(), list(taps.keys())

    case = ParityCase(name=cfg["name"], inputs={"x": x_np}, expected=expected, taps=taps)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{cfg['name']}.npz"
    weights_path = FIXTURE_DIR / f"{cfg['name']}_weights.npz"
    save_case(case, fixture_path)
    weights = {f"backbone.{k}": _np(v) for k, v in backbone.state_dict().items()}
    weights.update({f"head.{k}": _np(v) for k, v in head.state_dict().items()})
    np.savez(weights_path, **weights)

    print(f"target=da3 cfg={cfg['name']} sky={dpt_cfg['use_sky_head']} output_dim={dpt_cfg['output_dim']}")
    print(f"  fixture -> {fixture_path} ({fixture_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  weights -> {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print("  expected shapes: " + ", ".join(f"{k}{v.shape}" for k, v in expected.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["dinov2", "da3"], default="dinov2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.target == "dinov2":
        mint_dinov2(args.seed)
    elif args.target == "da3":
        mint_da3(args.seed)


if __name__ == "__main__":
    main()
