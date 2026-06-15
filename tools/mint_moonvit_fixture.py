"""Mint MoonViT tiny golden fixtures (out-of-band; needs torch + transformers).

Runtime `mlx-cv` tests consume committed `.npz` files and do not import torch or
transformers. Use a throwaway environment, for example:

    uv run --with torch --with transformers python tools/mint_moonvit_fixture.py
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures"
REF_DIR = REPO / "references" / "LocateAnything-3B"

sys.path.insert(0, str(REPO / "src"))

from mlx_cv.parity import (  # noqa: E402
    MOONVIT_FIXTURE_CONFIG,
    ParityCase,
    moonvit_fixed_inputs,
    moonvit_tap_order,
    save_case,
)


def _require_or_die():
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - exercised on mint hosts only.
        raise SystemExit(
            "mint_moonvit_fixture.py requires torch and transformers in the mint environment. "
            "Install them in a throwaway env; do not add them to pyproject runtime deps. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return torch, transformers


def _import_reference_moonvit():
    package_name = "locateanything_ref"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REF_DIR)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    model_mod = importlib.import_module(f"{package_name}.modeling_vit")
    return model_mod


def _np(x) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    if arr.dtype == np.complex128:
        return arr.astype(np.complex64)
    return arr


def _atomic_savez(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def _block_attention_visible(grid_hws: np.ndarray) -> np.ndarray:
    lengths = grid_hws[:, 0] * grid_hws[:, 1]
    cu = np.concatenate([np.zeros((1,), dtype=np.int32), np.cumsum(lengths, dtype=np.int32)])
    seq_len = int(cu[-1])
    visible = np.zeros((seq_len, seq_len), dtype=bool)
    for start, stop in zip(cu[:-1], cu[1:]):
        visible[start:stop, start:stop] = True
    return visible


def main() -> None:
    torch, transformers = _require_or_die()
    ref = _import_reference_moonvit()
    cfg = MOONVIT_FIXTURE_CONFIG
    inputs_np = moonvit_fixed_inputs()

    torch.manual_seed(int(cfg["seed"]))
    ref_cfg = ref.MoonViTConfig(
        patch_size=cfg["patch_size"],
        init_pos_emb_height=cfg["init_pos_emb_height"],
        init_pos_emb_width=cfg["init_pos_emb_width"],
        num_attention_heads=cfg["num_attention_heads"],
        num_hidden_layers=cfg["num_hidden_layers"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        merge_kernel_size=tuple(cfg["merge_kernel_size"]),
        attn_implementation=cfg["attn_implementation"],
    )
    ref_cfg._attn_implementation = cfg["attn_implementation"]
    model = ref.MoonVitPretrainedModel(ref_cfg)
    model.eval()

    implementations = {block.attn_implementation for block in model.encoder.blocks}
    if implementations != {cfg["attn_implementation"]}:
        raise RuntimeError(f"MoonViT fixture must use sdpa attention, got {implementations}")

    pixel_values = torch.from_numpy(inputs_np["pixel_values"])
    grid_hws = torch.from_numpy(inputs_np["grid_hws"]).to(torch.int32)

    taps: dict[str, np.ndarray] = {}
    with torch.no_grad():
        hidden = model.patch_embed(pixel_values, grid_hws)
        taps["patch_embed"] = _np(hidden)
        rope = model.encoder.rope_2d.get_freqs_cis(grid_hws=grid_hws)
        taps["rope_freqs_cis"] = _np(rope)
        taps["attention_mask_visible"] = _block_attention_visible(inputs_np["grid_hws"])

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden.device, dtype=grid_hws.dtype),
                grid_hws[:, 0] * grid_hws[:, 1],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
        for i, block in enumerate(model.encoder.blocks):
            hidden = block(hidden, cu_seqlens, rope_freqs_cis=rope)
            taps[f"block_{i:02d}"] = _np(hidden)
        hidden = model.encoder.final_layernorm(hidden)
        taps["norm"] = _np(hidden)
        merged = ref.patch_merger(hidden, grid_hws, merge_kernel_size=tuple(cfg["merge_kernel_size"]))
        for i, item in enumerate(merged):
            taps[f"merged_{i:02d}"] = _np(item)

    expected = {k: taps[k] for k in taps if k.startswith("merged_")}
    expected["norm"] = taps["norm"]
    expected_order = moonvit_tap_order(depth=cfg["num_hidden_layers"], num_images=len(cfg["grid_hws"]))
    if list(taps) != expected_order:
        raise RuntimeError(f"unexpected tap order: {list(taps)} != {expected_order}")

    versions = {
        "seed": cfg["seed"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "reference": str(REF_DIR.relative_to(REPO)),
    }
    case = ParityCase(name=cfg["name"], inputs=inputs_np, expected=expected, taps=taps)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{cfg['name']}.npz"
    weights_path = FIXTURE_DIR / f"{cfg['name']}_weights.npz"
    tmp_fixture = fixture_path.with_name(f"{fixture_path.name}.tmp.npz")
    save_case(case, tmp_fixture)
    tmp_fixture.replace(fixture_path)

    weights = {k: _np(v) for k, v in model.state_dict().items()}
    weights["__versions_json__"] = np.asarray(json.dumps(versions, sort_keys=True))
    weights["__config_json__"] = np.asarray(json.dumps(cfg, sort_keys=True))
    _atomic_savez(weights_path, **weights)

    print(f"fixture -> {fixture_path} ({fixture_path.stat().st_size / 1e6:.2f} MB)")
    print(f"weights -> {weights_path} ({weights_path.stat().st_size / 1e6:.2f} MB)")
    print(f"versions -> {versions}")


if __name__ == "__main__":
    main()
