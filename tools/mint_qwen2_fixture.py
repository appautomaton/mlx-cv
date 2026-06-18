"""Mint Qwen2 tiny golden fixtures (out-of-band; needs torch + transformers).

Runtime `mlx-cv` tests consume committed `.npz` files and do not import torch or
transformers. Use a throwaway environment, for example:

    uv run --with torch --with transformers python tools/mint_qwen2_fixture.py
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

from mlx_cv.parity import ParityCase, QWEN2_FIXTURE_CONFIG, qwen2_fixed_inputs, save_case  # noqa: E402


def _require_or_die():
    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - exercised on mint hosts only.
        raise SystemExit(
            "mint_qwen2_fixture.py requires torch and transformers in the mint environment. "
            "Install them in a throwaway env; do not add them to pyproject runtime deps. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return torch, transformers


def _import_reference_qwen2():
    package_name = "locateanything_ref"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REF_DIR)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    cfg_mod = importlib.import_module(f"{package_name}.configuration_qwen2")
    model_mod = importlib.import_module(f"{package_name}.modeling_qwen2")
    return cfg_mod.Qwen2Config, model_mod.Qwen2ForCausalLM


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32)


def _visible(mask: np.ndarray) -> np.ndarray:
    return np.isfinite(mask) & (mask > -1e20)


def _atomic_savez(path: pathlib.Path, **arrays) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def main() -> None:
    torch, transformers = _require_or_die()
    RefConfig, RefForCausalLM = _import_reference_qwen2()
    cfg = QWEN2_FIXTURE_CONFIG
    inputs_np = qwen2_fixed_inputs()

    torch.manual_seed(int(cfg["seed"]))
    ref_cfg = RefConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        hidden_act=cfg["hidden_act"],
        max_position_embeddings=cfg["max_position_embeddings"],
        initializer_range=0.02,
        rms_norm_eps=cfg["rms_norm_eps"],
        use_cache=cfg["use_cache"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        rope_theta=cfg["rope_theta"],
        use_sliding_window=False,
        sliding_window=cfg["max_position_embeddings"],
        max_window_layers=cfg["num_hidden_layers"],
        attention_dropout=cfg["attention_dropout"],
        attn_implementation=cfg["attn_implementation"],
        block_size=cfg["block_size"],
        causal_attn=cfg["causal_attn"],
        text_mask_token_id=cfg["text_mask_token_id"],
    )
    ref_cfg._attn_implementation = cfg["attn_implementation"]
    model = RefForCausalLM(ref_cfg)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.eval()

    captured: dict[str, object] = {}
    original_forward = model.model.layers[0].self_attn.forward

    def capture_first_layer_mask(*args, **kwargs):
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            captured["attention_mask"] = kwargs["attention_mask"].detach().cpu().float().numpy()
        return original_forward(*args, **kwargs)

    model.model.layers[0].self_attn.forward = capture_first_layer_mask

    input_ids = torch.from_numpy(inputs_np["input_ids"]).long()
    position_ids = torch.from_numpy(inputs_np["position_ids"]).long()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = out.hidden_states[-1]
    logits = out.logits
    attention_mask = captured.get("attention_mask")
    if attention_mask is None:
        raise RuntimeError("failed to capture reference first-layer attention mask")

    versions = {
        "seed": cfg["seed"],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "reference": str(REF_DIR.relative_to(REPO)),
    }
    expected = {
        "hidden_states": _np(hidden),
        "logits": _np(logits),
        "attention_mask_visible": _visible(attention_mask),
    }
    taps = {"attention_mask": attention_mask.astype(np.float32)}
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
