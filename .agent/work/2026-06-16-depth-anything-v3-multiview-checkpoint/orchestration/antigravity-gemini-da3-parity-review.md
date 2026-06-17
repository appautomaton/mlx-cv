# Antigravity Gemini DA3 Parity Review

- Date: 2026-06-17
- Model: `Gemini 3.5 Flash (High)`
- Session: `c22e1d64-fb0f-46c3-9188-34437b60b837`
- Mode: read-only review through Antigravity CLI bridge
- Verdict: `CHANGES_NEEDED`

## Reviewed Concern

The original DA3 multi-view plan was marked verified after a synthetic fixed three-view upstream-vs-MLX gate. Real-image follow-up checks showed the gate was insufficient:

- SOH 2-image demo: `confidence` max abs error `0.2812455893` vs tolerance `0.05`.
- Robot video 3-frame demo: `confidence` max abs error `0.1753456593` vs tolerance `0.05`; `intrinsics` max abs error `12.3303222656` vs tolerance `12.0`.

## Findings

1. Upstream PyTorch DINOv2 uses learned absolute positional embedding interpolation with `interpolate_offset=0.1` and PyTorch bicubic `align_corners=False` coordinate semantics. Local MLX used direct `th / gh`, `tw / gw` scale factors through `nn.Upsample(mode="cubic")`.
2. Upstream DualDPT constructs `LayerNorm(32)` for every `output_conv2_aux` level. The DA3-SMALL checkpoint only stores level-0 LayerNorm weights, and PyTorch non-strict loading leaves levels 1-3 at default `weight=1`, `bias=0`. Local MLX used `Identity()` for levels 1-3.
3. Agy reported that an in-memory scratch patch reduced:
   - confidence error from `0.2812` to `0.0019`;
   - intrinsics error from `12.33` to `2.0904`;
   - Block-0 input error from `0.1358` to about `0.00001`.

## Handoff

Use the corrective Slices 9-11 in `PLAN.md`:

- Slice 9: PyTorch-compatible DA3 absolute positional embedding interpolation.
- Slice 10: DualDPT auxiliary LayerNorm and documented default-key strict-load semantics.
- Slice 11: real-image/video parity gate plus truthful docs/status.
