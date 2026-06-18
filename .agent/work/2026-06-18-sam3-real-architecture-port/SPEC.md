# SPEC: SAM3 Real-Architecture MLX Re-Port (Image + Video)

Change: `2026-06-18-sam3-real-architecture-port` — Stage: frame

## Objective

Replace the reduced clean-room SAM3 MLX models with faithful re-implementations of the **real** SAM 3 architecture so the downloaded `facebook/sam3` weights load 1:1 and pass an upstream-vs-MLX numeric parity gate against the `transformers` reference — mirroring how LocateAnything reached `UPSTREAM_PASSED`. Flip `sam3_image` and `sam3_video` in `parity-status.json` from `BLOCKED:…architecture re-port` to `UPSTREAM_PASSED`.

## Background — the measured gap

`facebook/sam3` is `transformers`-native (`architectures=['Sam3VideoModel']`, `transformers_version 5.0.0.dev0`; shipped in `transformers>=5.10`). Its `model.safetensors` holds **1797 weight tensors**: `detector_model` 1468, `tracker_model` 307, `tracker_neck` 22. Our current MLX models are reduced approximations (~91 image, ~53 video tensors) whose namespaces and module counts do not match, so the real weights cannot load.

Subsystem tensor counts (real → ours):

| Group | Subsystem | Real | Ours | Status |
|---|---|---:|---:|---|
| detector | vision_encoder (`sam3_vit_model` windowed-RoPE ViT + FPN) | 538 | ~plain ViT | reduced |
| detector | text_encoder (`CLIPTextConfig`) | 389 | small | reduced |
| detector | detr_decoder (6 layers, 200 queries) | 247 | — | absent |
| detector | detr_encoder (6 layers) | 156 | — | absent |
| detector | geometry_encoder (3-layer ROI) | 94 | — | absent |
| detector | mask_decoder (FPN pixel decoder) | 32 | small | reduced |
| detector | dot_product_scoring | 10 | — | absent |
| detector | text_projection | 2 | ~present | partial |
| tracker | mask_decoder | 131 | small | reduced |
| tracker | memory_attention (4-layer transformer) | 106 | — | absent |
| tracker | memory_encoder | 40 | small | reduced |
| tracker | prompt_encoder | 14 | — | absent |
| tracker | object_pointer_proj + embeddings | ~16 | — | absent |
| tracker | tracker_neck | 22 | — | absent |

## Architecture identity (porting source-of-truth)

`transformers` modeling source (readable, pip-installable, keys map 1:1 to `model.safetensors`):
- `models/sam3/modeling_sam3.py` — `Sam3Model` image detector (config `Sam3Config`).
- `models/sam3_lite_text/modeling_sam3_lite_text.py` — the CLIP-style text tower.
- `models/sam3_video/modeling_sam3_video.py` — `Sam3VideoModel` (detector + tracker + association logic).
- `models/sam3_tracker_video/modeling_sam3_tracker_video.py` — SAM2-style video tracker (config `Sam3TrackerVideoConfig`).

`Sam3Config` sub-configs: `Sam3VisionConfig{backbone=Sam3ViTConfig(hidden 1024, 32 layers, 16 heads, patch 14, image 1008, window_size 24, global_attn_indexes [7,15,23,31], rope_theta 1e4), fpn_hidden 256}`, `CLIPTextConfig(vocab 49408, hidden 1024, 24 layers, 16 heads, max_pos 32)`, `Sam3GeometryEncoderConfig(256, 3 layers, roi 7)`, `Sam3DETREncoderConfig(256, 6 layers)`, `Sam3DETRDecoderConfig(256, 6 layers, 200 queries)`, `Sam3MaskDecoderConfig(256, 3 upsampling stages)`. `Sam3VideoConfig = {detector_config: Sam3Config, tracker_config: Sam3TrackerVideoConfig}` + detection/association/hotstart heuristics (post-processing, not weights).

## Scope

**In:** Faithful MLX re-implementation of the SAM3 image detector and video tracker (inference path only); converters for the real `detector_model.*` / `tracker_model.*` / `tracker_neck.*` safetensors namespaces; `transformers`-based reference capture in `tools/` + numeric parity gates for both; `parity-status.json` flips on real PASS.

**Out:** Training, loss, matcher, data pipelines. The SAM3 Agent / MLLM wrapper. Any `torch`/`transformers`/`references/` import inside `src/mlx_cv/` (reference capture stays in `tools/` + tests only). Downloading or committing weights (user-supplied, gated, out-of-git).

## Anti-Goals

- **Not** a re-derivation from the paper: the `transformers` modeling source is the authoritative architecture; we port it, we do not reinvent it.
- **Not** a behavioral rewrite of the reduced models in place: faithful modules are added/grown to match real shapes; we do not paper over the gap by loosening tolerances or sampling tensors.
- **Not** a performance/optimization pass: numeric parity first; speed is out of scope for this change.
- **Not** a training-capable port: inference path only.

## Acceptance Criteria

- **AC1 (configs):** Faithful MLX config dataclasses mirror every `Sam3Config`/`Sam3VideoConfig` sub-config; a `from_hf_config` parser round-trips the real `facebook/sam3` `config.json` with no lossy defaults; committed test.
- **AC2 (reference spine):** `tools/` reference capture loads the real model via `transformers` (`Sam3Model` / `Sam3VideoModel`) and records deterministic taps; returns a precise component-specific blocker (never a fake pass) when weights/`transformers` are absent; committed honest-blocker + injected-capture tests.
- **AC3 (image detector):** Each detector subsystem ports to MLX, loads its real `detector_model.<sub>.*` tensors with exact shape match, and matches the `transformers` per-subsystem tap within documented tolerances.
- **AC4 (image gate):** End-to-end `Sam3Model` MLX parity vs `transformers` passes within documented tolerances on a fixed input; `sam3_image` → `UPSTREAM_PASSED` with a `passed_gate` command.
- **AC5 (video tracker):** Each tracker subsystem (incl. the 106-tensor memory_attention transformer) ports + loads + matches taps; `tracker_neck` + `tracker_model.*` converter covers all keys or fails loudly.
- **AC6 (video gate):** End-to-end `Sam3VideoModel` MLX parity vs `transformers` on a short fixed clip; `sam3_video` → `UPSTREAM_PASSED`.
- **AC7 (hygiene):** No `torch`/`transformers`/`references` imports in `src/mlx_cv/`; full `uv run pytest` green; `git diff --check` clean; converters reject unsupported variants with clear errors.

## Constraints

- Checkpoint-first / trust-by-parity: no synthetic passes; a gate is `UPSTREAM_PASSED` only after a real numeric comparison.
- MLX-native runtime; reuse existing primitives (`ViTBackbone`+RoPE, `Attention`, `TransformerBlock`, `PatchEmbed`, necks, RFDETR decoder layer, CLIP-style blocks) rather than rebuild.
- Reference pinned to `transformers>=5.10,<6` via `uv run --with` in tools/tests only.
- Serial, low-blast-radius execution (other agents may touch the repo).

## Risks

- **Size.** ~1797 tensors / ~14k lines of reference modeling code; multi-session. Mitigation: dependency-ordered slices, each independently verifiable by per-subsystem shape + tap parity before end-to-end.
- **Windowed ViT + RoPE detail.** The `sam3_vit_model` windowed/global attention + 2D RoPE is the largest single port. Mitigation: dedicated slice; validate against the `transformers` vision-tower tap first.
- **HF prefix mapping.** `facebook/sam3` safetensors use `detector_model.*`; `Sam3Model` expects unprefixed keys. Mitigation: converter handles both; reference harness verifies which `from_pretrained` path transformers uses.
- **Gated weights.** Real PASS runs out-of-sandbox with user-supplied weights; CI stays at honest-blocker.
