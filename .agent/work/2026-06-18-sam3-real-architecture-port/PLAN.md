# PLAN: SAM3 Real-Architecture MLX Re-Port (Image + Video)

Change: `2026-06-18-sam3-real-architecture-port` — Stage: plan — Spec: `SPEC.md`

## Goal

Re-implement the real SAM 3 image detector and video tracker in MLX so `facebook/sam3` weights load 1:1 and pass an upstream-vs-MLX numeric parity gate against the `transformers` reference, flipping `sam3_image` and `sam3_video` to `UPSTREAM_PASSED` (SPEC AC1–AC7).

## Architecture Approach

- **Reference = `transformers`, not the research repo.** Reference capture in `tools/sam3_upstream.py` loads `Sam3Model` / `Sam3VideoModel` via `from_pretrained(facebook/sam3)` under `uv run --with "transformers>=5.10,<6" --with torch` (tools/tests only). This mirrors `tools/locateanything_upstream.py` and gives 1:1 key correspondence to `model.safetensors`. The existing `tools/sam3_image_upstream.py` (research-repo reference) stays as a secondary path; the new transformers gate is authoritative.
- **Faithful modules mirror `transformers` submodule structure** so the converter is a mechanical `detector_model.<sub>.* → <sub>.*` remap (LocateAnything pattern, 769/769). New tensor compute lives under `models/sam3/`, `backbones/`, `heads/`; `core/` stays MLX-native and import-light.
- **Reuse over rebuild:** `ViTBackbone`+`RoPEStrategy` (vision), `Attention`/`TransformerBlock`/`MlpFFN` (encoders/text), RFDETR decoder layer + necks (DETR/FPN), existing `SAM3FeatureNeck`/`SAM3MaskDecoder` skeletons grown to faithful shapes.
- **Per-slice verification is two-stage:** (1) load the real subsystem tensors with exact shape match, (2) match the `transformers` per-subsystem tap within documented tolerances — before any end-to-end claim. No synthetic passes.

## Requirement Traceability

| SPEC AC | Satisfying slices |
|---|---|
| AC1 configs | Slice 1 |
| AC2 reference spine | Slice 1 |
| AC3 image subsystems | Slices 2–6 |
| AC4 image gate | Slice 7 |
| AC5 video subsystems | Slices 8–10 |
| AC6 video gate | Slice 11 |
| AC7 hygiene | all slices; final gate in Slices 7 & 11 |

## Ordered Slice Sequence

### Slice 1: Config ingestion + transformers reference spine
**Objective:** Faithful MLX detector/tracker config dataclasses mirroring `Sam3Config`/`Sam3VideoConfig` sub-configs + `from_hf_config(config.json)` parser; transformers-based reference capture in `tools/sam3_upstream.py`.
**Acceptance criteria:** `from_hf_config` round-trips the real `facebook/sam3` `config.json` (committed test, no weights). Reference capture loads `Sam3Model` via transformers and records ≥1 deterministic tap when available; returns a precise blocker otherwise (committed honest-blocker + injected-capture tests). No torch/transformers imports in `src/`.
**Verification:** `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test pytest tests/test_sam3_config_ingest.py tests/test_sam3_upstream_hf.py tests/test_runtime_dependency_guards.py -q`
**Touches:** `src/mlx_cv/models/sam3/config.py` (+ subsystem configs), `tools/sam3_upstream.py`, new tests.
**Status:** pending

### Slice 2: Vision encoder — windowed-RoPE ViT + FPN (538)
**Objective:** Port `sam3_vit_model` (windowed/global attention + 2D RoPE, 32 layers) and the `sam3_vision_model` FPN; converter for `detector_model.vision_encoder.*`.
**Acceptance criteria:** Loads all 538 vision tensors with exact shapes; vision-tower feature-map tap matches transformers within tolerance on a fixed image.
**Verification:** `… pytest tests/test_sam3_vision_real.py -q`
**Touches:** `backbones/vision/sam3/`, `backbones/vision/necks/sam3.py`, `models/sam3/convert.py`.
**Depends on:** Slice 1. **Status:** pending

### Slice 3: Text encoder (389) + text_projection (2)
**Objective:** Port the CLIP-style text tower (`CLIPTextConfig`) + text projection; converter for `detector_model.text_encoder.*` / `text_projection`.
**Acceptance criteria:** Loads all 391 tensors; text-embedding tap matches transformers within tolerance for a fixed prompt.
**Verification:** `… pytest tests/test_sam3_text_real.py -q`
**Touches:** `models/sam3/text.py`, `models/sam3/convert.py`. **Depends on:** Slice 1. **Status:** pending

### Slice 4: DETR encoder (156) + geometry encoder (94) + scoring (10)
**Objective:** Port the 6-layer DETR encoder, 3-layer ROI geometry encoder, and dot-product scoring head; converter for those namespaces.
**Acceptance criteria:** Loads all 260 tensors; encoder-output and scoring taps match transformers within tolerance.
**Verification:** `… pytest tests/test_sam3_detr_encoder_real.py -q`
**Touches:** `heads/detection/` or `models/sam3/`, `models/sam3/convert.py`. **Depends on:** Slices 2–3. **Status:** pending

### Slice 5: DETR decoder (247)
**Objective:** Port the 6-layer, 200-query DETR decoder; converter for `detector_model.detr_decoder.*`.
**Acceptance criteria:** Loads all 247 tensors; decoder query/box taps match transformers within tolerance.
**Verification:** `… pytest tests/test_sam3_detr_decoder_real.py -q`
**Touches:** `models/sam3/multiplex_decoder.py`/new, `models/sam3/convert.py`. **Depends on:** Slice 4. **Status:** pending

### Slice 6: Mask decoder — FPN pixel decoder (32)
**Objective:** Port the mask decoder (3 upsampling stages); converter for `detector_model.mask_decoder.*`.
**Acceptance criteria:** Loads all 32 tensors; mask-logits tap matches transformers within tolerance.
**Verification:** `… pytest tests/test_sam3_mask_decoder_real.py -q`
**Touches:** `heads/segmentation/sam3.py`, `models/sam3/convert.py`. **Depends on:** Slice 5. **Status:** pending

### Slice 7: Sam3Model assembly + image parity gate
**Objective:** Wire detector end-to-end; full numeric parity vs transformers `Sam3Model`; flip `sam3_image`.
**Acceptance criteria:** End-to-end boxes/scores/masks match transformers within documented tolerances on a fixed input; `parity-status.json` `sam3_image` → `UPSTREAM_PASSED` with `passed_gate.command`; honest blocker preserved without weights.
**Verification:** `… pytest tests/test_sam3_upstream_parity.py tests/test_sam3_model.py -q` (+ real-weights gate out-of-sandbox)
**Touches:** `models/sam3/modeling.py`, `tools/sam3_upstream.py`, `parity-status.json`. **Depends on:** Slices 2–6. **Status:** pending

### Slice 8: Tracker neck (22) + memory encoder (40)
**Objective:** Port the SAM2-style memory encoder + tracker neck; converter for `tracker_neck.*` / `tracker_model.memory_encoder.*`.
**Acceptance criteria:** Loads all 62 tensors; memory-encoder tap matches transformers within tolerance.
**Verification:** `… pytest tests/test_sam3_tracker_memory_real.py -q`
**Touches:** `models/sam3/video_memory.py`, `models/sam3/convert.py`. **Depends on:** Slice 7. **Status:** pending

### Slice 9: Memory-attention transformer (106) + object pointers + embeddings
**Objective:** Port the 4-layer memory-attention transformer (self + cross attention + RoPE), `object_pointer_proj`, and the no-memory/occlusion embeddings; converter.
**Acceptance criteria:** Loads all ~122 tensors; memory-attention output tap matches transformers within tolerance.
**Verification:** `… pytest tests/test_sam3_memory_attention_real.py -q`
**Touches:** `models/sam3/video_memory.py`/new, `models/sam3/convert.py`. **Depends on:** Slice 8. **Status:** pending

### Slice 10: Tracker mask decoder (131) + prompt encoder (14)
**Objective:** Port the tracker SAM mask decoder + prompt encoder; converter for `tracker_model.mask_decoder.*` / `prompt_encoder.*`.
**Acceptance criteria:** Loads all 145 tensors; tracker mask tap matches transformers within tolerance.
**Verification:** `… pytest tests/test_sam3_tracker_mask_real.py -q`
**Touches:** `models/sam3/multiplex_decoder.py`, `models/sam3/convert.py`. **Depends on:** Slice 9. **Status:** pending

### Slice 11: Sam3VideoModel assembly + streaming + video parity gate
**Objective:** Wire detector + tracker + memory/association streaming; full numeric parity vs transformers `Sam3VideoModel`; flip `sam3_video`.
**Acceptance criteria:** End-to-end masks/identities match transformers within documented tolerances on a short fixed clip; `parity-status.json` `sam3_video` → `UPSTREAM_PASSED`; honest blocker preserved without weights.
**Verification:** `… pytest tests/test_sam3_video_upstream_parity.py tests/test_sam3_video_tracking.py -q` (+ real-weights gate out-of-sandbox)
**Touches:** `models/sam3/video_model.py`, `tools/sam3_video_upstream.py`, `parity-status.json`. **Depends on:** Slices 7–10. **Status:** pending

## Execution Routing And Topology

- **Default execution:** direct, serial; continue through approved slices once each slice's verification passes.
- **Subagent routes:** Slices 2, 5, 9 (largest single ports) are subagent candidates if context pressure warrants; otherwise direct.
- **Parallel-safe groups:** none. Serial per the non-aggressive-concurrency preference (other agents may touch the repo); the recent multi-agent fan-out hit upstream 429s, reinforcing serial.
- **Checkpoints:** after Slice 7 (image gate PASS) is a natural human checkpoint before the video half.
- **External access:** real PASS runs out-of-sandbox with user-supplied gated weights; no weights downloaded or committed.

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Config ingest | `… pytest tests/test_sam3_config_ingest.py -q` |
| Reference spine | `… pytest tests/test_sam3_upstream_hf.py -q` |
| Image subsystems | `… pytest tests/ -k "sam3 and real" -q` |
| Image gate | `… pytest tests/test_sam3_upstream_parity.py tests/test_sam3_model.py -q` |
| Video gate | `… pytest tests/test_sam3_video_upstream_parity.py -q` |
| Runtime hygiene | `… pytest tests/test_runtime_dependency_guards.py -q` |
| Full regression | `… pytest -q` |
| Diff hygiene | `git diff --check` |

(`…` = `UV_CACHE_DIR=/tmp/mlx-cv-uv-cache uv run --extra test`.)
