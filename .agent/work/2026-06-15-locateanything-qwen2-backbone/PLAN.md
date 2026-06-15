# PLAN: LocateAnything Qwen2.5 LLM Backbone

Change: `2026-06-15-locateanything-qwen2-backbone` - Stage: execute - Spec: `SPEC.md` - Design: `DESIGN.md`

## Goal
Implement the bounded Qwen2.5 language backbone from `SPEC.md`: canonical config, MLX decoder body, tied logits, GQA, block masks, append-only KV-cache, convert/load rules, and tiny reference parity, without broadening into full LocateAnything VLM assembly.

## Architecture Approach
See `DESIGN.md`. The main invariants are: `core/` and Stage-1 LocateAnything imports stay mlx-free; Qwen2 parameter paths remain compatible with `language_model.model.*`; `language_model.lm_head.weight` remains tied/dropped; mask and cache behavior are proven with tiny fixtures before any later PBD or multimodal work.

## Ordered Slice Sequence

### Slice 1: Canonical Config and LLM Registration

**Objective:** Move Qwen2 config ownership to `backbones/llm/qwen2/`, preserve LocateAnything defaults through an alias/re-export, and register a minimal LLM builder surface.

**Acceptance criteria:**
- `backbones/llm/qwen2/config.py` defines mlx-free `Qwen2Config` with LocateAnything-3B defaults plus fields needed by the model body (`hidden_act`, `attention_dropout`, `use_cache=False`, `pad_token_id`, `text_mask_token_id`, and local SDPA/manual-attention implementation naming).
- `models/locateanything/config.py` reuses that config class from the config submodule so `LocateAnythingConfig.text_config` does not drift and `import mlx_cv.models.locateanything` does not import `mlx`.
- `LocateAnythingConfig.text_mask_token_id` is either sourced from `text_config` or explicitly asserted equal to `text_config.text_mask_token_id`.
- `import mlx_cv.backbones.llm.qwen2` and `import mlx_cv.backbones.llm.qwen2.config` both remain mlx-free because Python runs package `__init__.py` before submodule imports.
- Slice 1 does not register a placeholder builder under `"qwen2.5-3b"`; registration happens once when the concrete builder exists.
- Existing LocateAnything config tests still pass, and new tests cover config defaults, reference `_attn_implementation="magi"` reconciliation to the supported local SDPA/manual path, `use_cache=False`, no placeholder registration, and mlx-free import smoke.

**Touches:** `src/mlx_cv/backbones/llm/qwen2/{__init__.py,config.py}`, `src/mlx_cv/models/locateanything/config.py`, `tests/test_la_config.py`, `tests/test_qwen2_config.py`, `tests/test_registry.py`.

**Produces:** Single Qwen2 config source of truth and mlx-free LocateAnything config reuse, with no Qwen2 registry side effect yet.

**Verification:** `uv run pytest tests/test_la_config.py tests/test_qwen2_config.py tests/test_registry.py`

**Status:** Complete.

**Evidence:** `uv run pytest tests/test_la_config.py tests/test_qwen2_config.py tests/test_registry.py` -> 13 passed.

### Slice 2: Qwen2 Leaves - RMSNorm, SwiGLU, RoPE, Repeat-KV

**Objective:** Implement the decoder leaf operations used by Qwen2 and prove them against reference formulas, hand-constructed arrays, or local numpy recomputation.

**Acceptance criteria:**
- `Qwen2RMSNorm` matches reference RMS normalization behavior and dtype handling for fixed arrays.
- `Qwen2MLP` implements `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Qwen2 bias layout matches reference: RMSNorm weight only; `gate/up/down/o` projections have no bias.
- RoPE helpers match the Qwen/Llama half-split convention (`cat(freqs, freqs)` plus `rotate_half` across `dim//2`) with explicit `position_ids`, including a prefix-drop/reset gather case.
- `repeat_kv` expands `(B, kv_heads, T, D)` to attention heads without copying semantics mistakes.
- No leaf code lives in `core/`.

**Touches:** `src/mlx_cv/backbones/llm/qwen2/modeling.py`, `tests/test_qwen2_layers.py`.

**Produces:** Tested Qwen2 leaf modules for the model stack.

**Verification:** `uv run pytest tests/test_qwen2_layers.py`

**Depends on:** Slice 1

### Slice 3: Block Masks and GQA Attention

**Objective:** Add Qwen2 attention with additive 4D masks and port LocateAnything's block-mask helpers.

**Acceptance criteria:**
- `Qwen2Attention` supports separate q/k/v projections, GQA (`num_attention_heads > num_key_value_heads`), explicit `position_ids`, and additive masks shaped `(B, 1, Q, K)`.
- `q_proj`, `k_proj`, and `v_proj` include bias while `o_proj` does not, and convert coverage later preserves those bias keys.
- The attention output matches an in-test numpy recomputation for fixed tiny weights, inputs, position IDs, GQA repetition, and additive mask; torch-oracle numerical parity starts in Slice 4 after fixture minting.
- Mask helpers reproduce hand-constructed `0.0`/masked visibility expectations for prefix-drop block-diff masks and inference generation-window masks, treating both `-inf` and dtype `finfo.min` as masked rather than comparing raw dtype/value equality.
- A local base 4D causal additive mask builder matches the transformer utility behavior the reference calls before applying LocateAnything-specific helpers.
- `causal_attn=True` and `causal_attn=False` are both covered where the reference helper behavior differs.
- Training block-diff mask coverage is helper-only in this change; model forward wires inference mask branches only.

**Touches:** `src/mlx_cv/backbones/llm/qwen2/{modeling.py,masks.py}`, `tests/test_qwen2_attention.py`, `tests/test_qwen2_masks.py`.

**Produces:** Self-contained mask-aware GQA attention tests and LocateAnything block-mask utilities without requiring torch-generated fixtures.

**Verification:** `uv run pytest tests/test_qwen2_attention.py tests/test_qwen2_masks.py`

**Depends on:** Slice 2

### Slice 4: Decoder Stack, Atomic Fixture Mint, and Tied Logits

**Objective:** Compose decoder layers into `Qwen2Model` and `Qwen2ForCausalLM`, mint the tiny reference activations/weights fixtures atomically before parity assertions, and verify no-cache tied-logit parity.

**Acceptance criteria:**
- `tools/mint_qwen2_fixture.py` creates both `qwen2_tiny_fixture.npz` and `qwen2_tiny_fixture_weights.npz` from the same seeded SDPA/fp32 reference instance before `tests/test_qwen2_parity.py` is expected to pass.
- The mint tool records seed plus torch/transformers versions and fails with a clear message if the mint host lacks a compatible transformers private API needed by the reference.
- The fixture pins `batch_size=1`, `attention_mask=None`, a no-cache SDLM input that contains and ends with `text_mask_token_id`, and fixed position IDs that exercise prefix-drop gather.
- `Qwen2Model(input_ids=...)` and `Qwen2Model(inputs_embeds=...)` return the same hidden states when embeddings are equivalent.
- Decoder layer residual order matches the reference: input RMSNorm -> attention -> residual -> post-attention RMSNorm -> MLP -> residual.
- `Qwen2ForCausalLM` computes logits by tying to `embed_tokens.weight`; no independent `lm_head.weight` is required.
- Tiny no-cache forward hidden states/logits match the reference fixture within tolerance.
- Model forward dispatch uses the no-cache inference mask path (`base 4D causal mask` plus `update_causal_mask_with_pad_non_visible_2d`) when appropriate; helper-only tests from Slice 3 are not the wiring proof.
- Concrete `build_qwen2` registers `"qwen2.5-3b"` with `kind="llm"` exactly once when `mlx_cv.backbones.llm.qwen2.modeling` is imported; package-root/config-only imports and LocateAnything Stage-1 imports must not trigger this registration or import `mlx`.

**Touches:** `src/mlx_cv/backbones/llm/qwen2/modeling.py`, `src/mlx_cv/parity/fixtures.py`, `tools/mint_qwen2_fixture.py`, `tests/fixtures/qwen2_tiny_fixture.npz`, `tests/fixtures/qwen2_tiny_fixture_weights.npz`, `tests/test_qwen2_model.py`, `tests/test_qwen2_parity.py`.

**Produces:** Usable no-cache Qwen2 language model, tied-logit output, concrete LLM registry builder, and committed tiny Qwen2 reference fixtures.

**Verification:** `uv run pytest tests/test_qwen2_model.py tests/test_qwen2_parity.py`

**Depends on:** Slice 3

### Slice 5: Append-Only KV Cache

**Objective:** Add cache objects and wire cached one-token decode through attention and the decoder stack.

**Acceptance criteria:**
- Cache state stores un-repeated K/V tensors per layer as `(B, kv_heads, T, head_dim)`.
- Cache update happens after RoPE and before KV-head repetition, matching reference ordering.
- One-token AR cached decode hidden states/logits match the equivalent full-sequence suffix within tolerance only for the reference early-return AR path, using a fixture whose last token is not `text_mask_token_id`.
- SDLM cache generation-window masking uses `update_causal_mask_for_one_gen_window_2d` and is tested as a separate mask-dispatch fixture, not as the AR full-suffix equality case.
- Attention mask width accounts for prior cached sequence length and fails clearly on incompatible shapes.

**Touches:** `src/mlx_cv/backbones/llm/qwen2/{cache.py,modeling.py}`, `tests/test_qwen2_cache.py`, `tests/test_qwen2_parity.py`.

**Produces:** Generation-ready append-only cache for later PBD work.

**Verification:** `uv run pytest tests/test_qwen2_cache.py tests/test_qwen2_parity.py`

**Depends on:** Slice 4

### Slice 6: Convert, Load, and Fixture Minting

**Objective:** Add declarative Qwen2 weight conversion/loading against the already-minted tiny fixture and prove tied `lm_head` dropping is lossless.

**Acceptance criteria:**
- `convert_qwen2_state_dict` maps reference Qwen2 keys to local paths, preserving `model.layers.*` structure and dropping tied `lm_head` intentionally.
- `load_qwen2_weights(model, path)` loads the tiny reference weights through shared convert machinery.
- The fixture from Slice 4 or a conversion test proves `lm_head.weight` is tied to or identical with `embed_tokens.weight`, or omits `lm_head.weight`, so dropping it is lossless.
- q/k/v projection bias keys are converted and loaded; o/gate/up/down bias keys are absent as expected.
- Loaded MLX model matches fixture no-cache, mask, and cache-step outputs within tolerance.

**Touches:** `src/mlx_cv/backbones/llm/qwen2/convert.py`, `src/mlx_cv/backbones/llm/qwen2/__init__.py`, `tests/test_qwen2_convert.py`, `tests/test_qwen2_parity.py`.

**Produces:** Qwen2 conversion/load path over the committed tiny parity artifacts.

**Verification:** `uv run pytest tests/test_qwen2_convert.py tests/test_qwen2_parity.py`

**Depends on:** Slice 5

### Slice 7: LocateAnything Integration and Guardrails

**Objective:** Close the change by proving Qwen2 registration, Stage-1 LocateAnything compatibility, dependency cleanliness, and full suite health.

**Acceptance criteria:**
- After explicitly importing `mlx_cv.backbones.llm.qwen2.modeling` (or a lazy modeling symbol that imports that submodule), `BACKBONES.list(kind="llm")` includes `qwen2.5-3b`; vision listings are unchanged.
- Registration happens once with the concrete builder from Slice 4; no placeholder registration can collide with `Registry.register` duplicate-key behavior.
- Existing `tests/test_la_config.py`, `tests/test_la_convert.py`, and `tests/test_la_decode.py` pass unchanged or with only necessary config-alias updates.
- `models/locateanything`, `mlx_cv.backbones.llm.qwen2`, and `mlx_cv.backbones.llm.qwen2.config` remain importable without `mlx`; `core/` remains mlx-free by regex and `sys.modules` smoke.
- `pyproject.toml` contains no `torch` or `transformers`.
- Full `uv run pytest` passes.

**Touches:** `src/mlx_cv/backbones/llm/__init__.py`, `src/mlx_cv/models/locateanything/`, `tests/test_la_config.py`, `tests/test_la_convert.py`, `tests/test_la_decode.py`, `tests/test_registry.py`, package guard tests if needed.

**Produces:** Final integration proof and package boundary guardrails.

**Verification:** `uv run pytest && uv run python -c "from pathlib import Path; s=Path('pyproject.toml').read_text(); assert 'torch' not in s and 'transformers' not in s" && uv run python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"`

**Depends on:** Slice 6

## Execution Routing and Topology

Default: direct, serial, continuation after each slice verification passes.

Overrides:
- Slice 6: subagent recommended only if parity drift diagnosis expands beyond the planned Qwen2 files and fixtures.

Checkpoints: none.

Parallel-safe groups: none. The implementation is intentionally bottom-up: config -> leaves -> attention/masks -> stack -> cache -> convert/parity -> guardrails.

## Requirement Traceability

| SPEC acceptance | Satisfying slices |
| --- | --- |
| AC1 canonical config, local attention/default reconciliation, and LocateAnything reuse | Slice 1 |
| AC2 RMSNorm/SwiGLU/RoPE/GQA and bias-layout unit coverage | Slices 2, 3 |
| AC3 atomic fixture mint and no-cache hidden/logit parity | Slice 4 |
| AC4 cache-step parity | Slice 5 |
| AC5 block-mask helpers and model-forward dispatch | Slices 3, 4, 5 |
| AC6 convert/load and tied lm_head drop | Slice 6 |
| AC7 concrete LLM registry and mlx-free Stage-1 imports | Slices 1, 4, 7 |
| AC8 focused/full tests and dependency guards | Slice 7 |

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Config and registry | `uv run pytest tests/test_la_config.py tests/test_qwen2_config.py tests/test_registry.py` |
| Leaves | `uv run pytest tests/test_qwen2_layers.py` |
| Attention and masks | `uv run pytest tests/test_qwen2_attention.py tests/test_qwen2_masks.py` |
| Model and no-cache parity | `uv run pytest tests/test_qwen2_model.py tests/test_qwen2_parity.py` |
| Cache parity | `uv run pytest tests/test_qwen2_cache.py tests/test_qwen2_parity.py` |
| Convert and final parity | `uv run pytest tests/test_qwen2_convert.py tests/test_qwen2_parity.py` |
| Final suite | `uv run pytest` |
| Dependency guard | `uv run python -c "from pathlib import Path; s=Path('pyproject.toml').read_text(); assert 'torch' not in s and 'transformers' not in s"` |
| Core MLX-free guard | `uv run python -c "import sys, mlx_cv.core; assert not any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules)"` |

## Risks

- Cache correctness is the main runtime risk: RoPE position IDs, mask widths, and un-repeated KV storage must agree with full forward and reference cache semantics.
- Block masks can silently pass shape checks while getting visibility wrong. Slice 3 must compare actual `0.0`/`-inf` mask positions, not only output shapes.
- Tied logits must remain explicit so dropping `language_model.lm_head.weight` in the existing LocateAnything converter stays correct.
- The reference Magi path is not a valid local parity oracle; fixture minting must force SDPA/fp32 and keep Magi as deferred scope.
- Fixture minting must happen before the first parity assertion and both Qwen2 `.npz` files must come from the same seeded reference instance.
- Slice 3 must stay independent of torch-generated artifacts; any reference-oracle parity begins only after Slice 4 mints fixtures.
- The mint host, not runtime package dependencies, must provide torch plus a compatible transformers release; runtime tests consume committed `.npz` fixtures and do not import those packages.
- This plan does not make LocateAnything usable end to end. It intentionally produces the LLM foundation only; MoonViT/projector/PBD require later framed changes.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan is sliced in the right dependency order and ties each risky Qwen2 behavior to focused verification before full integration.
- Concern: Execution can still drift if the canonical config does not explicitly reconcile the reference `_attn_implementation="magi"` setting with the local manual-SDPA path and `use_cache` default.
- Action: Proceed to `auto-execute`, making Slice 1 pin and test the supported local attention/cache defaults before implementing masks or cache parity.
- Verified: Read PLAN/SPEC/DESIGN, checked local LocateAnything config/convert/stub/registry files, inspected reference Qwen2 leaves, decoder/cache order, block-mask helpers, and config defaults.

## Review: Claude Code

- Verdict: needs_correction
- Findings: Claude Code identified that the original plan did not pin the reference parity oracle to SDPA/fp32, omitted the base 4D causal mask builder used before LocateAnything mask helpers, left model-forward mask dispatch ownership unclear, risked duplicate placeholder/concrete registry registration, and did not prove tied `lm_head` dropping was lossless.
- Action: Applied plan/spec/design tailoring before execution: Slice 1 now owns local SDPA/default reconciliation and import smoke, Slices 3-5 own helper plus dispatch coverage, fixture minting and tied-logit proof are explicitly assigned in the later corrected slices, and Slice 7 owns single concrete registration.
- Evidence: Claude Code Opus session `cd94e5e3-88b6-4ac4-97c7-6e14b963ea3f` inspected the active plan trio, local LocateAnything/Qwen2/registry stubs, and `references/LocateAnything-3B/{config.json,configuration_qwen2.py,modeling_qwen2.py,mask_sdpa_utils.py}`.

## Review: Claude Code Recheck

- Verdict: needs_correction
- Findings: Claude Code verified the first corrections but found a hard ordering blocker: Slice 4 used Qwen2 parity fixtures before the mint tool and weights fixture appeared in Slice 6, which would stall serial execution and could split activation and weight fixtures across different seeded reference instances.
- Action: Applied a second plan tailoring: Slice 4 now owns atomic SDPA/fp32 fixture minting for both `.npz` files before first parity, Slice 6 now only owns convert/load against those fixtures, mask branch-driving fixture token conditions are explicit, bias layout is specified, mask visibility comparison handles `finfo.min`, and `text_mask_token_id` source consistency is covered.
- Evidence: Claude Code Opus session `cd94e5e3-88b6-4ac4-97c7-6e14b963ea3f` re-read the revised plan trio and checked reference Qwen2 bias layout, mask dispatch, mask helpers, logits/tie behavior, config defaults, and local LocateAnything config/convert exports.

## Review: Claude Code Third Check

- Verdict: needs_correction
- Findings: Claude Code found the same artifact-ordering class one slice earlier: Slice 3 expected reference attention/mask artifacts before Slice 4 mints any torch-oracle fixtures, and concrete registration ownership was still implicit.
- Action: Applied a third plan tailoring: Slice 3 now uses self-contained numpy/hand-expected attention and mask tests with all torch-oracle parity deferred to Slice 4, training block-diff remains helper-only, `find_pred_pos_from_input_ids` is deferred as training-loss scope, concrete `build_qwen2` registration is assigned to Slice 4, and Slice 1/Slice 7 registry tests are explicitly separated.
- Evidence: Claude Code Opus session `cd94e5e3-88b6-4ac4-97c7-6e14b963ea3f` re-read the twice-corrected plan trio and checked reference Qwen2 mask dispatch, attention implementation, bias layout, training-only helper usage, config defaults, and local registry/import surfaces.

## Review: Claude Code Fourth Check

- Verdict: needs_correction
- Findings: Claude Code found a Python import-mechanics contradiction: `models/locateanything/config.py` importing `backbones.llm.qwen2.config` still executes `qwen2/__init__.py`, so package-root import cannot both trigger modeling registration and remain mlx-free.
- Action: Applied the exact correction: `qwen2/__init__.py` and `qwen2.config` must stay mlx-free and export config only, concrete `build_qwen2` registration is triggered by importing `mlx_cv.backbones.llm.qwen2.modeling`, and Slice 1/Slice 7 import-smoke and registry assertions now reflect that split.
- Evidence: Claude Code Opus session `cd94e5e3-88b6-4ac4-97c7-6e14b963ea3f` checked the third-tailored plan plus local package/import surfaces, registry import-time behavior, and reference Qwen2 config/modeling files.

## Review: Claude Code Fifth Check

- Verdict: approved_with_risks
- Findings: Claude Code found no remaining blockers after the fourth correction, but noted one mint-host risk: fixture minting needs torch plus a compatible transformers version because the reference imports private transformers APIs.
- Action: Added the mint-host dependency and fixture provenance requirement to SPEC/DESIGN/PLAN, and clarified that final base-mask parity is checked against fixture-captured masks by visibility rather than a live transformers utility.
- Evidence: Claude Code Opus session `cd94e5e3-88b6-4ac4-97c7-6e14b963ea3f` checked the fourth-tailored plan, reference import dependencies, Magi fallback feasibility, local import discipline, registry behavior, and Qwen2/mask reference code.
