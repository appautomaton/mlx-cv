# SPEC: mlx-cv 0.0.3 and Hugging Face Model Publishing

Change: `2026-07-18-mlx-cv-003-hugging-face-release`

**Bet:** A strict, reusable Hub contract plus reproducible model packages will let Apple Silicon users load the verified BF16 LocateAnything and SAM 3.1 models directly without inheriting conversion-time complexity.

## Bounded goal

Prepare `mlx-cv==0.0.3` and two independently verifiable public Hugging Face packages for `appautomaton/locateanything-3b-bf16-mlx` and `appautomaton/sam3.1-multiplex-bf16-mlx`, stopping before external publication for explicit human authorization.

## Broader intent

Make App Automaton model releases consistent across projects: final-layout Safetensors, coherent runtime APIs, tracked model cards, licenses and provenance, deterministic staging, and cautious sequential uploads.

## Work scale and shape

Large cross-subsystem release migration spanning runtime resolution, model APIs, artifact contracts, release tooling, documentation, packaging, and real-model regression verification.

## Selected lenses

- Product: simple and consistent `from_pretrained` usage.
- Engineering: strict local/Hub resolution and reproducible staging.
- Security: no remote code and explicit remote mutation checkpoint.
- Runtime: final-layout BF16 execution on MLX Metal.
- Content: accurate model cards, licenses, provenance, limitations, and backlinks.

## Stakeholders

Apple Silicon users of `mlx-cv`, App Automaton maintainers, and downstream users of the two Hugging Face repositories.

## Required outcome

1. A lazy Hub resolver accepts local package directories, exact Hugging Face repository IDs, revisions, offline cache behavior, and documented aliases without executing remote code.
2. `LocateAnythingPipeline.from_pretrained`, `SAM3Processor.from_pretrained`, and `SAM3VideoSession.from_pretrained` load self-contained model packages while preserving supported direct local-checkpoint paths.
3. LocateAnything is restaged as a strict 769-tensor final-layout BF16 Safetensors package with source checksum and complete metadata; SAM 3.1 uses its existing strict 1963-tensor BF16 checkpoint contract.
4. A registry-driven release command can list, stage, verify, upload, and verify remote packages, using atomic ignored staging directories, SHA256 manifests, exact allowlists, one upload worker, and safe resume semantics.
5. Tracked App Automaton model cards describe MLX, BF16 unquantized weights, base models, tasks, Apple Silicon requirements, `mlx-cv==0.0.3`, parity evidence, limitations, memory/latency observations, licenses, and project backlinks.
6. Package metadata, source version, tests, README, and release notes consistently name version 0.0.3; wheel and sdist pass packaging checks and a clean install smoke test.
7. Local package verification includes the four recorded LocateAnything real images sequentially and the existing SAM 3.1 real image/video gates.
8. No PyPI release, GitHub release, Hugging Face repository creation, upload, visibility change, or deletion occurs before a human-action checkpoint.

## Acceptance criteria

- Unit tests cover aliases, exact IDs, revisions, local paths, offline cache failures, optional dependencies, strict metadata, allowlists, manifests, card YAML, license presence, and safe upload/resume behavior.
- The full test suite passes.
- `python -m build` and `python -m twine check dist/*` pass, followed by a clean-wheel import/from-pretrained smoke test.
- LocateAnything BF16 outputs preserve the recorded generated tokens and geometry for four real images when the required local assets are available.
- SAM 3.1 package-based image and video gates meet the already-approved parity thresholds when required local assets are available.
- Local staged roots contain only the declared package files and their manifest checksums verify.
- Execution stops with commands and evidence ready for the human to authorize external release actions.

## Constraints and risks

- The main worktree contains unrelated Automaton updates and an unrelated `uv.lock` rewrite; release implementation must be isolated and must not absorb them.
- Multi-gigabyte weights stay outside Git. Staging must stream/copy safely and remain ignored.
- LocateAnything uses the NVIDIA non-commercial license; SAM 3.1 public redistribution must include the complete Meta SAM License prominently.
- The official SAM 3.1 repository is gated, but the user explicitly selected a public App Automaton derivative repository with bundled license and provenance.
- `https://appautomaton.github.io/mlx-cv/` is not a valid backlink at framing time; use the GitHub project and App Automaton homepage unless the project site becomes live.
- Hugging Face dependencies remain optional and fail with actionable errors.

## Scope coverage

Included: `mlx-cv` 0.0.3, two BF16 model repositories, runtime Hub loading, staging/verification/upload tooling, cards, licenses, documentation, packaging, and local regression verification.

Deferred: actual PyPI/GitHub/Hugging Face publication until explicit authorization.

## Anti-goals

- Do not publish RF-DETR, Depth Anything 3, int8, 4-bit, NPZ, PyTorch checkpoints, upstream source trees, training assets, or large example media.
- Do not add `trust_remote_code`, silently reuse a mismatched existing remote repository, delete remote files, or change repository visibility automatically.
- Do not call BF16 a quantized format; it is an unquantized reduced-precision checkpoint.
- Do not claim universal accuracy, latency, or memory guarantees from the recorded local samples.
