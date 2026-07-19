# PLAN: mlx-cv 0.0.3 and Hugging Face Model Publishing

Change: `2026-07-18-mlx-cv-003-hugging-face-release` — Spec: `SPEC.md`

## Goal

Prepare a verified `mlx-cv==0.0.3` release and two safe, self-contained BF16 Hugging Face model packages, then stop at the external-publication checkpoint.

## Architecture approach

Use one lazy resolver for local directories, exact Hub IDs, aliases, revisions, and offline cache semantics. Model packages carry final-layout Safetensors and normalized runtime assets; ordinary loading performs validation but no source-format conversion. A declarative registry drives staging, verification, cards, uploads, and remote verification so repository names and file contracts cannot drift across scripts.

## Ordered slice sequence

### Slice 1: Hub resolution and package-native model APIs

**Objective:** Implement the shared resolver and package-based LocateAnything and SAM 3.1 constructors without remote code execution.
**Acceptance criteria:** Local directories, exact repository IDs, aliases, revisions, and offline behavior work; optional dependencies fail clearly; LocateAnything returns the existing `Result`; SAM direct-checkpoint compatibility remains supported.
**Verification:** `.venv/bin/python -m pytest -q tests/test_hub_resolver.py tests/test_locateanything_pipeline.py tests/test_sam31_pretrained.py`
**Depends on:** none
**Checkpoint after:** none
**Status:** complete
**Evidence:** Added the lazy local/alias/exact-ID/revision/offline resolver, package-native LocateAnything pipeline, SAM image/video constructors, and upstream chat-template formatting; 10 focused tests passed.

### Slice 2: Strict BF16 artifact and staging contracts

**Objective:** Define strict LocateAnything metadata and reproducible, atomic, allowlisted packages for both models.
**Acceptance criteria:** LocateAnything emits 769 final-layout BF16 tensors with source checksum and format contract; SAM validates its 1963-tensor contract; staged packages contain required configs/tokenizer/BPE/license/card files only; SHA256 manifests verify.
**Verification:** `.venv/bin/python -m pytest -q tests/test_locateanything_safetensors.py tests/test_huggingface_release.py`
**Depends on:** Slice 1
**Checkpoint after:** none
**Status:** complete
**Evidence:** Finalized the 769-tensor LocateAnything BF16 checkpoint with strict metadata and source SHA256, retained the strict 1963-tensor SAM contract, and staged both real packages atomically with verified allowlists and SHA256 manifests.

### Slice 3: Registry-driven release command and model cards

**Objective:** Add tracked App Automaton cards and a single command for list, stage, verify, upload, and verify-remote workflows.
**Acceptance criteria:** Both repositories are registry-defined with lowercase precision names; cards have valid YAML and accurate licenses/backlinks/limitations; upload is sequential with one worker; existing repos are refused unless exact `--resume`; no command deletes or changes visibility.
**Verification:** `.venv/bin/python -m pytest -q tests/test_huggingface_release.py && .venv/bin/python tools/huggingface_release.py list`
**Depends on:** Slice 2
**Checkpoint after:** none
**Status:** complete
**Evidence:** Added the two tracked cards and the registry-driven list/stage/verify/upload/verify-remote command with one worker, safe existing-repo refusal, exact resume semantics, and no delete/visibility operations.

### Slice 4: Version 0.0.3, documentation, and distributions

**Objective:** Make source, package metadata, README, release notes, and build artifacts consistently release `mlx-cv==0.0.3` with the new Hub workflow.
**Acceptance criteria:** All authoritative version references are 0.0.3; README documents install, aliases, repository table, licenses, and release workflow; wheel/sdist metadata is correct; clean-wheel smoke test passes.
**Verification:** `.venv/bin/python -m pytest -q tests/test_version.py tests/test_readme.py && rm -rf /tmp/mlx-cv-dist && .venv/bin/python -m build --outdir /tmp/mlx-cv-dist && .venv/bin/python -m twine check /tmp/mlx-cv-dist/*`
**Depends on:** Slice 3
**Checkpoint after:** none
**Status:** complete
**Evidence:** Bumped all authoritative version references and lock metadata to 0.0.3, documented Hub use and releases, built wheel/sdist, passed Twine, and passed an isolated wheel install/API smoke test.

### Slice 5: Local regression and release-candidate verification

**Objective:** Verify staged packages, the full suite, distribution install, and available real BF16 parity gates sequentially without exhausting the machine.
**Acceptance criteria:** Full tests and diff hygiene pass; staged manifests verify; four LocateAnything samples match recorded token/geometry baselines; SAM package gates meet existing thresholds; unavailable heavyweight prerequisites are reported distinctly rather than counted as passes.
**Verification:** `.venv/bin/python -m pytest -q && git diff --check && .venv/bin/python tools/huggingface_release.py verify --all`
**Depends on:** Slice 4
**Checkpoint after:** human-action
**Checkpoint reason:** PyPI/GitHub release creation, public Hugging Face repository creation, and uploads mutate external state and require explicit authorization plus confirmed credentials.
**Status:** complete — checkpoint reached
**Evidence:** Full suite passed (431 passed, 12 expected skips); required SAM package parity passed 5 tests; the four-case LocateAnything real-image gate reproduced exact tokens and geometry; both final staged manifests and diff hygiene passed.

### Slice 6: Authorized public release and remote verification

**Objective:** After authorization, publish PyPI 0.0.3, verify a fresh install, then publish LocateAnything followed by SAM 3.1 and verify each remote snapshot.
**Acceptance criteria:** PyPI serves 0.0.3; both exact public repositories contain only verified files; remote manifests, licenses, YAML, Safetensors metadata, fresh-cache downloads, and inference checks pass; no automatic remote deletion or visibility change occurs.
**Verification:** `.venv/bin/python tools/huggingface_release.py verify-remote --all --fresh-cache`
**Depends on:** Slice 5 and explicit human authorization
**Checkpoint after:** human-verify
**Checkpoint reason:** The user reviews public release URLs and verification evidence.
**Status:** pending human authorization

## Execution routing and topology

- Direct, serial execution with automatic continuation through Slice 5.
- Slice 6 begins only after the explicit human-action checkpoint.
- Parallel-safe groups: none.
- Multi-gigabyte model operations and real-image gates run one at a time.
- Implementation occurs in an isolated clean worktree because the primary worktree contains unrelated edits.

## Aggregate verification

| Gate | Command |
|---|---|
| Hub APIs | `.venv/bin/python -m pytest -q tests/test_hub_resolver.py tests/test_locateanything_pipeline.py tests/test_sam31_pretrained.py` |
| Artifacts and release tool | `.venv/bin/python -m pytest -q tests/test_locateanything_safetensors.py tests/test_huggingface_release.py` |
| Packaging | `.venv/bin/python -m build --outdir /tmp/mlx-cv-dist && .venv/bin/python -m twine check /tmp/mlx-cv-dist/*` |
| Full regression | `.venv/bin/python -m pytest -q` |
| Staged packages | `.venv/bin/python tools/huggingface_release.py verify --all` |
| Hygiene | `git diff --check` |
