"""LocateAnything checkpoint admission and upstream parity gate helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


LOCATEANYTHING_CHECKPOINT_ENV = "MLX_CV_LOCATEANYTHING_CHECKPOINT"
LOCATEANYTHING_REQUIRED_GATE_ENV = "MLX_CV_REQUIRE_LOCATEANYTHING_GATE"
LOCATEANYTHING_REFERENCE_PATH = Path("references/LocateAnything-3B")


@dataclass(frozen=True)
class LocateAnythingGateResult:
    status: str
    checkpoint_env: str
    required_gate_env: str
    reference_path: str
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    blocked_reason: str | None = None
    admitted: bool = False

    @property
    def blocked(self) -> bool:
        return self.status.startswith("BLOCKED:")


def required_gate_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(LOCATEANYTHING_REQUIRED_GATE_ENV) == "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _block(reason: str, *, environ: Mapping[str, str]) -> LocateAnythingGateResult:
    return LocateAnythingGateResult(
        status=f"BLOCKED:{reason}",
        checkpoint_env=LOCATEANYTHING_CHECKPOINT_ENV,
        required_gate_env=LOCATEANYTHING_REQUIRED_GATE_ENV,
        reference_path=str(LOCATEANYTHING_REFERENCE_PATH),
        checkpoint_path=environ.get(LOCATEANYTHING_CHECKPOINT_ENV),
        blocked_reason=reason,
    )


def _admit(path: Path, *, environ: Mapping[str, str], sha256: str | None = None) -> LocateAnythingGateResult:
    return LocateAnythingGateResult(
        status="ADMITTED",
        checkpoint_env=LOCATEANYTHING_CHECKPOINT_ENV,
        required_gate_env=LOCATEANYTHING_REQUIRED_GATE_ENV,
        reference_path=str(LOCATEANYTHING_REFERENCE_PATH),
        checkpoint_path=str(path),
        checkpoint_sha256=sha256,
        admitted=True,
    )


def _index_shards(index_path: Path) -> list[str] | None:
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None
    shards = sorted(set(index.get("weight_map", {}).values()))
    return shards or None


def evaluate_locateanything_gate(
    *,
    environ: Mapping[str, str] | None = None,
    min_shard_bytes: int = 1_000_000,
) -> LocateAnythingGateResult:
    env = os.environ if environ is None else environ
    checkpoint = env.get(LOCATEANYTHING_CHECKPOINT_ENV)
    if not checkpoint:
        return _block(f"{LOCATEANYTHING_CHECKPOINT_ENV} is unset", environ=env)

    path = Path(checkpoint)
    if not path.exists():
        return _block(f"{LOCATEANYTHING_CHECKPOINT_ENV} does not point to an existing path: {path}", environ=env)

    if path.is_file():
        if path.suffix not in {".npz", ".safetensors"}:
            return _block(f"unsupported LocateAnything checkpoint format: {path.suffix or path.name}", environ=env)
        if path.stat().st_size < min_shard_bytes:
            return _block(f"{path} is not a usable LocateAnything checkpoint file", environ=env)
        return _admit(path, environ=env, sha256=_sha256(path))

    if not path.is_dir():
        return _block(f"{LOCATEANYTHING_CHECKPOINT_ENV} is neither a file nor a directory: {path}", environ=env)

    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        return _block(f"LocateAnything checkpoint directory is missing {index_path.name}: {path}", environ=env)

    shards = _index_shards(index_path)
    if not shards:
        return _block(f"LocateAnything safetensors index has no weight_map entries: {index_path}", environ=env)

    missing = [shard for shard in shards if not (path / shard).exists()]
    if missing:
        return _block(f"LocateAnything checkpoint directory is missing shard(s): {', '.join(missing[:3])}", environ=env)

    stub_shards = [shard for shard in shards if (path / shard).stat().st_size < min_shard_bytes]
    if stub_shards:
        return _block(
            f"LocateAnything checkpoint shard(s) are LFS stubs or incomplete: {', '.join(stub_shards[:3])}",
            environ=env,
        )

    return _admit(path, environ=env)


def status_dict(result: LocateAnythingGateResult) -> dict:
    out = asdict(result)
    out["model"] = "locateanything"
    out["display_name"] = "LocateAnything-3B"
    out["claim_level"] = "external_blocker" if result.blocked else "checkpoint_admitted"
    return out
