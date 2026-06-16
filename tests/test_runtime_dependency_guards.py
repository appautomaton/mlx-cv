import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11.
    import tomli as tomllib


RUNTIME_DEPENDENCY_BLOCKLIST = ("torch", "transformers", "triton", "cuda")
RUNTIME_IMPORT_BLOCKLIST = RUNTIME_DEPENDENCY_BLOCKLIST + (
    "ftfy",
    "huggingface_hub",
    "iopath",
    "references",
    "requests",
    "rfdetr",
    "urllib",
)
PARITY_STATUS = Path(".agent/work/2026-06-16-release-parity-hardening/parity-status.json")


def test_pyproject_runtime_dependencies_exclude_reference_frameworks():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    text = "\n".join(pyproject["project"]["dependencies"]).lower()
    for name in RUNTIME_DEPENDENCY_BLOCKLIST:
        assert name not in text


def test_runtime_package_sources_do_not_hard_import_reference_frameworks():
    import_re = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)
    for path in Path("src/mlx_cv").rglob("*.py"):
        text = path.read_text()
        imports = {m.group(1).split(".", 1)[0] for m in import_re.finditer(text)}
        assert not (imports & set(RUNTIME_IMPORT_BLOCKLIST)), path


def test_runtime_package_sources_do_not_inject_reference_paths():
    sys_path_re = re.compile(r"\bsys\.path\.(?:insert|append)\s*\(")
    for path in Path("src/mlx_cv").rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            assert not sys_path_re.search(line), f"{path}:{lineno}"


def test_package_root_import_does_not_load_reference_frameworks():
    code = (
        "import sys\n"
        "import mlx_cv\n"
        "blocked = ('torch', 'transformers', 'triton', 'ftfy', 'iopath')\n"
        "assert not any(m == b or m.startswith(b + '.') for b in blocked for m in sys.modules)\n"
    )
    subprocess.check_call([sys.executable, "-c", code])


def test_release_parity_status_matrix_is_bounded():
    status = json.loads(PARITY_STATUS.read_text())
    assert status["phase"] == "release-parity-hardening"
    assert status["default_tolerance"] == {
        "atol": 0.0001,
        "rtol": 0.0001,
        "max_without_replan": 0.001,
    }
    assert set(status["models"]) == {"locateanything", "rfdetr", "sam3_image"}

    for model in status["models"].values():
        value = model["status"]
        assert value in {"LOCAL_FIXTURE_ONLY", "UPSTREAM_PASSED"} or value.startswith("BLOCKED:")
        assert model["checkpoint_env"]
        assert model["reference_path"]
        assert model["local_fixture"]
        assert "license" in model["license_note"].lower()
