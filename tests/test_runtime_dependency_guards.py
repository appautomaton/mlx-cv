import re
import subprocess
import sys
from pathlib import Path


RUNTIME_DEPENDENCY_BLOCKLIST = ("torch", "transformers", "triton", "cuda")
RUNTIME_IMPORT_BLOCKLIST = RUNTIME_DEPENDENCY_BLOCKLIST + ("ftfy", "iopath")


def test_pyproject_runtime_dependencies_exclude_reference_frameworks():
    text = Path("pyproject.toml").read_text().lower()
    for name in RUNTIME_DEPENDENCY_BLOCKLIST:
        assert name not in text


def test_runtime_package_sources_do_not_hard_import_reference_frameworks():
    import_re = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)
    for path in Path("src/mlx_cv").rglob("*.py"):
        text = path.read_text()
        imports = {m.group(1).split(".", 1)[0] for m in import_re.finditer(text)}
        assert not (imports & set(RUNTIME_IMPORT_BLOCKLIST)), path


def test_package_root_import_does_not_load_reference_frameworks():
    code = (
        "import sys\n"
        "import mlx_cv\n"
        "blocked = ('torch', 'transformers', 'triton', 'ftfy', 'iopath')\n"
        "assert not any(m == b or m.startswith(b + '.') for b in blocked for m in sys.modules)\n"
    )
    subprocess.check_call([sys.executable, "-c", code])
