import ast
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11.
    import tomli as tomllib


FORBIDDEN_RUNTIME_IMPORTS = {"tools", "torch", "torchvision", "transformers"}
REPOSITORY_ONLY_PACKAGES = {
    "build",
    "pytest",
    "torch",
    "torchvision",
    "transformers",
    "twine",
}
RUNTIME_DEPENDENCY_BLOCKLIST = (
    "torch",
    "torchvision",
    "transformers",
    "triton",
    "cuda",
)
RUNTIME_IMPORT_BLOCKLIST = RUNTIME_DEPENDENCY_BLOCKLIST + (
    "ftfy",
    "huggingface_hub",
    "iopath",
    "references",
    "requests",
    "rfdetr",
    "tools",
    "urllib",
)


def test_repository_tooling_stays_outside_runtime_source_tree():
    candidates = (
        Path("src/mlx_cv/hub/release.py"),
        *Path("src/mlx_cv/parity").glob("*.py"),
    )
    forbidden_sources = [path for path in candidates if path.is_file()]
    assert not forbidden_sources, forbidden_sources


def _requirement_name(requirement: str) -> str:
    return requirement.split(";", 1)[0].split("[", 1)[0].split("@", 1)[0].strip()


def test_published_dependencies_exclude_repository_only_packages():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    project = pyproject["project"]
    requirements = list(project.get("dependencies", ()))
    for extra_requirements in project.get("optional-dependencies", {}).values():
        requirements.extend(extra_requirements)

    forbidden = {
        name
        for requirement in requirements
        if (name := _requirement_name(requirement).lower())
        in REPOSITORY_ONLY_PACKAGES
    }
    assert not forbidden, (
        "repository-only packages leaked into package metadata: "
        f"{sorted(forbidden)}"
    )


def test_runtime_package_sources_do_not_hard_import_external_or_repository_modules():
    # Top-level imports make optional integrations mandatory at module import
    # time. Indented imports inside explicit Hub/reference entry points remain
    # lazy and are covered by their actionable dependency-error tests.
    import_re = re.compile(r"^(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)
    for path in Path("src/mlx_cv").rglob("*.py"):
        text = path.read_text()
        imports = {m.group(1).split(".", 1)[0] for m in import_re.finditer(text)}
        assert not (imports & set(RUNTIME_IMPORT_BLOCKLIST)), path


def test_runtime_source_does_not_import_reference_or_repository_tooling():
    violations: list[str] = []
    for path in sorted(Path("src/mlx_cv").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = {node.module.split(".", 1)[0]}
            else:
                continue
            forbidden = imports & FORBIDDEN_RUNTIME_IMPORTS
            if forbidden:
                violations.append(f"{path}:{node.lineno}: {sorted(forbidden)}")

    assert not violations, (
        "reference-framework or repository-tool imports found in runtime source:\n"
        + "\n".join(violations)
    )


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
