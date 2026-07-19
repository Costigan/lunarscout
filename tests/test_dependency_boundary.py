from __future__ import annotations

import ast
from pathlib import Path
import tomllib


FORBIDDEN_IMPORT_ROOTS = {"backend", "lunar_analyst"}


def test_lunarscout_source_does_not_import_lunar_analyst_modules() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "lunarscout"
    violations: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{path.relative_to(source_root)} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(f"{path.relative_to(source_root)} imports from {node.module}")

    assert violations == []


def test_python_only_package_metadata_has_no_managed_runtime_dependency() -> None:
    repository = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((repository / "pyproject.toml").read_text())
    project = metadata["project"]
    dependencies = tuple(project["dependencies"])
    extras = project["optional-dependencies"]
    all_requirements = dependencies + tuple(
        requirement
        for values in extras.values()
        for requirement in values
    )

    assert any(requirement.startswith("numba") for requirement in dependencies)
    assert any(requirement.startswith("spiceypy") for requirement in dependencies)
    assert not any("pythonnet" in requirement.lower() for requirement in all_requirements)
    assert not any(requirement.startswith("h5py") for requirement in all_requirements)
    assert not any(requirement.startswith("hdf5plugin") for requirement in all_requirements)
