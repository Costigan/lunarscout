from __future__ import annotations

import ast
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
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
    assert "rasterio>=1.4.4,<1.6" in dependencies
    assert any(requirement.startswith("spiceypy") for requirement in dependencies)
    assert project["version"] == "0.1.0rc1"
    assert any(requirement.startswith("build") for requirement in extras["dev"])
    assert any(requirement.startswith("twine") for requirement in extras["dev"])
    assert not any("pythonnet" in requirement.lower() for requirement in all_requirements)
    assert not any(requirement.startswith("h5py") for requirement in all_requirements)
    assert not any(requirement.startswith("hdf5plugin") for requirement in all_requirements)


def test_root_version_comes_from_installed_distribution_metadata() -> None:
    import lunarscout as ls

    assert ls.__version__ == version("lunarscout")


def test_curated_root_does_not_import_or_export_managed_runtime() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, lunarscout as ls; "
                "assert not hasattr(ls, 'native'); "
                "assert not hasattr(ls, 'GenerateHorizons'); "
                "assert 'lunarscout.native' not in sys.modules; "
                "assert 'lunarscout.native_horizon' not in sys.modules; "
                "assert 'lunarscout._native_runtime.bootstrap' not in sys.modules"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
