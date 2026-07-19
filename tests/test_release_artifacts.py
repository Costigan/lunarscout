from __future__ import annotations

import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts.build_release_artifacts import inspect_sdist, inspect_wheel


def _write_wheel(path: Path, *, extra: str | None = None) -> None:
    dist_info = "lunarscout-0.1.0rc1.dist-info"
    names = {
        "lunarscout/__init__.py": b"",
        "lunarscout/data/spice/default_kernels.toml": b"",
        f"{dist_info}/METADATA": b"",
        f"{dist_info}/WHEEL": b"",
        f"{dist_info}/RECORD": b"",
        f"{dist_info}/licenses/LICENSE": b"",
    }
    if extra is not None:
        names[extra] = b""
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, payload in names.items():
            archive.writestr(name, payload)


def _write_sdist(path: Path, *, extra: str | None = None) -> None:
    root = "lunarscout-0.1.0rc1"
    names = [
        "LICENSE",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "src/lunarscout/__init__.py",
        "src/lunarscout/data/spice/default_kernels.toml",
    ]
    if extra is not None:
        names.append(extra)
    with tarfile.open(path, mode="w:gz") as archive:
        for relative in names:
            info = tarfile.TarInfo(f"{root}/{relative}")
            info.size = 0
            archive.addfile(info, io.BytesIO())


def test_distribution_inspectors_accept_minimal_artifacts(tmp_path: Path) -> None:
    wheel = tmp_path / "lunarscout.whl"
    sdist = tmp_path / "lunarscout.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    assert inspect_wheel(wheel)["entries"] == 6
    assert inspect_sdist(sdist)["entries"] == 6


@pytest.mark.parametrize(
    "extra",
    [
        "lunarscout/native_product.py",
        "lunarscout/generated.tif",
        "tests/test_unexpected.py",
    ],
)
def test_wheel_inspector_rejects_unexpected_content(
    tmp_path: Path, extra: str
) -> None:
    wheel = tmp_path / "lunarscout.whl"
    _write_wheel(wheel, extra=extra)

    with pytest.raises(ValueError):
        inspect_wheel(wheel)


@pytest.mark.parametrize(
    "extra",
    [
        "native/moonlib.csproj",
        "build/lib/lunarscout/stale.py",
        "tests/test_unexpected.py",
    ],
)
def test_sdist_inspector_rejects_unexpected_content(
    tmp_path: Path, extra: str
) -> None:
    sdist = tmp_path / "lunarscout.tar.gz"
    _write_sdist(sdist, extra=extra)

    with pytest.raises(ValueError):
        inspect_sdist(sdist)
