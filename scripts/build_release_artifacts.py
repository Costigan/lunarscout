#!/usr/bin/env python3
"""Build and inspect Lunarscout distributions from a clean source snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
REQUIRED_PACKAGE_DATA = "lunarscout/data/spice/default_kernels.toml"
FORBIDDEN_PARTS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "build",
    "dist",
    "native",
}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".cbin",
    ".cs",
    ".csproj",
    ".dll",
    ".h5",
    ".hdf5",
    ".journal",
    ".pyc",
    ".pyo",
    ".tif",
    ".tiff",
}
FORBIDDEN_NAME_FRAGMENTS = ("_native_runtime", "moonlib", "native_", "pythonnet")


def _run(
    command: list[str], *, cwd: Path = REPOSITORY, echo: bool = True
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if echo and result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        if not echo and result.stdout:
            print(result.stdout, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout
        )
    return result.stdout


def _git_output(*arguments: str) -> str:
    return _run(["git", *arguments], echo=False).strip()


def _copy_tracked_snapshot(destination: Path) -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY,
        check=True,
        stdout=subprocess.PIPE,
    )
    for encoded_path in result.stdout.split(b"\0"):
        if not encoded_path:
            continue
        relative = Path(os.fsdecode(encoded_path))
        source = REPOSITORY / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _validate_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member path: {name}")
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & FORBIDDEN_PARTS:
        raise ValueError(f"forbidden artifact path: {name}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"forbidden artifact file type: {name}")
    if any(fragment in name.lower() for fragment in FORBIDDEN_NAME_FRAGMENTS):
        raise ValueError(f"forbidden managed-runtime artifact path: {name}")
    return path


def inspect_wheel(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    if not names:
        raise ValueError(f"wheel is empty: {path}")

    dist_info_roots = {
        PurePosixPath(name).parts[0]
        for name in names
        if PurePosixPath(name).parts
        and PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    if len(dist_info_roots) != 1:
        raise ValueError(f"wheel must contain exactly one .dist-info tree: {path}")
    dist_info = next(iter(dist_info_roots))

    for name in names:
        member = _validate_member_path(name)
        if name.endswith("/"):
            continue
        root = member.parts[0]
        if root == "lunarscout":
            if member.suffix not in {".py", ".toml"}:
                raise ValueError(f"unexpected package file in wheel: {name}")
        elif root != dist_info:
            raise ValueError(f"unexpected top-level wheel path: {name}")

    required = {
        "lunarscout/__init__.py",
        REQUIRED_PACKAGE_DATA,
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/RECORD",
        f"{dist_info}/licenses/LICENSE",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"wheel is missing required files: {missing}")
    return {"entries": len(names), "dist_info": dist_info}


def inspect_sdist(path: Path) -> dict[str, object]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
    names = [member.name for member in members]
    directories = {member.name for member in members if member.isdir()}
    if not names:
        raise ValueError(f"source distribution is empty: {path}")

    roots = {PurePosixPath(name).parts[0] for name in names if name}
    if len(roots) != 1:
        raise ValueError(f"sdist must contain exactly one root directory: {path}")
    root = next(iter(roots))
    allowed_root_files = {
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "setup.cfg",
    }

    for name in names:
        member = _validate_member_path(name)
        if name in directories:
            continue
        if len(member.parts) <= 1:
            continue
        relative = PurePosixPath(*member.parts[1:])
        if len(relative.parts) == 1 and relative.name in allowed_root_files:
            continue
        if relative.parts[:2] == ("src", "lunarscout"):
            continue
        if relative.parts[:2] == ("src", "lunarscout.egg-info"):
            continue
        raise ValueError(f"unexpected source-distribution path: {name}")

    required = {
        f"{root}/LICENSE",
        f"{root}/MANIFEST.in",
        f"{root}/README.md",
        f"{root}/pyproject.toml",
        f"{root}/src/lunarscout/__init__.py",
        f"{root}/src/{REQUIRED_PACKAGE_DATA}",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"sdist is missing required files: {missing}")
    return {"entries": len(names), "root": root}


def _artifact_record(path: Path, inspection: dict[str, object]) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        **inspection,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory",
        type=Path,
        help="new or empty output directory outside the repository",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a development build and record that it is not releasable",
    )
    parser.add_argument(
        "--skip-twine",
        action="store_true",
        help="skip twine check for a development-only diagnostic build",
    )
    parser.add_argument(
        "--upload-target",
        choices=("none", "testpypi"),
        default="none",
        help="record the intended target; this script never uploads",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    output = arguments.output_directory.resolve()
    if output == REPOSITORY or REPOSITORY in output.parents:
        raise SystemExit("output directory must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")

    dirty_text = _git_output("status", "--porcelain=v1")
    dirty = bool(dirty_text)
    if dirty and not arguments.allow_dirty:
        raise SystemExit("release builds require a clean working tree")

    commit = _git_output("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="lunarscout-release-source-") as temporary:
        source = Path(temporary)
        _copy_tracked_snapshot(source)
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--wheel",
                "--outdir",
                str(output),
            ],
            cwd=source,
        )

    wheels = sorted(output.glob("*.whl"))
    sdists = sorted(output.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("build must produce exactly one wheel and one .tar.gz sdist")
    if not arguments.skip_twine:
        _run([sys.executable, "-m", "twine", "check", str(wheels[0]), str(sdists[0])])

    artifacts = [
        _artifact_record(wheels[0], inspect_wheel(wheels[0])),
        _artifact_record(sdists[0], inspect_sdist(sdists[0])),
    ]
    report = {
        "schema": "lunarscout-release-artifacts-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "dirty_worktree": dirty,
        "candidate_artifacts": not dirty and not arguments.skip_twine,
        "upload_target": arguments.upload_target,
        "python": sys.version,
        "platform": platform.platform(),
        "artifacts": artifacts,
    }
    report_path = output / "release-artifacts.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
