from __future__ import annotations

from collections.abc import Iterable
import hashlib
import os
from pathlib import Path
import tempfile
import tomllib
from urllib.request import Request, urlopen

from .errors import SpiceKernelError


_DEFAULT_META_KERNEL_ENV = "LUNARSCOUT_SPICE_META_KERNEL"
_DEFAULT_KERNEL_DIR_ENV = "LUNARSCOUT_SPICE_KERNEL_DIR"
_USER_AGENT = "lunarscout-spice-kernel-downloader/0.1"
_DEFAULT_KERNELS_LOADED = False
_AUTOLOAD_ENABLED = True
_DEFAULT_FURNISHED_PATHS: list[Path] = []


def _spiceypy():
    try:
        import spiceypy
    except ImportError as exc:
        raise SpiceKernelError(
            "SpiceyPy is required for SPICE kernel operations.",
            code="spiceypy_unavailable",
        ) from exc
    return spiceypy


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_kernel_manifest_path() -> Path:
    repository_path = _repository_root() / "data" / "spice" / "default_kernels.toml"
    if repository_path.exists():
        return repository_path
    return Path(__file__).resolve().parent / "data" / "spice" / "default_kernels.toml"


def default_kernel_directory() -> Path:
    override = os.environ.get(_DEFAULT_KERNEL_DIR_ENV)
    if override:
        return Path(override)
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "lunarscout" / "spice" / "kernels"
    return Path.home() / ".local" / "share" / "lunarscout" / "spice" / "kernels"


def load_default_kernel_manifest(path: str | Path | None = None) -> dict:
    manifest_path = Path(path) if path is not None else default_kernel_manifest_path()
    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpiceKernelError(
            "Default SPICE kernel manifest was not found.",
            code="spice_manifest_missing",
            details={"path": str(manifest_path)},
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise SpiceKernelError(
            "Default SPICE kernel manifest is invalid TOML.",
            code="spice_manifest_invalid",
            details={"path": str(manifest_path), "error": str(exc)},
        ) from exc
    if not isinstance(data.get("kernels"), list):
        raise SpiceKernelError(
            "Default SPICE kernel manifest must contain a kernels list.",
            code="spice_manifest_invalid",
            details={"path": str(manifest_path)},
        )
    return data


def default_kernel_entries(path: str | Path | None = None) -> list[dict]:
    data = load_default_kernel_manifest(path)
    kernels = []
    for index, raw in enumerate(data["kernels"]):
        if not isinstance(raw, dict):
            raise SpiceKernelError(
                "Default SPICE kernel entry must be a table.",
                code="spice_manifest_invalid",
                details={"index": index},
            )
        for field in ("id", "filename", "url", "kind", "load_order"):
            if field not in raw:
                raise SpiceKernelError(
                    "Default SPICE kernel entry is missing a required field.",
                    code="spice_manifest_invalid",
                    details={"index": index, "field": field},
                )
        kernels.append(dict(raw))
    return sorted(kernels, key=lambda item: int(item["load_order"]))


def _default_meta_kernel_path() -> Path:
    override = os.environ.get(_DEFAULT_META_KERNEL_ENV)
    if override:
        return Path(override)
    download_default_kernels()
    return _generate_default_meta_kernel()


def _quote_meta_kernel_value(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _generate_default_meta_kernel() -> Path:
    kernel_directory = default_kernel_directory()
    entries = default_kernel_entries()
    missing = [
        str(kernel_directory / str(entry["filename"]))
        for entry in entries
        if not (kernel_directory / str(entry["filename"])).exists()
    ]
    if missing:
        raise SpiceKernelError(
            "Default SPICE kernels are missing.",
            code="spice_default_kernels_missing",
            details={
                "kernel_directory": str(kernel_directory),
                "missing": missing,
                "environment_variable": _DEFAULT_KERNEL_DIR_ENV,
            },
        )

    meta_directory = Path(tempfile.gettempdir()) / "lunarscout_spice"
    meta_directory.mkdir(parents=True, exist_ok=True)
    meta_kernel = meta_directory / "lunarscout_default.tm"
    kernel_lines = [
        f"    '$KERNELS/{entry['filename']}'"
        for entry in entries
    ]
    meta_kernel.write_text(
        "\n".join(
            [
                "KPL/MK",
                "",
                "\\begindata",
                f"PATH_VALUES = ( {_quote_meta_kernel_value(str(kernel_directory))} )",
                "PATH_SYMBOLS = ( 'KERNELS' )",
                "KERNELS_TO_LOAD = (",
                *kernel_lines,
                ")",
                "",
                "\\begintext",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return meta_kernel


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            with urlopen(request, timeout=60.0) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        temporary_path.replace(destination)
    except Exception as exc:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise SpiceKernelError(
            "Unable to download default SPICE kernel.",
            code="spice_kernel_download_failed",
            details={"url": url, "destination": str(destination), "error": str(exc)},
        ) from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_kernel_checksum(entry: dict, path: Path) -> None:
    expected = entry.get("sha256")
    if not expected:
        return
    actual = _file_sha256(path)
    if actual.lower() != str(expected).lower():
        raise SpiceKernelError(
            "Cached or downloaded SPICE kernel checksum does not match the manifest.",
            code="spice_kernel_checksum_mismatch",
            details={
                "path": str(path),
                "filename": str(entry["filename"]),
                "expected_sha256": str(expected),
                "actual_sha256": actual,
            },
        )


def download_default_kernels(
    *,
    overwrite: bool = False,
    kernel_directory: str | Path | None = None,
) -> list[Path]:
    directory = Path(kernel_directory) if kernel_directory is not None else default_kernel_directory()
    downloaded: list[Path] = []
    for entry in default_kernel_entries():
        destination = directory / str(entry["filename"])
        if destination.exists() and not overwrite:
            _verify_kernel_checksum(entry, destination)
            continue
        _download_file(str(entry["url"]), destination)
        _verify_kernel_checksum(entry, destination)
        downloaded.append(destination)
    return downloaded


def _paths(path_or_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        return [Path(path_or_paths)]
    try:
        return [Path(path) for path in path_or_paths]
    except TypeError as exc:
        raise SpiceKernelError(
            "SPICE kernels must be a path or iterable of paths.",
            code="spice_kernel_invalid_paths",
            details={"type": type(path_or_paths).__name__},
        ) from exc


def furnish(
    path_or_paths: str | Path | Iterable[str | Path],
    *,
    disable_autoload: bool = True,
) -> None:
    spiceypy = _spiceypy()
    paths = _paths(path_or_paths)
    if not paths:
        raise SpiceKernelError(
            "At least one SPICE kernel path is required.",
            code="spice_kernel_empty_paths",
        )
    for path in paths:
        spiceypy.furnsh(str(path))
    if disable_autoload:
        set_autoload_enabled(False)


def ensure_default_kernels() -> None:
    global _DEFAULT_KERNELS_LOADED
    if _DEFAULT_KERNELS_LOADED or not _AUTOLOAD_ENABLED:
        return
    path = _default_meta_kernel_path()
    if not path.exists():
        raise SpiceKernelError(
            "Default SPICE meta-kernel is not available.",
            code="spice_default_meta_kernel_missing",
            details={
                "path": str(path),
                "environment_variable": _DEFAULT_META_KERNEL_ENV,
            },
        )
    spiceypy = _spiceypy()
    spiceypy.furnsh(str(path))
    _DEFAULT_FURNISHED_PATHS.clear()
    _DEFAULT_FURNISHED_PATHS.append(path)
    _DEFAULT_KERNELS_LOADED = True


def unload_default_kernels() -> None:
    global _DEFAULT_KERNELS_LOADED
    spiceypy = _spiceypy()
    for path in reversed(_DEFAULT_FURNISHED_PATHS):
        spiceypy.unload(str(path))
    _DEFAULT_FURNISHED_PATHS.clear()
    _DEFAULT_KERNELS_LOADED = False


def reload_default_kernels() -> None:
    set_autoload_enabled(True)
    unload_default_kernels()
    ensure_default_kernels()


def clear_kernels() -> None:
    global _DEFAULT_KERNELS_LOADED, _AUTOLOAD_ENABLED
    spiceypy = _spiceypy()
    spiceypy.kclear()
    _DEFAULT_FURNISHED_PATHS.clear()
    _DEFAULT_KERNELS_LOADED = False
    _AUTOLOAD_ENABLED = True


def default_kernels_loaded() -> bool:
    return _DEFAULT_KERNELS_LOADED


def autoload_enabled() -> bool:
    return _AUTOLOAD_ENABLED


def set_autoload_enabled(enabled: bool) -> None:
    global _AUTOLOAD_ENABLED
    _AUTOLOAD_ENABLED = bool(enabled)
