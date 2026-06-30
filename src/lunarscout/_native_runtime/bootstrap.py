from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


MOONLIB_DLL_ENV = "LUNARSCOUT_MOONLIB_DLL"
DOTNET_RUNTIME_CONFIG_ENV = "LUNARSCOUT_DOTNET_RUNTIME_CONFIG"
NUGET_PACKAGES_ENV = "NUGET_PACKAGES"
_LINUX_RID = "linux-x64"


class NativeBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeBootstrapConfig:
    moonlib_dll: Path | None = None
    dotnet_runtime_config: Path | None = None
    expected_target_framework: str = "net10.0"
    build_profile: str = "debug"
    dll_resolver_search_dirs: tuple[Path, ...] = field(default_factory=tuple)
    dll_resolver_imports: tuple[tuple[str, Path], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NativeRuntimeHandle:
    runtime: str
    moonlib_dll: Path
    dotnet_runtime_config: Path | None
    expected_target_framework: str
    smoke_check: dict[str, Any]


_BOOTSTRAP_CACHE: NativeRuntimeHandle | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def reset_bootstrap_cache() -> None:
    global _BOOTSTRAP_CACHE
    _BOOTSTRAP_CACHE = None


def load_native_bootstrap_config() -> NativeBootstrapConfig:
    moonlib_raw = os.environ.get(MOONLIB_DLL_ENV)
    runtime_raw = os.environ.get(DOTNET_RUNTIME_CONFIG_ENV)
    moonlib = Path(moonlib_raw).expanduser().resolve() if moonlib_raw else None
    runtime_config = Path(runtime_raw).expanduser().resolve() if runtime_raw else None
    search_dirs: list[Path] = []
    if moonlib is not None:
        search_dirs.append(moonlib.parent)
    return NativeBootstrapConfig(
        moonlib_dll=moonlib,
        dotnet_runtime_config=runtime_config,
        dll_resolver_search_dirs=tuple(search_dirs),
    )


def _candidate_moonlib_dirs(repo_root: Path) -> list[Path]:
    return [
        repo_root / "native" / "moonlib" / "bin" / "Debug" / "net10.0" / _LINUX_RID,
        repo_root / "native" / "moonlib" / "bin" / "Debug" / "net10.0",
        repo_root / "native" / "moonlib" / "bin" / "Release" / "net10.0" / _LINUX_RID,
        repo_root / "native" / "moonlib" / "bin" / "Release" / "net10.0",
    ]


def resolve_moonlib_dll(config: NativeBootstrapConfig | None = None) -> Path:
    cfg = config or load_native_bootstrap_config()
    if cfg.moonlib_dll is not None:
        if not cfg.moonlib_dll.is_file():
            raise NativeBootstrapError(f"Configured moonlib DLL does not exist: {cfg.moonlib_dll}")
        return cfg.moonlib_dll

    for directory in _candidate_moonlib_dirs(_repo_root()):
        candidate = directory / "moonlib.dll"
        if candidate.is_file():
            return candidate.resolve()

    raise NativeBootstrapError(
        "Unable to locate moonlib.dll. Build native/moonlib or set LUNARSCOUT_MOONLIB_DLL."
    )


def resolve_runtimeconfig(
    config: NativeBootstrapConfig | None = None,
    *,
    moonlib_dll: Path | None = None,
) -> Path | None:
    cfg = config or load_native_bootstrap_config()
    if cfg.dotnet_runtime_config is not None:
        if not cfg.dotnet_runtime_config.is_file():
            raise NativeBootstrapError(
                f"Configured .NET runtimeconfig does not exist: {cfg.dotnet_runtime_config}"
            )
        return cfg.dotnet_runtime_config
    dll = moonlib_dll or resolve_moonlib_dll(cfg)
    candidate = dll.with_name("moonlib.runtimeconfig.json")
    return candidate if candidate.is_file() else None


def _configure_probe_path(moonlib_dll: Path) -> None:
    directory = str(moonlib_dll.parent)
    current_path = os.environ.get("PATH", "")
    if directory not in current_path.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current_path
    if sys.platform.startswith("linux"):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if directory not in current_ld.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = directory + os.pathsep + current_ld
    if directory not in sys.path:
        sys.path.insert(0, directory)


def run_bridge_smoke_check() -> dict[str, Any]:
    try:
        import moonlib  # type: ignore

        smoke = getattr(moonlib, "BridgeSmoke", None)
        bridge = getattr(moonlib, "MoonlibBridge", None)
        spice_output = None if smoke is None else int(smoke.SpiceSmokeTest(1))
        gdal_probe = False
        if bridge is not None:
            try:
                _ = bridge.GdalSmokeTest()
                gdal_probe = True
            except Exception:
                gdal_probe = False
        return {
            "type": "bridge_smoke",
            "spice_output": spice_output,
            "gdal_config_probe": gdal_probe,
        }
    except Exception as exc:
        return {"type": "failed", "error": str(exc)}


def bootstrap_status() -> dict[str, Any]:
    return {
        "loaded": _BOOTSTRAP_CACHE is not None,
        "moonlib_dll": None if _BOOTSTRAP_CACHE is None else str(_BOOTSTRAP_CACHE.moonlib_dll),
        "smoke_check": None if _BOOTSTRAP_CACHE is None else _BOOTSTRAP_CACHE.smoke_check,
    }


def bootstrap_pythonnet(
    config: NativeBootstrapConfig | None = None,
    *,
    force: bool = False,
    verify_bridge_smoke: bool = True,
) -> NativeRuntimeHandle:
    global _BOOTSTRAP_CACHE
    if _BOOTSTRAP_CACHE is not None and not force:
        return _BOOTSTRAP_CACHE

    cfg = config or load_native_bootstrap_config()
    moonlib_dll = resolve_moonlib_dll(cfg)
    runtime_config = resolve_runtimeconfig(cfg, moonlib_dll=moonlib_dll)

    try:
        from pythonnet import load as pythonnet_load
    except Exception as exc:
        raise NativeBootstrapError(
            "pythonnet is unavailable. Install lunarscout[native] or pythonnet."
        ) from exc

    try:
        if runtime_config is None:
            pythonnet_load("coreclr")
        else:
            pythonnet_load("coreclr", runtime_config=str(runtime_config))
    except Exception as exc:
        raise NativeBootstrapError("Failed to initialize pythonnet coreclr runtime.") from exc

    try:
        import clr  # type: ignore
    except Exception as exc:
        raise NativeBootstrapError("pythonnet clr module is unavailable after runtime load.") from exc

    _configure_probe_path(moonlib_dll)

    try:
        clr.AddReference(str(moonlib_dll))  # type: ignore[attr-defined]
    except Exception as exc:
        raise NativeBootstrapError(f"Failed to add moonlib assembly reference: {moonlib_dll}") from exc

    smoke_check = run_bridge_smoke_check() if verify_bridge_smoke else {"type": "skipped"}
    handle = NativeRuntimeHandle(
        runtime="coreclr",
        moonlib_dll=moonlib_dll,
        dotnet_runtime_config=runtime_config,
        expected_target_framework=cfg.expected_target_framework,
        smoke_check=smoke_check,
    )
    _BOOTSTRAP_CACHE = handle
    return handle


def import_moonlib(
    *,
    force_bootstrap: bool = False,
    verify_bridge_smoke: bool = True,
) -> Any:
    bootstrap_pythonnet(force=force_bootstrap, verify_bridge_smoke=verify_bridge_smoke)
    try:
        import moonlib  # type: ignore
    except Exception as exc:
        raise NativeBootstrapError("Failed to import moonlib after pythonnet bootstrap.") from exc
    return moonlib

