from __future__ import annotations

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .errors import NativeBootstrapError, NativeUnavailableError


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dotnet_probe() -> dict[str, Any]:
    executable = shutil.which("dotnet")
    if executable is None:
        return {
            "available": False,
            "executable": None,
            "runtimes": [],
            "reason": "The dotnet executable is not on PATH.",
        }
    runtimes: list[str] = []
    reason: str | None = None
    try:
        completed = subprocess.run(
            [executable, "--list-runtimes"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if completed.returncode == 0:
            runtimes = [
                line.strip()
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
        else:
            reason = completed.stderr.strip() or f"dotnet exited with {completed.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    return {
        "available": bool(runtimes),
        "executable": executable,
        "runtimes": runtimes,
        "reason": reason,
    }


def _bootstrap_module():
    try:
        return importlib.import_module("lunarscout._native_runtime.bootstrap")
    except Exception as exc:
        raise NativeUnavailableError(
            "The Lunarscout native bootstrap adapter is unavailable.",
            code="native_bootstrap_adapter_unavailable",
            details={"error": str(exc)},
        ) from exc


def _payload_probe() -> tuple[dict[str, Any], Any | None, Any | None]:
    try:
        bootstrap = _bootstrap_module()
        config = bootstrap.load_native_bootstrap_config()
        moonlib_dll = Path(bootstrap.resolve_moonlib_dll(config)).resolve()
        return (
            {"available": True, "path": str(moonlib_dll), "reason": None},
            bootstrap,
            config,
        )
    except NativeUnavailableError as exc:
        return (
            {"available": False, "path": None, "reason": str(exc)},
            None,
            None,
        )
    except Exception as exc:
        return (
            {"available": False, "path": None, "reason": str(exc)},
            None,
            None,
        )


def _configured_native_library(
    config: Any | None,
    moonlib_path: str | None,
    filenames: tuple[str, ...],
) -> Path | None:
    candidates: list[Path] = []
    if config is not None:
        candidates.extend(Path(path) for path in config.dll_resolver_search_dirs)
        candidates.extend(
            Path(path).parent for _name, path in config.dll_resolver_imports
        )
    if moonlib_path:
        candidates.append(Path(moonlib_path).parent)
    for directory in candidates:
        for filename in filenames:
            path = directory.expanduser().resolve() / filename
            if path.is_file():
                return path
    return None


def status() -> dict[str, Any]:
    """Inspect native prerequisites without initializing pythonnet or CLR."""

    pythonnet = {
        "available": _module_available("pythonnet"),
        "loaded": "pythonnet" in sys.modules,
    }
    dotnet = _dotnet_probe()
    moonlib, bootstrap, config = _payload_probe()
    bootstrap_status: dict[str, Any] = {"loaded": False}
    if bootstrap is not None:
        try:
            bootstrap_status = dict(bootstrap.bootstrap_status())
        except Exception as exc:
            bootstrap_status = {"loaded": False, "error": str(exc)}

    smoke = bootstrap_status.get("smoke_check")
    smoke = smoke if isinstance(smoke, dict) else {}
    loaded = bool(bootstrap_status.get("loaded"))
    cspice_path = _configured_native_library(
        config,
        moonlib.get("path"),
        ("libcspice.so", "cspice.dll", "libcspice.dylib"),
    )
    cspice_verified = loaded and smoke.get("spice_output") == 2
    cspice = {
        "available": bool(cspice_path) or cspice_verified,
        "path": str(cspice_path) if cspice_path is not None else None,
        "verified": cspice_verified,
    }
    gdal_verified = loaded and smoke.get("gdal_config_probe") is True
    gdal = {
        "available": _module_available("osgeo") or gdal_verified,
        "python_bindings": _module_available("osgeo"),
        "verified_by_native_smoke": gdal_verified,
    }
    moonlib["loaded"] = loaded
    moonlib["verified"] = (
        loaded and bool(smoke) and smoke.get("type") != "skipped"
    )

    components = {
        "pythonnet": pythonnet,
        "dotnet": dotnet,
        "moonlib": moonlib,
        "cspice": cspice,
        "gdal": gdal,
    }
    available = all(
        bool(components[name]["available"])
        for name in ("pythonnet", "dotnet", "moonlib", "cspice", "gdal")
    )
    return {
        "available": available,
        "loaded": loaded,
        "components": components,
        "bootstrap": bootstrap_status,
    }


def is_available() -> bool:
    """Return whether native prerequisites are discoverable without loading them."""

    return bool(status()["available"])


def _component_for_error(message: str) -> str:
    normalized = message.lower()
    if "pythonnet" in normalized or " clr " in f" {normalized} ":
        return "pythonnet"
    if "cspice" in normalized or "spice" in normalized:
        return "cspice"
    if "gdal" in normalized or "proj" in normalized:
        return "gdal"
    if (
        "coreclr" in normalized
        or "runtimeconfig" in normalized
        or "target framework" in normalized
    ):
        return "dotnet"
    if "moonlib" in normalized or "assembly" in normalized or "native artifact" in normalized:
        return "moonlib"
    return "native_runtime"


def initialize(
    *,
    force: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """Explicitly initialize the Lunarscout native runtime."""

    bootstrap = _bootstrap_module()
    try:
        bootstrap.bootstrap_pythonnet(
            force=force,
            verify_bridge_smoke=verify,
        )
    except Exception as exc:
        component = _component_for_error(str(exc))
        error_type = (
            NativeUnavailableError
            if component != "native_runtime"
            else NativeBootstrapError
        )
        raise error_type(
            f"Native initialization failed for {component}: {exc}",
            code=f"native_{component}_unavailable"
            if error_type is NativeUnavailableError
            else "native_bootstrap_failed",
            details={"component": component, "error": str(exc)},
        ) from exc
    return status()


def _create_moonlib_bridge(
    *,
    force: bool = False,
    verify: bool = True,
):
    """Create the sole supported production moonlib entry object lazily."""

    bootstrap = _bootstrap_module()
    try:
        moonlib = bootstrap.import_moonlib(
            force_bootstrap=force,
            verify_bridge_smoke=verify,
        )
        return moonlib.MoonlibBridge()
    except Exception as exc:
        component = _component_for_error(str(exc))
        raise NativeBootstrapError(
            f"Unable to create MoonlibBridge: {exc}",
            code="native_bridge_creation_failed",
            details={"component": component, "error": str(exc)},
        ) from exc


from .native_temporal import (  # noqa: E402 - keeps backend/pythonnet loading lazy
    NativeLightmapBufferPatch,
    NativeTemporalProgress,
    TemporalAllocationEstimate,
    estimate_temporal_allocation,
    generate_temporal_signal,
    stream_lightmap_buffers,
)
from .native_horizon import (  # noqa: E402 - keeps backend/pythonnet loading lazy
    GenerateHorizons,
    NativeHorizonProgress,
)
from .native_product import (  # noqa: E402 - keeps backend/pythonnet loading lazy
    NativeProductProgress,
)
