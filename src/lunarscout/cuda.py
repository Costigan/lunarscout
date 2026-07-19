"""Lazy CUDA capability and diagnostic helpers.

Importing this module does not import Numba or initialize CUDA. Capability
probing begins only when :func:`is_available` or :func:`status` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any


@dataclass(frozen=True, slots=True)
class CudaStatus:
    """Read-only result of an explicit Numba CUDA capability probe."""

    available: bool
    numba_version: str | None
    device_name: str | None
    compute_capability: tuple[int, int] | None
    reason: str | None
    numba_cuda_version: str | None = None
    cuda_toolkit_version: str | None = None
    cuda_driver_version: str | None = None
    free_memory_bytes: int | None = None
    total_memory_bytes: int | None = None


def _numba_modules() -> tuple[Any, Any]:
    from ._cuda_runtime import import_numba_cuda

    return import_numba_cuda()


def _device_name(device: Any) -> str | None:
    value = getattr(device, "name", None)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return None if value is None else str(value)


def _distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_versions(numba_cuda: Any) -> tuple[str | None, str | None]:
    module_path = str(getattr(numba_cuda, "__file__", ""))
    if "numba_cuda" not in module_path:
        return None, None
    return _distribution_version("numba-cuda"), _distribution_version("cuda-toolkit")


def _driver_version(numba_cuda: Any) -> str | None:
    try:
        value = numba_cuda.cudadrv.driver.driver.get_version()
        return ".".join(str(int(part)) for part in value)
    except Exception:
        return None


def _memory_info(context: Any) -> tuple[int | None, int | None]:
    try:
        value = context.get_memory_info()
        return int(value.free), int(value.total)
    except Exception:
        return None, None


def is_available() -> bool:
    """Return whether Numba reports a usable CUDA device.

    This is an explicit capability probe and may initialize CUDA driver state.
    It does not prove which backend a later product invocation used.
    """

    try:
        _numba, numba_cuda = _numba_modules()
        return bool(numba_cuda.is_available())
    except Exception:
        return False


def status() -> CudaStatus:
    """Return lazy CUDA capability details without raising probe failures."""

    try:
        numba, numba_cuda = _numba_modules()
    except Exception as exc:
        return CudaStatus(False, None, None, None, str(exc))

    version = str(getattr(numba, "__version__", "unknown"))
    numba_cuda_version, cuda_toolkit_version = _runtime_versions(numba_cuda)
    try:
        if not numba_cuda.is_available():
            return CudaStatus(
                False,
                version,
                None,
                None,
                "Numba CUDA is unavailable.",
                numba_cuda_version=numba_cuda_version,
                cuda_toolkit_version=cuda_toolkit_version,
            )
        context = numba_cuda.current_context()
        device = context.device
        free_memory_bytes, total_memory_bytes = _memory_info(context)
        capability_value = getattr(device, "compute_capability", None)
        capability = (
            None
            if capability_value is None
            else (int(capability_value[0]), int(capability_value[1]))
        )
        return CudaStatus(
            True,
            version,
            _device_name(device),
            capability,
            None,
            numba_cuda_version=numba_cuda_version,
            cuda_toolkit_version=cuda_toolkit_version,
            cuda_driver_version=_driver_version(numba_cuda),
            free_memory_bytes=free_memory_bytes,
            total_memory_bytes=total_memory_bytes,
        )
    except Exception as exc:
        return CudaStatus(
            False,
            version,
            None,
            None,
            str(exc),
            numba_cuda_version=numba_cuda_version,
            cuda_toolkit_version=cuda_toolkit_version,
        )
