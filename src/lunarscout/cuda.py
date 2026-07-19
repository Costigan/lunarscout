"""Lazy CUDA capability and diagnostic helpers.

Importing this module does not import Numba or initialize CUDA. Capability
probing begins only when :func:`is_available` or :func:`status` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CudaStatus:
    """Read-only result of an explicit Numba CUDA capability probe."""

    available: bool
    numba_version: str | None
    device_name: str | None
    compute_capability: tuple[int, int] | None
    reason: str | None


def _numba_modules() -> tuple[Any, Any]:
    import numba
    from numba import cuda as numba_cuda

    return numba, numba_cuda


def _device_name(device: Any) -> str | None:
    value = getattr(device, "name", None)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return None if value is None else str(value)


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
    try:
        if not numba_cuda.is_available():
            return CudaStatus(
                False,
                version,
                None,
                None,
                "Numba CUDA is unavailable.",
            )
        context = numba_cuda.current_context()
        device = context.device
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
        )
    except Exception as exc:
        return CudaStatus(False, version, None, None, str(exc))
