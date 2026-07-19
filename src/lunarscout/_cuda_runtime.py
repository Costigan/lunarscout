"""Lazy import boundary for Lunarscout's supported CUDA installation profile."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any


CUDA_INSTALL_HINT = 'Install the CUDA runtime with: pip install "lunarscout[cuda]"'


def import_numba_cuda() -> tuple[Any, Any]:
    """Import the supported Numba-CUDA target without probing a device."""

    try:
        version("numba-cuda")
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            f"The Lunarscout CUDA runtime is not installed. {CUDA_INSTALL_HINT}"
        ) from exc

    import numba
    from numba import cuda as numba_cuda

    return numba, numba_cuda
