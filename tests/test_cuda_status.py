from __future__ import annotations

from types import SimpleNamespace

import lunarscout as ls
from lunarscout._cuda_runtime import CUDA_INSTALL_HINT


def test_cuda_capability_helpers_are_available_in_focused_namespace() -> None:
    assert ls.CudaStatus is ls.cuda.CudaStatus
    assert callable(ls.cuda.is_available)
    assert callable(ls.cuda.status)


def test_cuda_unavailable_probe_is_nonthrowing(monkeypatch) -> None:
    fake_numba = SimpleNamespace(__version__="test-numba")
    fake_cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(
        ls.cuda,
        "_numba_modules",
        lambda: (fake_numba, fake_cuda),
    )

    assert ls.cuda.is_available() is False
    assert ls.cuda.status() == ls.CudaStatus(
        available=False,
        numba_version="test-numba",
        device_name=None,
        compute_capability=None,
        reason="Numba CUDA is unavailable.",
    )


def test_cuda_available_status_reports_selected_device(monkeypatch) -> None:
    device = SimpleNamespace(name=b"Example GPU", compute_capability=(9, 1))
    memory = SimpleNamespace(free=12_000, total=16_000)
    fake_numba = SimpleNamespace(__version__="test-numba")
    fake_cuda = SimpleNamespace(
        __file__="/site-packages/numba_cuda/numba/cuda/__init__.py",
        is_available=lambda: True,
        current_context=lambda: SimpleNamespace(
            device=device, get_memory_info=lambda: memory
        ),
        cudadrv=SimpleNamespace(
            driver=SimpleNamespace(
                driver=SimpleNamespace(get_version=lambda: (12, 9))
            )
        ),
    )
    versions = {"numba-cuda": "test-cuda", "cuda-toolkit": "12.9"}
    monkeypatch.setattr(
        ls.cuda,
        "_numba_modules",
        lambda: (fake_numba, fake_cuda),
    )
    monkeypatch.setattr(ls.cuda, "_distribution_version", versions.get)

    assert ls.cuda.is_available() is True
    assert ls.cuda.status() == ls.CudaStatus(
        available=True,
        numba_version="test-numba",
        device_name="Example GPU",
        compute_capability=(9, 1),
        reason=None,
        numba_cuda_version="test-cuda",
        cuda_toolkit_version="12.9",
        cuda_driver_version="12.9",
        free_memory_bytes=12_000,
        total_memory_bytes=16_000,
    )


def test_cuda_probe_failure_is_nonthrowing(monkeypatch) -> None:
    def fail():
        raise RuntimeError("driver probe failed")

    monkeypatch.setattr(ls.cuda, "_numba_modules", fail)

    assert ls.cuda.is_available() is False
    assert ls.cuda.status() == ls.CudaStatus(
        available=False,
        numba_version=None,
        device_name=None,
        compute_capability=None,
        reason="driver probe failed",
    )


def test_base_install_cuda_probe_explains_the_cuda_extra(monkeypatch) -> None:
    def missing_runtime():
        raise ModuleNotFoundError(
            f"The Lunarscout CUDA runtime is not installed. {CUDA_INSTALL_HINT}"
        )

    monkeypatch.setattr(ls.cuda, "_numba_modules", missing_runtime)

    assert ls.cuda.is_available() is False
    assert ls.cuda.status().reason == (
        'The Lunarscout CUDA runtime is not installed. Install the CUDA runtime '
        'with: pip install "lunarscout[cuda]"'
    )
