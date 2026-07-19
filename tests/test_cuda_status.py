from __future__ import annotations

from types import SimpleNamespace

import lunarscout as ls


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
    fake_numba = SimpleNamespace(__version__="test-numba")
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        current_context=lambda: SimpleNamespace(device=device),
    )
    monkeypatch.setattr(
        ls.cuda,
        "_numba_modules",
        lambda: (fake_numba, fake_cuda),
    )

    assert ls.cuda.is_available() is True
    assert ls.cuda.status() == ls.CudaStatus(
        available=True,
        numba_version="test-numba",
        device_name="Example GPU",
        compute_capability=(9, 1),
        reason=None,
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
