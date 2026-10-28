"""A pool whose worker dies fails with an error instead of hanging."""

from __future__ import annotations

import multiprocessing
import os
import signal

import pytest

from lunarscout._numba_horizon.pool_guard import guarded_results
from lunarscout.errors import ProductCalculationError


def _square(n: int) -> int:
    return n * n


def _die_on_three(n: int) -> int:
    if n == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return n


def test_results_pass_through_when_no_worker_dies() -> None:
    with multiprocessing.get_context("fork").Pool(2) as pool:
        results = guarded_results(pool, pool.imap_unordered(_square, range(10), chunksize=1), poll_seconds=0.2)
        assert sorted(results) == [n * n for n in range(10)]


def test_a_worker_killed_by_a_signal_raises_instead_of_hanging() -> None:
    pool = multiprocessing.get_context("fork").Pool(2)
    try:
        results = guarded_results(pool, pool.imap_unordered(_die_on_three, range(6), chunksize=1), poll_seconds=0.2)
        with pytest.raises(ProductCalculationError) as error:
            list(results)
        assert error.value.code == "worker_died"
        assert error.value.details["exitcode"] == -signal.SIGKILL and "signal 9" in str(error.value)
    finally:
        pool.terminate()
        pool.join()
