"""Results of a ``multiprocessing.Pool`` that fail instead of hanging when a worker process dies (private).

A worker killed by a signal (a segmentation fault, the out-of-memory killer) takes the task it was running with it. The
pool starts a replacement worker, but ``imap_unordered`` never gets that task's result and waits for it forever, with
every other worker idle. :func:`guarded_results` polls for results and, when none arrives, checks the workers the pool
started with: one that has exited is an error.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterator
from typing import Any

from ..errors import ProductCalculationError


def guarded_results(pool: Any, results: Iterator, *, poll_seconds: float = 5.0) -> Iterator:
    """The items of ``results`` (an ``imap`` or ``imap_unordered`` iterator of ``pool``), raising
    :class:`ProductCalculationError` (``code="worker_died"``) if one of the pool's worker processes exits while results
    are still due. Call it right after the pool is created, before any worker can die and be replaced."""
    workers = list(pool._pool)                              # the processes the pool started; replacements are new ones
    while True:
        try:
            yield results.next(timeout=poll_seconds)
        except StopIteration:
            return
        except multiprocessing.TimeoutError:
            dead = [(w.pid, w.exitcode) for w in workers if w.exitcode is not None]
            if dead:
                pid, code = dead[0]
                how = f"was killed by signal {-code}" if code < 0 else f"exited with status {code}"
                raise ProductCalculationError(
                    f"a worker process {how} (pid {pid}) while its task was running; the task's result is lost",
                    code="worker_died", details={"pid": pid, "exitcode": code, "dead_workers": len(dead)}) from None
