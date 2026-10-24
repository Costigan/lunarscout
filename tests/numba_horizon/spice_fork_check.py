"""Run by test_safe_haven_blocks.py in a fresh process: do five forked workers, set up the way
``run_pass_two`` sets them up, read the same SPICE kernels without corrupting each other?

Prints one JSON line, ``{"wrong": n, "errors": [...]}``, and exits 0. Exits 3 with a reason when
SPICE or the default kernels are not installed."""

from __future__ import annotations

import json
import multiprocessing
import sys

import numpy as np

try:
    import spiceypy  # noqa: F401
    from lunarscout import spice
    from lunarscout._numba_horizon import safe_haven_blocks as blocks
    from lunarscout._numba_horizon.safe_haven_windows import spice_vector_source
except ImportError as error:
    print(f"spiceypy is not installed: {error}")
    raise SystemExit(3)

BATCHES: list = []
TRUTH: list = []


def task(seed):
    sun = spice_vector_source("sun")
    rng = np.random.default_rng(seed)
    wrong, errors = 0, []
    for _ in range(300):
        index = int(rng.integers(len(BATCHES)))
        try:
            wrong += not np.array_equal(sun(BATCHES[index]), TRUTH[index])
        except Exception as error:                      # SpiceINVALIDRADIUS, without the fix
            errors.append(type(error).__name__)
    return wrong, errors


def main() -> int:
    if not spice._default_meta_kernel_path().exists():
        print("the default SPICE kernels are not installed")
        return 3
    sun = spice_vector_source("sun")
    start = np.datetime64("2024-01-01T00:00:00", "us")
    BATCHES[:] = [start + np.arange(45) * np.timedelta64(32, "m") + np.timedelta64(int(k) * 37, "h")
                  for k in range(1500)]
    TRUTH[:] = [sun(batch) for batch in BATCHES]        # the parent loads the kernels here
    kernels = blocks._loaded_spice_kernels()
    if not kernels:
        print("no SPICE kernels were loaded")
        return 3
    with multiprocessing.get_context("fork").Pool(
            5, initializer=blocks._worker_init, initargs=(1, kernels)) as pool:
        results = pool.map(task, range(10))
    print(json.dumps({"wrong": sum(wrong for wrong, _ in results),
                      "errors": [error for _, errors in results for error in errors]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
