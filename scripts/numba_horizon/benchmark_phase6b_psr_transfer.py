#!/usr/bin/env python3
"""Measure pageable and pinned 94 MiB PSR horizon decode/transfer boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from numba import cuda

from lunarscout._numba_horizon.file_format import (
    AZIMUTH_COUNT,
    HorizonTileStore,
    PATCH_SIZE,
    read_horizon_tile_with_timings,
)


DEFAULT_SCENARIO = Path("/e/lunar_analyst_scenarios/mons-mouton")
HORIZON_SHAPE = (PATCH_SIZE, PATCH_SIZE, AZIMUTH_COUNT)


def _summary(values: list[float]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        "total_seconds": float(samples.sum()),
        "mean_seconds": float(samples.mean()),
        "median_seconds": float(np.median(samples)),
        "minimum_seconds": float(samples.min()),
        "maximum_seconds": float(samples.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.iterations < 1:
        raise ValueError("iterations must be positive")

    store = HorizonTileStore(arguments.scenario.expanduser().resolve() / "horizons")
    path = store.find_existing_path(0, 0, 0.0)
    if path is None:
        raise RuntimeError("the 0,0 horizon tile is missing")

    pageable = np.empty(HORIZON_SHAPE, dtype=np.float32)
    pinned = cuda.pinned_array(HORIZON_SHAPE, dtype=np.float32)
    device = cuda.device_array(HORIZON_SHAPE, dtype=np.float32)
    stream = cuda.stream()

    read_horizon_tile_with_timings(path, output=pageable)
    read_horizon_tile_with_timings(path, output=pinned)
    device.copy_to_device(pageable)
    cuda.synchronize()

    pageable_decode = []
    pinned_decode = []
    pageable_transfer = []
    pinned_transfer = []
    pinned_async_transfer = []
    for _ in range(arguments.iterations):
        started = time.perf_counter()
        read_horizon_tile_with_timings(path, output=pageable)
        pageable_decode.append(time.perf_counter() - started)

        started = time.perf_counter()
        read_horizon_tile_with_timings(path, output=pinned)
        pinned_decode.append(time.perf_counter() - started)

        started = time.perf_counter()
        device.copy_to_device(pageable)
        cuda.synchronize()
        pageable_transfer.append(time.perf_counter() - started)

        started = time.perf_counter()
        device.copy_to_device(pinned)
        cuda.synchronize()
        pinned_transfer.append(time.perf_counter() - started)

        started = time.perf_counter()
        device.copy_to_device(pinned, stream=stream)
        stream.synchronize()
        pinned_async_transfer.append(time.perf_counter() - started)

    report = {
        "schema": "lunarscout-numba-phase6b-psr-transfer-v1",
        "input": {
            "horizon_path": str(path.resolve()),
            "horizon_bytes": int(np.prod(HORIZON_SHAPE)) * 4,
            "iterations": arguments.iterations,
        },
        "pageable_decode": _summary(pageable_decode),
        "pinned_decode": _summary(pinned_decode),
        "pageable_h2d": _summary(pageable_transfer),
        "pinned_h2d": _summary(pinned_transfer),
        "pinned_async_h2d_and_sync": _summary(pinned_async_transfer),
        "identity": {
            "pageable_pinned_equal": bool(np.array_equal(pageable, pinned)),
        },
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
