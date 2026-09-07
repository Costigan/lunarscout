"""Benchmark reference Dijkstra against private Numba block relaxation.

This is development evidence, not a public API example. The benchmark uses a
deterministic synthetic projected lunar grid and verifies full-field parity
before reporting timings.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
from time import perf_counter

import numba
import numpy as np
from pyproj import CRS

import lunarscout as ls
from lunarscout.trajectory._block_cpu import block_static_travel_time
from lunarscout.trajectory._static_reference import dijkstra_field
from lunarscout.trajectory._validation import prepare_static_problem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 384])
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def _georef(size: int) -> ls.GeoReference:
    crs = CRS.from_user_input("ESRI:103878")
    return ls.GeoReference(
        projection_wkt=crs.to_wkt(),
        projection_proj4=crs.to_proj4(),
        affine_transform=(-1920.0, 20.0, 2.0, 1920.0, -1.0, -20.0),
        width=size,
        height=size,
        pixel_size_x=20.0,
        pixel_size_y=-20.0,
        nodata=None,
    )


def _problem(size: int):
    rows, columns = np.indices((size, size), dtype=np.float64)
    elevation = 0.025 * columns + 0.01 * rows
    elevation += 2.0 * np.sin(columns / 17.0) * np.cos(rows / 23.0)
    traversable = np.ones((size, size), dtype=np.bool_)
    for column in range(31, size - 1, 47):
        traversable[:, column] = False
        traversable[(column * 7) % size, column] = True
    model = ls.trajectory.StaticTravelModel(
        speed_m_per_h=36.0,
        slip=ls.trajectory.SlipFunction(
            (-0.5, 0.0, 0.5),
            (0.8, 1.0, 2.0),
            extrapolation="constant",
        ),
    )
    return prepare_static_problem(
        traversable,
        _georef(size),
        (0, 0),
        elevation=elevation,
        model=model,
    )


def _timed(call, repeats: int) -> tuple[list[float], object]:
    durations: list[float] = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        durations.append(perf_counter() - start)
    return durations, result


def main() -> None:
    args = _parser().parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        raise SystemExit("sizes must be >= 2 and repeats must be positive")
    if any(size < 1 for size in args.block_sizes):
        raise SystemExit("block sizes must be positive")

    # Compile the fixed-signature kernel before collecting warm measurements.
    warm_problem = _problem(2)
    cold_start = perf_counter()
    block_static_travel_time(warm_problem, block_width=1, block_height=1)
    jit_seconds = perf_counter() - cold_start

    cases = []
    for size in args.sizes:
        problem = _problem(size)
        reference_seconds, reference = _timed(
            lambda: dijkstra_field(problem), args.repeats
        )
        assert isinstance(reference, np.ndarray)
        block_cases = []
        for block_size in args.block_sizes:
            block_seconds, block_result = _timed(
                lambda block_size=block_size: block_static_travel_time(
                    problem,
                    block_width=block_size,
                    block_height=block_size,
                ),
                args.repeats,
            )
            np.testing.assert_array_equal(
                np.isfinite(block_result.travel_time_hours),
                np.isfinite(reference),
            )
            finite = np.isfinite(reference)
            maximum_delta = float(
                np.max(
                    np.abs(
                        block_result.travel_time_hours[finite] - reference[finite]
                    )
                )
            )
            block_cases.append(
                {
                    "block_size": block_size,
                    "seconds": block_seconds,
                    "median_seconds": float(np.median(block_seconds)),
                    "activation_count": block_result.activation_count,
                    "relaxation_pass_count": block_result.relaxation_pass_count,
                    "maximum_finite_delta_hours": maximum_delta,
                    "state_bytes": int(
                        block_result.travel_time_hours.nbytes
                        + (
                            0
                            if block_result.predecessor_x is None
                            else block_result.predecessor_x.nbytes
                        )
                        + (
                            0
                            if block_result.predecessor_y is None
                            else block_result.predecessor_y.nbytes
                        )
                        + block_result.block_visits.nbytes
                        + block_result.block_improvements.nbytes
                    ),
                }
            )
        cases.append(
            {
                "size": size,
                "reference_seconds": reference_seconds,
                "reference_median_seconds": float(np.median(reference_seconds)),
                "block_cases": block_cases,
            }
        )

    report = {
        "benchmark": "trajectory_static_cpu",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "jit_warmup_seconds": jit_seconds,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "repeats": args.repeats,
        "cases": cases,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
