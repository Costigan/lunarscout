#!/usr/bin/env python3
"""Benchmark the bounded Phase 3 host-side segment preparation prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numba
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.contract import (  # noqa: E402
    SEGMENT_FIELDS,
    load_reference_artifact,
)
from lunarscout._numba_horizon.geometry import (  # noqa: E402
    RAY_SAMPLE_FIELDS,
    DemGrid,
    ProjectionParameters,
    build_ray_samples,
    fit_ray_segment,
)
from lunarscout._numba_horizon.geometry_numba import (  # noqa: E402
    fit_segments_parallel,
    fit_segments_serial,
    generate_segments_parallel,
    generate_segments_serial,
)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _timings(function, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return values


def _summary(values: list[float], jobs: int) -> dict[str, object]:
    median = statistics.median(values)
    return {
        "seconds": values,
        "median_seconds": median,
        "jobs_per_second_from_median": jobs / median,
    }


def _load_jobs():
    data = REPOSITORY / "tests" / "data" / "numba_horizon"
    artifact = load_reference_artifact(
        data / "phase1_reference_rays.json", data / "phase1_reference_rays.npz"
    )
    jobs = []
    for case in artifact.metadata["cases"]:
        observer = case["observer"]
        dems = []
        for dem_info in case["dems"]:
            prefix = f"{case['id']}__dem_{dem_info['index']}"
            dems.append(
                DemGrid(
                    artifact.arrays[f"{prefix}__elevation_m"],
                    artifact.arrays[f"{prefix}__geo_transform"],
                    ProjectionParameters.from_array(
                        artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
                    ),
                )
            )
        terrain = dems[0].elevation(observer["pixel_x"], observer["pixel_y"])
        radius = dems[0].projection.radius_m + terrain
        for fit_index, fit in enumerate(case["ray_fit_passes"]):
            prefix = f"{case['id']}__ray_fit_pass_{fit_index}"
            samples = np.column_stack(
                [
                    artifact.arrays[f"{prefix}__sample_{field}"]
                    for field in RAY_SAMPLE_FIELDS
                ]
            )
            jobs.append(
                (
                    samples,
                    float(fit["map_resolution_m"]),
                    artifact.arrays[f"{prefix}__observer_vector_moon_centered_m"],
                    radius,
                    float(fit["requested_start_distance_m"]),
                    float(fit["ray_limit_m"]),
                    artifact.arrays[f"{prefix}__nominal_direction_moon_centered"],
                    dems[fit["dem_index"]],
                )
            )
    return jobs


def _batch(jobs, count: int):
    samples = np.zeros((count, 16, len(RAY_SAMPLE_FIELDS)), dtype=np.float64)
    counts = np.empty(count, dtype=np.int64)
    resolutions = np.empty(count, dtype=np.float64)
    observers = np.empty((count, 3), dtype=np.float64)
    radii = np.empty(count, dtype=np.float64)
    starts = np.empty(count, dtype=np.float64)
    for index in range(count):
        sample, resolution, observer, radius, start, _, _, _ = jobs[index % len(jobs)]
        counts[index] = len(sample)
        samples[index, : len(sample)] = sample
        resolutions[index] = resolution
        observers[index] = observer
        radii[index] = radius
        starts[index] = start
    return samples, counts, resolutions, observers, radii, starts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-jobs", type=int, default=65_536)
    parser.add_argument("--python-jobs", type=int, default=2_048)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-3-host-benchmark.json",
    )
    arguments = parser.parse_args()
    jobs = _load_jobs()
    compiled = _batch(jobs, arguments.compiled_jobs)
    python_jobs = [jobs[index % len(jobs)] for index in range(arguments.python_jobs)]

    def run_python():
        return np.stack(
            [fit_ray_segment(samples, resolution, observer, radius, start)
             for samples, resolution, observer, radius, start, _, _, _ in python_jobs]
        )

    generation_job = jobs[0]
    _, generation_resolution, generation_observer, generation_radius, generation_start, generation_maximum, generation_direction, generation_dem = generation_job
    generation_observers = np.repeat(
        generation_observer[np.newaxis, :], arguments.compiled_jobs, axis=0
    )
    generation_directions = np.repeat(
        generation_direction[np.newaxis, :], arguments.compiled_jobs, axis=0
    )
    generation_starts = np.full(
        arguments.compiled_jobs, generation_start, dtype=np.float64
    )
    generation_maximums = np.full(
        arguments.compiled_jobs, generation_maximum, dtype=np.float64
    )
    generation_radii = np.full(
        arguments.compiled_jobs, generation_radius, dtype=np.float64
    )
    projection = np.array(
        (
            generation_dem.projection.radius_m,
            generation_dem.projection.latitude_origin_rad,
            generation_dem.projection.longitude_origin_rad,
            generation_dem.projection.scale,
            generation_dem.projection.false_easting_m,
            generation_dem.projection.false_northing_m,
        ),
        dtype=np.float64,
    )
    generation_arguments = (
        generation_dem.elevation_m,
        generation_dem.geo_transform,
        projection,
        generation_observers,
        generation_directions,
        generation_starts,
        generation_maximums,
        generation_resolution,
        generation_radii,
    )

    def run_python_generation():
        results = []
        for _ in range(arguments.python_jobs):
            samples = build_ray_samples(
                generation_observer, generation_direction, generation_start,
                generation_maximum, generation_dem,
            )
            results.append(
                fit_ray_segment(
                    samples, generation_resolution, generation_observer,
                    generation_radius, generation_start,
                )
            )
        return np.stack(results)

    # Compile before measuring warm execution.
    serial_result = fit_segments_serial(*compiled)
    requested_threads = min(8, os.cpu_count() or 1)
    numba.set_num_threads(requested_threads)
    parallel_result = fit_segments_parallel(*compiled)
    if not np.array_equal(serial_result, parallel_result):
        raise RuntimeError("compiled parallel output differs from compiled serial output")
    generation_serial = generate_segments_serial(*generation_arguments)
    generation_parallel = generate_segments_parallel(*generation_arguments)
    if not np.array_equal(generation_serial, generation_parallel):
        raise RuntimeError("parallel full segment generation differs from serial output")

    python_result = run_python()
    python_times = _timings(run_python, arguments.repeats)
    serial_times = _timings(lambda: fit_segments_serial(*compiled), arguments.repeats)
    parallel_times = _timings(lambda: fit_segments_parallel(*compiled), arguments.repeats)
    python_generation_result = run_python_generation()
    python_generation_times = _timings(run_python_generation, arguments.repeats)
    generation_serial_times = _timings(
        lambda: generate_segments_serial(*generation_arguments), arguments.repeats
    )
    generation_parallel_times = _timings(
        lambda: generate_segments_parallel(*generation_arguments), arguments.repeats
    )

    production_segments_per_patch = 324 * 1_440 * 4
    bytes_per_segment = len(SEGMENT_FIELDS) * np.dtype(np.float32).itemsize
    report = {
        "schema_version": 1,
        "scope": {
            "included": [
                "ray-segment coefficient fitting from precomputed C# oracle samples",
                "complete DEM chord sampling plus segment fitting for a repeated synthetic oracle ray",
            ],
            "excluded": [
                "complete subpatch assembly",
                "CUDA horizon ray casting",
                "pyramid traversal",
                "final horizon generation",
            ],
            "warm_execution": True,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "logical_cpu_count": os.cpu_count(),
            "numba_parallel_threads": requested_threads,
        },
        "inputs": {
            "distinct_csharp_oracle_fits": len(jobs),
            "compiled_jobs": arguments.compiled_jobs,
            "python_jobs": arguments.python_jobs,
            "repeats": arguments.repeats,
        },
        "results": {
            "python_serial": _summary(python_times, arguments.python_jobs),
            "numba_serial": _summary(serial_times, arguments.compiled_jobs),
            "numba_parallel": _summary(parallel_times, arguments.compiled_jobs),
            "parallel_equals_serial_bitwise": True,
            "python_output_sha256": _sha256(python_result),
            "compiled_serial_output_sha256": _sha256(serial_result),
            "compiled_parallel_output_sha256": _sha256(parallel_result),
            "peak_process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "complete_segment_generation": {
                "workload": "repeated synthetic C# oracle ray on one DEM",
                "python_serial": _summary(
                    python_generation_times, arguments.python_jobs
                ),
                "numba_serial": _summary(
                    generation_serial_times, arguments.compiled_jobs
                ),
                "numba_parallel": _summary(
                    generation_parallel_times, arguments.compiled_jobs
                ),
                "parallel_equals_serial_bitwise": True,
                "python_output_sha256": _sha256(python_generation_result),
                "compiled_serial_output_sha256": _sha256(generation_serial),
                "compiled_parallel_output_sha256": _sha256(generation_parallel),
            },
        },
        "production_cache_projection": {
            "basis": "324 centers * 1440 azimuths * 4 DEMs * 18 float32 fields",
            "segments_per_patch": production_segments_per_patch,
            "segment_tensor_bytes_per_patch": production_segments_per_patch
            * bytes_per_segment,
            "segment_tensor_bytes_for_six_patch_queue": production_segments_per_patch
            * bytes_per_segment
            * 6,
            "qualification": "tensor payload only; Python objects, samples, DEMs, and CUDA buffers excluded",
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
