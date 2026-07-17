#!/usr/bin/env python3
"""Compare Python host geometry with a C# real-terrain segment capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import numba
import rasterio


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.geometry import (  # noqa: E402
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    build_subpatch_segments,
    build_subpatch_segments_numba,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _evaluate(segment: np.ndarray, distances_km: np.ndarray):
    delta = distances_km - segment[12]
    x = segment[2].astype(np.float64) + sum(
        segment[3 + power].astype(np.float64) * delta**power
        for power in range(1, 5)
    )
    y = segment[3].astype(np.float64) + sum(
        segment[7 + power].astype(np.float64) * delta**power
        for power in range(1, 5)
    )
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csharp_capture", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-3-real-terrain.json",
    )
    arguments = parser.parse_args()
    capture = json.loads(arguments.csharp_capture.read_text(encoding="utf-8"))
    input_path = Path(capture["input_path"])

    with rasterio.open(input_path) as dataset:
        elevation = np.ascontiguousarray(dataset.read(1), dtype=np.float32)
        transform = np.ascontiguousarray(dataset.transform.to_gdal(), dtype=np.float64)
        crs = dataset.crs.to_dict()
    raster_sha256 = _array_sha256(elevation)
    fixture_manifest = json.loads(
        (REPOSITORY / "docs" / "numba-horizon-phase-1-real-terrain-fixtures.json")
        .read_text(encoding="utf-8")
    )
    expected_raster_sha256 = fixture_manifest["small_automatically_acquired_fixture"][
        "output_raster_float32_sha256"
    ]
    if raster_sha256 != expected_raster_sha256:
        raise RuntimeError("real-terrain raster does not match the Phase 1 fixture manifest")
    projection = ProjectionParameters(
        radius_m=float(crs["R"]),
        latitude_origin_rad=float(np.deg2rad(crs["lat_0"])),
        longitude_origin_rad=float(np.deg2rad(crs["lon_0"])),
        scale=float(crs.get("k", crs.get("k_0", 1.0))),
        false_easting_m=float(crs.get("x_0", 0.0)),
        false_northing_m=float(crs.get("y_0", 0.0)),
    )
    dem = DemGrid(elevation, transform, projection)
    config = capture["configuration"]
    convergence = GridConvergenceInput(*capture["grid_convergence"])
    build_arguments = dict(
        tile_column=config["tile_column"],
        tile_row=config["tile_row"],
        tile_width=config["tile_width"],
        azimuth_count=config["azimuth_count"],
        maximum_distance_m=config["maximum_distance_m"],
        observer_elevation_m=config["observer_elevation_m"],
        subpatch_size=config["subpatch_size"],
        grid_convergence=convergence,
    )
    started = time.perf_counter()
    python_segments, centers, _ = build_subpatch_segments(
        [dem],
        **build_arguments,
    )
    python_seconds = time.perf_counter() - started
    numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
    started = time.perf_counter()
    numba_serial, _, _ = build_subpatch_segments_numba(
        [dem], **build_arguments, parallel=False
    )
    numba_serial_cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    numba_parallel, _, _ = build_subpatch_segments_numba(
        [dem], **build_arguments, parallel=True
    )
    numba_parallel_cold_seconds = time.perf_counter() - started
    serial_warm = []
    parallel_warm = []
    for _ in range(5):
        started = time.perf_counter()
        build_subpatch_segments_numba([dem], **build_arguments, parallel=False)
        serial_warm.append(time.perf_counter() - started)
        started = time.perf_counter()
        build_subpatch_segments_numba([dem], **build_arguments, parallel=True)
        parallel_warm.append(time.perf_counter() - started)
    if not np.array_equal(numba_serial, numba_parallel):
        raise RuntimeError("Numba serial and parallel real-terrain tensors differ")

    production_shape_arguments = dict(
        tile_column=192,
        tile_row=192,
        tile_width=128,
        azimuth_count=1440,
        maximum_distance_m=5000.0,
        observer_elevation_m=0.0,
        subpatch_size=8,
        grid_convergence=convergence,
    )
    production_shape_times = []
    production_shape_segments = None
    for _ in range(3):
        started = time.perf_counter()
        production_shape_segments, _, _ = build_subpatch_segments_numba(
            [dem], **production_shape_arguments, parallel=True
        )
        production_shape_times.append(time.perf_counter() - started)
    production_shape_max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    retained_patch_tensors = []
    retained_patch_started = time.perf_counter()
    for tile_column, tile_row in (
        (0, 0), (64, 0), (128, 0), (192, 0), (256, 0), (320, 0)
    ):
        retained, _, _ = build_subpatch_segments_numba(
            [dem, dem, dem, dem],
            **(
                production_shape_arguments
                | {"tile_column": tile_column, "tile_row": tile_row}
            ),
            parallel=True,
        )
        retained_patch_tensors.append(retained)
    retained_patch_seconds = time.perf_counter() - retained_patch_started
    retained_patch_bytes = sum(array.nbytes for array in retained_patch_tensors)
    retained_patch_hashes = [_array_sha256(array) for array in retained_patch_tensors]
    retained_patch_max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    csharp_segments = np.asarray(capture["segments"], dtype=np.float32).reshape(
        python_segments.shape
    )
    python_centers = np.asarray(
        [[getattr(center, field) for field in center.__dataclass_fields__] for center in centers],
        dtype=np.int64,
    )
    csharp_centers = np.asarray(capture["centers"], dtype=np.int64)
    if not np.array_equal(python_centers, csharp_centers):
        raise RuntimeError("Python and C# subpatch-center layouts differ")

    def path_errors(actual_segments):
        values = []
        for actual_segment, csharp_segment in zip(
            actual_segments.reshape(-1, 18), csharp_segments.reshape(-1, 18)
        ):
            start = max(float(actual_segment[12]), float(csharp_segment[12]))
            end = min(float(actual_segment[13]), float(csharp_segment[13]))
            distances = np.linspace(start, end, 257, dtype=np.float64)
            actual_x, actual_y = _evaluate(actual_segment, distances)
            csharp_x, csharp_y = _evaluate(csharp_segment, distances)
            values.extend(np.hypot(actual_x - csharp_x, actual_y - csharp_y))
        return np.asarray(values, dtype=np.float64)

    errors = path_errors(python_segments)
    numba_errors = path_errors(numba_parallel)
    coefficient_error = np.abs(
        python_segments.astype(np.float64) - csharp_segments.astype(np.float64)
    )
    accepted_pixel_error = 1e-4
    report = {
        "schema_version": 1,
        "scope": "host-side real-terrain ray-segment preparation; no horizon kernel",
        "input": {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "raster_float32_sha256": raster_sha256,
            "shape_y_x": list(elevation.shape),
            "dtype": str(elevation.dtype),
            "geo_transform": transform.tolist(),
            "projection": capture["projection"],
        },
        "configuration": config,
        "csharp": {
            "selected_accelerator_name": capture["selected_accelerator_name"],
            "selected_accelerator_type": capture["selected_accelerator_type"],
            "segment_sha256": _array_sha256(csharp_segments),
        },
        "python": {
            "implementation": "ordinary Python/NumPy Phase 3 geometry",
            "segment_sha256": _array_sha256(python_segments),
            "elapsed_seconds": python_seconds,
        },
        "numba": {
            "version": numba.__version__,
            "threads": numba.get_num_threads(),
            "serial_segment_sha256": _array_sha256(numba_serial),
            "parallel_segment_sha256": _array_sha256(numba_parallel),
            "serial_equals_parallel_bitwise": True,
            "serial_first_call_seconds_including_compilation": numba_serial_cold_seconds,
            "parallel_first_call_seconds_including_compilation": numba_parallel_cold_seconds,
            "serial_warm_seconds": serial_warm,
            "serial_warm_median_seconds": statistics.median(serial_warm),
            "parallel_warm_seconds": parallel_warm,
            "parallel_warm_median_seconds": statistics.median(parallel_warm),
            "bounded_production_shape_benchmark": {
                "qualification": (
                    "128x128 patch shape and 1440 azimuth bins, but one 512x512 "
                    "real DEM and a bounded 5 km distance; not a production horizon run"
                ),
                "configuration": production_shape_arguments | {
                    "grid_convergence": list(convergence)
                },
                "segment_count": int(np.prod(production_shape_segments.shape[:-1])),
                "output_tensor_bytes": production_shape_segments.nbytes,
                "warm_seconds": production_shape_times,
                "warm_median_seconds": statistics.median(production_shape_times),
                "output_sha256": _array_sha256(production_shape_segments),
                "process_max_rss_kib_after_run": production_shape_max_rss,
            },
            "retained_six_patch_tensor_batch": {
                "qualification": (
                    "Six production-shaped four-DEM tensors are retained, but "
                    "the same bounded LOLA DEM is repeated and later DEM passes "
                    "have no remaining ray distance; this measures tensor/cache "
                    "memory, not four-DEM computation throughput"
                ),
                "patch_count": len(retained_patch_tensors),
                "dem_axes_per_patch": 4,
                "retained_tensor_bytes": retained_patch_bytes,
                "elapsed_seconds": retained_patch_seconds,
                "process_max_rss_kib": retained_patch_max_rss,
                "output_sha256_by_patch": retained_patch_hashes,
            },
        },
        "comparison": {
            "segment_count": int(np.prod(python_segments.shape[:-1])),
            "dense_samples_per_segment": 257,
            "maximum_absolute_coefficient_difference": float(coefficient_error.max()),
            "mean_path_error_pixels": float(errors.mean()),
            "p95_path_error_pixels": float(np.percentile(errors, 95)),
            "p99_path_error_pixels": float(np.percentile(errors, 99)),
            "maximum_path_error_pixels": float(errors.max()),
            "accepted_maximum_path_error_pixels": accepted_pixel_error,
            "accepted": bool(errors.max() <= accepted_pixel_error),
            "numba_maximum_path_error_pixels": float(numba_errors.max()),
            "numba_accepted": bool(numba_errors.max() <= accepted_pixel_error),
        },
        "grid_convergence": capture["grid_convergence"],
        "qualification": (
            "C# calculated convergence and selected CUDA while building diagnostic "
            "pyramids; the compared values are CPU-prepared ray segments, not CUDA "
            "horizon output. The current production subpatch kernel does not apply "
            "the convergence values."
        ),
    }
    if not report["comparison"]["accepted"] or not report["comparison"]["numba_accepted"]:
        raise RuntimeError(
            "real-terrain path error exceeds the accepted pixel tolerance"
        )
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
