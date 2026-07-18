#!/usr/bin/env python3
"""Benchmark the four landed mission-duration products on real terrain."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.lightmap_cuda import LightmapCudaSession
from lunarscout._numba_horizon.mission_duration import (
    monthly_candidate_intervals,
    reduce_longest_candidate_duration_stream,
)
from lunarscout._numba_horizon.mission_duration_pipeline import (
    run_sun_elevation_duration_product,
    run_sun_elevation_earth_elevation_duration_product,
    run_sunlight_duration_product,
    run_sunlight_earth_elevation_duration_product,
)
from lunarscout._numba_horizon.pipeline import enumerate_patches
from lunarscout._numba_horizon.product_vectors import generate_moon_me_vectors
from lunarscout._numba_horizon.psr_pipeline import _inventory_identity
from lunarscout.georeference import GeoReference


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dem(path: Path) -> DemGrid:
    with rasterio.open(path) as dataset:
        elevation = np.ascontiguousarray(dataset.read(1), dtype=np.float32)
        transform = np.ascontiguousarray(
            dataset.transform.to_gdal(), dtype=np.float64
        )
        crs = dataset.crs.to_dict()
    return DemGrid(
        elevation,
        transform,
        ProjectionParameters(
            radius_m=float(crs["R"]),
            latitude_origin_rad=float(np.deg2rad(crs["lat_0"])),
            longitude_origin_rad=float(np.deg2rad(crs["lon_0"])),
            scale=float(crs.get("k", crs.get("k_0", 1.0))),
            false_easting_m=float(crs.get("x_0", 0.0)),
            false_northing_m=float(crs.get("y_0", 0.0)),
        ),
    )


def _load_georef(path: Path) -> GeoReference:
    with rasterio.open(path) as dataset:
        transform = tuple(float(value) for value in dataset.transform.to_gdal())
        return GeoReference(
            projection_wkt=dataset.crs.to_wkt(),
            projection_proj4=dataset.crs.to_proj4(),
            affine_transform=transform,
            width=dataset.width,
            height=dataset.height,
            pixel_size_x=transform[1],
            pixel_size_y=transform[5],
            nodata=dataset.nodata,
        )


def _conditions(
    sun_tiles,
    sun_threshold: float,
    earth_tiles=None,
    earth_threshold: float = 0.0,
):
    if earth_tiles is None:
        for sun_tile in sun_tiles:
            yield np.asarray(sun_tile) >= sun_threshold
    else:
        for sun_tile, earth_tile in zip(sun_tiles, earth_tiles, strict=True):
            yield (np.asarray(sun_tile) >= sun_threshold) & (
                np.asarray(earth_tile) >= earth_threshold
            )


def _time_one_patch(
    *,
    sun_calculator,
    earth_calculator,
    dem,
    horizons,
    sun_vectors,
    earth_vectors,
    sun_threshold,
    earth_threshold,
    times,
    intervals,
) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    sun_tiles = sun_calculator(
        dem, horizons, sun_vectors, tile_y=0, tile_x=0
    )
    earth_tiles = (
        earth_calculator(dem, horizons, earth_vectors, tile_y=0, tile_x=0)
        if earth_calculator is not None
        else None
    )
    result = np.stack(
        reduce_longest_candidate_duration_stream(
            _conditions(
                sun_tiles,
                sun_threshold,
                earth_tiles,
                earth_threshold,
            ),
            times_utc=times,
            evaluation_start_utc=times[0],
            evaluation_stop_utc=times[-1],
            start_intervals=intervals,
            output_unit="days",
        )
    )
    return time.perf_counter() - started, result


def _compare_products(cpu_path: Path, cuda_path: Path) -> dict[str, object]:
    mismatch_count = 0
    maximum_absolute_delta = 0.0
    mismatch_samples: list[dict[str, object]] = []
    with rasterio.open(cpu_path) as cpu, rasterio.open(cuda_path) as cuda:
        if cpu.count != cuda.count or cpu.dtypes != cuda.dtypes:
            raise RuntimeError("CPU and CUDA product layouts differ")
        cpu_mask = cpu.dataset_mask()
        cuda_mask = cuda.dataset_mask()
        timestamps_match = all(
            cpu.tags(index).get("TIMESTAMP_UTC")
            == cuda.tags(index).get("TIMESTAMP_UTC")
            for index in range(1, cpu.count + 1)
        )
        intervals_match = all(
            (
                cpu.tags(index).get("CANDIDATE_START_UTC"),
                cpu.tags(index).get("CANDIDATE_STOP_UTC"),
                cpu.tags(index).get("DURATION_UNIT"),
            )
            == (
                cuda.tags(index).get("CANDIDATE_START_UTC"),
                cuda.tags(index).get("CANDIDATE_STOP_UTC"),
                cuda.tags(index).get("DURATION_UNIT"),
            )
            for index in range(1, cpu.count + 1)
        )
        for index in range(1, cpu.count + 1):
            cpu_values = cpu.read(index).astype(np.float64)
            cuda_values = cuda.read(index).astype(np.float64)
            delta = cpu_values - cuda_values
            mismatch_count += int(np.count_nonzero(delta))
            maximum_absolute_delta = max(
                maximum_absolute_delta, float(np.max(np.abs(delta)))
            )
            if len(mismatch_samples) < 20:
                for row, column in np.argwhere(delta != 0):
                    mismatch_samples.append(
                        {
                            "band": index,
                            "row": int(row),
                            "column": int(column),
                            "cpu_days": float(cpu_values[row, column]),
                            "cuda_days": float(cuda_values[row, column]),
                            "delta_days": float(delta[row, column]),
                        }
                    )
                    if len(mismatch_samples) == 20:
                        break
        return {
            "dtype": cpu.dtypes[0],
            "band_count": cpu.count,
            "mask_mismatch_count": int(np.count_nonzero(cpu_mask != cuda_mask)),
            "valid_pixels": int(np.count_nonzero(cpu_mask)),
            "value_mismatch_count": mismatch_count,
            "maximum_absolute_delta_days": maximum_absolute_delta,
            "mismatch_samples": mismatch_samples,
            "timestamps_match": timestamps_match,
            "candidate_interval_metadata_match": intervals_match,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--horizon-root", type=Path, required=True)
    parser.add_argument("--time-count", type=int, default=2925)
    parser.add_argument("--time-batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.time_count < 2:
        parser.error("--time-count must be at least two")

    full_dem = _load_dem(arguments.dem)
    full_georef = _load_georef(arguments.dem)
    size = 256
    dem = DemGrid(
        np.ascontiguousarray(full_dem.elevation_m[:size, :size]),
        full_dem.geo_transform.copy(),
        full_dem.projection,
    )
    georef = replace(full_georef, width=size, height=size)
    start = datetime(2027, 1, 1, tzinfo=timezone.utc)
    times = tuple(
        start + index * timedelta(hours=6)
        for index in range(arguments.time_count)
    )
    intervals = monthly_candidate_intervals(times[0], times[-1])

    vector_started = time.perf_counter()
    sun_vectors = generate_moon_me_vectors("sun", times).vectors_m
    sun_vector_seconds = time.perf_counter() - vector_started
    vector_started = time.perf_counter()
    earth_vectors = generate_moon_me_vectors("earth", times).vectors_m
    earth_vector_seconds = time.perf_counter() - vector_started

    store = HorizonTileStore(arguments.horizon_root)
    horizon_started = time.perf_counter()
    horizons = store.read(0, 0, 0.0)
    one_horizon_read_seconds = time.perf_counter() - horizon_started
    if horizons is None:
        raise RuntimeError("missing real horizon tile at row=0 col=0")

    cpu_sun = LightmapCpuSession(time_batch_size=arguments.time_batch_size)
    cpu_earth = LightmapCpuSession(time_batch_size=arguments.time_batch_size)
    from numba import cuda

    cuda.select_device(0)
    free_before_sessions, gpu_total_bytes = cuda.current_context().get_memory_info()
    gpu_sun = LightmapCudaSession(time_batch_size=arguments.time_batch_size)
    gpu_earth = LightmapCudaSession(time_batch_size=arguments.time_batch_size)
    free_after_sessions, _ = cuda.current_context().get_memory_info()

    list(
        cpu_sun.iter_patch_fraction_tiles(
            dem, horizons, sun_vectors[:1], tile_y=0, tile_x=0
        )
    )
    list(
        cpu_sun.iter_patch_margin_tiles(
            dem, horizons, sun_vectors[:1], tile_y=0, tile_x=0
        )
    )
    list(
        gpu_sun.iter_patch_fraction_tiles(
            dem, horizons, sun_vectors[:1], tile_y=0, tile_x=0
        )
    )
    list(
        gpu_sun.iter_patch_margin_tiles(
            dem, horizons, sun_vectors[:1], tile_y=0, tile_x=0
        )
    )

    configurations = {
        "sunlight": {
            "sun_signal": "fraction",
            "sun_threshold": 0.5,
            "earth": False,
            "function": run_sunlight_duration_product,
            "function_arguments": {"sunlight_fraction_threshold": 0.5},
        },
        "sun_elevation": {
            "sun_signal": "margin",
            "sun_threshold": 0.0,
            "earth": False,
            "function": run_sun_elevation_duration_product,
            "function_arguments": {"sun_elevation_threshold_deg": 0.0},
        },
        "sunlight_earth_elevation": {
            "sun_signal": "fraction",
            "sun_threshold": 0.5,
            "earth": True,
            "function": run_sunlight_earth_elevation_duration_product,
            "function_arguments": {
                "sunlight_fraction_threshold": 0.5,
                "earth_elevation_threshold_deg": 0.0,
                "earth_vectors_m": earth_vectors,
            },
        },
        "sun_elevation_earth_elevation": {
            "sun_signal": "margin",
            "sun_threshold": 0.0,
            "earth": True,
            "function": run_sun_elevation_earth_elevation_duration_product,
            "function_arguments": {
                "sun_elevation_threshold_deg": 0.0,
                "earth_elevation_threshold_deg": 0.0,
                "earth_vectors_m": earth_vectors,
            },
        },
    }

    report_products: dict[str, object] = {}
    with tempfile.TemporaryDirectory(
        prefix="lunarscout-phase6b-mission-duration-"
    ) as temp:
        root = Path(temp)
        for name, configuration in configurations.items():
            sun_cpu_calculator = (
                cpu_sun.iter_patch_fraction_tiles
                if configuration["sun_signal"] == "fraction"
                else cpu_sun.iter_patch_margin_tiles
            )
            sun_gpu_calculator = (
                gpu_sun.iter_patch_fraction_tiles
                if configuration["sun_signal"] == "fraction"
                else gpu_sun.iter_patch_margin_tiles
            )
            cpu_calculation_seconds, cpu_values = _time_one_patch(
                sun_calculator=sun_cpu_calculator,
                earth_calculator=(
                    cpu_earth.iter_patch_margin_tiles
                    if configuration["earth"]
                    else None
                ),
                dem=dem,
                horizons=horizons,
                sun_vectors=sun_vectors,
                earth_vectors=earth_vectors,
                sun_threshold=float(configuration["sun_threshold"]),
                earth_threshold=0.0,
                times=times,
                intervals=intervals,
            )
            gpu_calculation_seconds, gpu_values = _time_one_patch(
                sun_calculator=sun_gpu_calculator,
                earth_calculator=(
                    gpu_earth.iter_patch_margin_tiles
                    if configuration["earth"]
                    else None
                ),
                dem=dem,
                horizons=horizons,
                sun_vectors=sun_vectors,
                earth_vectors=earth_vectors,
                sun_threshold=float(configuration["sun_threshold"]),
                earth_threshold=0.0,
                times=times,
                intervals=intervals,
            )
            calculation_delta = cpu_values.astype(np.float64) - gpu_values
            del cpu_values, gpu_values

            cpu_path = root / f"{name}-cpu.tif"
            cuda_path = root / f"{name}-cuda.tif"
            common = {
                "dem": dem,
                "georef": georef,
                "horizon_store": store,
                "times_utc": times,
                "evaluation_start_utc": times[0],
                "evaluation_stop_utc": times[-1],
                "start_intervals": intervals,
                "sun_vectors_m": sun_vectors,
                "output_unit": "days",
                "time_batch_size": arguments.time_batch_size,
            }
            function = configuration["function"]
            function_arguments = configuration["function_arguments"]
            pipeline_started = time.perf_counter()
            function(
                **common,
                **function_arguments,
                output_path=cpu_path,
                backend="cpu",
            )
            cpu_pipeline_seconds = time.perf_counter() - pipeline_started
            pipeline_started = time.perf_counter()
            function(
                **common,
                **function_arguments,
                output_path=cuda_path,
                backend="cuda",
            )
            cuda_pipeline_seconds = time.perf_counter() - pipeline_started
            product_report = _compare_products(cpu_path, cuda_path)
            product_report.update(
                {
                    "sun_signal": configuration["sun_signal"],
                    "sun_threshold": configuration["sun_threshold"],
                    "earth_elevation_threshold_deg": (
                        0.0 if configuration["earth"] else None
                    ),
                    "cpu_calculation_seconds_one_patch": cpu_calculation_seconds,
                    "cuda_calculation_seconds_one_patch": gpu_calculation_seconds,
                    "calculation_mismatch_count": int(
                        np.count_nonzero(calculation_delta)
                    ),
                    "calculation_maximum_absolute_delta_days": float(
                        np.max(np.abs(calculation_delta))
                    ),
                    "cpu_pipeline_seconds": cpu_pipeline_seconds,
                    "cuda_pipeline_seconds": cuda_pipeline_seconds,
                    "cpu_pipeline_patches_per_second": 4 / cpu_pipeline_seconds,
                    "cuda_pipeline_patches_per_second": 4 / cuda_pipeline_seconds,
                    "cpu_file_bytes": cpu_path.stat().st_size,
                    "cuda_file_bytes": cuda_path.stat().st_size,
                    "cpu_file_sha256": _sha256(cpu_path),
                    "cuda_file_sha256": _sha256(cuda_path),
                }
            )
            report_products[name] = product_report

    report = {
        "schema": "lunarscout-numba-phase6b-mission-duration-benchmark-v1",
        "region": {"width": size, "height": size, "patch_count": 4},
        "time_count": len(times),
        "time_step_hours": 6,
        "candidate_interval_count": len(intervals),
        "candidate_start_intervals_utc": [
            [item.start_utc.isoformat(), item.stop_utc.isoformat()]
            for item in intervals
        ],
        "evaluation_start_utc": times[0].isoformat(),
        "evaluation_stop_utc": times[-1].isoformat(),
        "output_unit": "days",
        "time_batch_size": arguments.time_batch_size,
        "sun_vector_seconds": sun_vector_seconds,
        "earth_vector_seconds": earth_vector_seconds,
        "one_horizon_read_seconds": one_horizon_read_seconds,
        "products": report_products,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "gpu_total_bytes": int(gpu_total_bytes),
        "two_gpu_session_buffer_bytes": int(
            free_before_sessions - free_after_sessions
        ),
        "dem_path": str(arguments.dem.resolve()),
        "dem_sha256": _sha256(arguments.dem),
        "horizon_root": str(arguments.horizon_root.resolve()),
        "horizon_inventory_identity": _inventory_identity(
            store, enumerate_patches(size, size), 0.0
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": __import__("numba").__version__,
            "rasterio": rasterio.__version__,
            "cuda_device_name": str(cuda.get_current_device().name),
            "cuda_compute_capability": list(
                cuda.get_current_device().compute_capability
            ),
        },
        "loaded_managed_modules": [
            name
            for name in ("clr", "pythonnet", "moonlib")
            if name in sys.modules
        ],
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
