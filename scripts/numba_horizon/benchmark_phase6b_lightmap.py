#!/usr/bin/env python3
"""Benchmark bounded CPU/CUDA lightmaps on a 2-by-2 real patch region."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.lightmap_cpu import LightmapCpuSession
from lunarscout._numba_horizon.lightmap_cuda import LightmapCudaSession
from lunarscout._numba_horizon.lightmap_pipeline import run_lightmap_product
from lunarscout._numba_horizon.product_vectors import generate_moon_me_vectors
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


def _timed_tiles(calculator, dem, horizons, vectors) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    values = np.stack(
        tuple(
            calculator(
                dem, horizons, vectors, tile_y=0, tile_x=0
            )
        )
    )
    return time.perf_counter() - started, values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--horizon-root", type=Path, required=True)
    parser.add_argument("--time-count", type=int, default=256)
    parser.add_argument("--time-batch-size", type=int, default=32)
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.time_count < 1:
        parser.error("--time-count must be positive")

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
    vector_started = time.perf_counter()
    vectors = generate_moon_me_vectors("sun", times).vectors_m
    vector_seconds = time.perf_counter() - vector_started
    store = HorizonTileStore(arguments.horizon_root)
    horizon_started = time.perf_counter()
    horizons = store.read(0, 0, 0.0)
    horizon_seconds = time.perf_counter() - horizon_started
    if horizons is None:
        raise RuntimeError("missing real horizon tile at row=0 col=0")

    cpu = LightmapCpuSession(time_batch_size=arguments.time_batch_size)
    from numba import cuda

    cuda.select_device(0)
    free_before_gpu_session, gpu_total_bytes = (
        cuda.current_context().get_memory_info()
    )
    gpu = LightmapCudaSession(time_batch_size=arguments.time_batch_size)
    free_after_gpu_session, _ = cuda.current_context().get_memory_info()
    list(cpu.iter_patch_tiles(dem, horizons, vectors[:1], tile_y=0, tile_x=0))
    list(gpu.iter_patch_tiles(dem, horizons, vectors[:1], tile_y=0, tile_x=0))
    cpu_kernel_seconds, cpu_values = _timed_tiles(
        cpu.iter_patch_tiles, dem, horizons, vectors
    )
    gpu_kernel_seconds, gpu_values = _timed_tiles(
        gpu.iter_patch_tiles, dem, horizons, vectors
    )
    delta = cpu_values.astype(np.int16) - gpu_values.astype(np.int16)
    nonzero = delta[delta != 0]
    kernel_mismatch_count = int(nonzero.size)
    kernel_max_abs_byte_delta = (
        int(np.max(np.abs(nonzero))) if nonzero.size else 0
    )
    del cpu_values, gpu_values, delta, nonzero

    with tempfile.TemporaryDirectory(prefix="lunarscout-phase6b-lightmap-") as temp:
        root = Path(temp)
        cpu_path = root / "cpu.tif"
        started = time.perf_counter()
        run_lightmap_product(
            dem=dem,
            georef=georef,
            horizon_store=store,
            output_path=cpu_path,
            times_utc=times,
            sun_vectors_m=vectors,
            backend="cpu",
            time_batch_size=arguments.time_batch_size,
        )
        cpu_pipeline_seconds = time.perf_counter() - started
        gpu_path = root / "gpu.tif"
        started = time.perf_counter()
        run_lightmap_product(
            dem=dem,
            georef=georef,
            horizon_store=store,
            output_path=gpu_path,
            times_utc=times,
            sun_vectors_m=vectors,
            backend="cuda",
            time_batch_size=arguments.time_batch_size,
        )
        gpu_pipeline_seconds = time.perf_counter() - started
        product_mismatch_count = 0
        product_max_abs_byte_delta = 0
        with rasterio.open(cpu_path) as cpu_dataset, rasterio.open(
            gpu_path
        ) as gpu_dataset:
            cpu_mask = cpu_dataset.dataset_mask()
            gpu_mask = gpu_dataset.dataset_mask()
            cpu_profile = cpu_dataset.profile
            for band_index in range(1, cpu_dataset.count + 1):
                product_delta = (
                    cpu_dataset.read(band_index).astype(np.int16)
                    - gpu_dataset.read(band_index).astype(np.int16)
                )
                product_mismatch_count += int(np.count_nonzero(product_delta))
                product_max_abs_byte_delta = max(
                    product_max_abs_byte_delta,
                    int(np.max(np.abs(product_delta))),
                )
        report = {
            "schema": "lunarscout-numba-phase6b-lightmap-benchmark-v1",
            "region": {"width": size, "height": size, "patch_count": 4},
            "time_count": len(times),
            "time_step_hours": 6,
            "time_batch_size": arguments.time_batch_size,
            "vector_seconds": vector_seconds,
            "one_horizon_read_seconds": horizon_seconds,
            "cpu_kernel_seconds_one_patch": cpu_kernel_seconds,
            "gpu_kernel_seconds_one_patch": gpu_kernel_seconds,
            "cpu_pipeline_seconds": cpu_pipeline_seconds,
            "gpu_pipeline_seconds": gpu_pipeline_seconds,
            "cpu_pipeline_patches_per_second": 4 / cpu_pipeline_seconds,
            "gpu_pipeline_patches_per_second": 4 / gpu_pipeline_seconds,
            "kernel_mismatch_count": kernel_mismatch_count,
            "kernel_max_abs_byte_delta": kernel_max_abs_byte_delta,
            "product_mismatch_count": product_mismatch_count,
            "product_max_abs_byte_delta": product_max_abs_byte_delta,
            "mask_mismatch_count": int(np.count_nonzero(cpu_mask != gpu_mask)),
            "valid_pixels": int(np.count_nonzero(cpu_mask)),
            "dtype": cpu_profile["dtype"],
            "band_count": cpu_profile["count"],
            "cpu_file_bytes": cpu_path.stat().st_size,
            "gpu_file_bytes": gpu_path.stat().st_size,
            "cpu_file_sha256": _sha256(cpu_path),
            "gpu_file_sha256": _sha256(gpu_path),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpu_total_bytes": int(gpu_total_bytes),
            "gpu_session_buffer_bytes": int(
                free_before_gpu_session - free_after_gpu_session
            ),
        }
    arguments.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
