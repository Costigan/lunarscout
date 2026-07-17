#!/usr/bin/env python3
"""Benchmark full-cycle PSR vectors, reduction, and one real horizon patch."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import HorizonTileStore, read_horizon_tile
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.product_vectors import generate_moon_me_vectors
from lunarscout._numba_horizon.psr import reduce_sun_vectors_for_psr
from lunarscout._numba_horizon.psr_cuda import PsrCudaSession
from lunarscout._numba_horizon.psr_pipeline import run_psr_product
from lunarscout.georeference import GeoReference
from lunarscout.spice_geometry import iter_times


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dem(path: Path) -> DemGrid:
    with rasterio.open(path) as dataset:
        elevation = np.ascontiguousarray(dataset.read(1), dtype=np.float32)
        transform = np.ascontiguousarray(dataset.transform.to_gdal(), dtype=np.float64)
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
        projection_wkt = dataset.crs.to_wkt()
        with rasterio.Env():
            projection_proj4 = dataset.crs.to_proj4()
        return GeoReference(
            projection_wkt=projection_wkt,
            projection_proj4=projection_proj4,
            affine_transform=transform,
            width=dataset.width,
            height=dataset.height,
            pixel_size_x=transform[1],
            pixel_size_y=transform[5],
            nodata=dataset.nodata,
        )


def _angular_separation_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cross = np.linalg.norm(np.cross(left, right), axis=1)
    dot = np.sum(left * right, axis=1)
    return np.degrees(np.arctan2(cross, dot))


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--horizon", type=Path, required=True)
    parser.add_argument("--horizon-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args()

    print("Loading DEM and compressed horizon...", flush=True)
    dem = _load_dem(arguments.dem)
    georef = _load_georef(arguments.dem)
    horizons = read_horizon_tile(arguments.horizon)
    times = tuple(
        iter_times(
            "1970-01-01T00:00:00Z",
            "2044-01-01T00:00:00Z",
            timedelta(hours=6),
        )
    )
    print(f"Generating {len(times)} exact utc2et Sun vectors...", flush=True)
    started = time.perf_counter()
    exact = generate_moon_me_vectors("sun", times)
    exact_seconds = time.perf_counter() - started

    print("Generating anchored-linear Sun vectors...", flush=True)
    started = time.perf_counter()
    linear = generate_moon_me_vectors(
        "sun",
        times,
        ensure_kernels=False,
        time_conversion="linear_from_anchor",
    )
    linear_seconds = time.perf_counter() - started

    vector_position_difference = np.linalg.norm(
        exact.vectors_m - linear.vectors_m, axis=1
    )
    vector_angle_difference = _angular_separation_deg(
        exact.vectors_m, linear.vectors_m
    )

    print("Reducing exact vectors at five DEM viewpoints...", flush=True)
    started = time.perf_counter()
    exact_reduced, exact_indices = reduce_sun_vectors_for_psr(
        dem, exact.vectors_m
    )
    exact_reduction_seconds = time.perf_counter() - started
    print("Reducing anchored-linear vectors...", flush=True)
    started = time.perf_counter()
    linear_reduced, linear_indices = reduce_sun_vectors_for_psr(
        dem, linear.vectors_m
    )
    linear_reduction_seconds = time.perf_counter() - started
    exact_set = set(int(value) for value in exact_indices)
    linear_set = set(int(value) for value in linear_indices)

    print("Comparing both reductions on one real horizon patch...", flush=True)
    from numba import cuda

    context_started = time.perf_counter()
    cuda.select_device(0)
    free_before_session, total_gpu_bytes = cuda.current_context().get_memory_info()
    context_seconds = time.perf_counter() - context_started
    session_started = time.perf_counter()
    session = PsrCudaSession()
    session_seconds = time.perf_counter() - session_started
    free_after_session, _ = cuda.current_context().get_memory_info()
    # Populate the compiled cache and fixed buffers before retained timings.
    session.compute_patch(
        dem, horizons, linear_reduced, tile_y=0, tile_x=0
    )
    free_after_warmup, _ = cuda.current_context().get_memory_info()
    exact_cuda = []
    linear_cuda = []
    exact_output = None
    linear_output = None
    for _ in range(3):
        started = time.perf_counter()
        exact_output = session.compute_patch(
            dem, horizons, exact_reduced, tile_y=0, tile_x=0
        )
        exact_cuda.append(time.perf_counter() - started)
        started = time.perf_counter()
        linear_output = session.compute_patch(
            dem, horizons, linear_reduced, tile_y=0, tile_x=0
        )
        linear_cuda.append(time.perf_counter() - started)
    assert exact_output is not None and linear_output is not None

    print("Running both vector modes through the 16-patch GeoTIFF pipeline...", flush=True)
    horizon_store = HorizonTileStore(arguments.horizon_root)
    with tempfile.TemporaryDirectory(prefix="lunarscout-phase6b-psr-pipeline-") as temp:
        output_root = Path(temp)
        exact_product = output_root / "psr-exact.tif"
        started = time.perf_counter()
        run_psr_product(
            dem=dem,
            georef=georef,
            horizon_store=horizon_store,
            output_path=exact_product,
            sun_vectors_m=exact.vectors_m,
            patch_calculator=session.compute_patch,
        )
        exact_pipeline_seconds = time.perf_counter() - started
        linear_product = output_root / "psr-linear.tif"
        started = time.perf_counter()
        run_psr_product(
            dem=dem,
            georef=georef,
            horizon_store=horizon_store,
            output_path=linear_product,
            sun_vectors_m=linear.vectors_m,
            patch_calculator=session.compute_patch,
        )
        linear_pipeline_seconds = time.perf_counter() - started
        with rasterio.open(exact_product) as dataset:
            exact_product_values = dataset.read(1)
            exact_product_mask = dataset.dataset_mask()
        with rasterio.open(linear_product) as dataset:
            linear_product_values = dataset.read(1)
            linear_product_mask = dataset.dataset_mask()
        exact_product_bytes = exact_product.stat().st_size
        linear_product_bytes = linear_product.stat().st_size
        exact_product_file_sha256 = _sha256(exact_product)
        linear_product_file_sha256 = _sha256(linear_product)

    csharp_path = (
        Path(__file__).resolve().parents[2]
        / "tests/data/numba_horizon/phase6b_spice_csharp.json"
    )
    csharp = json.loads(csharp_path.read_text(encoding="utf-8"))
    sample_times = [sample["timestamp_utc"] for sample in csharp["samples"]]
    csharp_sun = np.asarray([sample["sun_m"] for sample in csharp["samples"]])
    csharp_earth = np.asarray([sample["earth_m"] for sample in csharp["samples"]])
    python_sun_exact = generate_moon_me_vectors(
        "sun", sample_times, ensure_kernels=False
    ).vectors_m
    python_sun_linear = generate_moon_me_vectors(
        "sun",
        sample_times,
        ensure_kernels=False,
        time_conversion="linear_from_anchor",
    ).vectors_m
    python_earth_exact = generate_moon_me_vectors(
        "earth", sample_times, ensure_kernels=False
    ).vectors_m
    python_earth_linear = generate_moon_me_vectors(
        "earth",
        sample_times,
        ensure_kernels=False,
        time_conversion="linear_from_anchor",
    ).vectors_m

    forbidden = sorted(
        name
        for name in sys.modules
        if name == "clr"
        or name.startswith("clr.")
        or name == "pythonnet"
        or name.startswith("pythonnet.")
        or name == "moonlib"
        or name.startswith("moonlib.")
    )
    report = {
        "schema": "lunarscout-numba-phase6b-psr-vectors-v1",
        "dem": {"path": str(arguments.dem), "sha256": _sha256(arguments.dem)},
        "horizon": {
            "path": str(arguments.horizon),
            "sha256": _sha256(arguments.horizon),
        },
        "time_start_utc": times[0].isoformat(),
        "time_stop_utc": times[-1].isoformat(),
        "time_step_hours": 6,
        "time_count": len(times),
        "exact_vector_seconds": exact_seconds,
        "linear_vector_seconds": linear_seconds,
        "linear_over_exact_vector_time_ratio": linear_seconds / exact_seconds,
        "exact_linear_position_difference_m": _summary(vector_position_difference),
        "exact_linear_angle_difference_deg": _summary(vector_angle_difference),
        "exact_reduced_count": int(exact_indices.size),
        "linear_reduced_count": int(linear_indices.size),
        "reduced_index_intersection_count": len(exact_set & linear_set),
        "exact_only_reduced_indices": len(exact_set - linear_set),
        "linear_only_reduced_indices": len(linear_set - exact_set),
        "exact_reduction_seconds": exact_reduction_seconds,
        "linear_reduction_seconds": linear_reduction_seconds,
        "cuda_session_seconds": session_seconds,
        "cuda_context_seconds": context_seconds,
        "gpu_total_bytes": int(total_gpu_bytes),
        "gpu_session_buffer_bytes": int(free_before_session - free_after_session),
        "gpu_session_and_vector_buffer_bytes": int(
            free_before_session - free_after_warmup
        ),
        "exact_cuda_seconds": exact_cuda,
        "linear_cuda_seconds": linear_cuda,
        "exact_cuda_median_seconds": float(np.median(exact_cuda)),
        "linear_cuda_median_seconds": float(np.median(linear_cuda)),
        "exact_linear_psr_mismatch_count": int(
            np.count_nonzero(exact_output != linear_output)
        ),
        "exact_psr_sha256": hashlib.sha256(exact_output.tobytes()).hexdigest(),
        "linear_psr_sha256": hashlib.sha256(linear_output.tobytes()).hexdigest(),
        "pipeline_patch_count": 16,
        "exact_pipeline_seconds": exact_pipeline_seconds,
        "linear_pipeline_seconds": linear_pipeline_seconds,
        "exact_pipeline_patches_per_second": 16.0 / exact_pipeline_seconds,
        "linear_pipeline_patches_per_second": 16.0 / linear_pipeline_seconds,
        "exact_linear_pipeline_mismatch_count": int(
            np.count_nonzero(exact_product_values != linear_product_values)
        ),
        "exact_linear_pipeline_mask_mismatch_count": int(
            np.count_nonzero(exact_product_mask != linear_product_mask)
        ),
        "exact_pipeline_valid_pixels": int(
            np.count_nonzero(exact_product_mask == 255)
        ),
        "exact_pipeline_invalid_pixels": int(
            np.count_nonzero(exact_product_mask == 0)
        ),
        "exact_product_bytes": exact_product_bytes,
        "linear_product_bytes": linear_product_bytes,
        "exact_product_file_sha256": exact_product_file_sha256,
        "linear_product_file_sha256": linear_product_file_sha256,
        "selected_csharp_sun_linear_max_difference_m": float(
            np.max(np.linalg.norm(python_sun_linear - csharp_sun, axis=1))
        ),
        "selected_csharp_earth_linear_max_difference_m": float(
            np.max(np.linalg.norm(python_earth_linear - csharp_earth, axis=1))
        ),
        "selected_csharp_sun_exact_angle_difference_deg": _summary(
            _angular_separation_deg(python_sun_exact, csharp_sun)
        ),
        "selected_csharp_earth_exact_angle_difference_deg": _summary(
            _angular_separation_deg(python_earth_exact, csharp_earth)
        ),
        "peak_host_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "forbidden_modules_loaded": forbidden,
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
