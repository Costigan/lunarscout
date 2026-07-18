#!/usr/bin/env python3
"""Measure sustained serial PSR stages on a bounded all-valid real batch."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import threading
import time

import numpy as np
import rasterio

from lunarscout._numba_horizon.file_format import HorizonTileStore
from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters
from lunarscout._numba_horizon.product_vectors import generate_moon_me_vectors
from lunarscout._numba_horizon.psr import reduce_sun_vectors_for_psr
from lunarscout._numba_horizon.psr_cuda import PsrCudaSession
from lunarscout._numba_horizon.psr_pipeline import (
    PsrPipelineMetrics,
    PsrTiming,
    run_psr_product,
)
from lunarscout.georeference import GeoReference
from lunarscout.spice_geometry import iter_times


DEFAULT_SCENARIO = Path("/e/lunar_analyst_scenarios/mons-mouton")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_grid(path: Path, width: int, height: int) -> tuple[DemGrid, GeoReference]:
    with rasterio.open(path) as dataset:
        if width > dataset.width or height > dataset.height:
            raise ValueError("requested benchmark rectangle exceeds the DEM")
        elevation = np.ascontiguousarray(
            dataset.read(1, window=((0, height), (0, width))), dtype=np.float32
        )
        transform = tuple(float(value) for value in dataset.transform.to_gdal())
        projection_wkt = dataset.crs.to_wkt()
        projection_proj4 = dataset.crs.to_proj4()
        crs = dataset.crs.to_dict()
        nodata = dataset.nodata
    dem = DemGrid(
        elevation,
        np.asarray(transform, dtype=np.float64),
        ProjectionParameters(
            radius_m=float(crs["R"]),
            latitude_origin_rad=float(np.deg2rad(crs["lat_0"])),
            longitude_origin_rad=float(np.deg2rad(crs["lon_0"])),
            scale=float(crs.get("k", crs.get("k_0", 1.0))),
            false_easting_m=float(crs.get("x_0", 0.0)),
            false_northing_m=float(crs.get("y_0", 0.0)),
        ),
    )
    georef = GeoReference(
        projection_wkt=projection_wkt,
        projection_proj4=projection_proj4,
        affine_transform=transform,
        width=width,
        height=height,
        pixel_size_x=transform[1],
        pixel_size_y=transform[5],
        nodata=nodata,
    )
    return dem, georef


class _ResourceSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes = 0
        self.peak_gpu_memory_mib = 0
        self.gpu_utilization_samples: list[int] = []

    def _run(self) -> None:
        pid = str(os.getpid())
        while not self._stop.wait(0.05):
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        self.peak_rss_bytes = max(
                            self.peak_rss_bytes, int(line.split()[1]) * 1024
                        )
                        break
            except OSError:
                pass
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2 and fields[0] == pid:
                    self.peak_gpu_memory_mib = max(
                        self.peak_gpu_memory_mib, int(fields[1])
                    )
            utilization = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in utilization.stdout.splitlines():
                try:
                    self.gpu_utilization_samples.append(int(line.strip()))
                except ValueError:
                    pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        self._thread.join()


def _stage_summary(events: list[PsrTiming]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        grouped[event.stage].append(event.seconds)
    result = {}
    for stage, samples in sorted(grouped.items()):
        values = np.asarray(samples, dtype=np.float64)
        result[stage] = {
            "count": len(samples),
            "total_seconds": float(values.sum()),
            "mean_seconds": float(values.mean()),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95)),
            "minimum_seconds": float(values.min()),
            "maximum_seconds": float(values.max()),
        }
    return result


def _wait_summary(values: tuple[float, ...]) -> dict[str, float | int]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0:
        return {"count": 0, "total_seconds": 0.0}
    return {
        "count": int(samples.size),
        "total_seconds": float(samples.sum()),
        "mean_seconds": float(samples.mean()),
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "maximum_seconds": float(samples.max()),
    }


def _product_snapshot(path: Path) -> dict:
    with rasterio.open(path) as dataset:
        values = dataset.read(1)
        mask = dataset.dataset_mask()
        metadata = {
            "profile": dataset.profile,
            "dataset_tags": dataset.tags(),
            "band_tags": dataset.tags(1),
            "transform": tuple(dataset.transform),
            "crs_wkt": dataset.crs.to_wkt(),
            "nodata": dataset.nodata,
        }
    return {
        "values": values,
        "mask": mask,
        "metadata": metadata,
        "file_sha256": _sha256(path),
        "file_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--patch-rows", type=int, default=4)
    parser.add_argument("--patch-columns", type=int, default=4)
    parser.add_argument(
        "--pipeline-mode", choices=("serial", "bounded"), default="serial"
    )
    parser.add_argument("--decoded-horizon-capacity", type=int, default=2)
    parser.add_argument("--writer-queue-capacity", type=int, default=1)
    parser.add_argument("--reader-worker-count", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.patch_rows < 1 or arguments.patch_columns < 1:
        raise ValueError("patch row and column counts must be positive")

    scenario = arguments.scenario.expanduser().resolve()
    dem_path = scenario / "dem.tif"
    horizon_root = scenario / "horizons"
    width = arguments.patch_columns * 128
    height = arguments.patch_rows * 128
    dem, georef = _load_grid(dem_path, width, height)
    store = HorizonTileStore(horizon_root)
    paths = []
    for tile_y in range(0, height, 128):
        for tile_x in range(0, width, 128):
            path = store.find_existing_path(tile_y, tile_x, 0.0)
            if path is None:
                raise RuntimeError(f"missing valid horizon tile at {tile_y},{tile_x}")
            paths.append(path)

    times = tuple(
        iter_times(
            "1970-01-01T00:00:00Z",
            "2044-01-01T00:00:00Z",
            timedelta(hours=6),
        )
    )
    vector_started = time.perf_counter()
    vectors = generate_moon_me_vectors("sun", times).vectors_m
    vector_seconds = time.perf_counter() - vector_started
    reduced, reduced_indices = reduce_sun_vectors_for_psr(dem, vectors)
    warm_horizon = store.read(0, 0, 0.0)
    if warm_horizon is None:
        raise RuntimeError("warm-up horizon disappeared")
    PsrCudaSession().compute_patch(
        dem, warm_horizon, reduced, tile_y=0, tile_x=0
    )

    events: list[PsrTiming] = []
    metrics_values: list[PsrPipelineMetrics] = []
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    with tempfile.TemporaryDirectory(prefix="lunarscout-psr-instrumentation-") as temp:
        temp_root = Path(temp)
        control_path = temp_root / "control.tif"
        control_started = time.perf_counter()
        run_psr_product(
            dem=dem,
            georef=georef,
            horizon_store=store,
            output_path=control_path,
            sun_vectors_m=vectors,
            backend="cuda",
            pipeline_mode="serial",
        )
        control_seconds = time.perf_counter() - control_started
        control = _product_snapshot(control_path)

        instrumented_path = temp_root / "instrumented.tif"
        cpu_before = resource.getrusage(resource.RUSAGE_SELF)
        instrumented_started = time.perf_counter()
        with _ResourceSampler() as sampler:
            run_psr_product(
                dem=dem,
                georef=georef,
                horizon_store=store,
                output_path=instrumented_path,
                sun_vectors_m=vectors,
                backend="cuda",
                timing_callback=events.append,
                metrics_callback=metrics_values.append,
                pipeline_mode=arguments.pipeline_mode,
                decoded_horizon_capacity=arguments.decoded_horizon_capacity,
                writer_queue_capacity=arguments.writer_queue_capacity,
                reader_worker_count=arguments.reader_worker_count,
            )
        instrumented_seconds = time.perf_counter() - instrumented_started
        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        instrumented = _product_snapshot(instrumented_path)

    patch_count = arguments.patch_rows * arguments.patch_columns
    total_compressed_horizon_bytes = sum(path.stat().st_size for path in paths)
    total_decoded_horizon_bytes = patch_count * 128 * 128 * 1440 * 4
    stage_summary = _stage_summary(events)
    decompression_seconds = stage_summary["cbin_decompression"]["total_seconds"]
    cpu_seconds = (
        cpu_after.ru_utime
        + cpu_after.ru_stime
        - cpu_before.ru_utime
        - cpu_before.ru_stime
    )
    gpu_utilization = np.asarray(sampler.gpu_utilization_samples, dtype=np.float64)
    if len(metrics_values) != 1:
        raise RuntimeError("PSR pipeline did not report exactly one metrics record")
    metrics = metrics_values[0]
    report = {
        "schema": "lunarscout-numba-phase6b-psr-pipeline-instrumentation-v1",
        "input": {
            "scenario": str(scenario),
            "dem_path": str(dem_path),
            "dem_sha256": _sha256(dem_path),
            "horizon_root": str(horizon_root),
            "total_compressed_horizon_bytes": total_compressed_horizon_bytes,
            "horizon_paths": [str(path) for path in paths],
            "horizon_sha256": [_sha256(path) for path in paths],
            "width": width,
            "height": height,
            "patch_rows": arguments.patch_rows,
            "patch_columns": arguments.patch_columns,
            "patch_count": patch_count,
            "all_valid": True,
            "vector_count": len(times),
            "reduced_vector_count": int(reduced_indices.size),
            "reduced_vector_indices_sha256": hashlib.sha256(
                reduced_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "vector_generation_seconds": vector_seconds,
        "control": {
            "elapsed_seconds": control_seconds,
            "patches_per_second": patch_count / control_seconds,
            "file_sha256": control["file_sha256"],
            "file_bytes": control["file_bytes"],
        },
        "instrumented": {
            "pipeline_mode": arguments.pipeline_mode,
            "elapsed_seconds": instrumented_seconds,
            "patches_per_second": patch_count / instrumented_seconds,
            "file_sha256": instrumented["file_sha256"],
            "file_bytes": instrumented["file_bytes"],
            "stages": stage_summary,
            "data_throughput": {
                "compressed_input_mib_per_pipeline_second": (
                    total_compressed_horizon_bytes
                    / instrumented_seconds
                    / (1024 * 1024)
                ),
                "decoded_output_gib_per_pipeline_second": (
                    total_decoded_horizon_bytes
                    / instrumented_seconds
                    / (1024 * 1024 * 1024)
                ),
                "decoded_output_gib_per_aggregate_decompression_second": (
                    total_decoded_horizon_bytes
                    / decompression_seconds
                    / (1024 * 1024 * 1024)
                ),
            },
        },
        "resource_usage": {
            "process_cpu_seconds": cpu_seconds,
            "process_cpu_core_equivalents": cpu_seconds / instrumented_seconds,
            "process_cpu_percent_of_one_core": 100.0 * cpu_seconds / instrumented_seconds,
            "logical_cpu_count": os.cpu_count(),
            "sampled_peak_host_rss_bytes": sampler.peak_rss_bytes,
            "process_max_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
            * 1024,
            "sampled_peak_process_gpu_memory_mib": sampler.peak_gpu_memory_mib,
            "gpu_utilization_sample_count": int(gpu_utilization.size),
            "gpu_utilization_mean_percent": (
                float(gpu_utilization.mean()) if gpu_utilization.size else None
            ),
            "gpu_utilization_p95_percent": (
                float(np.percentile(gpu_utilization, 95))
                if gpu_utilization.size
                else None
            ),
            "gpu_utilization_maximum_percent": (
                int(gpu_utilization.max()) if gpu_utilization.size else None
            ),
        },
        "identity": {
            "pixel_mismatch_count": int(
                np.count_nonzero(control["values"] != instrumented["values"])
            ),
            "mask_mismatch_count": int(
                np.count_nonzero(control["mask"] != instrumented["mask"])
            ),
            "metadata_equal": control["metadata"] == instrumented["metadata"],
            "file_hash_equal": control["file_sha256"] == instrumented["file_sha256"],
        },
        "bounds": {
            "configured_decoded_horizon_capacity": metrics.decoded_horizon_capacity,
            "configured_reader_worker_count": metrics.reader_worker_count,
            "configured_writer_queue_capacity": metrics.writer_queue_capacity,
            "maximum_live_decoded_horizons": metrics.maximum_live_decoded_horizons,
            "maximum_reader_queue_depth": metrics.maximum_reader_queue_depth,
            "maximum_writer_queue_depth": metrics.maximum_writer_queue_depth,
            "maximum_live_decoded_horizon_bytes": (
                metrics.maximum_live_decoded_horizons * 128 * 128 * 1440 * 4
            ),
            "configured_maximum_decoded_horizon_bytes": (
                metrics.decoded_horizon_capacity * 128 * 128 * 1440 * 4
            ),
            "reader_enqueue_wait": _wait_summary(
                metrics.reader_enqueue_wait_seconds
            ),
            "cuda_dequeue_wait": _wait_summary(metrics.cuda_dequeue_wait_seconds),
            "writer_enqueue_wait": _wait_summary(
                metrics.writer_enqueue_wait_seconds
            ),
            "writer_dequeue_wait": _wait_summary(
                metrics.writer_dequeue_wait_seconds
            ),
        },
        "notes": [
            "The control run is serial; the instrumented run uses the requested pipeline mode and bounds.",
            "Horizon lookup includes the current structural completeness scan of each candidate file.",
            "kernel_execution_gpu is CUDA event time; cuda_synchronization_boundary is wall time that includes waiting for the same kernel and is non-additive.",
            "TIFF outputs were created under an isolated temporary directory and removed after comparison.",
            f"Process max RSS before the measured pair was {int(usage_before.ru_maxrss) * 1024} bytes.",
        ],
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
