#!/usr/bin/env python3
"""Measure Phase 6 serial or bounded-pipeline horizon generation with file output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import rasterio


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
DEFAULT_INPUTS = (
    Path("/d/lunar_analyst_scenarios/test_scenario/dem.tif"),
    Path("/d/datasets/viper_v71_2024_medium/other/dem.tif"),
    Path("/d/viper/maps/gsfc/site_20v2/Site20v2_final_adj_5mpp_surf.tif"),
    Path("/d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif"),
)


def _load_dem(path: Path):
    from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters

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


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _GpuMemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak_mib = 0
        self.samples = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        pid = str(__import__("os").getpid())
        while not self._stop.wait(0.05):
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
                    self.peak_mib = max(self.peak_mib, int(fields[1]))
                    self.samples += 1

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._stop.set()
        self._thread.join()


def _run(arguments: argparse.Namespace) -> dict:
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from lunarscout._numba_horizon.contract import (
        ContractConfiguration,
        HorizonBuffers,
        SegmentTensor,
    )
    from lunarscout._numba_horizon.cuda_backend import CudaSession
    from lunarscout._numba_horizon.file_format import HorizonTileStore
    from lunarscout._numba_horizon.geometry import (
        GridConvergenceInput,
        build_subpatch_segments_numba,
    )
    from lunarscout._numba_horizon.pipeline import (
        enumerate_patches,
        run_bounded_pipeline,
    )
    from lunarscout._numba_horizon.pyramid import load_max_pyramid_cache

    def progress(message: str) -> None:
        print(f"[phase6 {arguments.mode}] {message}", flush=True)

    paths = [Path(value).resolve() for value in arguments.inputs]
    progress("loading DEMs and pyramid caches")
    load_started = time.perf_counter()
    dems = [_load_dem(path) for path in paths]
    pyramids = [
        load_max_pyramid_cache(dem, path.with_suffix(".pyr.bin"))
        for dem, path in zip(dems, paths, strict=True)
    ]
    dem_and_pyramid_load_seconds = time.perf_counter() - load_started
    patches = enumerate_patches(dems[0].width, dems[0].height)[: arguments.patch_count]
    output_directory = Path(arguments.output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    store = HorizonTileStore(output_directory)
    session_started = time.perf_counter()
    session = CudaSession(production_concurrency=arguments.gpu_workers)
    session_construction_seconds = time.perf_counter() - session_started

    preparation_times: list[float] = []
    cuda_times: list[float] = []
    degree_times: list[float] = []
    write_times: list[float] = []
    pass_kernel_times: list[list[float]] = []
    production_timings: list[dict[str, float]] = []

    def prepare(patch):
        started = time.perf_counter()
        values, _, _ = build_subpatch_segments_numba(
            dems,
            tile_column=patch.tile_x,
            tile_row=patch.tile_y,
            tile_width=patch.kernel_width,
            azimuth_count=1440,
            maximum_distance_m=1_000_000.0,
            observer_elevation_m=0.0,
            subpatch_size=8,
            grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
            parallel=True,
        )
        configuration = ContractConfiguration(
            patch.kernel_width,
            patch.kernel_height,
            1440,
            8,
            len(dems),
            dems[0].width,
            dems[0].height,
        )
        ids = np.broadcast_to(
            np.arange(len(dems), dtype=np.int32), values.shape[:-1]
        ).copy()
        result = SegmentTensor(values, ids, configuration)
        preparation_times.append(time.perf_counter() - started)
        return result

    def compute(patch, segments):
        started = time.perf_counter()
        slopes, pass_times = session.subpatch_hierarchical_all_passes(
            segments.values,
            pyramids,
            tile_column=patch.tile_x,
            tile_row=patch.tile_y,
            tile_width=patch.kernel_width,
            tile_height=patch.kernel_height,
            subpatch_size=8,
        )
        cuda_times.append(time.perf_counter() - started)
        pass_kernel_times.append(pass_times)
        production_timings.append(dict(session.last_production_timings))
        return slopes

    def finalize(_patch, slopes):
        started = time.perf_counter()
        degrees = HorizonBuffers(slopes).degrees()
        degree_times.append(time.perf_counter() - started)
        return degrees

    def process(patch, segments):
        return finalize(patch, compute(patch, segments))

    progress("warming segment compiler, CUDA kernel, resident pyramids, and compressor")
    warm_segments = prepare(patches[0])
    warm_degrees = process(patches[0], warm_segments)
    warm_write_started = time.perf_counter()
    warm_path = store.write(
        patches[0].tile_y,
        patches[0].tile_x,
        0.0,
        warm_degrees,
        compress=arguments.compress,
        valid_width=patches[0].width,
        valid_height=patches[0].height,
    )
    startup = {
        "dem_and_pyramid_load_seconds": dem_and_pyramid_load_seconds,
        "cuda_session_construction_seconds": session_construction_seconds,
        "first_segment_generation_seconds": preparation_times[-1],
        "first_cuda_compile_and_execution_seconds": cuda_times[-1],
        "first_cuda_timings": dict(production_timings[-1]),
        "first_degree_conversion_seconds": degree_times[-1],
        "first_compression_and_write_seconds": (
            time.perf_counter() - warm_write_started
        ),
    }
    preparation_times.clear()
    cuda_times.clear()
    degree_times.clear()
    pass_kernel_times.clear()
    production_timings.clear()
    warm_path.unlink()

    progress(f"measuring {len(patches)} patches")
    started = time.perf_counter()
    maximum_queue_depth = 0
    pipeline_metrics = None
    with _GpuMemorySampler() as gpu_memory:
        if arguments.mode == "serial":
            output_paths = []
            for index, patch in enumerate(patches, start=1):
                segments = prepare(patch)
                degrees = process(patch, segments)
                write_started = time.perf_counter()
                output_paths.append(
                    store.write(
                        patch.tile_y,
                        patch.tile_x,
                        0.0,
                        degrees,
                        compress=arguments.compress,
                        valid_width=patch.width,
                        valid_height=patch.height,
                    )
                )
                write_times.append(time.perf_counter() - write_started)
                progress(f"generated {index}/{len(patches)}")
        else:
            writer_enabled = arguments.mode == "writer-pipeline"
            pipeline_result = run_bounded_pipeline(
                patches,
                store=store,
                prepare_patch=prepare,
                processor_factory=lambda _worker_id: (
                    compute if writer_enabled else process
                ),
                finalize_patch=finalize if writer_enabled else None,
                compress=arguments.compress,
                skip_existing=False,
                prepared_queue_capacity=arguments.queue_capacity,
                writer_queue_capacity=(
                    arguments.writer_queue_capacity if writer_enabled else None
                ),
                worker_count=arguments.gpu_workers,
                progress_stream=sys.stdout,
            )
            output_paths = list(pipeline_result.output_paths)
            write_times.append(pipeline_result.write_seconds)
            maximum_queue_depth = pipeline_result.maximum_prepared_queue_depth
            pipeline_metrics = {
                "preparation_seconds": pipeline_result.preparation_seconds,
                "compute_seconds": pipeline_result.compute_seconds,
                "finalization_seconds": pipeline_result.finalization_seconds,
                "write_seconds": pipeline_result.write_seconds,
                "maximum_prepared_queue_depth": (
                    pipeline_result.maximum_prepared_queue_depth
                ),
                "maximum_writer_queue_depth": (
                    pipeline_result.maximum_writer_queue_depth
                ),
                "producer_enqueue_wait_seconds": list(
                    pipeline_result.producer_enqueue_wait_seconds
                ),
                "consumer_dequeue_wait_seconds": list(
                    pipeline_result.consumer_dequeue_wait_seconds
                ),
                "writer_enqueue_wait_seconds": list(
                    pipeline_result.writer_enqueue_wait_seconds
                ),
                "writer_dequeue_wait_seconds": list(
                    pipeline_result.writer_dequeue_wait_seconds
                ),
            }
    elapsed = time.perf_counter() - started

    files = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _hash_path(path),
        }
        for path in sorted(output_paths)
    ]
    return {
        "schema_version": 1,
        "startup": startup,
        "mode": arguments.mode,
        "gpu_workers": arguments.gpu_workers,
        "compressed": arguments.compress,
        "patch_count": len(patches),
        "elapsed_seconds": elapsed,
        "patches_per_second": len(patches) / elapsed,
        "wall_seconds_per_patch": elapsed / len(patches),
        "preparation_seconds": preparation_times,
        "cuda_call_seconds": cuda_times,
        "degree_conversion_seconds": degree_times,
        "write_seconds": write_times,
        "pass_kernel_seconds": pass_kernel_times,
        "production_cuda_timings": production_timings,
        "maximum_prepared_queue_depth": maximum_queue_depth,
        "pipeline_metrics": pipeline_metrics,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_process_gpu_memory_mib": gpu_memory.peak_mib,
        "gpu_memory_samples": gpu_memory.samples,
        "files": files,
        "total_output_bytes": sum(item["bytes"] for item in files),
        "inputs": [str(path) for path in paths],
        "native_runtime_modules_loaded": sorted(
            name
            for name in sys.modules
            if name == "clr"
            or name.startswith("pythonnet")
            or name == "moonlib"
            or name.startswith("moonlib.")
        ),
        "notes": [
            "Warm timing follows one unrecorded segment/CUDA/compression warm-up patch.",
            "CUDA call includes segment upload, kernel events, synchronization, and result copy.",
            f"Pipeline mode uses one producer and {arguments.gpu_workers} CUDA worker(s)/work-buffer slot(s).",
            "Writer-pipeline mode moves degree conversion, compression, and staged writing to a one-item bounded writer queue.",
            "Pyramids and fixed-shape transient segment/output device buffers remain resident and are reused.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("serial", "pipeline", "writer-pipeline"), required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--patch-count", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=1)
    parser.add_argument("--writer-queue-capacity", type=int, default=1)
    parser.add_argument("--gpu-workers", type=int, default=1)
    parser.add_argument("--compress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("inputs", nargs="*", default=DEFAULT_INPUTS)
    arguments = parser.parse_args()
    report = _run(arguments)
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
