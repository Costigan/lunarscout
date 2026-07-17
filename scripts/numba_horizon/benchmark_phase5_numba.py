#!/usr/bin/env python3
"""Capture an unoptimized Phase 5 Numba production-scale benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import rasterio


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
DEFAULT_OUTPUT = REPOSITORY / "docs" / "numba-horizon-phase-5-numba-benchmark.json"
DEFAULT_CSHARP_REPORT = (
    REPOSITORY / "docs" / "numba-horizon-phase-5-csharp-corrected-benchmark.json"
)
DEFAULT_INPUTS = (
    Path("/d/lunar_analyst_scenarios/test_scenario/dem.tif"),
    Path("/d/datasets/viper_v71_2024_medium/other/dem.tif"),
    Path("/d/viper/maps/gsfc/site_20v2/Site20v2_final_adj_5mpp_surf.tif"),
    Path("/d/viper/maps/lola/LDEM_80S_20M-2017-06-15-processed.tif"),
)


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_dem(path: Path):
    from lunarscout._numba_horizon.geometry import DemGrid, ProjectionParameters

    with rasterio.open(path) as dataset:
        elevation = np.ascontiguousarray(dataset.read(1), dtype=np.float32)
        transform = np.ascontiguousarray(dataset.transform.to_gdal(), dtype=np.float64)
        crs = dataset.crs.to_dict()
    projection = ProjectionParameters(
        radius_m=float(crs["R"]),
        latitude_origin_rad=float(np.deg2rad(crs["lat_0"])),
        longitude_origin_rad=float(np.deg2rad(crs["lon_0"])),
        scale=float(crs.get("k", crs.get("k_0", 1.0))),
        false_easting_m=float(crs.get("x_0", 0.0)),
        false_northing_m=float(crs.get("y_0", 0.0)),
    )
    return DemGrid(elevation, transform, projection)


def _summary(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "seconds": values,
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
        "mean_seconds": statistics.mean(values),
        "population_standard_deviation_seconds": statistics.pstdev(values),
    }


def _run_worker(arguments: argparse.Namespace) -> dict:
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import numba
    from lunarscout._numba_horizon.contract import (
        ContractConfiguration,
        HorizonBuffers,
        SegmentTensor,
    )
    from lunarscout._numba_horizon.cuda_backend import CudaSession
    from lunarscout._numba_horizon.geometry import (
        GridConvergenceInput,
        build_subpatch_segments_numba,
    )
    from lunarscout._numba_horizon.pyramid import load_max_pyramid_cache

    process_started = time.perf_counter()

    def progress(stage: str, message: str) -> None:
        elapsed = time.perf_counter() - process_started
        print(f"[phase5 +{elapsed:9.3f}s] {stage}: {message}", flush=True)

    progress(
        "worker",
        f"started with {len(arguments.inputs)} DEMs, {arguments.patch_count} patches, "
        f"and {arguments.repeats} warm repeats",
    )
    inputs = [Path(value).resolve() for value in arguments.inputs]
    caches = [path.with_suffix(".pyr.bin") for path in inputs]
    input_metadata = []
    for index, (dem_path, cache_path) in enumerate(
        zip(inputs, caches, strict=True), start=1
    ):
        progress("input_hash", f"hashing DEM {index}/{len(inputs)}: {dem_path}")
        started = time.perf_counter()
        input_metadata.append(
            {
                "path": str(dem_path),
                "size_bytes": dem_path.stat().st_size,
                "sha256": _hash_path(dem_path),
                "pyramid_cache_path": str(cache_path),
                "pyramid_cache_size_bytes": cache_path.stat().st_size,
                "pyramid_cache_sha256": _hash_path(cache_path),
            }
        )
        progress(
            "input_hash",
            f"completed DEM {index}/{len(inputs)} in {time.perf_counter() - started:.3f}s",
        )

    started = time.perf_counter()
    dems = []
    for index, path in enumerate(inputs, start=1):
        progress("dem_load", f"loading DEM {index}/{len(inputs)}: {path}")
        item_started = time.perf_counter()
        dems.append(_load_dem(path))
        progress(
            "dem_load",
            f"loaded DEM {index}/{len(inputs)} in {time.perf_counter() - item_started:.3f}s",
        )
    dem_load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pyramids = []
    for index, (dem, cache) in enumerate(
        zip(dems, caches, strict=True), start=1
    ):
        progress("pyramid_cache", f"loading cache {index}/{len(caches)}: {cache}")
        item_started = time.perf_counter()
        pyramids.append(load_max_pyramid_cache(dem, cache))
        progress(
            "pyramid_cache",
            f"loaded cache {index}/{len(caches)} in {time.perf_counter() - item_started:.3f}s",
        )
    pyramid_cache_load_seconds = time.perf_counter() - started

    configuration = ContractConfiguration(
        tile_width=128,
        tile_height=128,
        azimuth_count=1440,
        subpatch_size=8,
        dem_count=len(dems),
        primary_width=dems[0].width,
        primary_height=dems[0].height,
    )

    def build_tensor(tile_column: int) -> SegmentTensor:
        values, _, _ = build_subpatch_segments_numba(
            dems,
            tile_column=tile_column,
            tile_row=0,
            tile_width=128,
            azimuth_count=1440,
            maximum_distance_m=1_000_000.0,
            observer_elevation_m=0.0,
            subpatch_size=8,
            grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
            parallel=True,
        )
        ids = np.broadcast_to(
            np.arange(len(dems), dtype=np.int32), values.shape[:-1]
        ).copy()
        return SegmentTensor(values, ids, configuration)

    progress("first_segment", "starting segment generation and CPU Numba compilation")
    started = time.perf_counter()
    first_tensor = build_tensor(0)
    first_segment_generation_seconds = time.perf_counter() - started
    progress("first_segment", f"completed in {first_segment_generation_seconds:.3f}s")
    progress("cuda_init", "initializing Numba CUDA session")
    started = time.perf_counter()
    session = CudaSession()
    session_initialization_seconds = time.perf_counter() - started
    progress(
        "cuda_init",
        f"selected {session.info.name} in {session_initialization_seconds:.3f}s",
    )

    def run_cuda(tensor: SegmentTensor, tile_column: int, label: str):
        progress("cuda_pass", f"{label}: uploading resources and starting all DEM passes")
        slopes, pass_seconds = session.subpatch_hierarchical_all_passes(
            tensor.values,
            pyramids,
            tile_column=tile_column,
            tile_row=0,
            tile_width=configuration.tile_width,
            tile_height=configuration.tile_height,
            subpatch_size=configuration.subpatch_size,
        )
        progress(
            "cuda_pass",
            f"{label}: completed all DEM passes; kernel seconds "
            + ", ".join(f"{value:.3f}" for value in pass_seconds),
        )
        return HorizonBuffers(slopes), pass_seconds

    progress("first_cuda", "starting CUDA compilation and first production patch")
    started = time.perf_counter()
    first, first_pass_seconds = run_cuda(first_tensor, 0, "first patch")
    first_cuda_compile_and_execution_seconds = time.perf_counter() - started
    progress("first_cuda", f"completed in {first_cuda_compile_and_execution_seconds:.3f}s")
    started = time.perf_counter()
    first_degrees = first.degrees()
    first_degree_conversion_seconds = time.perf_counter() - started
    first_hashes = {
        "slope_sha256": _hash_array(first.slopes),
        "degree_sha256": _hash_array(first_degrees),
    }
    progress(
        "first_output",
        f"converted and hashed first output; degree conversion took "
        f"{first_degree_conversion_seconds:.3f}s",
    )

    single_times = []
    single_degree_times = []
    single_pass_times = []
    single_hashes = []
    for repeat in range(arguments.repeats):
        progress("warm_single", f"starting repeat {repeat + 1}/{arguments.repeats}")
        started = time.perf_counter()
        result, pass_times = run_cuda(
            first_tensor, 0, f"warm single {repeat + 1}/{arguments.repeats}"
        )
        cuda_elapsed = time.perf_counter() - started
        single_times.append(cuda_elapsed)
        single_pass_times.append(pass_times)
        started = time.perf_counter()
        degrees = result.degrees()
        degree_elapsed = time.perf_counter() - started
        single_degree_times.append(degree_elapsed)
        single_hashes.append(
            {"slope_sha256": _hash_array(result.slopes), "degree_sha256": _hash_array(degrees)}
        )
        progress(
            "warm_single",
            f"completed repeat {repeat + 1}/{arguments.repeats}: "
            f"CUDA {cuda_elapsed:.3f}s, degree conversion {degree_elapsed:.3f}s",
        )
    if any(item != first_hashes for item in single_hashes):
        raise RuntimeError("warm single-patch output changed")

    multi_runs = []
    expected_patch_hashes = None
    for repeat in range(arguments.repeats):
        progress("warm_multi", f"starting repeat {repeat + 1}/{arguments.repeats}")
        run_started = time.perf_counter()
        segment_seconds = []
        cuda_seconds = []
        cuda_pass_seconds = []
        degree_seconds = []
        patch_hashes = []
        for patch_index in range(arguments.patch_count):
            tile_column = patch_index * 128
            progress(
                "warm_multi_segment",
                f"repeat {repeat + 1}/{arguments.repeats}, "
                f"patch {patch_index + 1}/{arguments.patch_count}: starting",
            )
            started = time.perf_counter()
            tensor = build_tensor(tile_column)
            segment_elapsed = time.perf_counter() - started
            segment_seconds.append(segment_elapsed)
            progress(
                "warm_multi_cuda",
                f"repeat {repeat + 1}/{arguments.repeats}, "
                f"patch {patch_index + 1}/{arguments.patch_count}: segments completed "
                f"in {segment_elapsed:.3f}s; starting CUDA",
            )
            started = time.perf_counter()
            result, pass_times = run_cuda(
                tensor,
                tile_column,
                f"multi {repeat + 1}/{arguments.repeats} patch "
                f"{patch_index + 1}/{arguments.patch_count}",
            )
            cuda_elapsed = time.perf_counter() - started
            cuda_seconds.append(cuda_elapsed)
            cuda_pass_seconds.append(pass_times)
            started = time.perf_counter()
            degrees = result.degrees()
            degree_elapsed = time.perf_counter() - started
            degree_seconds.append(degree_elapsed)
            patch_hashes.append(
                {
                    "patch_index": patch_index,
                    "tile_column": tile_column,
                    "tile_row": 0,
                    "slope_sha256": _hash_array(result.slopes),
                    "degree_sha256": _hash_array(degrees),
                }
            )
            progress(
                "warm_multi_patch",
                f"repeat {repeat + 1}/{arguments.repeats}, "
                f"patch {patch_index + 1}/{arguments.patch_count}: completed; "
                f"CUDA {cuda_elapsed:.3f}s, degree conversion {degree_elapsed:.3f}s",
            )
        elapsed = time.perf_counter() - run_started
        if expected_patch_hashes is None:
            expected_patch_hashes = patch_hashes
        elif patch_hashes != expected_patch_hashes:
            raise RuntimeError("multi-patch output changed between warm repeats")
        multi_runs.append(
            {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "patches_per_second": arguments.patch_count / elapsed,
                "segment_generation_seconds": segment_seconds,
                "cuda_seconds_including_transfers": cuda_seconds,
                "cuda_kernel_seconds_by_pass": cuda_pass_seconds,
                "degree_conversion_seconds": degree_seconds,
                "patch_hashes": patch_hashes,
            }
        )
        progress(
            "warm_multi",
            f"completed repeat {repeat + 1}/{arguments.repeats} in {elapsed:.3f}s "
            f"({arguments.patch_count / elapsed:.4f} patches/s)",
        )

    multi_elapsed = [item["elapsed_seconds"] for item in multi_runs]

    pipeline_runs = []
    expected_pipeline_hashes = expected_patch_hashes
    for repeat in range(arguments.repeats):
        progress("warm_pipeline", f"starting repeat {repeat + 1}/{arguments.repeats}")
        run_started = time.perf_counter()
        segment_seconds = []
        segment_ready_seconds = []
        cuda_seconds = []
        cuda_pass_seconds = []
        degree_seconds = []
        patch_hashes = []

        def build_timed(tile_column: int):
            segment_started = time.perf_counter()
            tensor = build_tensor(tile_column)
            return tensor, time.perf_counter() - segment_started

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase5-segments") as producer:
            pending = producer.submit(build_timed, 0)
            for patch_index in range(arguments.patch_count):
                tile_column = patch_index * 128
                progress(
                    "warm_pipeline_segment",
                    f"repeat {repeat + 1}/{arguments.repeats}, "
                    f"patch {patch_index + 1}/{arguments.patch_count}: waiting for producer",
                )
                ready_started = time.perf_counter()
                tensor, segment_elapsed = pending.result()
                segment_ready_elapsed = time.perf_counter() - ready_started
                segment_seconds.append(segment_elapsed)
                segment_ready_seconds.append(segment_ready_elapsed)
                if patch_index + 1 < arguments.patch_count:
                    pending = producer.submit(build_timed, (patch_index + 1) * 128)
                progress(
                    "warm_pipeline_cuda",
                    f"repeat {repeat + 1}/{arguments.repeats}, "
                    f"patch {patch_index + 1}/{arguments.patch_count}: segments took "
                    f"{segment_elapsed:.3f}s, consumer waited {segment_ready_elapsed:.3f}s; "
                    "starting CUDA while producer prepares the next patch",
                )
                started = time.perf_counter()
                result, pass_times = run_cuda(
                    tensor,
                    tile_column,
                    f"pipeline {repeat + 1}/{arguments.repeats} patch "
                    f"{patch_index + 1}/{arguments.patch_count}",
                )
                cuda_elapsed = time.perf_counter() - started
                cuda_seconds.append(cuda_elapsed)
                cuda_pass_seconds.append(pass_times)
                started = time.perf_counter()
                degrees = result.degrees()
                degree_elapsed = time.perf_counter() - started
                degree_seconds.append(degree_elapsed)
                patch_hashes.append(
                    {
                        "patch_index": patch_index,
                        "tile_column": tile_column,
                        "tile_row": 0,
                        "slope_sha256": _hash_array(result.slopes),
                        "degree_sha256": _hash_array(degrees),
                    }
                )
                progress(
                    "warm_pipeline_patch",
                    f"repeat {repeat + 1}/{arguments.repeats}, "
                    f"patch {patch_index + 1}/{arguments.patch_count}: completed; "
                    f"CUDA {cuda_elapsed:.3f}s, degree conversion {degree_elapsed:.3f}s",
                )
        elapsed = time.perf_counter() - run_started
        if patch_hashes != expected_pipeline_hashes:
            raise RuntimeError("pipelined multi-patch output differs from serial output")
        pipeline_runs.append(
            {
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "patches_per_second": arguments.patch_count / elapsed,
                "segment_generation_seconds": segment_seconds,
                "consumer_segment_wait_seconds": segment_ready_seconds,
                "cuda_seconds_including_transfers": cuda_seconds,
                "cuda_kernel_seconds_by_pass": cuda_pass_seconds,
                "degree_conversion_seconds": degree_seconds,
                "patch_hashes": patch_hashes,
            }
        )
        progress(
            "warm_pipeline",
            f"completed repeat {repeat + 1}/{arguments.repeats} in {elapsed:.3f}s "
            f"({arguments.patch_count / elapsed:.4f} patches/s)",
        )

    pipeline_elapsed = [item["elapsed_seconds"] for item in pipeline_runs]
    report = {
        "schema_version": 1,
        "report_kind": "numba_phase_5_production_benchmark",
        "configuration": {
            "patch_count": arguments.patch_count,
            "patches": [
                {"index": i, "tile_column": i * 128, "tile_row": 0, "width": 128, "height": 128}
                for i in range(arguments.patch_count)
            ],
            "azimuth_bins": 1440,
            "observer_elevation_m": 0.0,
            "hierarchy_enabled": True,
            "dem_pass_count": len(dems),
            "cuda_streams": 1,
            "serial_host_pipeline_concurrency": 1,
            "pipelined_segment_producers": 1,
            "pipelined_gpu_consumers": 1,
            "pipelined_queue_capacity": 1,
            "warm_repeats": arguments.repeats,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "logical_cpu_count": os.cpu_count(),
            "numba_threads": numba.get_num_threads(),
            "cuda_device_name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
        },
        "inputs": input_metadata,
        "startup": {
            "dem_load_seconds": dem_load_seconds,
            "warm_pyramid_cache_load_seconds": pyramid_cache_load_seconds,
            "first_segment_generation_including_numba_cpu_compilation_seconds": first_segment_generation_seconds,
            "cuda_session_initialization_seconds": session_initialization_seconds,
            "first_cuda_compile_and_execution_seconds": first_cuda_compile_and_execution_seconds,
            "first_cuda_kernel_seconds_by_pass": first_pass_seconds,
            "first_degree_conversion_seconds": first_degree_conversion_seconds,
            "first_output_hashes": first_hashes,
        },
        "warm_single_patch": {
            "cuda_seconds_including_transfers": _summary(single_times),
            "cuda_kernel_seconds_by_pass": single_pass_times,
            "degree_conversion_seconds": _summary(single_degree_times),
            "output_hashes_stable": True,
            "output_hashes": first_hashes,
        },
        "warm_multi_patch": {
            "execution_model": "Serial segment generation followed by CUDA for each patch.",
            "runs": multi_runs,
            "elapsed": _summary(multi_elapsed),
            "patches_per_second_from_median_elapsed": arguments.patch_count / statistics.median(multi_elapsed),
            "outputs_stable": True,
        },
        "warm_multi_patch_pipelined": {
            "execution_model": "One CPU segment producer prepares patch N+1 while one CUDA consumer processes patch N.",
            "runs": pipeline_runs,
            "elapsed": _summary(pipeline_elapsed),
            "patches_per_second_from_median_elapsed": arguments.patch_count
            / statistics.median(pipeline_elapsed),
            "outputs_match_serial": True,
        },
        "runtime": {
            "worker_total_seconds": time.perf_counter() - process_started,
            "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "limitations": [
            "Immutable DEM pyramids remain device-resident for the session; segments and output buffers are still allocated and transferred per patch.",
            "The serial measurement is retained separately from the one-producer/one-CUDA-consumer pipelined measurement.",
            "The pipelined measurement overlaps segment generation with CUDA work but still uses one CUDA default stream; it does not reproduce the C# four concurrent CUDA workers.",
            "Warm pyramid timing reads existing C#-format float32 mip caches and is not a fresh-pyramid build.",
            "CUDA timing includes allocation, host-to-device transfer, synchronization, and device-to-host transfer.",
            "Compression and horizon-file writing are outside this kernel-prototype benchmark.",
        ],
    }
    progress("report", f"writing worker report to {arguments.worker_output}")
    arguments.worker_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    progress("worker", "completed successfully")
    return report


def _sample_process(pid: int) -> tuple[int | None, int | None]:
    host_rss = None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                host_rss = int(line.split()[1]) * 1024
                break
    except (FileNotFoundError, ProcessLookupError):
        pass
    gpu_mib = None
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] == str(pid):
                try:
                    gpu_mib = int(fields[1])
                except ValueError:
                    pass
    return host_rss, gpu_mib


def _run_parent(arguments: argparse.Namespace) -> int:
    worker_output = Path("/tmp/lunarscout-numba-horizon-phase5-worker.json")
    worker_output.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-output",
        str(worker_output),
        "--patch-count",
        str(arguments.patch_count),
        "--repeats",
        str(arguments.repeats),
        "--inputs",
        *[str(path) for path in arguments.inputs],
    ]
    started = time.perf_counter()
    print(f"[phase5 launcher] starting worker: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []

    def relay_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)

    relay = threading.Thread(target=relay_output, name="phase5-output-relay")
    relay.start()
    host_samples = []
    gpu_samples = []
    while process.poll() is None:
        host_rss, gpu_mib = _sample_process(process.pid)
        if host_rss is not None:
            host_samples.append(host_rss)
        if gpu_mib is not None:
            gpu_samples.append(gpu_mib)
        time.sleep(0.05)
    process.wait()
    relay.join()
    output = "".join(output_lines)
    if process.returncode:
        raise RuntimeError(f"Phase 5 worker failed ({process.returncode}):\n{output[-8000:]}")
    report = json.loads(worker_output.read_text(encoding="utf-8"))
    report["external_sampling"] = {
        "method": "Parent sampled /proc/<pid>/status and nvidia-smi every approximately 50 ms.",
        "launcher_elapsed_seconds": time.perf_counter() - started,
        "host_sample_count": len(host_samples),
        "peak_host_rss_bytes": max(host_samples) if host_samples else None,
        "gpu_sample_count": len(gpu_samples),
        "peak_process_gpu_memory_mib": max(gpu_samples) if gpu_samples else None,
        "minimum_process_gpu_memory_mib": min(gpu_samples) if gpu_samples else None,
        "sampling_limitation": "Polling can miss allocations shorter than the sample interval.",
    }
    report["acceptance_policy"] = {
        "warm_single_patch_latency_maximum_csharp_ratio": 2.0,
        "warm_multi_patch_throughput_minimum_csharp_ratio": 0.5,
        "peak_gpu_memory_maximum_csharp_ratio": 1.5,
        "peak_host_memory_maximum_csharp_ratio": 1.5,
        "first_use_maximum_csharp_ratio": 5.0,
        "comparison_status": "pending matched corrected-C# benchmark",
    }
    if DEFAULT_CSHARP_REPORT.is_file():
        csharp = json.loads(DEFAULT_CSHARP_REPORT.read_text(encoding="utf-8"))
        warm = next(
            item for item in csharp["runs"] if item["name"] == "warm_cached_pyramids"
        )
        csharp_patch_count = csharp["configuration"]["patch_count"]
        csharp_patch_seconds = warm["elapsed_seconds"] / csharp_patch_count
        numba_first_seconds = (
            report["startup"]["cuda_session_initialization_seconds"]
            + report["startup"]["first_cuda_compile_and_execution_seconds"]
        )
        csharp_first_seconds = (
            csharp["runtime"]["generator_initialization_seconds"]
            + csharp["runs"][0]["elapsed_seconds"] / csharp_patch_count
        )
        comparison = {
            "csharp_report": str(DEFAULT_CSHARP_REPORT.relative_to(REPOSITORY)),
            "csharp_patch_count": csharp_patch_count,
            "warm_single_patch_latency_csharp_ratio": report["warm_single_patch"]
            ["cuda_seconds_including_transfers"]["median_seconds"]
            / csharp_patch_seconds,
            "warm_multi_patch_throughput_csharp_ratio": report[
                "warm_multi_patch_pipelined"
            ]["patches_per_second_from_median_elapsed"]
            / warm["patches_per_second"],
            "peak_gpu_memory_csharp_ratio": report["external_sampling"]
            ["peak_process_gpu_memory_mib"]
            / csharp["gpu_memory"]["phases"]["warm_cached_pyramids"]
            ["peak_process_gpu_memory_mib"],
            "peak_host_memory_csharp_ratio": report["external_sampling"]
            ["peak_host_rss_bytes"]
            / warm["host_peak_working_set_bytes"],
            "first_use_csharp_ratio": numba_first_seconds / csharp_first_seconds,
        }
        comparison["all_provisional_gates_pass"] = (
            comparison["warm_single_patch_latency_csharp_ratio"] <= 2.0
            and comparison["warm_multi_patch_throughput_csharp_ratio"] >= 0.5
            and comparison["peak_gpu_memory_csharp_ratio"] <= 1.5
            and comparison["peak_host_memory_csharp_ratio"] <= 1.5
            and comparison["first_use_csharp_ratio"] <= 5.0
        )
        report["comparison"] = comparison
        report["acceptance_policy"]["comparison_status"] = (
            "pass" if comparison["all_provisional_gates_pass"] else "fail"
        )
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[phase5 launcher] wrote {arguments.output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--patch-count", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inputs", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    arguments = parser.parse_args()
    if arguments.patch_count < 1 or arguments.repeats < 1:
        parser.error("patch count and repeats must be positive")
    if arguments.worker:
        if arguments.worker_output is None:
            parser.error("--worker-output is required with --worker")
        _run_worker(arguments)
        return 0
    return _run_parent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
