"""Benchmark a representative file-backed native temporal series.

Inputs: --scenario with dem.tif, lighting/horizons, SPICE, and native runtime.
Outputs: a timestamped series and JSON report under the scenario analysis folder.
Resources: defaults to 3,800 float32 512 x 512 layers (about 3.7 GiB scratch).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import lunarscout as ls
import numpy as np
import rasterio

from _example_support import example_parser, require_native_scenario


def _time_axis(start: str, *, count: int, step_hours: float) -> ls.TimeRange:
    if count < 1:
        raise SystemExit("--layer-count must be positive.")
    first = ls.times(start, start, step_hours=step_hours).values[0]
    step_microseconds = int(round(step_hours * 3_600_000_000))
    stop = first + np.timedelta64(step_microseconds * (count - 1), "us")
    return ls.times(first, stop, step_hours=step_hours)


def _directory_metrics(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _rss_bytes() -> int:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux process RSS is unavailable.")


def _monitor_resources(
    scratch: Path,
    stop: threading.Event,
    peaks: dict[str, int],
) -> None:
    while not stop.wait(0.05):
        scratch_bytes = sum(
            path.stat().st_size
            for path in scratch.glob(".lunarscout-native-temporal-*")
            if path.is_file()
        )
        peaks["scratch_bytes"] = max(peaks["scratch_bytes"], scratch_bytes)
        peaks["rss_bytes"] = max(peaks["rss_bytes"], _rss_bytes())


def _timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def _processor_name() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> None:
    parser = example_parser(__doc__, native=True)
    parser.add_argument("--layer-count", type=int, default=3_800)
    parser.add_argument(
        "--signal",
        choices=(
            "sun_fraction",
            "sun_over_horizon_deg",
            "earth_over_horizon_deg",
        ),
        default="sun_over_horizon_deg",
    )
    parser.add_argument(
        "--output",
        default="analysis/native_performance_sun_margin.temporal",
    )
    parser.add_argument(
        "--report",
        default="analysis/native_performance_report.json",
    )
    parser.add_argument(
        "--scratch",
        default="analysis/native_performance_scratch",
    )
    parser.add_argument("--random-read-count", type=int, default=32)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip native generation and benchmark an existing completed series.",
    )
    args = parser.parse_args()

    scenario = require_native_scenario(args.scenario)
    times = _time_axis(
        args.start, count=args.layer_count, step_hours=args.step_hours
    )
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise SystemExit("Scenario DEM must be georeferenced.")
    output = scenario.output_path(args.output)
    report_path = scenario.output_path(args.report)
    scratch = scenario.output_path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    estimate = ls.native.estimate_temporal_allocation(
        signal=args.signal,
        times=times,
        georef=georef,
        storage="geotiff_series",
    )

    disk_before = shutil.disk_usage(output.parent)
    rss_before = _rss_bytes()
    peaks = {"scratch_bytes": 0, "rss_bytes": rss_before}
    progress_times: dict[str, float] = {}
    started = time.perf_counter()

    def progress(event: ls.native.NativeTemporalProgress) -> None:
        now = time.perf_counter()
        progress_times.setdefault(f"{event.stage}_first", now)
        progress_times[f"{event.stage}_last"] = now
        if event.stage in {"preflight", "complete"}:
            print(f"{event.percent:6.2f}% [{event.stage}] {event.message}")

    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=_monitor_resources,
        args=(scratch, monitor_stop, peaks),
        daemon=True,
    )
    monitor.start()
    generation_seconds: float | None = None
    try:
        if args.reuse_existing:
            series = ls.open_temporal_cube(output)
        else:
            operation = getattr(scenario, args.signal)
            series, generation_seconds = _timed(
                lambda: operation(
                    times=times,
                    storage="geotiff_series",
                    output=args.output,
                    scratch_directory=scratch,
                    observer_elevation_meters=args.observer_elevation_meters,
                    overwrite=args.overwrite,
                    progress_callback=progress,
                )
            )
    finally:
        monitor_stop.set()
        monitor.join()

    if series.shape != estimate.shape:
        raise RuntimeError(
            f"Completed series shape {series.shape} differs from {estimate.shape}."
        )
    series.close()

    validated, validation_seconds = _timed(
        lambda: ls.open_temporal_cube(output, validate_layers=True)
    )
    validated.close()
    metadata_only, metadata_open_seconds = _timed(
        lambda: ls.open_temporal_cube(output, validate_layers=False)
    )
    metadata_only.close()

    vrt_started = time.perf_counter()
    with rasterio.open(output / "series.vrt") as vrt:
        vrt_open_seconds = time.perf_counter() - vrt_started
        vrt_band_count = vrt.count
    if vrt_band_count != args.layer_count:
        raise RuntimeError("VRT does not expose the expected layer count.")

    random_count = min(max(1, args.random_read_count), args.layer_count)
    indices = np.random.default_rng(20270621).choice(
        args.layer_count, size=random_count, replace=False
    )
    layer_bytes = georef.width * georef.height * estimate.dtype.itemsize
    reader = ls.open_temporal_cube(
        output,
        validate_layers=False,
        layer_cache_bytes=layer_bytes * random_count,
        max_open_datasets=min(64, random_count),
    )

    def read_samples() -> float:
        checksum = 0.0
        for index in indices:
            layer, _layer_georef = reader.read_time(reader.time_for_layer(int(index)))
            checksum += float(layer[0, 0])
        return checksum

    cold_checksum, cold_read_seconds = _timed(read_samples)
    warm_checksum, warm_read_seconds = _timed(read_samples)
    reader.close()
    if cold_checksum != warm_checksum:
        raise RuntimeError("Cold and warm random-read checksums differ.")

    reducer = ls.open_temporal_cube(
        output,
        validate_layers=False,
        layer_cache_bytes=0,
        max_open_datasets=8,
    )
    reductions: dict[str, dict[str, float]] = {}
    for name, operation in (
        ("mean", ls.temporal_mean),
        ("min", ls.temporal_min),
        ("max", ls.temporal_max),
        ("std", ls.temporal_std),
    ):
        result, elapsed = _timed(lambda operation=operation: operation(reducer)[0])
        reductions[name] = {
            "seconds": elapsed,
            "minimum": float(np.nanmin(result)),
            "maximum": float(np.nanmax(result)),
        }
    reducer.close()

    file_count, output_size = _directory_metrics(output)
    disk_after = shutil.disk_usage(output.parent)
    stage_seconds: dict[str, float] = {}
    preflight = progress_times.get("preflight_first")
    first_write = progress_times.get("write_series_first")
    last_write = progress_times.get("write_series_last")
    complete = progress_times.get("complete_first")
    if preflight is not None and first_write is not None:
        stage_seconds["native_stream_and_scratch_assembly_approx"] = (
            first_write - preflight
        )
    if first_write is not None and last_write is not None:
        stage_seconds["geotiff_writing_approx"] = last_write - first_write
    if last_write is not None and complete is not None:
        stage_seconds["vrt_finalize_and_validation_approx"] = complete - last_write

    report = {
        "scenario": str(scenario.root),
        "signal": args.signal,
        "output": str(output),
        "shape": list(estimate.shape),
        "dtype": str(estimate.dtype),
        "time_range": {
            "start": str(times.values[0]),
            "stop": str(times.values[-1]),
            "step_hours": times.step_hours,
            "count": times.time_count,
        },
        "allocation_bytes": estimate.estimated_bytes,
        "runtime_seconds": {
            "generation_total": generation_seconds,
            "series_validation": validation_seconds,
            "metadata_only_open": metadata_open_seconds,
            "vrt_open": vrt_open_seconds,
            "random_reads_application_cache_cold": cold_read_seconds,
            "random_reads_application_cache_warm": warm_read_seconds,
            "reductions": reductions,
            "wall_total": time.perf_counter() - started,
            "generation_stages": stage_seconds,
        },
        "random_reads": {
            "count": random_count,
            "indices": [int(value) for value in indices],
            "checksum": cold_checksum,
            "cold_definition": "new Lunarscout reader with an empty full-layer cache; OS page cache was not dropped",
            "warm_definition": "same reader and samples retained in the Lunarscout full-layer cache",
        },
        "resource_bytes": {
            "rss_before": rss_before,
            "peak_rss_observed": peaks["rss_bytes"],
            "peak_scratch_observed": peaks["scratch_bytes"],
            "output_size": output_size,
            "filesystem_free_before": disk_before.free,
            "filesystem_free_after": disk_after.free,
        },
        "output_file_count": file_count,
        "scratch_files_remaining": len(
            list(scratch.glob(".lunarscout-native-temporal-*"))
        ),
        "host": {
            "platform": os.uname().sysname + " " + os.uname().release,
            "processor": _processor_name(),
            "logical_cpu_count": os.cpu_count(),
        },
        "notes": [
            "Stage timings derived from progress callbacks are approximate.",
            "Peak RSS includes Python, Python.NET, .NET, Moonlib, GDAL, and native runtime allocations.",
            "The benchmark does not access scenario.db, register products, or reproject data.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
