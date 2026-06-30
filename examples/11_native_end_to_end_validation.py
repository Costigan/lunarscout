"""Validate native solar-fraction memory/file parity and series lifecycle.

Inputs: --scenario with dem.tif, lighting/horizons, SPICE, and native runtime.
Outputs: analysis/native_validation_sun_fraction.temporal and a JSON report.
Resources: one float32 memory cube, one float32 scratch cube, and final TIFFs.
"""

from __future__ import annotations

import hashlib
import json
import resource
import threading
import time
from pathlib import Path

import lunarscout as ls
import numpy as np
import rasterio
from rasterio.crs import CRS

from _example_support import example_parser, native_times, require_native_scenario


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _scratch_monitor(root: Path, stop: threading.Event, peak: list[int]) -> None:
    while not stop.wait(0.01):
        size = sum(
            path.stat().st_size
            for path in root.glob(".lunarscout-native-temporal-*")
            if path.is_file()
        )
        peak[0] = max(peak[0], size)


def _run_timed(operation):
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def main() -> None:
    args = example_parser(__doc__, native=True).parse_args()
    scenario = require_native_scenario(args.scenario)
    times = native_times(args)
    _dem, georef = ls.read_geotiff(scenario.dem_path())
    if georef is None:
        raise SystemExit("Scenario DEM must be georeferenced.")

    output = scenario.output_path("analysis/native_validation_sun_fraction.temporal")
    report_path = scenario.output_path("analysis/native_validation_report.json")
    scratch = scenario.output_path("analysis/native_validation_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    estimate = ls.native.estimate_temporal_allocation(
        signal="sun_fraction", times=times, georef=georef, storage="memory"
    )

    cube, memory_seconds = _run_timed(
        lambda: scenario.sun_fraction(
            times=times,
            storage="memory",
            observer_elevation_meters=args.observer_elevation_meters,
        )
    )

    stop = threading.Event()
    scratch_peak = [0]
    monitor = threading.Thread(
        target=_scratch_monitor, args=(scratch, stop, scratch_peak), daemon=True
    )
    monitor.start()
    try:
        series, initial_file_seconds = _run_timed(
            lambda: scenario.sun_fraction(
                times=times,
                storage="geotiff_series",
                output="analysis/native_validation_sun_fraction.temporal",
                scratch_directory=scratch,
                observer_elevation_meters=args.observer_elevation_meters,
                overwrite=True,
            )
        )
    finally:
        stop.set()
        monitor.join()

    before_cancel = _tree_digests(output)
    series.close()
    cancel_requested = False

    def cancel_after_first_tile(progress: ls.native.NativeTemporalProgress) -> None:
        nonlocal cancel_requested
        if progress.stage == "native_stream":
            cancel_requested = True

    cancellation_started = time.perf_counter()
    try:
        scenario.sun_fraction(
            times=times,
            storage="geotiff_series",
            output="analysis/native_validation_sun_fraction.temporal",
            scratch_directory=scratch,
            observer_elevation_meters=args.observer_elevation_meters,
            overwrite=True,
            progress_callback=cancel_after_first_tile,
            cancellation_requested=lambda: cancel_requested,
        )
    except ls.NativeTemporalError as exc:
        if exc.code != "native_temporal_cancelled":
            raise
        cancellation_code = exc.code
    else:
        raise RuntimeError("Native validation cancellation did not interrupt generation.")
    cancellation_seconds = time.perf_counter() - cancellation_started

    after_cancel = _tree_digests(output)
    if after_cancel != before_cancel:
        raise RuntimeError("Cancelled overwrite changed the completed output series.")
    if list(scratch.glob(".lunarscout-native-temporal-*")):
        raise RuntimeError("Cancelled generation left native scratch files behind.")
    if list(output.parent.glob(f".{output.name}.staging-*")):
        raise RuntimeError("Cancelled generation left staging directories behind.")

    restarted, restart_seconds = _run_timed(
        lambda: scenario.sun_fraction(
            times=times,
            storage="geotiff_series",
            output="analysis/native_validation_sun_fraction.temporal",
            scratch_directory=scratch,
            observer_elevation_meters=args.observer_elevation_meters,
            overwrite=True,
        )
    )

    file_values = np.stack(
        [restarted.read_layer(index)[0] for index in range(restarted.shape[0])]
    )
    if not np.array_equal(cube.values, file_values):
        difference = np.abs(cube.values - file_values)
        raise RuntimeError(
            f"Memory/file solar fractions differ; max_abs_difference={difference.max()}"
        )
    quantized = np.rint(file_values * np.float32(255.0)).astype(np.uint8)
    converted = np.empty_like(file_values)
    np.multiply(quantized, 1.0 / 255.0, out=converted, casting="unsafe")
    if file_values.dtype != np.float32 or not np.array_equal(file_values, converted):
        raise RuntimeError("Solar fractions are not exact float32 uint8/255 conversions.")
    if float(file_values.min()) < 0.0 or float(file_values.max()) > 1.0:
        raise RuntimeError("Solar fractions fall outside [0, 1].")

    manifest_bytes = (output / "manifest.json").read_bytes()
    completion = json.loads((output / "COMPLETE").read_text(encoding="utf-8"))
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if completion["manifest_sha256"] != manifest_digest:
        raise RuntimeError("Completion digest does not match manifest bytes.")

    layer_metadata: list[dict[str, object]] = []
    manifest_crs = CRS.from_wkt(restarted.georef.projection_wkt)
    for index, layer_path in enumerate(restarted.layer_paths):
        with rasterio.open(layer_path) as dataset:
            layer_metadata.append(
                {
                    "path": layer_path.relative_to(output).as_posix(),
                    "size_bytes": layer_path.stat().st_size,
                    "dtype": dataset.dtypes[0],
                    "nodata": dataset.nodata,
                    "dimensions": [dataset.height, dataset.width],
                    "transform": tuple(float(value) for value in dataset.transform),
                    "projection_matches_manifest": dataset.crs == manifest_crs,
                }
            )
            if not np.array_equal(dataset.read(1), file_values[index]):
                raise RuntimeError(f"Independent Rasterio layer read differs at index {index}.")

    if restarted.vrt_path is None:
        raise RuntimeError("Completed native series does not have a VRT.")
    with rasterio.open(restarted.vrt_path) as vrt:
        if vrt.count != restarted.shape[0]:
            raise RuntimeError("Rasterio could not open the complete temporal VRT.")
        descriptions = list(vrt.descriptions)
        for index in range(restarted.shape[0]):
            if not np.array_equal(vrt.read(index + 1), file_values[index]):
                raise RuntimeError(f"VRT band differs from its TIFF at index {index}.")
    restarted.close()

    evidence = {
        "scenario": str(scenario.root),
        "time_range": {
            "start": str(times.values[0]),
            "stop": str(times.values[-1]),
            "step_hours": times.step_hours,
            "count": times.time_count,
        },
        "shape": list(estimate.shape),
        "dtype": str(file_values.dtype),
        "allocation_bytes": estimate.estimated_bytes,
        "runtime_seconds": {
            "memory": memory_seconds,
            "initial_file_backed": initial_file_seconds,
            "cancelled_overwrite": cancellation_seconds,
            "restart_file_backed": restart_seconds,
        },
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scratch_peak_bytes_observed": scratch_peak[0],
        "scratch_files_remaining": 0,
        "output_size_bytes": _directory_size(output),
        "output_file_count": len(before_cancel),
        "memory_file_pixel_equal": True,
        "uint8_to_float32_exact": True,
        "value_range": [float(file_values.min()), float(file_values.max())],
        "cancellation_code": cancellation_code,
        "cancelled_overwrite_preserved": True,
        "restart_completed": True,
        "manifest_sha256": manifest_digest,
        "layer_metadata": layer_metadata,
        "vrt_band_descriptions": descriptions,
        "qgis_validation": "manual validation remains",
    }
    report_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
