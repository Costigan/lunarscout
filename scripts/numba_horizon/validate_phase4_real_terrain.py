#!/usr/bin/env python3
"""Compare a bounded real-terrain Numba horizon buffer with C# CUDA output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import rasterio


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
ACCEPTED_ANGULAR_ERROR_DEGREES = 5e-3
REFERENCE_SOLAR_DIAMETER_DEGREES = 0.5
ACCEPTED_SUNLIGHT_FRACTION_ERROR = 0.01
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.contract import (  # noqa: E402
    ContractConfiguration,
    HorizonBuffers,
    SegmentTensor,
)
from lunarscout._numba_horizon.cuda_backend import CudaSession  # noqa: E402
from lunarscout._numba_horizon.generator import generate_patch_horizons  # noqa: E402
from lunarscout._numba_horizon.geometry import (  # noqa: E402
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    build_subpatch_segments,
)
from lunarscout._numba_horizon.pyramid import build_max_pyramid  # noqa: E402
from lunarscout._numba_horizon.hierarchy import traverse_hierarchy  # noqa: E402
from lunarscout._numba_horizon.kernel_math import sample_bilinear  # noqa: E402


def _hash_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _solar_visible_fraction(
    center_above_horizon_degrees: np.ndarray,
    radius_degrees: float = REFERENCE_SOLAR_DIAMETER_DEGREES / 2.0,
) -> np.ndarray:
    """Visible fraction of a uniform circular solar disk above a straight horizon."""
    normalized = np.asarray(center_above_horizon_degrees, dtype=np.float64) / radius_degrees
    clipped = np.clip(normalized, -1.0, 1.0)
    partial = 0.5 + (
        np.arcsin(clipped) + clipped * np.sqrt(np.maximum(0.0, 1.0 - clipped**2))
    ) / np.pi
    return np.where(normalized <= -1.0, 0.0, np.where(normalized >= 1.0, 1.0, partial))


def _load_dem(path: Path) -> tuple[DemGrid, np.ndarray]:
    with rasterio.open(path) as dataset:
        elevation = np.ascontiguousarray(dataset.read(1), dtype=np.float32)
        transform = np.ascontiguousarray(
            dataset.transform.to_gdal(), dtype=np.float64
        )
        crs = dataset.crs.to_dict()
    projection = ProjectionParameters(
        radius_m=float(crs["R"]),
        latitude_origin_rad=float(np.deg2rad(crs["lat_0"])),
        longitude_origin_rad=float(np.deg2rad(crs["lon_0"])),
        scale=float(crs.get("k", crs.get("k_0", 1.0))),
        false_easting_m=float(crs.get("x_0", 0.0)),
        false_northing_m=float(crs.get("y_0", 0.0)),
    )
    return DemGrid(elevation, transform, projection), elevation


def _group_metrics(error: np.ndarray, mask: np.ndarray) -> dict:
    values = error[mask]
    return {
        "count": int(values.size),
        "mean_signed_degrees": float(np.mean(values)),
        "maximum_absolute_degrees": float(np.max(np.abs(values))),
    }


def _compact_comparison(slopes, degrees, csharp_slopes, csharp_degrees) -> dict:
    sentinel_mismatch = np.isneginf(slopes) != np.isneginf(csharp_slopes)
    finite = np.isfinite(degrees) & np.isfinite(csharp_degrees)
    absolute = np.abs(
        degrees[finite].astype(np.float64) - csharp_degrees[finite].astype(np.float64)
    )
    return {
        "slope_sentinel_mismatches": int(np.count_nonzero(sentinel_mismatch)),
        "maximum_absolute_angular_error_degrees": float(np.max(absolute)),
        "mean_absolute_angular_error_degrees": float(np.mean(absolute)),
        "p95_absolute_angular_error_degrees": float(np.percentile(absolute, 95)),
        "p99_absolute_angular_error_degrees": float(np.percentile(absolute, 99)),
        "p99_9_absolute_angular_error_degrees": float(np.percentile(absolute, 99.9)),
        "counts_above_degrees": {
            "1e-6": int(np.count_nonzero(absolute > 1e-6)),
            "1e-5": int(np.count_nonzero(absolute > 1e-5)),
            "1e-4": int(np.count_nonzero(absolute > 1e-4)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csharp_metadata", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4-real-terrain.json",
    )
    parser.add_argument("--warm-repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.warm_repeats < 1:
        parser.error("--warm-repeats must be at least one")
    metadata = json.loads(arguments.csharp_metadata.read_text(encoding="utf-8"))
    config = metadata["configuration"]
    input_paths = [
        Path(path) for path in metadata.get(
            "input_paths", [metadata["input_path"]]
        )
    ]
    slope_path = Path(metadata["output"]["slope_path"])
    degree_path = Path(metadata["output"]["degree_path"])
    segment_path = Path(metadata["output"]["segment_path"])
    if _hash_bytes(slope_path) != metadata["output"]["slope_sha256"]:
        raise RuntimeError("C# slope buffer hash does not match its metadata")
    if _hash_bytes(degree_path) != metadata["output"]["degree_sha256"]:
        raise RuntimeError("C# degree buffer hash does not match its metadata")
    if _hash_bytes(segment_path) != metadata["output"]["segment_sha256"]:
        raise RuntimeError("C# segment tensor hash does not match its metadata")
    pass_paths = [
        Path(path) for path in metadata["output"].get("per_dem_slope_paths", [])
    ]
    pass_hashes = metadata["output"].get("per_dem_slope_sha256", [])
    if len(pass_paths) != len(pass_hashes):
        raise RuntimeError("C# per-DEM slope path/hash counts differ")
    for path, expected_hash in zip(pass_paths, pass_hashes, strict=True):
        if _hash_bytes(path) != expected_hash:
            raise RuntimeError(f"C# per-DEM slope hash does not match: {path}")
    shape = tuple(metadata["output"]["shape"])
    csharp_slopes = np.fromfile(slope_path, dtype="<f4").reshape(shape)
    csharp_degrees = np.fromfile(degree_path, dtype="<f4").reshape(shape)
    csharp_segments = np.fromfile(segment_path, dtype="<f4").reshape(
        metadata["output"]["segment_shape"]
    )

    loaded = [_load_dem(path) for path in input_paths]
    dems = [item[0] for item in loaded]
    elevations = [item[1] for item in loaded]
    pyramids = [build_max_pyramid(dem) for dem in dems]
    started = time.perf_counter()
    segments, _, _ = build_subpatch_segments(
        dems, tile_column=config["tile_column"], tile_row=config["tile_row"],
        tile_width=config["tile_width"], azimuth_count=config["azimuth_count"],
        maximum_distance_m=1_000_000, observer_elevation_m=0,
        subpatch_size=config["subpatch_size"],
        grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
    )
    host_seconds = time.perf_counter() - started
    configuration = ContractConfiguration(
        tile_width=config["tile_width"], tile_height=config["tile_height"],
        azimuth_count=config["azimuth_count"], subpatch_size=config["subpatch_size"],
        dem_count=len(dems), primary_width=dems[0].width,
        primary_height=dems[0].height,
    )
    ids = np.broadcast_to(
        np.arange(len(dems), dtype=np.int32), segments.shape[:-1]
    ).copy()
    generated_tensor = SegmentTensor(segments, ids, configuration)
    session = CudaSession()
    started = time.perf_counter()
    generated = generate_patch_horizons(
        session, generated_tensor, pyramids,
        tile_column=config["tile_column"], tile_row=config["tile_row"],
    )
    cuda_seconds = time.perf_counter() - started
    degrees = generated.degrees()
    generated_passes = [
        session.subpatch_hierarchical_pass(
            segments, pyramids[0], pyramid,
            tile_column=config["tile_column"], tile_row=config["tile_row"],
            tile_width=config["tile_width"], tile_height=config["tile_height"],
            subpatch_size=config["subpatch_size"], pass_index=index,
        )
        for index, pyramid in enumerate(pyramids)
    ]
    csharp_passes = [
        np.fromfile(path, dtype="<f4").reshape(shape) for path in pass_paths
    ]
    warm_hashes = []
    warm_seconds = []
    for _ in range(arguments.warm_repeats):
        started = time.perf_counter()
        repeated = generate_patch_horizons(
            session, generated_tensor, pyramids,
            tile_column=config["tile_column"], tile_row=config["tile_row"],
        )
        warm_seconds.append(time.perf_counter() - started)
        warm_hashes.append(_hash_array(repeated.slopes))
        if not np.array_equal(repeated.slopes, generated.slopes):
            raise RuntimeError("warm Numba CUDA output is not deterministic")
    exact_segments_generated = generate_patch_horizons(
        session, SegmentTensor(csharp_segments, ids, configuration), pyramids,
        tile_column=config["tile_column"], tile_row=config["tile_row"],
    )
    exact_segments_degrees = exact_segments_generated.degrees()
    selected_segment = np.asarray(
        metadata["selected_interpolated_segment"], dtype=np.float32
    )
    selected_pixel = metadata["traversal_trace"]["pixel_index"]
    selected_azimuth = metadata["traversal_trace"]["azimuth_index"]
    selected_pass = metadata["traversal_trace"].get("dem_pass", len(dems) - 1)
    selected_column = selected_pixel % config["tile_width"]
    selected_row = selected_pixel // config["tile_width"]
    device_selected_segment = session.subpatch_interpolation(
        csharp_segments, pyramids[0], pyramids[selected_pass],
        tile_column=config["tile_column"], tile_row=config["tile_row"],
        tile_width=config["tile_width"], subpatch_size=config["subpatch_size"],
        pass_index=selected_pass, pixels=np.array([selected_pixel]),
        azimuths=np.array([selected_azimuth]),
    )[0]
    selected_observer = sample_bilinear(
        pyramids[0].level0,
        config["tile_column"] + selected_column,
        config["tile_row"] + selected_row,
    )
    selected_cpu = traverse_hierarchy(
        selected_segment, pyramids[selected_pass],
        observer_z_m=float(selected_observer),
        radius_m=dems[selected_pass].projection.radius_m,
        map_resolution_m=dems[selected_pass].map_resolution_m,
        pass_index=selected_pass,
    )
    selected_gpu_maximum, selected_gpu_traces, selected_gpu_counts = session.hierarchical(
        selected_segment[np.newaxis], pyramids[selected_pass],
        np.array([selected_observer]),
        np.array([dems[selected_pass].projection.radius_m]),
        dems[selected_pass].map_resolution_m, pass_index=selected_pass,
    )
    selected_gpu_trace = selected_gpu_traces[0, : selected_gpu_counts[0]]
    device_segment_maximum, _, _ = session.hierarchical(
        device_selected_segment[np.newaxis], pyramids[selected_pass],
        np.array([selected_observer]),
        np.array([dems[selected_pass].projection.radius_m]),
        dems[selected_pass].map_resolution_m, pass_index=selected_pass,
    )
    trace_fields = (
        "s_km", "true_distance_m", "level", "cell_x", "cell_y", "pixel_x",
        "pixel_y", "maximum_elevation_m", "sample_elevation_m", "sample_slope",
        "advance_km", "action",
    )
    selected_expected_trace = np.asarray(
        [[np.nan if row[field] is None else row[field] for field in trace_fields]
         for row in metadata["traversal_trace"]["rows"]],
        dtype=np.float32,
    )
    common_rows = min(len(selected_gpu_trace), len(selected_expected_trace))
    decision_difference = np.flatnonzero(
        np.any(
            selected_gpu_trace[:common_rows, 2:5]
            != selected_expected_trace[:common_rows, 2:5], axis=1
        )
        | (selected_gpu_trace[:common_rows, 11]
           != selected_expected_trace[:common_rows, 11])
    )
    first_decision_difference = (
        int(decision_difference[0]) if len(decision_difference) else None
    )
    distance_difference = np.flatnonzero(
        selected_gpu_trace[:common_rows, 0]
        != selected_expected_trace[:common_rows, 0]
    )
    first_distance_difference = (
        int(distance_difference[0]) if len(distance_difference) else None
    )
    expected_sample_rows = np.flatnonzero(
        np.isfinite(selected_expected_trace[:, 9])
    )
    gpu_sample_rows = np.flatnonzero(np.isfinite(selected_gpu_trace[:, 9]))
    expected_maximum_sample_row = int(expected_sample_rows[np.argmax(
        selected_expected_trace[expected_sample_rows, 9]
    )])
    gpu_maximum_sample_row = int(gpu_sample_rows[np.argmax(
        selected_gpu_trace[gpu_sample_rows, 9]
    )])
    if generated.slopes.shape != csharp_slopes.shape:
        raise RuntimeError("C# and Numba slope shapes differ")
    sentinel_mismatch = np.isneginf(generated.slopes) != np.isneginf(csharp_slopes)
    if np.any(sentinel_mismatch):
        raise RuntimeError("C# and Numba negative-infinity sentinels differ")
    finite = np.isfinite(degrees) & np.isfinite(csharp_degrees)
    signed = degrees.astype(np.float64) - csharp_degrees.astype(np.float64)
    absolute = np.abs(signed[finite])
    finite_angular_delta = signed[finite].astype(np.float64)
    half_delta = np.abs(finite_angular_delta) / 2.0
    worst_solar_fraction_by_bin = (
        _solar_visible_fraction(half_delta)
        - _solar_visible_fraction(-half_delta)
    )
    csharp_horizon_solar_fraction_error = np.abs(
        0.5 - _solar_visible_fraction(-finite_angular_delta)
    )
    flat_index = int(np.argmax(np.where(finite, np.abs(signed), -1.0)))
    pixel, azimuth = np.unravel_index(flat_index, degrees.shape)
    rows, columns = np.divmod(np.arange(config["tile_width"] * config["tile_height"]), config["tile_width"])
    edge_pixels = (
        (rows == 0) | (rows == config["tile_height"] - 1)
        | (columns == 0) | (columns == config["tile_width"] - 1)
    )
    seam_pixels = (
        np.isin(rows % config["subpatch_size"], (0, config["subpatch_size"] - 1))
        | np.isin(columns % config["subpatch_size"], (0, config["subpatch_size"] - 1))
    )
    all_bins = np.ones_like(signed, dtype=bool)
    report = {
        "schema_version": 1,
        "scope": "bounded hierarchy-enabled real-terrain C# CUDA versus Numba CUDA horizon comparison",
        "inputs": [
            {
                "path": str(path),
                "sha256": _hash_bytes(path),
                "raster_float32_sha256": _hash_array(elevation),
                "shape_y_x": list(elevation.shape),
            }
            for path, elevation in zip(input_paths, elevations, strict=True)
        ],
        "configuration": config,
        "acceptance": {
            "maximum_angular_error_degrees": ACCEPTED_ANGULAR_ERROR_DEGREES,
            "reference_solar_angular_diameter_degrees": (
                REFERENCE_SOLAR_DIAMETER_DEGREES
            ),
            "accepted_sunlight_fraction_error": ACCEPTED_SUNLIGHT_FRACTION_ERROR,
            "basis": (
                "User-approved conservative angular proxy: one percent of an "
                "approximately 0.5-degree solar diameter. A downstream solar-limb "
                "illumination comparison remains required."
            ),
            "observed_maximum_within_angular_criterion": bool(
                np.max(absolute) <= ACCEPTED_ANGULAR_ERROR_DEGREES
            ),
            "angular_margin_factor": float(
                ACCEPTED_ANGULAR_ERROR_DEGREES / np.max(absolute)
            ) if np.max(absolute) > 0 else None,
            "solar_disk_model": "uniform circular disk and locally straight horizon",
            "maximum_possible_absolute_sunlight_fraction_error": float(
                np.max(worst_solar_fraction_by_bin)
            ),
            "maximum_absolute_sunlight_fraction_error_when_center_on_csharp_horizon": float(
                np.max(csharp_horizon_solar_fraction_error)
            ),
            "observed_maximum_within_sunlight_fraction_criterion": bool(
                np.max(worst_solar_fraction_by_bin)
                <= ACCEPTED_SUNLIGHT_FRACTION_ERROR
            ),
        },
        "csharp": {
            "selected_accelerator_name": metadata["selected_accelerator_name"],
            "selected_accelerator_type": metadata["selected_accelerator_type"],
            "slope_sha256": metadata["output"]["slope_sha256"],
            "degree_sha256": metadata["output"]["degree_sha256"],
            "per_dem_slope_sha256": pass_hashes,
            "capture_elapsed_seconds": metadata["elapsed_seconds"],
        },
        "numba": {
            "device_name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
            "slope_sha256": _hash_array(generated.slopes),
            "degree_sha256": _hash_array(degrees),
            "per_dem_slope_sha256": [_hash_array(value) for value in generated_passes],
            "host_segment_seconds": host_seconds,
            "cuda_call_seconds_including_transfers": cuda_seconds,
            "warm_repeat_count": arguments.warm_repeats,
            "warm_repeat_seconds_including_transfers": warm_seconds,
            "warm_slope_sha256": warm_hashes,
            "warm_outputs_stable": len(set(warm_hashes)) == 1,
        },
        "comparison": {
            "per_dem_python_generated_segments": [
                _compact_comparison(
                    actual,
                    HorizonBuffers(actual).degrees(),
                    expected,
                    HorizonBuffers(expected).degrees(),
                )
                for actual, expected in zip(
                    generated_passes, csharp_passes, strict=True
                )
            ] if csharp_passes else [],
            "exact_csharp_segments": _compact_comparison(
                exact_segments_generated.slopes, exact_segments_degrees,
                csharp_slopes, csharp_degrees,
            ),
            "python_generated_segments": _compact_comparison(
                generated.slopes, degrees, csharp_slopes, csharp_degrees,
            ),
            "segment_maximum_absolute_field_difference": float(np.max(np.abs(
                segments.astype(np.float64) - csharp_segments.astype(np.float64)
            ))),
            "value_count": int(degrees.size),
            "finite_value_count": int(np.count_nonzero(finite)),
            "csharp_nonfinite_count": int(np.count_nonzero(~np.isfinite(csharp_degrees))),
            "numba_nonfinite_count": int(np.count_nonzero(~np.isfinite(degrees))),
            "slope_sentinel_mismatches": int(np.count_nonzero(sentinel_mismatch)),
            "maximum_absolute_angular_error_degrees": float(np.max(absolute)),
            "mean_absolute_angular_error_degrees": float(np.mean(absolute)),
            "median_absolute_angular_error_degrees": float(np.percentile(absolute, 50)),
            "p95_absolute_angular_error_degrees": float(np.percentile(absolute, 95)),
            "p99_absolute_angular_error_degrees": float(np.percentile(absolute, 99)),
            "p99_9_absolute_angular_error_degrees": float(np.percentile(absolute, 99.9)),
            "mean_signed_angular_error_degrees": float(np.mean(signed[finite])),
            "counts_above_degrees": {
                "1e-6": int(np.count_nonzero(absolute > 1e-6)),
                "1e-5": int(np.count_nonzero(absolute > 1e-5)),
                "1e-4": int(np.count_nonzero(absolute > 1e-4)),
                "5e-3_accepted": int(np.count_nonzero(
                    absolute > ACCEPTED_ANGULAR_ERROR_DEGREES
                )),
            },
            "largest_error_location": {
                "pixel_index": int(pixel),
                "tile_column": int(pixel % config["tile_width"]),
                "tile_row": int(pixel // config["tile_width"]),
                "azimuth_index": int(azimuth),
                "azimuth_degrees": float(azimuth * 0.25),
                "csharp_degrees": float(csharp_degrees[pixel, azimuth]),
                "numba_degrees": float(degrees[pixel, azimuth]),
            },
            "spatial_groups": {
                "tile_edge": _group_metrics(signed, edge_pixels[:, None] & all_bins),
                "subpatch_seam": _group_metrics(signed, seam_pixels[:, None] & all_bins),
                "other": _group_metrics(
                    signed, (~edge_pixels & ~seam_pixels)[:, None] & all_bins
                ),
            },
            "azimuth_quadrants": {
                str(index): _group_metrics(
                    signed,
                    all_bins
                    & (np.arange(config["azimuth_count"])[None, :] >= index * 360)
                    & (np.arange(config["azimuth_count"])[None, :] < (index + 1) * 360),
                )
                for index in range(4)
            },
            "selected_trace": {
                "dem_pass": selected_pass,
                "pixel_index": selected_pixel,
                "azimuth_index": selected_azimuth,
                "csharp_rows": len(selected_expected_trace),
                "numba_gpu_rows": len(selected_gpu_trace),
                "csharp_output_slope": float(
                    (csharp_passes[selected_pass] if csharp_passes else csharp_slopes)[
                        selected_pixel, selected_azimuth
                    ]
                ),
                "numba_gpu_direct_slope": float(selected_gpu_maximum[0]),
                "numpy_cpu_direct_slope": float(selected_cpu.maximum_slope),
                "numba_device_interpolated_segment_direct_slope": float(
                    device_segment_maximum[0]
                ),
                "device_interpolated_segment_maximum_field_difference": float(
                    np.max(np.abs(
                        device_selected_segment.astype(np.float64)
                        - selected_segment.astype(np.float64)
                    ))
                ),
                "device_interpolated_segment": device_selected_segment.tolist(),
                "csharp_interpolated_segment": selected_segment.tolist(),
                "first_level_cell_or_action_difference": first_decision_difference,
                "first_parameter_distance_difference": first_distance_difference,
                "csharp_row_at_first_parameter_distance_difference": (
                    selected_expected_trace[first_distance_difference].tolist()
                    if first_distance_difference is not None else None
                ),
                "numba_row_at_first_parameter_distance_difference": (
                    selected_gpu_trace[first_distance_difference].tolist()
                    if first_distance_difference is not None else None
                ),
                "csharp_row_before_first_parameter_distance_difference": (
                    selected_expected_trace[first_distance_difference - 1].tolist()
                    if first_distance_difference else None
                ),
                "numba_row_before_first_parameter_distance_difference": (
                    selected_gpu_trace[first_distance_difference - 1].tolist()
                    if first_distance_difference else None
                ),
                "csharp_maximum_sample_row_index": expected_maximum_sample_row,
                "csharp_maximum_sample_row": (
                    selected_expected_trace[expected_maximum_sample_row].tolist()
                ),
                "numba_maximum_sample_row_index": gpu_maximum_sample_row,
                "numba_maximum_sample_row": (
                    selected_gpu_trace[gpu_maximum_sample_row].tolist()
                ),
                "csharp_row_at_first_difference": (
                    selected_expected_trace[first_decision_difference].tolist()
                    if first_decision_difference is not None else None
                ),
                "numba_row_at_first_difference": (
                    selected_gpu_trace[first_decision_difference].tolist()
                    if first_decision_difference is not None else None
                ),
                "csharp_row_before_first_difference": (
                    selected_expected_trace[first_decision_difference - 1].tolist()
                    if first_decision_difference else None
                ),
                "numba_row_before_first_difference": (
                    selected_gpu_trace[first_decision_difference - 1].tolist()
                    if first_decision_difference else None
                ),
            },
        },
        "qualification": (
            f"This fixture measures one 16x16 real-terrain tile and {len(dems)} "
            "DEM pass(es) on one GPU. "
            "The timings are diagnostic single calls, not Phase 5 benchmarks."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
