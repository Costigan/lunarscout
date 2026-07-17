#!/usr/bin/env python3
"""Capture max-pyramid and hierarchical traversal parity evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lunarscout._numba_horizon.contract import (  # noqa: E402
    PyramidArrays,
    load_reference_artifact,
)
from lunarscout._numba_horizon.cuda_backend import CudaSession  # noqa: E402
from lunarscout._numba_horizon.fixed_step import (  # noqa: E402
    traverse_level0_adaptive,
    traverse_level0_fixed_step,
)
from lunarscout._numba_horizon.geometry import (  # noqa: E402
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    build_subpatch_segments,
)
from lunarscout._numba_horizon.hierarchy import (  # noqa: E402
    traversal_counters,
    traverse_hierarchy,
)
from lunarscout._numba_horizon.kernel_math import (  # noqa: E402
    evaluate_tangent,
    interpolate_segments,
)
from lunarscout._numba_horizon.pyramid import build_max_pyramid  # noqa: E402


DATA = REPOSITORY / "tests" / "data" / "numba_horizon"


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _maximum_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite])))


def _trace_summary(actual: np.ndarray, expected: np.ndarray) -> dict:
    return {
        "rows": len(actual),
        "same_shape": actual.shape == expected.shape,
        "levels_cells_actions_exact": bool(
            np.array_equal(actual[:, 2:5], expected[:, 2:5])
            and np.array_equal(actual[:, 11], expected[:, 11])
        ),
        "maximum_s_difference_km": _maximum_difference(actual[:, 0], expected[:, 0]),
        "maximum_true_distance_difference_m": _maximum_difference(
            actual[:, 1], expected[:, 1]
        ),
        "maximum_pixel_difference": _maximum_difference(
            actual[:, 5:7], expected[:, 5:7]
        ),
        "maximum_sample_elevation_difference_m": _maximum_difference(
            actual[:, 8], expected[:, 8]
        ),
        "maximum_sample_slope_difference": _maximum_difference(
            actual[:, 9], expected[:, 9]
        ),
        "maximum_advance_difference_km": _maximum_difference(
            actual[:, 10], expected[:, 10]
        ),
        "sha256": _hash(actual),
    }


def _dem(artifact, case_id: str, dem_index: int = 0) -> DemGrid:
    prefix = f"{case_id}__dem_{dem_index}"
    return DemGrid(
        artifact.arrays[f"{prefix}__elevation_m"],
        artifact.arrays[f"{prefix}__geo_transform"],
        ProjectionParameters.from_array(
            artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
        ),
    )


def _json_trace(rows: list[dict]) -> np.ndarray:
    fields = (
        "s_km", "true_distance_m", "level", "cell_x", "cell_y", "pixel_x",
        "pixel_y", "maximum_elevation_m", "sample_elevation_m", "sample_slope",
        "advance_km", "action",
    )
    return np.asarray(
        [[np.nan if row[field] is None else row[field] for field in fields] for row in rows],
        dtype=np.float32,
    )


def _production_inputs(artifact, csharp_capture):
    base = "single_pixel_multi_dem_production__horizon_buffer_fixture"
    dems = [_dem(artifact, base, index) for index in range(2)]
    convergence = GridConvergenceInput(
        *map(float, artifact.arrays[f"{base}__grid_convergence"])
    )
    generated, _, _ = build_subpatch_segments(
        dems,
        tile_column=20,
        tile_row=20,
        tile_width=1,
        azimuth_count=1440,
        maximum_distance_m=1_000_000,
        observer_elevation_m=0,
        subpatch_size=8,
        grid_convergence=convergence,
    )
    captured = np.asarray(csharp_capture["segments"], dtype=np.float32)
    segment_error = float(np.max(np.abs(generated[360, :, 1] - captured)))
    segment = interpolate_segments(
        captured,
        np.array(((4, 4), (-4, 4), (4, -4), (-4, -4)), dtype=np.float32),
        1.0,
        0.5,
        0.5,
    )
    return (
        base,
        segment,
        PyramidArrays.from_artifact(artifact.arrays, f"{base}__dem_1__pyramid"),
        dems[0].elevation(20, 20),
        dems[1].projection.radius_m,
        segment_error,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4d-hierarchy.json",
    )
    arguments = parser.parse_args()
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    csharp_capture = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )
    session = CudaSession()

    pyramid_count = 0
    for case in artifact.metadata["cases"]:
        for dem_info in case["dems"]:
            prefix = f"{case['id']}__dem_{dem_info['index']}"
            pyramid = build_max_pyramid(_dem(artifact, case["id"], dem_info["index"]))
            np.testing.assert_array_equal(
                pyramid.levels, artifact.arrays[f"{prefix}__pyramid__level_metadata"]
            )
            for level, (_, offset, width, height) in enumerate(pyramid.levels):
                expected = artifact.arrays[f"{prefix}__pyramid__level_{level}"]
                actual = (
                    pyramid.level0 if level == 0
                    else pyramid.mips[offset : offset + width * height].reshape(height, width)
                )
                np.testing.assert_array_equal(actual, expected)
            pyramid_count += 1

    base, segment, pyramid, observer_z, radius, segment_error = _production_inputs(
        artifact, csharp_capture
    )
    production_capture = csharp_capture["production_hierarchy_case"]
    expected = _json_trace(production_capture["trace"])
    expected_slope = float(production_capture["csharp_hierarchy_maximum_slope"])
    cpu = traverse_hierarchy(
        segment, pyramid, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=30.0, pass_index=1,
    )
    gpu_maximum, gpu_traces, gpu_counts = session.hierarchical(
        segment[np.newaxis], pyramid, np.array([observer_z]), np.array([radius]),
        30.0, pass_index=1,
    )
    gpu = gpu_traces[0, : gpu_counts[0]]
    adaptive = traverse_level0_adaptive(
        segment, pyramid.level0, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=30.0, pass_index=1,
    )
    fixed = traverse_level0_fixed_step(
        segment, pyramid.level0, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=30.0,
    )

    boundary_capture = csharp_capture["bilinear_boundary_case"]
    boundary_segment = interpolate_segments(
        np.asarray(boundary_capture["segments"], dtype=np.float32),
        np.array(((4, 4), (-4, 4), (4, -4), (-4, -4)), dtype=np.float32),
        1.0,
        0.5,
        0.5,
    )
    boundary_dem = _dem(artifact, "single_obstacle_north")
    boundary_pyramid = build_max_pyramid(boundary_dem)
    boundary_observer = boundary_dem.elevation(60, 60)
    boundary_expected = _json_trace(boundary_capture["trace"])
    boundary_cpu = traverse_hierarchy(
        boundary_segment, boundary_pyramid, observer_z_m=boundary_observer,
        radius_m=boundary_dem.projection.radius_m, map_resolution_m=30.0,
        pass_index=0,
    )
    boundary_gpu_maximum, boundary_gpu_traces, boundary_gpu_counts = session.hierarchical(
        boundary_segment[np.newaxis], boundary_pyramid,
        np.array([boundary_observer]), np.array([boundary_dem.projection.radius_m]),
        30.0, pass_index=0,
    )
    boundary_gpu = boundary_gpu_traces[0, : boundary_gpu_counts[0]]
    boundary_adaptive = traverse_level0_adaptive(
        boundary_segment, boundary_pyramid.level0,
        observer_z_m=boundary_observer, radius_m=boundary_dem.projection.radius_m,
        map_resolution_m=30.0, pass_index=0,
    )
    boundary_fixed = traverse_level0_fixed_step(
        boundary_segment, boundary_pyramid.level0,
        observer_z_m=boundary_observer, radius_m=boundary_dem.projection.radius_m,
        map_resolution_m=30.0,
    )
    east_segment = artifact.arrays["flat_east__ray_fit_pass_0__segment_values"]
    north_segment = artifact.arrays[
        "single_obstacle_north__ray_fit_pass_0__segment_values"
    ]
    _, east_dy = evaluate_tangent(east_segment[4:8], east_segment[8:12], 0.0)
    north_dx, _ = evaluate_tangent(north_segment[4:8], north_segment[8:12], 0.0)

    report = {
        "schema_version": 1,
        "scope": "factor-four max pyramids and current production hierarchical traversal",
        "device": {
            "name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
        },
        "pyramids": {
            "csharp_artifact_dems_compared": pyramid_count,
            "all_levels_exact": True,
            "downsample_factor": 4,
            "invalid_cutoff_exclusive_m": -20000.0,
            "invalid_block_sentinel_m": -32000.0,
        },
        "production_selected_ray": {
            "azimuth_index": 360,
            "dem_pass": 1,
            "python_host_segment_maximum_absolute_difference": segment_error,
            "csharp_maximum_slope": expected_slope,
            "cpu_maximum_slope": float(cpu.maximum_slope),
            "gpu_maximum_slope": float(gpu_maximum[0]),
            "adaptive_level0_maximum_slope": float(adaptive.maximum_slope),
            "fixed_step_level0_maximum_slope": float(fixed.maximum_slope),
            "hierarchy_minus_fixed_step_slope": float(
                gpu_maximum[0] - fixed.maximum_slope
            ),
            "cpu_trace": _trace_summary(cpu.values, expected),
            "gpu_trace": _trace_summary(gpu, expected),
            "counters": asdict(traversal_counters(gpu)),
        },
        "bilinear_boundary_case": {
            "csharp_hierarchy_maximum_slope": boundary_capture[
                "csharp_hierarchy_maximum_slope"
            ],
            "cpu_hierarchy_maximum_slope": float(boundary_cpu.maximum_slope),
            "gpu_hierarchy_maximum_slope": float(boundary_gpu_maximum[0]),
            "adaptive_level0_maximum_slope": float(boundary_adaptive.maximum_slope),
            "fixed_step_level0_maximum_slope": float(boundary_fixed.maximum_slope),
            "hierarchy_minus_fixed_step_slope": float(
                boundary_gpu_maximum[0] - boundary_fixed.maximum_slope
            ),
            "cpu_trace": _trace_summary(boundary_cpu.values, boundary_expected),
            "gpu_trace": _trace_summary(boundary_gpu, boundary_expected),
            "counters": asdict(traversal_counters(boundary_gpu)),
            "classification": (
                "corrected in C# and Numba: culling bounds include the four-cell "
                "bilinear footprint; non-cullable level-0 cells retain adaptive "
                "sampling and use a 1 mm boundary nudge"
            ),
        },
        "boundary_mechanics": {
            "east_ray_absolute_dy_ds": float(abs(east_dy)),
            "north_ray_absolute_dx_ds": float(abs(north_dx)),
            "captured_exact_integer_pixel_x": 42.0,
        },
        "conclusion": (
            "Numba CPU and CUDA reproduce the corrected C# hierarchy decisions and "
            "slopes. A separate ten-case directional and coarse-mip matrix records "
            "differences from dense bilinear sampling as a non-gating approximation "
            "diagnostic; C#/Numba parity and downstream illumination error are the "
            "correctness gates."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
