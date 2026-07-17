#!/usr/bin/env python3
"""Capture full subpatch, patch-edge, and multi-DEM CUDA parity evidence."""

from __future__ import annotations

import argparse
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
    ContractConfiguration,
    PyramidArrays,
    SegmentTensor,
    load_reference_artifact,
)
from lunarscout._numba_horizon.cuda_backend import CudaSession  # noqa: E402
from lunarscout._numba_horizon.generator import generate_patch_horizons  # noqa: E402
from lunarscout._numba_horizon.geometry import (  # noqa: E402
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    build_subpatch_segments,
)
from lunarscout._numba_horizon.hierarchy import traverse_hierarchy  # noqa: E402
from lunarscout._numba_horizon.kernel_math import sample_bilinear  # noqa: E402
from lunarscout._numba_horizon.pyramid import build_max_pyramid  # noqa: E402
from lunarscout._numba_horizon.subpatch import interpolate_pixel_segment  # noqa: E402


DATA = REPOSITORY / "tests" / "data" / "numba_horizon"


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite])))


def _dem(artifact, case_id: str, index: int) -> DemGrid:
    prefix = f"{case_id}__dem_{index}"
    return DemGrid(
        artifact.arrays[f"{prefix}__elevation_m"],
        artifact.arrays[f"{prefix}__geo_transform"],
        ProjectionParameters.from_array(
            artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
        ),
    )


def _selected_cpu_difference(
    gpu_passes, segments, pyramids, *, tile_column, tile_row, tile_width,
    subpatch_size, pixels, azimuths,
) -> tuple[float, int]:
    maximum = 0.0
    comparisons = 0
    for column, row in pixels:
        pixel = row * tile_width + column
        observer_z = sample_bilinear(
            pyramids[0].level0, tile_column + column, tile_row + row
        )
        for azimuth in azimuths:
            for pass_index, gpu in enumerate(gpu_passes):
                segment = interpolate_pixel_segment(
                    segments, pyramids[0], pyramids[pass_index],
                    tile_column=tile_column, tile_row=tile_row,
                    tile_width=tile_width, subpatch_size=subpatch_size,
                    pixel_column=column, pixel_row=row, azimuth=azimuth,
                    pass_index=pass_index,
                )
                active_map = pyramids[pass_index].map_parameters
                resolution = 0.5 * (
                    np.hypot(active_map[6], active_map[9])
                    + np.hypot(active_map[7], active_map[10])
                )
                cpu = traverse_hierarchy(
                    segment, pyramids[pass_index], observer_z_m=float(observer_z),
                    radius_m=float(pyramids[pass_index].projection_parameters[0]),
                    map_resolution_m=float(resolution), pass_index=pass_index,
                )
                gpu_value = float(gpu[pixel, azimuth])
                cpu_value = float(cpu.maximum_slope)
                if np.isneginf(gpu_value) and np.isneginf(cpu_value):
                    difference = 0.0
                elif np.isfinite(gpu_value) and np.isfinite(cpu_value):
                    difference = abs(gpu_value - cpu_value)
                else:
                    raise RuntimeError("CPU/GPU slope sentinel mismatch")
                maximum = max(maximum, difference)
                comparisons += 1
    return maximum, comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY / "docs" / "numba-horizon-phase-4e-subpatch.json",
    )
    arguments = parser.parse_args()
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    session = CudaSession()

    base = "single_pixel_multi_dem_production__horizon_buffer_fixture"
    production_dems = [_dem(artifact, base, index) for index in range(2)]
    production_pyramids = [
        PyramidArrays.from_artifact(artifact.arrays, f"{base}__dem_{index}__pyramid")
        for index in range(2)
    ]
    production_segments, _, _ = build_subpatch_segments(
        production_dems, tile_column=20, tile_row=20, tile_width=1,
        azimuth_count=1440, maximum_distance_m=1_000_000,
        observer_elevation_m=0, subpatch_size=8,
        grid_convergence=GridConvergenceInput(
            *map(float, artifact.arrays[f"{base}__grid_convergence"])
        ),
    )
    expected_passes = artifact.arrays[f"{base}__per_dem_slopes"].reshape(2, 1, 1440)
    production_passes = [
        session.subpatch_hierarchical_pass(
            production_segments, production_pyramids[0], production_pyramids[index],
            tile_column=20, tile_row=20, tile_width=1, tile_height=1,
            subpatch_size=8, pass_index=index,
        )
        for index in range(2)
    ]
    configuration = ContractConfiguration(
        tile_width=1, tile_height=1, azimuth_count=1440, subpatch_size=8,
        dem_count=2, primary_width=41, primary_height=41,
    )
    production_ids = np.broadcast_to(
        np.arange(2, dtype=np.int32), production_segments.shape[:-1]
    ).copy()
    generated = generate_patch_horizons(
        session,
        SegmentTensor(production_segments, production_ids, configuration),
        production_pyramids,
        tile_column=20,
        tile_row=20,
    )
    expected_final = artifact.arrays[f"{base}__final_slopes"]
    expected_degrees = artifact.arrays[f"{base}__final_degrees"]

    boundary_segments = artifact.arrays[
        "boundary_halo_multi_dem_16az__subpatch_fixture__segment_values"
    ]
    boundary_pyramids = [
        PyramidArrays.from_artifact(
            artifact.arrays, f"multi_dem_outer_obstacle_east__dem_{index}__pyramid"
        )
        for index in range(2)
    ]
    boundary_passes = [
        session.subpatch_hierarchical_pass(
            boundary_segments, boundary_pyramids[0], boundary_pyramids[index],
            tile_column=0, tile_row=0, tile_width=16, tile_height=16,
            subpatch_size=8, pass_index=index,
        )
        for index in range(2)
    ]
    boundary_error, boundary_count = _selected_cpu_difference(
        boundary_passes, boundary_segments, boundary_pyramids,
        tile_column=0, tile_row=0, tile_width=16, subpatch_size=8,
        pixels=((0, 0), (15, 0), (0, 15), (15, 15), (7, 7), (8, 8)),
        azimuths=(0, 4, 8, 12, 15),
    )

    projection = ProjectionParameters.from_array(
        artifact.arrays["flat_east__dem_0__pyramid__projection_parameters"]
    )
    full_dem = DemGrid(
        np.zeros((129, 129), dtype=np.float32),
        np.array((-1920.0, 30.0, 0.0, 1920.0, 0.0, -30.0), dtype=np.float64),
        projection,
    )
    full_pyramid = build_max_pyramid(full_dem)
    full_segments, _, _ = build_subpatch_segments(
        [full_dem], tile_column=0, tile_row=0, tile_width=128,
        azimuth_count=16, maximum_distance_m=2500, observer_elevation_m=0,
        subpatch_size=8, grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
    )
    full_pass = session.subpatch_hierarchical_pass(
        full_segments, full_pyramid, full_pyramid,
        tile_column=0, tile_row=0, tile_width=128, tile_height=128,
        subpatch_size=8, pass_index=0,
    )
    full_error, full_count = _selected_cpu_difference(
        [full_pass], full_segments, [full_pyramid],
        tile_column=0, tile_row=0, tile_width=128, subpatch_size=8,
        pixels=((0, 0), (127, 0), (0, 127), (127, 127), (63, 63), (64, 64)),
        azimuths=(0, 4, 8, 12, 15),
    )

    resolution_case = "multi_dem_different_resolutions"
    resolution_dems = [_dem(artifact, resolution_case, index) for index in range(2)]
    resolution_pyramids = [build_max_pyramid(dem) for dem in resolution_dems]
    resolution_segments, _, _ = build_subpatch_segments(
        resolution_dems, tile_column=20, tile_row=20, tile_width=1,
        azimuth_count=16, maximum_distance_m=2500, observer_elevation_m=0,
        subpatch_size=8, grid_convergence=GridConvergenceInput(0.0, 0.0, 0.0),
    )
    resolution_passes = [
        session.subpatch_hierarchical_pass(
            resolution_segments, resolution_pyramids[0], resolution_pyramids[index],
            tile_column=20, tile_row=20, tile_width=1, tile_height=1,
            subpatch_size=8, pass_index=index,
        )
        for index in range(2)
    ]
    resolution_error, resolution_count = _selected_cpu_difference(
        resolution_passes, resolution_segments, resolution_pyramids,
        tile_column=20, tile_row=20, tile_width=1, subpatch_size=8,
        pixels=((0, 0),), azimuths=tuple(range(16)),
    )

    report = {
        "schema_version": 1,
        "scope": "device subpatch interpolation, patch traversal, DEM accumulation, and degree conversion",
        "device": {
            "name": session.info.name,
            "compute_capability": list(session.info.compute_capability),
        },
        "csharp_production_fixture": {
            "pixels": 1,
            "azimuth_bins": 1440,
            "dem_passes": 2,
            "maximum_pass_slope_errors": [
                _max_abs(production_passes[index], expected_passes[index])
                for index in range(2)
            ],
            "maximum_final_slope_error": _max_abs(generated.slopes, expected_final),
            "maximum_final_degree_error": _max_abs(generated.degrees(), expected_degrees),
            "slope_sentinel_mismatches": int(np.count_nonzero(
                np.isneginf(generated.slopes) != np.isneginf(expected_final)
            )),
            "final_slope_sha256": _hash(generated.slopes),
            "final_degree_sha256": _hash(generated.degrees()),
        },
        "partial_patch_halo_fixture": {
            "shape": [16, 16, 16],
            "selected_cpu_gpu_comparisons": boundary_count,
            "maximum_selected_slope_error": boundary_error,
            "pass_sha256": [_hash(value) for value in boundary_passes],
        },
        "full_patch_fixture": {
            "shape": [128, 128, 16],
            "halo_centers": 324,
            "selected_cpu_gpu_comparisons": full_count,
            "maximum_selected_slope_error": full_error,
            "output_sha256": _hash(full_pass),
        },
        "different_resolution_fixture": {
            "map_resolutions_m": [30.0, 60.0],
            "selected_cpu_gpu_comparisons": resolution_count,
            "maximum_selected_slope_error": resolution_error,
            "pass_sha256": [_hash(value) for value in resolution_passes],
        },
        "near_field_reference_merge": {
            "current_public_python_path_enabled": False,
            "required_for_initial_replacement_parity": False,
            "status": (
                "deferred optional native mode; adopting the Numba backend must either "
                "keep it explicitly unsupported or evaluate it as separate scope"
            ),
        },
        "qualification": (
            "This is kernel correctness evidence. It is not a production scheduler, "
            "streaming pipeline, file writer, or performance benchmark."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
