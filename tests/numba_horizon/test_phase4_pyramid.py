from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pytest

from lunarscout._numba_horizon.contract import load_reference_artifact
from lunarscout._numba_horizon.contract import PyramidArrays
from lunarscout._numba_horizon.cuda_backend import CudaSession
from lunarscout._numba_horizon.geometry import (
    DemGrid,
    GridConvergenceInput,
    ProjectionParameters,
    build_subpatch_segments,
)
from lunarscout._numba_horizon.fixed_step import (
    traverse_level0_adaptive,
    traverse_level0_fixed_step,
)
from lunarscout._numba_horizon.hierarchy import (
    _bilinear_bound,
    traversal_counters,
    traverse_hierarchy,
)
from lunarscout._numba_horizon.kernel_math import evaluate_tangent, interpolate_segments
from lunarscout._numba_horizon.pyramid import (
    build_max_pyramid,
    load_max_pyramid_cache,
)


DATA = Path(__file__).resolve().parents[1] / "data" / "numba_horizon"


def test_python_max_pyramids_match_every_csharp_case() -> None:
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    for case in artifact.metadata["cases"]:
        for dem_info in case["dems"]:
            prefix = f"{case['id']}__dem_{dem_info['index']}"
            dem = DemGrid(
                artifact.arrays[f"{prefix}__elevation_m"],
                artifact.arrays[f"{prefix}__geo_transform"],
                ProjectionParameters.from_array(
                    artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
                ),
            )
            actual = build_max_pyramid(dem)
            np.testing.assert_array_equal(
                actual.levels,
                artifact.arrays[f"{prefix}__pyramid__level_metadata"],
            )
            np.testing.assert_array_equal(
                actual.map_parameters,
                artifact.arrays[f"{prefix}__pyramid__map_parameters"],
            )
            np.testing.assert_array_equal(
                actual.projection_parameters,
                artifact.arrays[f"{prefix}__pyramid__projection_parameters"],
            )
            for level, (_, offset, width, height) in enumerate(actual.levels):
                expected = artifact.arrays[f"{prefix}__pyramid__level_{level}"]
                if level == 0:
                    np.testing.assert_array_equal(actual.level0, expected)
                else:
                    np.testing.assert_array_equal(
                        actual.mips[offset : offset + width * height].reshape(height, width),
                        expected,
                    )


def test_invalid_value_pyramid_matches_csharp_cutoff_and_sentinel() -> None:
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    fixture = artifact.metadata["pyramid_fixtures"][0]
    prefix = f"{fixture['id']}__pyramid_fixture"
    elevation = artifact.arrays[f"{prefix}__elevation_m"]
    map_parameters = artifact.arrays[f"{prefix}__map_parameters"]
    transform = np.ascontiguousarray(map_parameters[5:11], dtype=np.float64)
    projection = ProjectionParameters.from_array(
        artifact.arrays[f"{prefix}__projection_parameters"]
    )
    actual = build_max_pyramid(DemGrid(elevation, transform, projection))
    for level in range(1, len(actual.levels)):
        _, offset, width, height = actual.levels[level]
        expected = artifact.arrays[f"{prefix}__level_{level}"]
        np.testing.assert_array_equal(
            actual.mips[offset : offset + width * height].reshape(height, width), expected
        )


def test_load_max_pyramid_cache_reads_csharp_float_payload(tmp_path: Path) -> None:
    elevation = np.arange(35, dtype=np.float32).reshape(5, 7)
    dem = DemGrid(
        elevation,
        np.array((0.0, 2.0, 0.0, 0.0, 0.0, -2.0), dtype=np.float64),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    expected = build_max_pyramid(dem)
    cache = tmp_path / "dem.pyr.bin"
    expected.mips.tofile(cache)

    actual = load_max_pyramid_cache(dem, cache)

    np.testing.assert_array_equal(actual.level0, expected.level0)
    np.testing.assert_array_equal(actual.mips, expected.mips)
    np.testing.assert_array_equal(actual.levels, expected.levels)
    np.testing.assert_array_equal(actual.map_parameters, expected.map_parameters)
    np.testing.assert_array_equal(
        actual.projection_parameters, expected.projection_parameters
    )


def test_load_max_pyramid_cache_rejects_wrong_length(tmp_path: Path) -> None:
    dem = DemGrid(
        np.zeros((5, 7), dtype=np.float32),
        np.array((0.0, 2.0, 0.0, 0.0, 0.0, -2.0), dtype=np.float64),
        ProjectionParameters(1_737_400.0, -np.pi / 2.0, 0.0, 1.0, 0.0, 0.0),
    )
    cache = tmp_path / "bad.pyr.bin"
    np.zeros(1, dtype=np.float32).tofile(cache)

    with pytest.raises(ValueError, match="expected"):
        load_max_pyramid_cache(dem, cache)


def test_bilinear_culling_bound_handles_edges_and_invalid_neighbors() -> None:
    level0 = np.array(
        ((1.0, 2.0, 3.0), (4.0, np.nan, -32000.0), (7.0, 8.0, 9.0)),
        dtype=np.float32,
    )
    pyramid = PyramidArrays(
        level0,
        np.empty(0, dtype=np.float32),
        np.array(((0, 0, 3, 3),), dtype=np.int32),
        np.zeros(11, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
    )
    assert _bilinear_bound(pyramid, 0, 0, 0) == np.float32(4.0)
    assert _bilinear_bound(pyramid, 0, 1, 0) == np.float32(3.0)
    assert _bilinear_bound(pyramid, 0, 1, 1) == np.float32(9.0)
    assert _bilinear_bound(pyramid, 0, 2, 2) == np.float32(9.0)

    invalid = PyramidArrays(
        np.full((2, 2), -32000.0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.array(((0, 0, 2, 2),), dtype=np.int32),
        np.zeros(11, dtype=np.float32),
        np.zeros(6, dtype=np.float32),
    )
    assert _bilinear_bound(invalid, 0, 0, 0) == np.float32(-32000.0)


def _production_hierarchy_inputs():
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    base = "single_pixel_multi_dem_production__horizon_buffer_fixture"
    dems = []
    for index in range(2):
        prefix = f"{base}__dem_{index}"
        dems.append(
            DemGrid(
                artifact.arrays[f"{prefix}__elevation_m"],
                artifact.arrays[f"{prefix}__geo_transform"],
                ProjectionParameters.from_array(
                    artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
                ),
            )
        )
    convergence = GridConvergenceInput(
        *map(float, artifact.arrays[f"{base}__grid_convergence"])
    )
    segments, _, _ = build_subpatch_segments(
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
    csharp_capture = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )
    csharp_segments = np.asarray(csharp_capture["segments"], dtype=np.float32)
    np.testing.assert_allclose(segments[360, :, 1], csharp_segments, rtol=0, atol=6e-8)
    shifts = np.array(((4, 4), (-4, 4), (4, -4), (-4, -4)), dtype=np.float32)
    segment = interpolate_segments(
        csharp_segments,
        shifts,
        1.0,
        0.5,
        0.5,
    )
    pyramid = PyramidArrays.from_artifact(
        artifact.arrays, f"{base}__dem_1__pyramid"
    )
    return (
        artifact,
        base,
        segment,
        pyramid,
        dems[0].elevation(20, 20),
        dems[1].projection.radius_m,
    )


def _assert_trace_matches_csharp(
    actual, expected, *, sample_elevation_atol=2e-8, slope_atol=2e-8
):
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual[:, 2:5], expected[:, 2:5])
    np.testing.assert_array_equal(actual[:, 11], expected[:, 11])
    np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=0, atol=4e-6)
    np.testing.assert_allclose(actual[:, 1], expected[:, 1], rtol=0, atol=6e-3)
    np.testing.assert_allclose(actual[:, 5:7], expected[:, 5:7], rtol=0, atol=2e-4)
    np.testing.assert_allclose(actual[:, 7], expected[:, 7], rtol=0, atol=2e-8)
    np.testing.assert_allclose(
        actual[:, 8], expected[:, 8], rtol=0, atol=sample_elevation_atol,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual[:, 9], expected[:, 9], rtol=0, atol=slope_atol, equal_nan=True
    )
    np.testing.assert_allclose(actual[:, 10], expected[:, 10], rtol=0, atol=6e-6)


def _trace_from_capture(capture):
    fields = (
        "s_km", "true_distance_m", "level", "cell_x", "cell_y", "pixel_x",
        "pixel_y", "maximum_elevation_m", "sample_elevation_m", "sample_slope",
        "advance_km", "action",
    )
    return np.asarray(
        [[np.nan if row[field] is None else row[field] for field in fields]
         for row in capture["trace"]],
        dtype=np.float32,
    )


def _synthetic_hierarchy_inputs(case_id: str):
    artifact = load_reference_artifact(
        DATA / "phase1_reference_rays.json", DATA / "phase1_reference_rays.npz"
    )
    case = next(item for item in artifact.metadata["cases"] if item["id"] == case_id)
    prefix = f"{case_id}__dem_0"
    dem = DemGrid(
        artifact.arrays[f"{prefix}__elevation_m"],
        artifact.arrays[f"{prefix}__geo_transform"],
        ProjectionParameters.from_array(
            artifact.arrays[f"{prefix}__pyramid__projection_parameters"]
        ),
    )
    observer = case["observer"]
    observer_z = float(dem.elevation(observer["pixel_x"], observer["pixel_y"]))
    observer_z += float(observer["elevation_m"])
    fit = case["ray_fit_passes"][0]
    return (
        artifact.arrays[f"{case_id}__ray_fit_pass_0__segment_values"],
        build_max_pyramid(dem),
        observer_z,
        dem.projection.radius_m,
        fit["map_resolution_m"],
    )


@pytest.mark.parametrize(
    "case_id",
    ("flat_east", "single_obstacle_north", "nodata_hole_east", "entirely_nodata_east_ray"),
)
def test_cpu_hierarchy_uses_conservative_bilinear_bound(case_id: str) -> None:
    segment, pyramid, observer_z, radius, resolution = _synthetic_hierarchy_inputs(case_id)
    adaptive = traverse_level0_adaptive(
        segment, pyramid.level0, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=resolution, pass_index=0,
    )
    hierarchy = traverse_hierarchy(
        segment, pyramid, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=resolution, pass_index=0,
    )
    if case_id == "single_obstacle_north":
        csharp = json.loads(
            (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
        )["bilinear_boundary_case"]
        assert hierarchy.maximum_slope == pytest.approx(
            csharp["csharp_hierarchy_maximum_slope"], abs=1e-6
        )
        fixed = traverse_level0_fixed_step(
            segment, pyramid.level0, observer_z_m=observer_z, radius_m=radius,
            map_resolution_m=resolution,
        )
        assert hierarchy.maximum_slope >= fixed.maximum_slope
        assert 0.03 < float(hierarchy.maximum_slope - adaptive.maximum_slope) < 0.04
    else:
        assert hierarchy.maximum_slope == pytest.approx(
            float(adaptive.maximum_slope), abs=3e-8
        )
    counters = traversal_counters(hierarchy.values)
    assert counters.iterations > 0
    assert counters.level0_samples > 0 or not np.isfinite(hierarchy.maximum_slope)
    if case_id == "entirely_nodata_east_ray":
        assert not np.isfinite(hierarchy.maximum_slope)


def test_hierarchy_trace_covers_cardinal_tangents_and_exact_block_boundary() -> None:
    east_segment, _, _, _, _ = _synthetic_hierarchy_inputs("flat_east")
    north_segment, _, _, _, _ = _synthetic_hierarchy_inputs("single_obstacle_north")
    _, east_dy = evaluate_tangent(east_segment[4:8], east_segment[8:12], 0.0)
    north_dx, _ = evaluate_tangent(north_segment[4:8], north_segment[8:12], 0.0)
    assert abs(east_dy) < 1e-8
    assert abs(north_dx) < 1e-8

    artifact, base, _, _, _, _ = _production_hierarchy_inputs()
    trace = artifact.arrays[f"{base}__traversal_trace"]
    assert trace[1, 5] == 42.0
    assert trace[1, 3] == 42.0
    assert trace[1, 11] == 1.0


def _bilinear_boundary_inputs():
    capture = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )["bilinear_boundary_case"]
    segments = np.asarray(capture["segments"], dtype=np.float32)
    segment = interpolate_segments(
        segments,
        np.array(((4, 4), (-4, 4), (4, -4), (-4, -4)), dtype=np.float32),
        1.0,
        0.5,
        0.5,
    )
    _, pyramid, observer_z, radius, resolution = _synthetic_hierarchy_inputs(
        "single_obstacle_north"
    )
    expected = _trace_from_capture(capture)
    return segment, pyramid, observer_z, radius, resolution, expected, capture


def test_cpu_hierarchy_reproduces_corrected_csharp_bilinear_boundary_trace() -> None:
    segment, pyramid, observer_z, radius, resolution, expected, capture = (
        _bilinear_boundary_inputs()
    )
    actual = traverse_hierarchy(
        segment, pyramid, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=resolution, pass_index=0,
    )
    _assert_trace_matches_csharp(
        actual.values, expected, sample_elevation_atol=2e-3, slope_atol=2e-6
    )
    assert actual.maximum_slope == pytest.approx(
        capture["csharp_hierarchy_maximum_slope"], abs=1e-6
    )


def test_production_hierarchy_and_adaptive_paths_are_measured_separately() -> None:
    _, _, segment, pyramid, observer_z, radius = _production_hierarchy_inputs()
    hierarchy = traverse_hierarchy(
        segment, pyramid, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=30.0, pass_index=1,
    )
    adaptive = traverse_level0_adaptive(
        segment, pyramid.level0, observer_z_m=observer_z, radius_m=radius,
        map_resolution_m=30.0, pass_index=1,
    )
    capture = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )["production_hierarchy_case"]
    assert hierarchy.maximum_slope == pytest.approx(
        capture["csharp_hierarchy_maximum_slope"], abs=3e-8
    )
    assert adaptive.maximum_slope == pytest.approx(0.10548653, abs=3e-8)
    assert hierarchy.maximum_slope != adaptive.maximum_slope
    assert traversal_counters(hierarchy.values).culled_blocks > 0


def test_cpu_hierarchy_reproduces_csharp_production_trace() -> None:
    artifact, base, segment, pyramid, observer_z, radius = _production_hierarchy_inputs()
    actual = traverse_hierarchy(
        segment,
        pyramid,
        observer_z_m=observer_z,
        radius_m=radius,
        map_resolution_m=30.0,
        pass_index=1,
    )
    capture = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )["production_hierarchy_case"]
    expected = _trace_from_capture(capture)
    _assert_trace_matches_csharp(
        actual.values, expected, sample_elevation_atol=0.5, slope_atol=2e-5
    )
    assert actual.maximum_slope == pytest.approx(
        capture["csharp_hierarchy_maximum_slope"], abs=3e-8
    )


@pytest.mark.skipif(
    os.environ.get("LUNARSCOUT_REQUIRE_NUMBA_CUDA") != "1",
    reason="set LUNARSCOUT_REQUIRE_NUMBA_CUDA=1 for the explicit real-GPU probe",
)
def test_cuda_hierarchy_reproduces_csharp_production_trace() -> None:
    artifact, base, segment, pyramid, observer_z, radius = _production_hierarchy_inputs()
    maximum, traces, counts = CudaSession().hierarchical(
        segment[np.newaxis], pyramid, np.array([observer_z]), np.array([radius]),
        30.0, pass_index=1,
    )
    capture = json.loads(
        (DATA / "phase4d_production_segments.json").read_text(encoding="utf-8")
    )["production_hierarchy_case"]
    expected = _trace_from_capture(capture)
    actual = traces[0, : counts[0]]
    _assert_trace_matches_csharp(actual, expected)
    assert maximum[0] == pytest.approx(
        capture["csharp_hierarchy_maximum_slope"], abs=3e-8
    )

    segment, pyramid, observer_z, radius, resolution, expected, capture = (
        _bilinear_boundary_inputs()
    )
    maximum, traces, counts = CudaSession().hierarchical(
        segment[np.newaxis], pyramid, np.array([observer_z]), np.array([radius]),
        resolution, pass_index=0,
    )
    _assert_trace_matches_csharp(
        traces[0, : counts[0]], expected,
        sample_elevation_atol=6e-4, slope_atol=1e-6,
    )
    assert maximum[0] == pytest.approx(
        capture["csharp_hierarchy_maximum_slope"], abs=1e-6
    )
